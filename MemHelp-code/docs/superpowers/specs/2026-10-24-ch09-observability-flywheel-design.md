# 第九章设计 · 可观测性与数据飞轮（Langfuse / 证据置信度 / 问题池 / 审核队列 / 评估趋势）

- 日期：2026-10-24
- 状态：已批准（2026-10-24）；Langfuse 部署方式按用户 m00504 改为 Cloud，本地容器已清理
- 关联：`dev-notes/ch09.md`、`sql/ch09-ddl.sql`、`docs/superpowers/plans/2026-10-24-ch09-observability-flywheel-plan.md`（待写）
- 前置章节：ch03（知识库落库）、ch04（评估集 / 低置信度池）、ch05（图执行器 / 置信度闸）、ch06（意图路由）、ch08（工具执行引擎）

---

## 1. 目标与非目标

### 1.1 目标

1. **Langfuse 全链路**：使用 Langfuse Cloud（日本区，凭据在 `.env`，不入库）；每条聊天请求一棵完整 trace 树，能点开看到每个节点的 prompt、模型调用、工具调用、检索结果、token 消耗与耗时。
2. **按意图看成本**：意图识别结果写进 trace，能按意图维度汇总 token / 成本，回答「钱花在哪类问题上」。
3. **三个入口统一入池**：检索证据置信度低、生成前自评不够、用户 👎，全部写 `low_confidence_questions`，各自标记 `source`；入池时把当轮召回片段（Top 条原文 + 得分）快照进新列 `retrieved_chunks`。
4. **正式证据置信度闸**：用 `evidence_confidence`（精排 Top1 分、有效证据数、Top1-Top2 分差）替代 ch05 的最简阈值版；阈值在 ch04 评估集上留一法校准，不拍脑袋；闸的位置仍在知识检索之后、Agent 之前。
5. **飞轮流水线**：问题标准化 → 对 `review_queue` 待审队列查重 → 命中累加次数 / 未命中新建 → 人工审核 → 通过后走 ch03 落库 + 向量化。
6. **评估趋势**：复用 ch04 评估集与指标，定期跑，每轮写 `eval_runs`，按时间展示趋势，指标下滑一眼可见。
7. **审核后台**：列出标准化问题、出现次数、示例答案；点开详情能看到归并进来的用户原话与当轮召回片段；可通过/驳回。

### 1.2 明确不做

- 低置信度问题按主题归类的微调分类器（下一步）。
- Langfuse 之外的链路存储（不自建 trace 表）。
- 多租户 / 权限系统；审核页无登录（内网演示）。
- 自动生成核准答案：`approved_answer` 必须人工确认或采用人工修改后的示例答案。
- 评估集的扩写与重标注（沿用 ch04 的 48 条）。

---

## 2. 关键决策

| # | 决策 | 选择 | 理由 / 代价 |
|---|---|---|---|
| D1 | Langfuse 部署 | **Langfuse Cloud**（`https://jp.cloud.langfuse.com`），`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` 写在 `.env`（不入库）；用户 m00504 决定不再自部署 | 已用云端凭据实测 `GET /api/public/projects` 200（项目 `My Project`）；本地 Docker 容器/镜像/卷已删除，仓库不再保留 `deploy/langfuse/` |
| D2 | SDK 版本 | `langfuse==5.0.4`（已 `uv add`） | 实测与 server 4.69 兼容；v5 是 OTel 架构，`CallbackHandler` 来自 `langfuse.langchain` |
| D3 | 回调挂载点 | `builder.compile(...).with_config({"callbacks": [handler]})`，**只挂一次** | 探针证明返回仍是 `CompiledStateGraph`，`aget_state` / `checkpointer` / `stream_mode="custom"` 全部保留，无需代理对象 |
| D4 | trace 根节点 | 在 `runner.astream_turn` 外层开一个 `chat-turn` 根 span，结束时补 `intent / route / gate / token_usage / stop_reason` | 意图在图中途才产生；根 span 一直活到本轮结束才能回写元数据 |
| D5 | 意图成本维度 | 意图已知后 `propagate_attributes(tags=["intent:<x>"], metadata={"intent": x})` 包住 Agent 执行；统计走 Metrics API v2 的 `tags` 维度 | 实测 Metrics API **不支持 `metadata.*` 维度**，只支持 `tags` 等固定维度；tags 能聚合 `totalTokens` / `totalCost` |
| D6 | 置信度模型 | 三特征逻辑回归（纯 Python 梯度下降，不引入 sklearn），阈值用 ch04 评估集留一法选 balanced accuracy 最高点 | 样本仅 48 条（8 条 refuse），留一法避免自拟合自评；模型/阈值/混淆矩阵落盘可审计 |
| D7 | 闸的输入 | 新增 `RagPipeline.retrieve_for_gate()`：只做理解+检索+top_n，**不提前按 0.3 拒答**，把低分证据也交出来 | 否则 `retrieve_only` 在低分时返回空证据，Top1/Top2 分差与有效证据数无从计算；旧 `retrieve_only` 保留给工具路径与既有测试 |
| D8 | 自评入口 | 闸通过后、进 Agent 前调用 `SufficiencyJudge`；不够则兜底 + 落池 `self_check` | 用户已确认（m00310）；主图原来不跑自评，ch04 的 self_check 只在旧工具路径，必须补上才闭环 |
| D9 | 飞轮时机 | 入池后**同步**做标准化+查重+入队，再结束请求 | 用户已确认（m00216）：刷新审核页立刻可见；代价是每轮多一次 LLM 调用（1-3 秒） |
| D10 | 查重范围 | 仅对 `review_status='待审'` 的最近 50 条做 LLM 判重；命中只累加次数、不新建行 | 已审核/已驳回的不再吸并；50 条上限控制 prompt 长度 |
| D11 | 审核通过落库 | 复用 `knowledge.upsert_chunk`（自然键幂等）+ 新增按 id 向量化 `vectorize_ids` | ch03 的落库状态机不能绕开；只向量化新块，不扫全表 pending |
| D12 | 👎 快照回捞 | 用 LangGraph `aget_state_history` 按 `session_id + user_message` 找最近一轮，读该轮 state 新增的 `retrieval_snapshot` | 消息行只存引用、不存分数；state 里新增带分数的快照是唯一可靠的「当轮」来源。真没走检索（闲聊/业务数据）则空 |
| D13 | 评估调度 | FastAPI lifespan 内 `asyncio` 周期任务，默认 24h；手动触发沿用后台子进程跑 `evals/run_ch04_evals.py` | 不引入 Celery/APScheduler；评估是重任务，子进程隔离，不阻塞事件循环 |
| D14 | 评估指标口径 | 固定 `hybrid_rerank` 策略：检索 `recall@5` / `mrr@10`，生成 `faithfulness` / `refusal_accuracy`，另存四桶 | 与 ch04 报告一致，趋势可比；其他策略不写趋势，避免一个指标多条线 |
| D15 | review 表外键 | `low_confidence_questions.matched_review_id` → `review_queue.id`，`ON DELETE SET NULL` | 与 DDL 一致；审核行被删不连带丢原话 |

---

## 3. 架构与数据流

```
聊天请求（POST /api/chat/stream）
  │
  ├─ Langfuse 根 span: chat-turn（session_id, question）
  │    └─ LangGraph（编译期挂 CallbackHandler）
  │         load_history → begin_turn → resolve_coref → classify_intent
  │           └─ propagate_attributes(tags=["intent:xxx"])   ← 意图成本维度
  │         route_intent
  │         knowledge_retrieve（手动 retriever span：query / 证据 / 分数 / degraded）
  │         confidence_gate（evidence_confidence）
  │           ├─ 不过 → fallback + 入池(source=retrieval_low_conf, 快照) → 飞轮
  │           └─ 过 → sufficiency_judge
  │                 ├─ 不够 → fallback + 入池(source=self_check, 快照) → 飞轮
  │                 └─ 够 → agent（模型 generation 自动 trace，工具调用手动 span）
  │         log_turn（持久化 + 输出 message_id）
  │
  └─ 根 span.update(metadata={intent, route, gate, tokens, stop_reason})

入池 → FlywheelService.ingest(low_conf_id)（同步）
  标准化（LLM）→ 查重（LLM vs 待审队列）→ 新建 review_queue / occurrence_count+1
  → 回写 matched_review_id

审核页 → 通过 → review_queue.status=通过 + upsert_chunk + 向量化 → 下一轮可检索

👎 → POST /api/feedback → 找回当轮 state 快照 → 入池(source=user_feedback) → 飞轮

定时/手动 → run_ch04_evals.py → eval_runs(triggered_by, dataset_size, metrics, created_at) → 趋势页
```

### 3.1 新增 / 改动模块

```
app/observability/
  __init__.py
  tracing.py            # Langfuse 客户端、CallbackHandler、根 span、意图 tag、工具/检索 span
  intent_cost.py        # Metrics API v2 → 按意图汇总 tokens/cost
app/agent/
  evidence_confidence.py# 三特征、逻辑回归模型、加载/预测、校准算法
  gate.py               # 保留 evaluate_gate；新增 evaluate_evidence_gate
  graph.py              # retrieve_for_gate / 自评节点 / 入池带快照 / 飞轮调用
  react.py              # 工具调用手动 span
app/flywheel/
  __init__.py
  normalizer.py         # 口语 → 标准 FAQ 问题 + 示例答案
  deduper.py            # 与待审队列判同义
  service.py            # ingest 编排 + 审核通过落库
  snapshot.py           # 👎 从 checkpoint 历史回捞当轮检索快照
app/eval_runs.py        # 跑评估、写 eval_runs、趋势
app/eval_schedule.py    # lifespan 周期调度
app/db/models.py        # ReviewQueue / EvalRun / LowConfidenceQuestion 新列
app/db/repositories.py  # 两个新仓库 + 低置信度仓库扩展
app/api.py              # 反馈、审核队列、评估趋势、意图成本端点 + 页面路由
app/config.py           # Langfuse / 飞轮 / 评估调度配置
web/review.html|js      # 审核后台（Vibe Coding）
web/app.js              # 👎 改为调后端
web/eval-trend.html|js  # 评估趋势页（简单表格 + 环比箭头）
sql/ch09-ddl.sql        # 已有，直接执行
scripts/calibrate_confidence.py   # 跑 ch04 评估集 → 模型 artifact + 报告
scripts/intent_cost.py            # CLI 打印意图成本
scripts/seed_ch09_demo.py         # 可选：准备演示数据
```

---

## 4. 数据模型

### 4.1 `review_queue`（DDL 已给，ORM 新增）

| 列 | 类型 | 说明 |
|---|---|---|
| id | BIGINT UNSIGNED PK | |
| normalized_question | VARCHAR(512) NOT NULL | 标准化 FAQ 问题 |
| ai_suggested_answer | TEXT NULL | 模型示例答案 |
| occurrence_count | INT UNSIGNED DEFAULT 1 | 查重命中累加 |
| review_status | ENUM('待审','通过','驳回') DEFAULT '待审' | |
| approved_answer | TEXT NULL | 人工核准答案 |
| created_at / updated_at | DATETIME | updated_at 随次数累加刷新 |

索引：`idx_review_status`。ORM 中索引名改成 `idx_review_queue_status`（SQLite 索引名全局唯一，避免与既有 `idx_status` 冲突），DDL 保持用户给定名字 —— 与 ch02/ch03 的既有处理一致。

### 4.2 `eval_runs`（DDL 已给，ORM 新增）

| 列 | 类型 | 说明 |
|---|---|---|
| id | BIGINT UNSIGNED PK | |
| triggered_by | ENUM('定时','手动') DEFAULT '定时' | |
| dataset_size | INT UNSIGNED NOT NULL | 本轮条数 |
| metrics | JSON NOT NULL | 见 §8 结构 |
| created_at | DATETIME | 趋势横轴 |

索引：`idx_created_at`。

### 4.3 `low_confidence_questions` 扩展

- `retrieved_chunks JSON NULL`：`[{"n":1,"chunk_id":123,"section_path":"...","category":"...","quote":"...","score":0.87}, ...]`；没走检索时为 `NULL`。
- `matched_review_id BIGINT UNSIGNED NULL`：归并到的审核行；未处理/新建失败为 `NULL`。
- 索引 `idx_matched_review_id` + 外键 `fk_lcq_review`。

ORM 需与 DDL 对齐；SQLite 测试库用 `Base.metadata.create_all` 建新表。

---

## 5. 可观测性设计

### 5.1 配置（`app/config.py` + `.env.example`）

```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
LANGFUSE_ENVIRONMENT=dev
LANGFUSE_FLUSH_AT=20
LANGFUSE_FLUSH_INTERVAL=3
```

`LANGFUSE_ENABLED` 默认 `true`；`enabled=false` 或三个连接项（public key / secret key / base URL）任一缺失时，所有 tracing helper 变成 no-op；聊天链路绝不因为 Langfuse 不可用而失败。当前 `.env` 只配了三个连接项（未写 `LANGFUSE_ENABLED`），走默认开启。

### 5.2 `app/observability/tracing.py`

```python
@dataclass(frozen=True)
class TracingState:
    enabled: bool
    client: Langfuse | None
    handler: CallbackHandler | None

def init_tracing(settings: Settings) -> TracingState: ...          # lifespan 调一次
def shutdown_tracing(state: TracingState) -> None: ...             # flush
def turn_span(*, session_id: str, question: str) -> ContextManager[Any]: ...  # no-op 兼容
def tag_intent(intent: str, *, confidence: float) -> ContextManager[None]: ...
def observation(name: str, *, as_type: str, input: Any = None) -> ContextManager[Any]: ...
```

- `turn_span` 创建 `chat-turn` 根 span：`input={"session_id","question"}`、`metadata={"session_id":...}`。
- `tag_intent` 内部 `propagate_attributes(tags=[f"intent:{intent}"], metadata={"intent": intent, "intent_confidence": confidence})`；在 `_run_agent_node` 开头进入，覆盖 Agent 内所有 generation/tool span。
- `observation` 供检索与工具用：`with observation("knowledge_retrieve", as_type="retriever", input={"query":...}) as span: ... span.update(output=..., metadata=...)`。
- 根 span 在 `astream_turn` 的 `finally` 里 `span.update(output=done, metadata={...})`，并 `client.flush()` 只在进程退出时做。

### 5.3 编译期挂回调

`build_agent_graph(*, checkpointer=None)` 末尾：

```python
compiled = builder.compile(checkpointer=checkpointer or memory_checkpointer())
tracing = init_tracing(get_settings())   # 已初始化则取单例
if tracing.handler is not None:
    compiled = compiled.with_config({"callbacks": [tracing.handler]})
return compiled
```

实测 `with_config` 返回 `CompiledStateGraph`，状态 API 与 custom stream 不受影响。

### 5.4 意图成本统计

新增 `GET /api/intent-cost?days=7` 与 `scripts/intent_cost.py`：

1. `GET /api/public/v2/metrics`，`query={"view":"observations","metrics":[{"measure":"totalTokens","aggregation":"sum"},{"measure":"totalCost","aggregation":"sum"}],"dimensions":[{"field":"tags"}],"fromTimestamp":...,"toTimestamp":...}`（`count` 不是已验证的 measure，调用次数用返回行数/事件数近似，不写进契约）。
2. 过滤 tags 中含 `intent:` 前缀的行，按意图聚合；同一行可能带多个 tag（如 `intent:refund` + 其他），以 `intent:` 开头的那一个为准。
3. 返回 `[{intent, tokens, cost, count}]`，按 tokens 降序；页面/CLI 直接展示「最烧钱意图」。

> 为什么不用 `metadata.intent` 维度：实测 400，`/api/public/v2/metrics` 的 dimensions 枚举不含 metadata。trace 详情里仍能看到 `metadata.intent`（根 span metadata），成本聚合用 tags。

---

## 6. 证据置信度闸

### 6.1 特征

对 `retrieve_for_gate()` 返回的证据列表（精排分降序）：

- `top1 = evidence[0].score`
- `top2 = evidence[1].score if len>1 else 0.0`
- `margin = max(0.0, top1 - top2)`
- `effective_count = sum(1 for e in evidence if e.score >= 0.5 * top1)`（top1>0 时；否则 0）

`effective_count` 用相对比例而非绝对 0.3：绝对相关度已由 top1 承载，相对计数回答的是「只有一块命中，还是多块互相印证」。

### 6.2 模型

```python
# app/agent/evidence_confidence.py
@dataclass(frozen=True)
class EvidenceFeatures:
    top1: float
    top2: float
    margin: float
    effective_count: int

@dataclass(frozen=True)
class ConfidenceModel:
    weights: tuple[float, float, float]   # top1, margin, effective_count
    intercept: float
    threshold: float
    version: str

    def score(self, f: EvidenceFeatures) -> float: ...   # sigmoid
    def passes(self, f: EvidenceFeatures) -> bool: ...   # score >= threshold
```

- 标准化用校准集均值/方差，参数一并写进 artifact。
- 模型 artifact：`evals/ch09_confidence_model.json`，字段 `version / trained_at / dataset / normalization / weights / intercept / threshold / loo_metrics / confusion_matrix`。
- 加载失败或文件缺失 → 退回 ch05 的 `evaluate_gate`（阈值 0.3），并记 warning；功能不中断。

### 6.3 校准流程（`scripts/calibrate_confidence.py`）

1. 读 `evals/ch04_cases.jsonl`（跳过 `#` 注释行），48 条：40 条 `behavior=answer` 记正样本，8 条 `refuse` 记负样本。
2. 对每条用例调用 `RagPipeline.retrieve_for_gate(query)`，采特征；检索异常记失败并报告，不静默丢样本。
3. **留一法**：每次留 1 条，用其余 47 条拟合逻辑回归（梯度下降，固定随机种子，3000 轮，学习率 0.1，L2=1e-3），预测留出样本；遍历 0.05~0.95 阈值，取 balanced accuracy 最高点。
4. 用全部 48 条重训最终模型；输出混淆矩阵、per-bucket 指标、每条用例的预测分。
5. 写 `evals/ch09_confidence_model.json` 和 `dev-notes/ch09-confidence-calibration.md`（人读报告）。
6. 若 LOO balanced accuracy < 0.70：不发布模型，报告中标注「保持 ch05 阈值闸」，并继续调特征/样本；**不硬上**。

### 6.4 闸的接入

- `RagPipeline` 新增：

```python
async def retrieve_for_gate(self, query: str, *, category=None) -> RetrievalOutcome:
    """理解+检索+top_n，不做分数阈值拒答；0 命中才 refused。"""
```

- `graph._knowledge_retrieve` 改调 `retrieve_for_gate`，并把原始证据（含 score）同时写入：
  - `evidence`（沿用 `evidence_payload`，给 Agent/引用用，不含分数）
  - `retrieval_snapshot`（新 state 字段，含分数，供落池/👎 回捞）
- `graph._confidence_gate`：
  - 0 命中 → 不过；
  - `rerank_degraded` → 直接通过（分数不可比），metadata 记 `confidence_degraded=true`；
  - 否则算特征，调 `ConfidenceModel.passes`；不过则 `low_confidence.add(..., retrieved_chunks=snapshot)` 后交给飞轮。
- 保留旧 `evaluate_gate` 与其测试；新逻辑放 `evaluate_evidence_gate(decision_inputs, model)`，便于单测。

### 6.5 自评节点

`knowledge` 路由在闸通过后新增节点 `knowledge_self_check`（或逻辑并入 `_confidence_gate` 之后的条件边）：调用 `SufficiencyJudge.judge(question, evidence)`；`sufficient=False` → 写 `retrieval_refused`/`self_check` 标记并路由到 `fallback`，同时在节点内入池（`source=self_check`，带快照）+ 飞轮。失败/超时按现有 fail-open 逻辑继续。

---

## 7. 数据飞轮

### 7.1 入口统一

新增 `app/flywheel/service.py`：

```python
class FlywheelService:
    def __init__(self, *, repositories, model, settings): ...
    async def ingest(self, low_confidence_id: int) -> FlywheelResult: ...
```

`FlywheelResult(low_confidence_id, review_id, matched: bool, degraded: bool, reason: str|None)`。

三个入口：

1. `_confidence_gate` 不通过：`low_confidence.add(..., retrieved_chunks=snapshot)` 返回 id → `await flywheel.ingest(id)`。
2. `knowledge_self_check` 不通过：同上，`source="self_check"`。
3. `POST /api/feedback`：落池后同样调用。

入池失败（DB/LLM）只 `logger.exception`，不影响兜底话术；审核页看不到这条时能在日志里追。

### 7.2 标准化（`normalizer.py`）

- 输入：`raw_question`、来源、当轮快照里的证据（可空）。
- Prompt 要求严格 JSON：`{"normalized_question": "...", "suggested_answer": "..."}`。
  - 标准问题 ≤ 60 字，FAQ 式（「如何…」「…怎么办」），保留关键实体（订单号脱敏）。
  - 示例答案 ≤ 300 字；证据里已有答案时可参考，不确定则给「待人工补充」的简短要点。
- 解析失败/超时 → 降级：`normalized_question=raw_question.strip()[:512]`、`suggested_answer=""`、`degraded=True`；不阻塞入队。

### 7.3 查重（`deduper.py`）

- 取 `review_status='待审'` 最近 50 行（id + normalized_question）。
- Prompt 给候选列表与待判问题，要求 JSON：`{"matched_id": <候选 id 或 null>, "same": true|false, "reason": "..."}`。
- 守卫：`matched_id` 必须出现在候选列表，否则视为未命中；解析失败 → 未命中。
- 命中 → `occurrence_count += 1`、`updated_at=now`，并把本条 `low_confidence_questions.matched_review_id` 指向该行。
- 未命中 → 新建 `review_queue` 行，`matched_review_id` 指新行。

### 7.4 审核通过 → 知识库

`POST /api/review-queue/{id}/approve`：

1. 仅 `review_status='待审'` 可操作；写 `approved_answer`（必填，允许等于示例答案）、状态改「通过」。
2. 落库：`knowledge.upsert_chunk(_MinedChunk(category=..., questions=normalized_question, answer=approved_answer, section_path="人工审核 / 数据飞轮", content_type="faq", is_key_clause=False, ordinal=0))`。
   - `category` 取该审核行关联快照里 Top1 的 category；没有则 `售后政策`。
3. 向量化：`vectorize_ids([chunk_id], chunk_repo=..., vector_store=..., embedder=...)`，只处理这一块。
4. 返回 `{ok, chunk_id, created|updated}`；向量化失败不回滚知识原文，返回 `vectorize_status=pending`，提示可重试（与 ch03 状态机一致）。

驳回：状态改「驳回」，不写知识库；同一问题再次入池时不会命中已驳回行（只查待审）。

### 7.5 👎 后端接入

- 持久化消息需要 id：`MessageRepository.append_many` 改为 flush 后返回 `list[int]`；`persist_turn` 返回 `PersistedTurn(user_message_id, answer_message_id)`；`_log_turn` 把 `message_id` 写进 state/done payload；SSE `done` 增加 `message_id`。
- 前端每个 bot 气泡记 `data-message-id`；👎 调 `POST /api/feedback`：

```json
{"session_id": "...", "message_id": 123, "rating": "down", "reason": "答非所问"}
```

- 后端：
  1. 按 id 取消息 → `conversation_id`；
  2. 取该消息之前最近一条 `role=user` 的 content 作为原话；
  3. `FlywheelService.snapshot_for_turn(session_id, user_message)`：`graph.aget_state_history(config)` 从新到旧找 `values["user_message"] == 原话` 的最近一轮，取 `retrieval_snapshot`；找不到则回退 `messages.citations`（分数置 null），再找不到就 `NULL`；
  4. `low_confidence.add(conversation_id, question, "user_feedback", reason, retrieved_chunks=snapshot)` → 飞轮。
- 幂等：同一 `message_id` 重复点 👎 不重复入池（进程内 `message_id → low_conf_id` 缓存即可；重启丢缓存可接受，也可查最近同会话同原话的 `user_feedback` 行做 5 分钟去重）。

---

## 8. 自动化评估

### 8.1 记录

`app/eval_runs.py`：

```python
async def run_evaluation(*, trigger: Literal["定时", "手动"], limit: int | None = None) -> int:
    """子进程跑 evals/run_ch04_evals.py --json <tmp>，读报告，写 eval_runs，返回 id。"""
```

`metrics` JSON：

```json
{
  "retrieval": {"recall@1":0.72,"recall@3":0.85,"recall@5":0.90,"recall@10":0.94,"mrr@10":0.80,
                 "buckets": {"A_policy": {...}, "...": {}}},
  "generation": {"faithfulness":0.88,"refusal_accuracy":0.75,"citation_validity":0.9,"citation_coverage":0.6},
  "meta": {"strategy":"hybrid_rerank","cases":48,"milvus_rows":68,"report_path":"..."}
}
```

### 8.2 周期调度

- `app/eval_schedule.py`：lifespan 启动 `asyncio.create_task(_loop())`；`EVAL_SCHEDULE_ENABLED=true` 时每 `EVAL_SCHEDULE_INTERVAL_HOURS` 跑一次，`triggered_by="定时"`；同一时刻只允许一个评估任务（进程内 `asyncio.Lock`）。
- `POST /api/eval-runs/run`：手动触发，后台任务，立即返回 202；`GET /api/eval-runs?limit=20`：返回列表与趋势（每指标序列 + 最近两次差值 + 方向）。
- 失败：写日志，不写半截行；调度器下一周期重试。

### 8.3 趋势页

`/eval-trend`：表格每行一轮（时间、触发方式、条数、recall@5、mrr@10、faithfulness、refusal_accuracy）；最后一行显示相对上一轮的箭头（↑ 绿 / ↓ 红 / → 灰）。无构建工具，原生 JS + 内联样式。

---

## 9. API 与前端

### 9.1 新增端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/feedback` | 👎 落池 + 飞轮 |
| GET | `/api/review-queue` | 列表（status/limit/offset） |
| GET | `/api/review-queue/{id}` | 详情：标准化问题、示例答案、原话列表、召回片段 |
| POST | `/api/review-queue/{id}/approve` | 通过并落库 |
| POST | `/api/review-queue/{id}/reject` | 驳回 |
| GET | `/api/eval-runs` | 评估历史 + 趋势 |
| POST | `/api/eval-runs/run` | 手动触发 |
| GET | `/api/intent-cost` | 按意图 token/cost |
| GET | `/review` | 审核后台页 |
| GET | `/eval-trend` | 趋势页 |

### 9.2 审核页（Vibe Coding）

- 表格列：标准化问题、出现次数、示例答案（截断）、状态、操作。
- 点行展开详情：用户原话（含 source/时间）、召回片段（Top 5：分数 + section_path + 原文，等宽字体）。
- 通过弹窗：预填示例答案，可编辑；提交后行变「通过」。
- 驳回二次确认；支持按状态过滤。
- 复用 `web/style.css` 的现有视觉，不引入框架。

### 9.3 聊天页 👎

- `web/app.js` 把 `localStorage` 记录改为 `POST /api/feedback`；成功后按钮锁定、note 显示「已反馈」；失败恢复可点并提示。
- 其余前端反馈逻辑（👍）保持纯前端，不上后端。

---

## 10. 错误处理与降级

| 场景 | 行为 |
|---|---|
| Langfuse 不可达 / 密钥错 | tracing 初始化失败只 warning，全部 helper no-op；聊天正常 |
| `retrieve_for_gate` 抛异常 | 走既有错误路径，回兜底话术；不因闸升级把聊天变 500 |
| 校准模型缺失 / 损坏 | 退回 ch05 阈值闸，`logger.warning` |
| 标准化 LLM 失败 | 用原话 + 空示例答案建行（degraded） |
| 查重 LLM 失败 | 不合并，新建行（宁可多一行，不误并） |
| 审核落库失败 | 状态不改「通过」，返回 500 与可读错误；已改则回滚状态 |
| 向量化失败 | 知识原文已写，状态 pending；接口返回可重试提示 |
| 评估子进程失败 | 不写 `eval_runs`，接口/日志标失败 |
| 👎 找不到当轮快照 | 照常入池，`retrieved_chunks=NULL`（闲聊/业务数据类属正常） |

---

## 11. 测试策略

### 11.1 后端（TDD）

- `tests/test_ch09_confidence.py`：特征计算（含 0/1 条证据、并列分）、sigmoid/阈值、逻辑回归在可分合成数据上收敛、留一法选阈值、模型 JSON 往返。
- `tests/test_ch09_gate.py`：`evaluate_evidence_gate` 各分支；graph 知识路由：0 命中/低分/降级/通过；通过后自评不够 → fallback + `self_check` 入池 + 快照；飞轮 fake 被调用。
- `tests/test_ch09_flywheel.py`：标准化 JSON 解析/降级；查重命中累加与未命中新建；非法 matched_id 守卫；`matched_review_id` 回写；`ingest` 幂等。
- `tests/test_ch09_repositories.py`：review_queue CRUD/计数、eval_runs 写入与排序、低置信度新列读写。
- `tests/test_ch09_feedback.py`：`append_many` 返回 id；`/api/feedback` 用 fake graph 的 `aget_state_history` 回捞快照；重复 👎 幂等；无检索 → NULL。
- `tests/test_ch09_review_api.py`：列表/详情/通过（fake upsert + vectorize）/驳回/重复通过。
- `tests/test_ch09_eval_runs.py`：报告 → metrics JSON；趋势方向；调度开关。
- `tests/test_ch09_tracing.py`：关闭时 no-op；开启时 `build_agent_graph` 的 `callbacks` 存在（monkeypatch handler）；不使用真实网络。
- 既有 `test_ch05_gate.py` / `test_ch05_graph.py` / `test_knowledge_pipeline.py` 随签名/行为同步更新，不删断言。

### 11.2 数据 / Prompt（替代 TDD 的验证）

- `scripts/calibrate_confidence.py` 跑真实 ch04 评估集，产出 `dev-notes/ch09-confidence-calibration.md`：LOO balanced accuracy、混淆矩阵、阈值、逐桶指标、失败样例。**验收看报告，不只看单测。**
- `scripts/verify_flywheel.py`：6 组标注样例（3 组同义、3 组不同）跑真实模型，打印判定与标准答案对比；全部判对才算通过；报告追加到 dev-notes。
- `scripts/verify_intent_cost.py`：造 3 类意图各 2 条请求后，拉 Metrics API 校验聚合结果与手工相加一致。

### 11.3 前端（Vibe Coding）

- 按用户描述调页面，不写单测；用真实浏览器手工走：列表 → 详情 → 通过 → 聊天复问；👎 → 队列出现。

---

## 12. 验收对照

| # | 验收标准 | 验证方式 |
|---|---|---|
| 1 | Langfuse 任意请求完整链路 | 打开 `https://jp.cloud.langfuse.com` → 选一条 chat-turn → 看到根 span / 节点 / generation（prompt+usage）/ 工具 / 检索 |
| 2 | 知识库没有的问题 → 兜底 + 待审 + 详情含原话与片段 | 演示脚本/手工提问 → 审核页详情 |
| 3 | 审核通过后再问能答对 | 审核页点通过 → 同问题复问 → 回答且引用新块 |
| 4 | 👎 落池 → 标准化查重 → 待审 | 聊天页点 👎 → `GET /api/review-queue` 出现 |
| 5 | 按意图 token 花费 | `GET /api/intent-cost` / `scripts/intent_cost.py` 输出表格，最烧钱意图可见 |
| 6 | 评估两轮 + 趋势 | `POST /api/eval-runs/run` ×2 → `/eval-trend` 两行 + 箭头 |

---

## 13. 风险与开放问题

1. **校准样本少**（8 条负样本）：LOO 方差大；若 balanced accuracy < 0.70 按 §6.3 不发布模型。需要在报告中如实写，不用测试集自评。
2. **Langfuse v4 events_only**：旧 traces API 404，只能用 observations/metrics v2；SDK 升级到 6.x 时需重验。
3. **同步飞轮延迟**：每轮知识问答 +1 次标准化/查重 LLM（1-3 秒）；若线上不可接受，后续改异步队列（本章不做）。
4. **checkpoint 历史回捞**：依赖 checkpointer 保留历史（SQLite saver 默认保留）；若历史被裁剪，👎 快照可能为空 —— 按「尽力回捞」处理并记录。
5. **审核通过写知识库的 category** 取快照 Top1 的 category；没有快照时落到「售后政策」，可能不够精确，审核页允许人工不改（本章不做分类器）。
6. **评估任务资源**：两轮评估会产生真实 LLM 调用；调度默认 24h，演示时手动触发。

---

## 14. 交付物

- 本 spec + plan + `dev-notes/ch09.md`（分阶段留痕）。
- Langfuse Cloud 接入说明：`.env` 三个变量（已配置、已实测），仓库不含任何密钥或自部署文件。
- 代码与测试；`sql/ch09-ddl.sql` 已应用。
- 校准报告 `dev-notes/ch09-confidence-calibration.md`、模型 artifact。
- 演示命令：起服务、问未知问题、审核通过、复问、👎、意图成本、评估两轮。
- 测试结果：`uv run pytest` 全绿 + 三个验证脚本输出。
