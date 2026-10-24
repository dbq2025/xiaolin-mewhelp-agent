# ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接上 Langfuse 全链路 trace、按意图统计 token，并把三类答不上的问题经标准化/查重送入审核队列，审核通过后回写知识库；同时把 ch04 评估集跑成可比较的趋势。

**Architecture:** 观测层用 Langfuse v4 + SDK 5（编译期挂 CallbackHandler + 手动根 span/tags）；置信度闸从单阈值升级为三特征逻辑回归，模型由 ch04 评估集留一法校准；入池与飞轮同步执行（标准化 → LLM 查重 → review_queue）；审核通过复用 ch03 的 `upsert_chunk` + 按 id 向量化；评估在后台子进程跑并写 `eval_runs`，前端是无构建的原生页面。

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy / LangGraph / Langfuse 5.0.4 / MySQL 8.4 / Milvus / 原生 HTML+JS。

**Spec:** `docs/superpowers/specs/2026-10-24-ch09-observability-flywheel-design.md`

## Global Constraints

- 只有 `app/config.py` 可以读 `.env` / `os.environ`；新配置项一律加 `Settings` 字段。
- 只有 `app/llm.py` 可以 `import langchain_openai`；其他模块用注入的模型对象。
- 错误文本出 API 前必须过 `app/errors.py::redact()`。
- 新增依赖仅允许已装的 `langfuse==5.0.4`；不引入 sklearn/pandas/apscheduler。
- 测试跑 SQLite 内存库，不连 MySQL/Milvus/真实模型；不使用 `tmp_path`（沙箱会锁目录）。
- 运行：`$env:UV_CACHE_DIR="F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"; & "C:\Users\dbq\.local\bin\uv.exe" run pytest -q -p no:cacheprovider`（工作目录 `MemHelp-code`）。
- 提交从工作区根执行，只 `git add MemHelp-code/...`；每条任务一次提交。
- 用户可见文案用中文；Langfuse 不可用时聊天必须照常工作。
- 每个任务收尾必须追记 `dev-notes/ch09.md`（四样：用户原话、关键产出、拒绝/纠偏、翻车返工），不许攒到最后。

## Review Focus

1. **Langfuse 不可达 / 密钥错**：聊天、落库、飞轮全部照常，tracing 静默降级为 no-op。
2. **重复 👎 与同一问题反复入池**：不重复建审核行，只累加 `occurrence_count`；同一 `message_id` 重复反馈不重复入池。
3. **审核通过后向量化失败**：知识原文已写、状态 pending，接口明确可重试，不能把整次通过回滚成 500 且丢原文。
4. **重排降级（RRF 融合分）**：闸直接放行并记 `degraded`，不能拿 0.03 量级的分去套 0.3 阈值误杀。
5. **评估趋势方向**：指标下降必须显示 ↓（红），不能因为数组顺序或空值显示成上升。

---

### Task 1: 数据层（review_queue / eval_runs / 问题池新列）

**Files:**
- Modify: `app/db/models.py`（`LowConfidenceQuestion` 后追加两个模型）
- Modify: `app/db/repositories.py`（`LowConfidenceQuestionRepository`、`Repositories`）
- Test: `tests/test_ch09_repositories.py`

**Interfaces:**
- Consumes: `app.db.models.LOW_CONFIDENCE_SOURCES`、`Base`
- Produces:
  - `ReviewQueue`、`EvalRun` ORM；`LowConfidenceQuestion.retrieved_chunks: list|None`、`.matched_review_id: int|None`
  - `ReviewRow(id, normalized_question, ai_suggested_answer, occurrence_count, review_status, approved_answer, created_at, updated_at)`
  - `ReviewQueueRepository.add(normalized_question, ai_suggested_answer) -> int`
  - `ReviewQueueRepository.list_rows(status=None, limit=50, offset=0) -> list[ReviewRow]`
  - `ReviewQueueRepository.get(review_id) -> ReviewRow | None`
  - `ReviewQueueRepository.pending_questions(limit=50) -> list[tuple[int, str]]`
  - `ReviewQueueRepository.increment(review_id) -> bool`
  - `ReviewQueueRepository.set_status(review_id, status, approved_answer=None) -> bool`
  - `EvalRunRepository.add(triggered_by, dataset_size, metrics) -> int`
  - `EvalRunRepository.list_recent(limit=20) -> list[EvalRunRow]`
  - `LowConfidenceQuestionRepository.add(..., retrieved_chunks=None) -> int`
  - `LowConfidenceQuestionRepository.get(question_id) -> LowConfidenceRow | None`
  - `LowConfidenceQuestionRepository.attach_review(question_id, review_id) -> None`
  - `Repositories.reviews` / `Repositories.eval_runs`

- [ ] **Step 1: 写失败测试**

```python
def test_review_queue_defaults_and_increment(fake_repositories):
    repo = fake_repositories.reviews
    review_id = repo.add("退款多久到账？", "1-3 个工作日")
    row = repo.get(review_id)
    assert row.review_status == "待审" and row.occurrence_count == 1
    assert repo.increment(review_id) is True
    assert repo.get(review_id).occurrence_count == 2
    assert [q for q in repo.pending_questions()] == [(review_id, "退款多久到账？")]


def test_low_confidence_snapshot_and_match(fake_repositories):
    lc = fake_repositories.low_confidence
    snapshot = [{"n": 1, "chunk_id": 7, "quote": "原文", "score": 0.42}]
    qid = lc.add(None, "怎么退", "retrieval_low_conf", "分低", retrieved_chunks=snapshot)
    rid = fake_repositories.reviews.add("如何退款？", "在订单页申请")
    lc.attach_review(qid, rid)
    row = lc.get(qid)
    assert row.retrieved_chunks == snapshot and row.matched_review_id == rid


def test_eval_runs_roundtrip(fake_repositories):
    run_id = fake_repositories.eval_runs.add("手动", 48, {"retrieval": {"recall@5": 0.9}})
    rows = fake_repositories.eval_runs.list_recent()
    assert rows[0].id == run_id and rows[0].metrics["retrieval"]["recall@5"] == 0.9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_repositories.py -q -p no:cacheprovider`
Expected: FAIL（`AttributeError: 'Repositories' object has no attribute 'reviews'`）

- [ ] **Step 3: 实现模型与仓库**

模型按 `sql/ch09-ddl.sql` 映射；SQLite 索引名用 `idx_review_queue_status`、`idx_lcq_matched_review`（库级唯一），MySQL DDL 名字不动。`LowConfidenceQuestion` 用 `ForeignKeyConstraint(["matched_review_id"], ["review_queue.id"], name="fk_lcq_review", ondelete="SET NULL")`。`_list_recent` 等返回新字段。`Repositories.from_session` 增加两个仓库；`ReviewQueueRepository` 与 `EvalRunRepository` 沿用现有 `_run` + 私有同步方法 + 公开 async 包装的写法。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ch09_repositories.py tests/test_db_repositories.py tests/test_db_models.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app/db/models.py MemHelp-code/app/db/repositories.py MemHelp-code/tests/test_ch09_repositories.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): review_queue/eval_runs 模型与仓库，问题池加快照列"
```

在 `dev-notes/ch09.md` 追加「阶段 2 · 任务 1 完成」，记录本任务实际改动与测试结果。

---

### Task 2: Langfuse 观测层（配置 / 编译期回调 / 根 span / 意图 tag / 工具与检索 span）

**Files:**
- Modify: `app/config.py`（Langfuse 五個字段）
- Create: `app/observability/__init__.py`、`app/observability/tracing.py`
- Modify: `app/agent/graph.py`（`build_agent_graph`、`_run_agent_node`、`_knowledge_retrieve`）
- Modify: `app/agent/react.py`（工具执行 span）
- Modify: `app/agent/runner.py`（根 span 包住一轮）
- Test: `tests/test_ch09_tracing.py`

**Interfaces:**
- Produces:
  - `Settings.langfuse_enabled: bool = False`、`langfuse_public_key/secret_key/base_url/environment`
  - `tracing.configure(settings) -> None`（幂等；失败只 warning）
  - `tracing.callback_handler() -> object | None`
  - `tracing.turn_span(session_id, question) -> TurnSpan`（`TurnSpan.finish(payload)`；未启用时 no-op）
  - `tracing.tag_intent(intent, confidence) -> ContextManager[None]`
  - `tracing.observation(name, as_type, input=None) -> ContextManager[Any | None]`
  - `tracing.reset_for_tests() -> None`
  - `tracing.shutdown() -> None`（`flush()`，即使异常也只 warning）
  - `build_agent_graph` 返回对象 `graph.config["callbacks"] == [handler]`（启用时）

- [ ] **Step 1: 写失败测试**

```python
def test_tracing_disabled_is_noop(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_handler", None)
    with tracing.turn_span("s1", "你好") as span:
        span.finish({"stop_reason": "chitchat"})
    with tracing.observation("retrieval", as_type="retriever"):
        pass


def test_graph_attaches_callback(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_handler", "HANDLER")
    graph = build_agent_graph(checkpointer=None)
    assert graph.config["callbacks"] == ["HANDLER"]
    assert hasattr(graph, "aget_state") and graph.checkpointer is not None
```

`test_turn_span_finishes_with_payload`：monkeypatch `_client` 为记录器，断言 `finish` 后 `update` 收到 `metadata.intent`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_tracing.py -q -p no:cacheprovider`
Expected: FAIL（`ModuleNotFoundError: app.observability`）

- [ ] **Step 3: 实现**

`tracing.py` 用模块级 `_client/_handler/_enabled`；`configure()` 读 `settings`，未启用或密钥空直接返回。`TurnSpan` 持有根 observation；`finish(payload)` 调 `update(output=payload, metadata={intent, route, gate, token_usage, stop_reason, message_id})`。`tag_intent` 用 `propagate_attributes(tags=[f"intent:{intent}"], metadata={"intent": intent, "intent_confidence": confidence})`；未启用 yield。`observation()` 用 `_client.start_as_current_observation(name=..., as_type=..., input=...)`，未启用 yield None。

`graph.py`：
- `build_agent_graph` 末尾：`compiled = builder.compile(...)`；`handler = callback_handler()`；有则 `compiled = compiled.with_config({"callbacks": [handler]})`；返回。
- `_run_agent_node`：在 `run_react_agent(...)` 外层套 `with tag_intent(state.get("intent",""), float(state.get("intent_confidence") or 0)):`。
- `_knowledge_retrieve`：`with observation("knowledge_retrieve", as_type="retriever", input={"query": ...}) as span:` 包住 `retrieve_only`，最后 `span.update(output={"chunks": [...], "top_score": ..., "degraded": ...})`。
- `react.py` 工具执行：`with observation(f"tool:{name}", as_type="tool", input=args) as span:`，执行后 `span.update(output=result.summary(), metadata={"ok": result.ok})`。
- `runner.astream_turn`：在 `payload` 构造后 `with turn_span(session_id, message) as span:` 包住两个 `async for`，记录 done payload，`finally` 里 `span.finish(done or {})`。

`config.py` 的 Langfuse 字段全部有默认值，保证既有测试不因缺省失败。
`api.py` 的 `_lifespan`：启动时 `tracing.configure(settings)`，关闭时 `tracing.shutdown()`（`flush()`，异常只 warning）。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_tracing.py tests/test_ch05_graph.py tests/test_api_sse.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/tests/test_ch09_tracing.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): Langfuse 编译期回调、根 span、意图 tag、工具与检索 span"
```

`dev-notes/ch09.md` 追加「阶段 3 · 任务 2 完成」。

---

### Task 3: 证据置信度模型（特征 / 逻辑回归 / 留一法 / artifact）

**Files:**
- Create: `app/agent/evidence_confidence.py`
- Modify: `app/config.py`（`evidence_confidence_model: str = "evals/ch09_confidence_model.json"`）
- Test: `tests/test_ch09_confidence.py`

**Interfaces:**
- Produces:
  - `EvidenceFeatures(top1, top2, margin, effective_count)`
  - `features_from_scores(scores: Sequence[float]) -> EvidenceFeatures`
  - `ConfidenceModel(weights, intercept, threshold, mean, std, version)`，方法 `score(f) -> float`、`passes(f) -> bool`
  - `fit_logistic(samples: Sequence[tuple[EvidenceFeatures, int]], *, epochs=3000, lr=0.1, l2=1e-3) -> ConfidenceModel`
  - `calibrate_loo(samples) -> tuple[ConfidenceModel, dict]`（返回最终模型 + LOO 报告）
  - `load_model(path: Path) -> ConfidenceModel | None`、`model_to_dict(model) -> dict`、`model_from_dict(data) -> ConfidenceModel`

- [ ] **Step 1: 写失败测试**

```python
def test_features_margin_and_effective_count():
    f = features_from_scores([0.9, 0.8, 0.2])
    assert (f.top1, f.top2, round(f.margin, 3), f.effective_count) == (0.9, 0.8, 0.1, 2)


def test_logistic_separates_synthetic_data():
    samples = [(features_from_scores([0.9, 0.85]), 1), (features_from_scores([0.2]), 0)] * 20
    model = fit_logistic(samples)
    assert model.score(features_from_scores([0.9])) > 0.5
    assert model.score(features_from_scores([0.1])) < 0.5


def test_loo_report_has_confusion_matrix():
    samples = [(features_from_scores([0.9, 0.8]), 1), (features_from_scores([0.1]), 0)] * 5
    model, report = calibrate_loo(samples)
    assert report["balanced_accuracy"] >= 0.9
    assert set(report["confusion"]) == {"tp", "fp", "tn", "fn"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_confidence.py -q -p no:cacheprovider`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

`sigmoid` 用 `math.exp` 并做溢出保护；标准化按训练集均值/标准差，标准差为 0 时置 1。梯度下降固定 3000 轮、学习率 0.1、L2 1e-3，权重初始化 0。`calibrate_loo`：对每个样本留出，训练后预测概率；遍历 `0.05..0.95`（步长 0.05）选 balanced accuracy 最大阈值（并列取更接近 0.5 的）；用全量样本重训最终模型。`load_model` 读 JSON 失败返回 None。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ch09_confidence.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app/agent/evidence_confidence.py MemHelp-code/tests/test_ch09_confidence.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 三特征证据置信度模型与留一法校准"
```

`dev-notes/ch09.md` 追加「阶段 4 · 任务 3 完成」。

---

### Task 4: 闸接入（retrieve_for_gate / 特征闸 / 自评节点 / 快照入池）

**Files:**
- Modify: `app/knowledge/pipeline.py`（新增 `retrieve_for_gate`）
- Modify: `app/agent/gate.py`（新增 `evaluate_evidence_gate`）
- Modify: `app/agent/state.py`（`retrieval_snapshot`、`evidence_confidence`、`self_check_degraded`）
- Modify: `app/agent/graph.py`（`_knowledge_retrieve`、`_confidence_gate`、`_knowledge_self_check`、路由）
- Test: `tests/test_ch09_gate.py`

**Interfaces:**
- Consumes: Task 1 `low_confidence.add(..., retrieved_chunks=...)`；Task 3 `EvidenceFeatures` / `features_from_scores` / `load_model`
- Produces:
  - `RagPipeline.retrieve_for_gate(query, *, category=None) -> RetrievalOutcome`（0 命中才 refused）
  - `evaluate_evidence_gate(*, evidence_count: int, features: EvidenceFeatures, confidence: float, rerank_degraded: bool, model: ConfidenceModel) -> GateDecision`
  - `_knowledge_self_check(state, config) -> dict`（写 `gate_passed`、`gate_reason`、`refusal_source`）
  - `GraphDeps.flywheel`（默认 None；Task 5 注入）

- [ ] **Step 1: 写失败测试**

```python
async def test_retrieve_for_gate_keeps_low_score_evidence():
    outcome = await pipeline.retrieve_for_gate("退货多久到账")
    assert outcome.refused is False
    assert outcome.evidence and outcome.top_score < 0.3


def test_gate_rejects_no_evidence_and_passes_degraded():
    model = _stub_model()
    assert not evaluate_evidence_gate(evidence_count=0, features=features_from_scores([]),
                                      confidence=0.0, rerank_degraded=False, model=model).passed
    assert evaluate_evidence_gate(evidence_count=3, features=features_from_scores([0.03, 0.02]),
                                  confidence=0.0, rerank_degraded=True, model=model).passed
```

Graph 级测试：`test_gate_failure_pools_with_snapshot` 用假检索器返回低分证据，跑图后断言 `low_confidence` 行 `source=retrieval_low_conf`、`retrieved_chunks[0]["score"] < 0.3`、且 fake flywheel 收到 id。`test_self_check_failure_pools` 用 SufficiencyJudge 假实现返回 `sufficient=False`，断言 `source=self_check` 且回兜底话术。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_gate.py -q -p no:cacheprovider`
Expected: FAIL（`AttributeError: retrieve_for_gate`）

- [ ] **Step 3: 实现**

`retrieve_for_gate` 复用 `_retrieve`，不比较 `_min_score`；0 命中返回 `_refuse`。`_knowledge_retrieve` 同时写 `retrieval_snapshot`（`n/chunk_id/section_path/category/quote/score`，quote 截 600 字）。`evaluate_evidence_gate` 在 `rerank_degraded` 或 `evidence_count == 0` 时短路；否则用 `confidence >= model.threshold` 判过，`reason` 带分数与阈值。模型路径从 `settings.evidence_confidence_model`（默认 `evals/ch09_confidence_model.json`）加载，失败退回 `evaluate_gate`。

`_knowledge_self_check` 调 `SufficiencyJudge`（`_deps(config).model`），`sufficient=False` 时写 `GateDecision(passed=False, reason=verdict.reason)` 并入池（`source=self_check`）+ 快照；`degraded=True` 视为通过并标记。路由：`confidence_gate` 的 `"agent"` 目标改为 `"knowledge_self_check"`，新增条件边 `knowledge_self_check -> {"agent","fallback"}`。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_gate.py tests/test_ch05_gate.py tests/test_ch05_graph.py tests/test_knowledge_pipeline.py -q -p no:cacheprovider`
Expected: PASS（旧 `evaluate_gate` 测试保持通过）

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/tests/test_ch09_gate.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 证据置信度闸与生成前自评入口，入池带召回快照"
```

`dev-notes/ch09.md` 追加「阶段 5 · 任务 4 完成」，写明阈值模型未就位时的降级行为。

---

### Task 5: 飞轮流水线（标准化 / 查重 / 入队 / 回写）

**Files:**
- Create: `app/flywheel/__init__.py`、`normalizer.py`、`deduper.py`、`service.py`
- Modify: `app/agent/graph.py`（`GraphDeps.flywheel` 注入与三处调用）
- Modify: `app/agent/runner.py`（`_config` 传 flywheel）
- Modify: `app/api.py`（构建 `FlywheelService` 并进 config）
- Test: `tests/test_ch09_flywheel.py`

**Interfaces:**
- Produces:
  - `NormalizedQuestion(normalized_question, suggested_answer, degraded=False)`
  - `async normalize_question(model, *, raw_question, evidence, timeout=20.0) -> NormalizedQuestion`
  - `DedupeResult(matched_id: int | None, reason: str, degraded: bool = False)`
  - `async find_duplicate(model, *, question, candidates, timeout=15.0) -> DedupeResult`
  - `FlywheelResult(low_confidence_id, review_id, matched, degraded, reason)`
  - `FlywheelService(repositories, model, settings)`；`async ingest(low_confidence_id) -> FlywheelResult`
  - `Repositories.reviews.pending_questions(limit)`；`low_confidence.attach_review(...)`

- [ ] **Step 1: 写失败测试**

```python
async def test_normalize_falls_back_on_bad_json():
    out = await normalize_question(BadModel(), raw_question="这个怎么弄", evidence=[])
    assert out.degraded is True and out.normalized_question == "这个怎么弄"


async def test_ingest_merges_duplicate(fake_repositories, stub_model):
    rid = fake_repositories.reviews.add("退款多久到账？", "1-3 个工作日")
    qid = fake_repositories.low_confidence.add(None, "钱啥时候回来", "user_feedback")
    service = FlywheelService(fake_repositories, stub_model(matched_id=rid), test_settings)
    result = await service.ingest(qid)
    assert result.matched is True and result.review_id == rid
    assert fake_repositories.reviews.get(rid).occurrence_count == 2
    assert fake_repositories.low_confidence.get(qid).matched_review_id == rid


async def test_ingest_creates_when_no_match(...):
    ...
    assert result.matched is False
    assert fake_repositories.reviews.get(result.review_id).normalized_question == "如何退款？"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_flywheel.py -q -p no:cacheprovider`
Expected: FAIL（`ModuleNotFoundError: app.flywheel`）

- [ ] **Step 3: 实现**

Prompt 放模块常量；要求模型只回 JSON。解析用 `json.loads(_strip_code_fence(text))`，失败走降级。`deduper` 校验 `matched_id` 必须 ∈ 候选 id 集合。

`ingest` 流程：`get(low_conf_id)` 不存在 → `FlywheelResult(..., degraded=True, reason="not_found")`；已有 `matched_review_id` → 直接返回；标准化 → 取 pending 50 → 判重 → 命中 `increment`，未命中 `reviews.add` → `low_confidence.attach_review`。任何异常 catch 后返回 degraded 结果并 `logger.exception`。

`graph.py`：`GraphDeps` 加 `flywheel: Any = None`；`_pool_and_flywheel(state, config, source, reason, conversation_id)` 辅助函数：入池后 `if deps.flywheel is not None: await deps.flywheel.ingest(id)`（异常只 warning）。`runner._config` 增加 `flywheel` 参数并在 `api.py` 构造 `FlywheelService(repositories, model, settings)` 注入。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_flywheel.py tests/test_ch09_gate.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/tests/test_ch09_flywheel.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 飞轮标准化、查重与审核队列写入"
```

`dev-notes/ch09.md` 追加「阶段 6 · 任务 5 完成」。

---

### Task 6: 👎 后端接入（message_id / 反馈接口 / checkpoint 回捞）

**Files:**
- Modify: `app/db/repositories.py`（`MessageRepository.append_many` 返回 id）
- Modify: `app/agent/persist.py`（`PersistedTurn`）
- Modify: `app/agent/graph.py`（`_log_turn` 返回 message_id）
- Modify: `app/agent/runner.py`（`done_payload` 带 message_id）
- Create: `app/flywheel/snapshot.py`
- Modify: `app/api.py`（`POST /api/feedback`）
- Modify: `web/app.js`（👎 调后端）
- Test: `tests/test_ch09_feedback.py`

**Interfaces:**
- Produces:
  - `MessageRepository.append_many(...) -> list[int]`
  - `PersistedTurn(user_message_id: int, answer_message_id: int)`
  - `persist_turn(...) -> PersistedTurn`
  - `async snapshot_for_turn(graph, *, session_id, user_message) -> list[dict] | None`
  - `FeedbackRequest(session_id: str, message_id: int, rating: Literal["up","down"], reason: str | None = None)`

- [ ] **Step 1: 写失败测试**

```python
async def test_append_many_returns_ids(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s", "anonymous")
    ids = await fake_repositories.messages.append_many(cid, [{"role": "user", "content": "hi"}])
    assert len(ids) == 1 and ids[0] > 0


async def test_feedback_pools_with_snapshot(client, fake_repositories):
    resp = await client.post("/api/feedback", json={"session_id": "s1", "message_id": 2, "rating": "down"})
    assert resp.status_code == 200 and resp.json()["ok"] is True
    row = (await fake_repositories.low_confidence.list_recent())[0]
    assert row.source == "user_feedback"
```

`test_feedback_idempotent`：同 `message_id` 两次只多一行。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_feedback.py -q -p no:cacheprovider`
Expected: FAIL（`404` / `append_many` 返回 None）

- [ ] **Step 3: 实现**

`_append_many` 在 commit 后 `flush()` 并返回 `[int(row.id) for row in rows]`。`persist_turn` 取返回 ids，`answer_message_id` 为最后一个 assistant 行；`_log_turn` 把 `message_id` 写进返回 payload；`done_payload` 增加 `"message_id": values.get("message_id")`。

`snapshot_for_turn`：`async for snap in graph.aget_state_history({"configurable": {"thread_id": session_id}})`，`values.get("user_message") == user_message` 时取 `retrieval_snapshot`，没有则 `None`。

接口：按 `message_id` 查消息行（新仓库方法 `MessageRepository.get(message_id) -> MessageRow | None`），取同会话前一条 user 行；快照先 checkpoint 后消息 `citations`（score 置 None）；`low_confidence.add(..., "user_feedback", reason, retrieved_chunks=snapshot)` → `flywheel.ingest`。`rating == "up"` 直接 `{"ok": True, "ignored": True}`。

前端：bot 气泡加 `dataset.messageId`（来自 done payload）；`attachFeedback` 点击 👎 时 `fetch('/api/feedback', ...)`，成功后锁定，失败恢复。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_feedback.py tests/test_api_sse.py tests/test_ch05_graph.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/web/app.js MemHelp-code/tests/test_ch09_feedback.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 👎 后端落池，按轮回捞检索快照"
```

`dev-notes/ch09.md` 追加「阶段 7 · 任务 6 完成」，记录回捞失败时的空快照行为。

---

### Task 7: 真实校准（脚本 / 跑评估集 / 报告 / artifact）

**Files:**
- Create: `scripts/calibrate_confidence.py`
- Produce: `evals/ch09_confidence_model.json`、`dev-notes/ch09-confidence-calibration.md`
- Test: `tests/test_ch09_calibration.py`

**Interfaces:**
- Consumes: `evidence_confidence.calibrate_loo`、`features_from_scores`、`Pipeline.retrieve_for_gate`
- Produces: `load_cases(path) -> list[dict]`、`collect_samples(cases, pipeline) -> list[tuple[EvidenceFeatures, int]]`、`main(argv) -> int`

- [ ] **Step 1: 写失败测试**

```python
def test_load_cases_skips_comments():
    cases = load_cases(Path("evals/ch04_cases.jsonl"))
    assert len(cases) == 48
    assert {c["behavior"] for c in cases} == {"answer", "refuse"}


def test_label_mapping():
    assert label_for("answer") == 1 and label_for("refuse") == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_calibration.py -q -p no:cacheprovider`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现脚本**

按 `evals/run_ch04_evals.py` 的方式建 MySQL session、`MilvusVectorStore`、`get_embedder`、`get_reranker`、`QueryUnderstander`、`HybridRetriever`，再建 `RagPipeline(..., strategy="hybrid_rerank")`。逐条 `retrieve_for_gate` 采分数；检索异常记 `failed_ids` 并继续。调 `calibrate_loo`；若 `balanced_accuracy < 0.70`，**不写** artifact，只写报告并 `return 2`。写报告含 LOO 混淆矩阵、阈值、逐桶统计、逐条预测、失败样例。打印最终路径与关键数字。

- [ ] **Step 4: 真实跑一遍并检查产物**

Run: `uv run python scripts/calibrate_confidence.py`
Expected: 生成 `evals/ch09_confidence_model.json`（或按阈值拒绝发布）与 `dev-notes/ch09-confidence-calibration.md`；报告里能看到 balanced accuracy、混淆矩阵、阈值。

- [ ] **Step 5: 跑单测 + 提交并追记**

Run: `uv run pytest tests/test_ch09_calibration.py tests/test_ch09_confidence.py -q -p no:cacheprovider`
Expected: PASS

```bash
git add MemHelp-code/scripts/calibrate_confidence.py MemHelp-code/evals/ch09_confidence_model.json MemHelp-code/dev-notes/ch09-confidence-calibration.md MemHelp-code/tests/test_ch09_calibration.py MemHelp-code/app/config.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 用 ch04 评估集校准证据置信度阈值"
```

`dev-notes/ch09.md` 追加「阶段 8 · 任务 7 校准结果」，如实写指标；若未达标，写明已退回阈值闸。

---

### Task 8: 审核 API 与知识库回写

**Files:**
- Modify: `app/knowledge/ingest.py`（`vectorize_ids`）
- Modify: `app/api.py`（列表/详情/通过/驳回 + `/review` 挂载）
- Create: `app/flywheel/review.py`（审核业务：组装详情、通过落库）
- Test: `tests/test_ch09_review_api.py`

**Interfaces:**
- Consumes: Task 1 仓库、`knowledge.upsert_chunk`、`_vector_store_for`、`get_embedder`
- Produces:
  - `async vectorize_ids(chunk_ids, *, chunk_repo, vector_store, embedder, batch_size=32) -> int`
  - `ReviewDetail(review: ReviewRow, questions: list[LowConfidenceRow], snapshot: list[dict])`
  - `async approve_review(...) -> dict`、`async reject_review(...) -> bool`
  - 端点：`GET /api/review-queue`、`GET /api/review-queue/{id}`、`POST /api/review-queue/{id}/approve`、`POST /api/review-queue/{id}/reject`

- [ ] **Step 1: 写失败测试**

```python
async def test_approve_writes_knowledge_and_marks(client, fake_repositories, monkeypatch):
    rid = fake_repositories.reviews.add("退款多久到账？", "1-3 个工作日")
    qid = fake_repositories.low_confidence.add(None, "钱啥时候回来", "user_feedback",
        retrieved_chunks=[{"n": 1, "category": "售后政策", "quote": "x", "score": 0.9}])
    fake_repositories.low_confidence.attach_review(qid, rid)
    resp = await client.post(f"/api/review-queue/{rid}/approve", json={"approved_answer": "1-3 个工作日"})
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake_repositories.reviews.get(rid).review_status == "通过"


async def test_reject_does_not_touch_knowledge(...):
    ...
```

另测：详情返回 `questions[0].raw_question` 与 `snapshot[0]["quote"]`；重复通过返回 409；向量化失败时返回 `vectorize_status="pending"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_review_api.py -q -p no:cacheprovider`
Expected: FAIL（404）

- [ ] **Step 3: 实现**

`vectorize_ids` 复用 ingest 内部取块/渲染/upsert/mark_done 逻辑，只处理给定 id。`review.py` 查询该 review 关联的 low_conf 行，`snapshot` 取第一行的 `retrieved_chunks`；`approve_review` 先校验状态，再 `upsert_chunk`，再 `vectorize_ids`；向量化异常不抛，返回 pending。API 用 `get_repositories`/`get_retriever` 依赖，错误用 `redact`。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_review_api.py tests/test_ch03_ingest.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/tests/test_ch09_review_api.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 审核队列 API 与通过后知识库回写"
```

`dev-notes/ch09.md` 追加「阶段 9 · 任务 8 完成」。

---

### Task 9: 审核后台页面（Vibe Coding）

**Files:**
- Create: `web/review/index.html`、`web/review/review.js`
- Modify: `app/api.py`（`/review` 挂载，模仿 `/rag-eval`）
- 不写单测；按用户描述手工调。

- [ ] **Step 1: 搭页面骨架**

表格列：标准化问题 / 次数 / 示例答案（截断）/ 状态 / 操作；顶部状态过滤（待审/通过/驳回/全部）；空状态文案「暂无待审问题」。

- [ ] **Step 2: 接列表与详情**

`fetch('/api/review-queue?status=待审')` 渲染；点行 `fetch('/api/review-queue/{id}')` 展开：原话列表（source/时间）、召回片段（分数 + section_path + 原文，等宽、可滚动）。

- [ ] **Step 3: 通过 / 驳回**

通过弹窗预填示例答案、可编辑；提交 `POST .../approve`；驳回二次确认 `POST .../reject`；成功后刷新列表并提示。

- [ ] **Step 4: 手工验证**

起服务后走：未知问题 → 审核页出现 → 详情有原话与片段 → 通过 → 复问答对。记录到 dev-notes。

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/web/review MemHelp-code/app/api.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 审核后台页面（列表/详情/通过/驳回）"
```

---

### Task 10: 按意图的 token 成本（聚合 / 接口 / CLI）

**Files:**
- Create: `app/observability/intent_cost.py`、`scripts/intent_cost.py`
- Modify: `app/api.py`（`GET /api/intent-cost`）
- Test: `tests/test_ch09_intent_cost.py`

**Interfaces:**
- Produces:
  - `aggregate_intent_costs(rows: Sequence[dict]) -> list[IntentCost]`
  - `IntentCost(intent, tokens, cost, observations)`
  - `async fetch_intent_costs(settings, *, days=7, client=None) -> list[IntentCost]`
  - `GET /api/intent-cost?days=7 -> {"items": [...], "total_tokens": int}`

- [ ] **Step 1: 写失败测试**

```python
def test_aggregate_sums_by_intent_tag():
    rows = [
        {"tags": ["intent:refund"], "sum_totalTokens": 300, "sum_totalCost": 0.03},
        {"tags": ["intent:refund", "env:dev"], "sum_totalTokens": 200, "sum_totalCost": 0.02},
        {"tags": [], "sum_totalTokens": 99, "sum_totalCost": 0.01},
    ]
    items = aggregate_intent_costs(rows)
    assert items[0].intent == "refund" and items[0].tokens == 500
    assert all(item.intent != "" for item in items)
```

HTTP 测试用 `httpx.MockTransport` 返回固定 JSON，断言 Basic Auth 与 URL 参数。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_intent_cost.py -q -p no:cacheprovider`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

`fetch_intent_costs` 调 `GET {base}/api/public/v2/metrics`，`params={"query": json.dumps({...})}`，Basic auth；`fromTimestamp` 为 UTC 的 `now - days`。一行多个 tags 时取第一个 `intent:` 前缀；没有则丢弃。CLI 打印表格（意图/观测数/tokens/成本），按 tokens 降序。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_intent_cost.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/scripts/intent_cost.py MemHelp-code/tests/test_ch09_intent_cost.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 按意图聚合 Langfuse token 成本"
```

---

### Task 11: 评估记录、手动触发与周期调度

**Files:**
- Create: `app/eval_runs.py`、`app/eval_schedule.py`
- Modify: `app/config.py`（`eval_schedule_enabled`、`eval_schedule_interval_hours`、`eval_script_path`）
- Modify: `app/api.py`（`GET /api/eval-runs`、`POST /api/eval-runs/run`、lifespan 启停调度）
- Test: `tests/test_ch09_eval_runs.py`

**Interfaces:**
- Produces:
  - `extract_metrics(report: dict) -> dict`
  - `async record_report(repositories, report, *, triggered_by) -> int`
  - `async run_evaluation(*, repositories, trigger, settings) -> int`
  - `summarize_trend(rows: Sequence[EvalRunRow]) -> dict`
  - `start_scheduler(app) -> asyncio.Task | None`、`async stop_scheduler(task) -> None`

- [ ] **Step 1: 写失败测试**

```python
def test_extract_metrics_uses_hybrid_rerank_overall(fixture_report):
    metrics = extract_metrics(fixture_report)
    assert metrics["retrieval"]["recall@5"] == 0.9
    assert metrics["generation"]["faithfulness"] == 0.88


def test_trend_marks_downward():
    trend = summarize_trend([_row(0.9), _row(0.8)])
    assert trend["faithfulness"]["direction"] == "down" and trend["faithfulness"]["delta"] < 0
```

`test_record_report_writes_row` 断言 `dataset_size` 与 `triggered_by`；`test_run_evaluation_uses_injected_report` monkeypatch `_run_report` 返回 fixture，断言写库。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_eval_runs.py -q -p no:cacheprovider`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

`extract_metrics` 读 `report["retrieval"]["hybrid_rerank"]["overall"]` 与 `report["generation"]["hybrid_rerank"]["overall"]`，缺键记 None。`_run_report` 用 `asyncio.create_subprocess_exec(sys.executable, "evals/run_ch04_evals.py", "--json", str(tmp), "--markdown", ...)`，1200 秒超时，失败抛 `RuntimeError`。`run_evaluation` 成功后写库。`summarize_trend` 按时间升序，`delta = 最后一轮 - 上一轮`，方向阈值 1e-9。调度器循环 `await asyncio.sleep(interval)` 后跑，异常只 log；启动时立即返回 task（第一轮等一个间隔）。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_ch09_eval_runs.py tests/test_rag_eval.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/app MemHelp-code/tests/test_ch09_eval_runs.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): eval_runs 记录、手动触发与周期调度"
```

---

### Task 12: 评估趋势页（Vibe Coding）

**Files:**
- Create: `web/eval-trend/index.html`、`web/eval-trend/trend.js`
- Modify: `app/api.py`（`/eval-trend` 挂载）

- [ ] **Step 1: 表格**

每行：时间 / 触发方式 / 条数 / recall@5 / mrr@10 / faithfulness / refusal_accuracy；无数据显示「—」。

- [ ] **Step 2: 趋势箭头**

每列最后一行旁显示相对上一轮：↑ 绿、↓ 红、→ 灰；没有上一轮不显示。

- [ ] **Step 3: 触发按钮**

「立即跑一轮」调 `POST /api/eval-runs/run`，轮询列表直到新行出现（最多 10 分钟），期间禁用按钮。

- [ ] **Step 4: 手工验证**

跑两轮后打开 `/eval-trend`，确认两行数字与箭头方向正确。

- [ ] **Step 5: 提交并追记**

```bash
git add MemHelp-code/web/eval-trend MemHelp-code/app/api.py MemHelp-code/dev-notes/ch09.md
git commit -m "feat(ch09): 评估趋势页面"
```

---

### Task 13: 端到端演示、清理与交付

**Files:**
- Create: `scripts/demo_ch09.py`
- Delete: `scripts/spike_langfuse.py`、`scripts/spike_langfuse_tags.py`、`scripts/spike_graph_config.py`、`scripts/spike_graph_trace.py`
- Modify: `dev-notes/ch09.md`（收尾阶段）
- Modify: `.env.example`（ch09 配置）

- [ ] **Step 1: 写演示脚本**

脚本按顺序：
1. `GET /healthz`；
2. 问一个知识库没有的问题（如「支持以旧换新吗？」）→ 断言回复是兜底话术；
3. `GET /api/review-queue?status=待审` 找到该问题 → 打印详情里的原话与 Top 片段；
4. `POST /api/review-queue/{id}/approve` → 再问同一问题 → 断言回复不再是兜底且引用包含新块；
5. 对一条回答 `POST /api/feedback`（down）→ 轮询审核队列出现；
6. `GET /api/intent-cost` 打印表格；
7. `POST /api/eval-runs/run` 两次（或检查已有两轮）→ `GET /api/eval-runs` 打印趋势。
每步失败抛异常并以非 0 退出；参数 `--base-url` 默认 `http://127.0.0.1:8000`。

- [ ] **Step 2: 起服务实测**

```powershell
$env:UV_CACHE_DIR="F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
& "C:\Users\dbq\.local\bin\uv.exe" run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

先对 MySQL 应用 `sql/ch09-ddl.sql`，再 `uv run python scripts/demo_ch09.py`；把实际输出贴进 dev-notes。

- [ ] **Step 3: 全量测试 + 清理**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: 全绿（若既有用例因签名变化失败，逐个修到绿，不删断言）。
删除四个 spike 脚本；确认 `dev-notes/ch09.md` 每个阶段都有记录。

- [ ] **Step 4: 写验收结论与交付说明**

`dev-notes/ch09.md` 追加「阶段 14 · finish」：六条验收逐条对应证据（Langfuse 链接、审核页操作、意图成本输出、两轮评估数字）、测试统计、演示命令。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code
git commit -m "feat(ch09): 端到端演示脚本与交付收尾"
```

---

## Self-Review

- **Spec coverage:** 需求 1→Task 2；2→Task 10；3→Task 4/5/6/7；4→Task 1/5/8/9；5→Task 11/12；6→Task 1/8/9；本章不做（主题分类器）未实现，符合 spec。验收 1–6 均在 Task 13 有对照。
- **类型一致:** `ReviewRow`/`LowConfidenceRow` 字段在 Task 1 定义，Task 5/6/8 复用；`EvidenceFeatures` Task 3 → Task 4；`FlywheelService.ingest` Task 5 → Task 4/6 调用点；`message_id` Task 6 → 前端 Task 6。
- **Review Focus:** 五条各自落到 Task 2（Langfuse 降级）、Task 5（查重幂等）、Task 8（向量化失败）、Task 4（degraded 放行）、Task 11（趋势方向）。
- **比例:** 计划以签名/测试断言为主，未贴实现全文；每个数据/Prompt 任务保留真实验证步骤（Task 7 校准、Task 13 端到端）。
