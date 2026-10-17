# MewHelp 第四章「混合检索 + 重排 + 评估体系」设计文档

> 状态：设计已获用户逐节认可（2026-10-17），待 spec 评审。
> 上游章节：`docs/superpowers/specs/2026-10-17-ch03-rag-design.md`（向量单路检索）。

---

## 0. 流程分类与范围

**分类：architectural。** 本章同时改数据层（Milvus 集合结构、MySQL 两张新表 + 一列）、
检索层（双路召回 + 融合 + 重排）、生成层（引用与拒答）与评估层（新评估集与报告），
并且改变跨章节的既有约定（Milvus 里「只存 id + 向量」→「存检索所需的副本」）。

**范围内（7 条功能需求）**

1. 混合检索：dense 与 Milvus 原生 BM25 各召 Top-50，`hybrid_search` + RRF 融合；
2. 重排：`bge-reranker-v2-m3` 精排 Top-10；塞进 prompt 时最相关的放首尾；
3. 元数据过滤：品类这类字段**先过滤再检索**；
4. Query 理解：口语问法改写归一到标准问法；同义词扩展**只在检索侧**做；
5. 生成质量控制：引用编号可回跳章节路径与原文、低置信度显式拒答、模型自评入池、
   System Prompt 列明禁止承诺的负面知识；
6. 评估体系：带难度梯度与 ground-truth 的评估集、四策略对比、按 query 类型分桶；
7. 前端：引用编号可点开看原文与章节路径，每条回答的 👍/👎 一次性反馈。

**范围外（用户指定）**

- 指代消解、多轮改写；
- LangGraph / Langfuse（本章管线是四步线性确定性流程，引入图运行时只增加间接层；
  观测另属后续章节）；
- `faith_cases` 的管理页面（只做最小查询/处置端点）；
- 👎 的后端落库（本轮纯前端采集）。

### 验收标准与判定方式

| # | 验收标准 | 判定方式（可复现） |
|---|---|---|
| 1 | 四策略对比报告能跑出数字 | `evals/reports/ch04-*.json` + `dev-notes/ch04-eval-report.md` 存在且含 4 策略 × 4 桶的 Recall/MRR 数字 |
| 2 | 带具体型号的问题，BM25 那路能命中 | `run_ch04_evals.py --only B_model --strategies bm25 -v` 逐条打印命中 chunk，且 `B_model` 桶 BM25 Recall@5 ≥ 0.8 |
| 3 | 引用编号能定位回原文，聊天页点引用能看到来源原文 | 页面点 `[n]` 弹出抽屉含 `section_path` 与原文；`messages.citations` 刷新后仍在 |
| 4 | 问知识库没有的内容，得到明确拒答且进池 | 页面收到拒答话术；`low_confidence_questions` 新增一行，`source` 与 `reason` 非空 |

---

## 1. 已确认的决策（对话中逐个敲定，全部选 A）

| # | 决策点 | 结论 | 否掉的选项 |
|---|---|---|---|
| 1 | 管线接线形态 | 沿用 ch02 模型路由，`query_faq` **内核整体换掉**，出参升级为带编号证据 | 新增「知识问答直通路径」（要多一个路由器、改 SSE 契约）；新增独立 `/api/rag/answer`（与「在现有聊天页上点引用」冲突） |
| 2 | Milvus 集合 | **原地重建** `knowledge`，存 `text/category/content_type/dense/sparse` | 新建 `knowledge_ch04`（两套集合、口径分裂）；Milvus 只加 text、品类回 MySQL 后过滤（与「先过滤再检索」相反） |
| 3 | 引用持久化 | 追加 `messages.citations JSON`，存**证据快照** | 不落库（刷新即失效）；复用 `tool_traces`（语义混淆） |
| 4 | reranker 服务 | 复用硅基流动同一把 Key 调云端 `POST {base}/rerank` | 本地 CrossEncoder（2GB 下载 + 沙箱风险）；两套都写（代码量翻倍） |
| 5 | 语料与评估集 | 新增《商品型号手册》、评估集 48 条四桶、gold 用**唯一锚串**标注 | 不动语料（验收 2 名不副实）；评估集只扩到 27 条（分桶统计意义弱） |
| 6 | `faith_cases` 范围 | 落表 + 最小查询/处置端点，不做页面 | 只落表无入口（DDL 里「人工点按钮改」永远没人按）；完整管理页（超出前端范围） |
| 7 | 入池口径 | `retrieval_low_conf` + `self_check` 两个入口都开，`user_feedback` 留给后续章节 | 只用 self_check（「检索不到」那类问题会丢）；只用阈值（与「让模型自评」冲突） |

---

## 2. 现状勘查与环境实测

### 2.1 仓库现状（读代码确认，非记忆）

| 项 | 现状 | 对本章的影响 |
|---|---|---|
| 检索入口 | 模型调 `query_faq`（ch02 单轮编排，`app/chat_tools.py`） | 引用映射要在这里捕获证据 |
| 出参契约 | `{ok, keyword, count, items:[{question, answer, category}]}`，被 `tests/test_tool_knowledge.py` 与 ch03 评估集钉住 | 只**增**字段，不改原有键 |
| Milvus | 集合 `knowledge` 只有 `id + vector`（`app/knowledge/vector_store.py`） | 必须重建集合；`VectorStore.upsert` 签名要升级 |
| 原文 | `knowledge_chunks` 有 `section_path`（`商品FAQ / 商品与配送 / 运费怎么算`）、`category`、`content_type` | 引用展示与过滤都靠它 |
| 向量化文本 | `chunker.render_embed_text()`：`分类：…\n问法：…\n答案：…` | dense 继续用它；BM25 用另一份不带标签的文本 |
| `messages` 表 | 有 `tool_calls` / `tool_traces`，**没有**存引用的列 | 需要 ALTER |
| 语料 | 三份 md 共 ~100 行，**无任何型号** | 验收 2 无料可测，需新增语料 |

### 2.2 环境实测（探针结果）

- Milvus 服务端 **2.6.22**，pymilvus **2.6.17**（`client.get_server_version()`）。
- `pymilvus` 暴露 `Function` / `FunctionType` / `AnnSearchRequest` / `RRFRanker`，
  `MilvusClient.hybrid_search` 存在。
- 现有集合：`['knowledge', 'ch03_integration_test']`。
- Milvus 健康检查 `http://127.0.0.1:9091/healthz` 返回 `OK`。
- Docker CLI 在沙箱里不可用（命名管道被拒），**不影响**直连 19530。

### 2.3 Context7 核对结果（官方文档，非记忆）

| 事实 | 结论 |
|---|---|
| BM25 全文检索的建集合方式 | `text` 字段 `enable_analyzer=True` + `analyzer_params={"type": "chinese"}`；`sparse` 字段为 `SPARSE_FLOAT_VECTOR`；`schema.add_function(Function(function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))`；**函数必须在建集合前加上**，事后无法补 |
| 稀疏索引参数 | `index_type="SPARSE_INVERTED_INDEX"`、`metric_type="BM25"`、`params={"inverted_index_algo":"DAAT_MAXSCORE","bm25_k1":1.2,"bm25_b":0.75}` |
| chinese analyzer | `{"type": "chinese"}`，官方内置、无额外参数；等价于 `jieba` 分词 + `cnalphanumonly` 过滤 |
| BM25 检索的入参 | `data` 传**原始查询文本**（不是稀疏向量），`anns_field` 指向 sparse 字段 |
| 混合检索 | `client.hybrid_search(reqs=[AnnSearchRequest(...), ...], ranker=RRFRanker(), limit=..., filter=...)`，`filter` 是顶层标量过滤表达式 |
| 重排服务 | 硅基流动 `POST {base}/rerank`，body `{model, query, documents, top_n, return_documents}`，模型名 `BAAI/bge-reranker-v2-m3` |

> 未在文档里 100% 确认、留待 Task 0 用**临时集合**实测的细节：`AnnSearchRequest`
> 的构造参数名（`param` vs `params`）、rerank 响应里 `results[].index` 的字段名、
> chinese analyzer 对 `XH-3000A` 这类型号的切分结果。三者都用真机探针钉住，
> 不猜。

### 2.4 用户更正与边界

- 前端部分**不套** brainstorm / TDD / code review，按 Vibe Coding 做（用户明确要求）。
- 技术选型（Milvus 原生 BM25、bge-reranker-v2-m3）定死，走不通要停下来问，不自行换方案。

---

## 3. 数据层设计

### 3.1 Milvus 集合 `knowledge`（ch04 重建）

| 字段 | 类型 / 参数 | 说明 |
|---|---|---|
| `id` | `VARCHAR(64)`，主键，`auto_id=False` | 仍是 `str(knowledge_chunks.id)`，沿用 ch03 的确定性主键与 `upsert` 幂等 |
| `text` | `VARCHAR(8192)`，`enable_analyzer=True`，`analyzer_params={"type":"chinese"}` | BM25 原文。内容 = `问法\n答案`（**不带**「分类/问法/答案」标签：标签是每个块都有的高频词，会稀释 BM25 的区分度） |
| `category` | `VARCHAR(255)` | 前置过滤标量字段 |
| `content_type` | `VARCHAR(32)` | 次级过滤（`faq` / `policy` / `manual` / `mined`） |
| `dense` | `FLOAT_VECTOR(dim=EMBEDDING_DIM)`，`AUTOINDEX` / `COSINE` | 文本仍用 `render_embed_text()`，与 ch03 语义一致 |
| `sparse` | `SPARSE_FLOAT_VECTOR` + `Function(BM25)` | 索引 `SPARSE_INVERTED_INDEX` / `BM25`，`k1=1.2`、`b=0.75` |

**口径变更（必须写进 README 与注释）**：ch03 的「Milvus 不存第二份真相」升级为
「**Milvus 是检索索引副本**」。`text` / `category` / `content_type` 是从
`knowledge_chunks` 派生出来的只读副本，不参与人工编辑；改原文必须重建索引，
否则 BM25 与过滤会按旧副本工作。

**集合版本自检**：`ensure_collection()` 在集合已存在时检查 (a) `dense` 维度、
(b) 是否存在 `text` 与 `sparse` 字段。缺任一 → 抛可执行错误：
「ch04 需要重建集合：`python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild`」。
这一条替代 ch03 只查维度的检查，避免「集合是旧的、BM25 静默全空」。

### 3.2 写入路径

`VectorStore` 协议升级（ch03 的 `upsert(ids, vectors)` 无法携带 text/category）：

```python
@dataclass(frozen=True)
class VectorRecord:
    id: str
    dense: Sequence[float]
    text: str
    category: str
    content_type: str | None

class VectorStore(Protocol):
    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...
    async def search_dense(self, vector, top_k, *, expr=None) -> list[VectorHit]: ...
    async def search_bm25(self, text, top_k, *, expr=None) -> list[VectorHit]: ...
    async def hybrid_search(self, dense_vector, bm25_text, top_k, *, expr=None) -> list[VectorHit]: ...
```

`search`（ch03 的单一入口）保留为 `search_dense` 的别名，ch03 的调用方与测试少改。

### 3.3 MySQL 变更

**新增表**（用户已提供 `sql/ch04-ddl.sql`，不重新设计）：
`low_confidence_questions`、`faith_cases`。

**新增列**（本章追加，写进 `sql/ch04-ddl.sql` 末尾）：

```sql
ALTER TABLE messages ADD COLUMN citations JSON NULL
  COMMENT '本轮引用快照:[{n,chunk_id,section_path,category,quote}];刷新后引用仍可点开';
```

**新增 ORM**：`Message.citations`、`LowConfidenceQuestion`、`FaithCase`
（逐列与 DDL 对齐，沿用 ch02/ch03 的「列名集合由测试逐个钉住」做法）。

**新增仓库**：

| 仓库 | 方法 | 语义 |
|---|---|---|
| `LowConfidenceRepository` | `add(conversation_id, raw_question, source, reason)` | 入池；`source` 由调用方决定，非法值在 flush 时抛 `LookupError` |
| | `list_recent(limit)` / `count_by_source()` | 演示与复盘用 |
| `FaithCaseRepository` | `upsert_from_judgement(...)` | 一题一行：`seen_count+1`、刷新答案/理由/角标快照、`last_seen_at` 更新；已「已解决」又被判出 → 退回「未解决」 |
| | `list_by_status(status)` / `update_status(eval_id, status, resolution)` | 最小处置入口；标「已解决/无需解决」必须带 `resolution`，退回时清空 |

### 3.4 重建与幂等

`build_knowledge.py --rebuild` 仍是唯一重建入口：先 drop 集合再清表（顺序不可反，
理由见 ch03 spec §3.3），然后按新 schema 建集合并写入 `VectorRecord`。
`vectorize_status` 状态机保持不变。

---

## 4. Query 理解（`app/knowledge/query.py`）

**目标**：把「钱啥时候能回来」这类口语问法归一到标准问法「退款到账时间」；
同义词扩展**只在检索侧**做，入库侧不拆存多份。

**两级策略**（避免每次检索都多一个 RTT）：

1. **静态归一表命中** → 直接返回。纯函数、零延迟、可单测。
   初始表覆盖评估集里的口语问法（退款到账/包邮门槛/退货运费/发货时效/赔付上限/换货期限）。
2. **未命中且判定为口语** → 调一次小 LLM 改写（`max_tokens` 收紧、超时 3s、
   按 query 哈希做进程内 LRU 缓存 512 条）。判定规则：含语气词/疑问代词，
   或长度 ≤ 12 字且未命中任何领域词。

**输出结构**：

```python
@dataclass(frozen=True)
class UnderstoodQuery:
    original: str
    standard: str          # 归一后的标准问法（用于 dense 与 BM25 的语义主体）
    bm25_text: str         # standard + 同义词扩展（**只**给 BM25 用）
    expanded: tuple[str, ...]  # 命中的同义词，进日志与报告
```

**为什么同义词不扩 dense**：向量检索对同义表达本就不敏感，把同义词拼进 dense 查询
会稀释语义中心（一句话变三个意思）。BM25 是词面匹配，扩同义词才有收益。

**失败语义**：改写超时/解析失败 → 用原话继续检索（记 `rewrite_degraded=true`）。
改写失败不该让用户查不到东西。

---

## 5. 混合检索与 RRF（`app/knowledge/hybrid.py`）

**一个接口，四策略**（评估脚本与线上共用同一份实现）：

```python
Strategy = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]

async def retrieve(
    strategy: Strategy,
    query: UnderstoodQuery,
    *,
    top_k: int = 10,
    category: str | None = None,
    content_type: str | None = None,
) -> list[RetrievedChunk]
```

| 策略 | 实现 | 返回 |
|---|---|---|
| `vector` | `search_dense(dense_vec, RAG_HYBRID_TOP_K)` | 按 COSINE 降序 |
| `bm25` | `search_bm25(bm25_text, RAG_HYBRID_TOP_K)` | 按 BM25 分降序 |
| `hybrid` | `hybrid_search(dense_vec, bm25_text, RAG_CANDIDATE_K)` → RRF | 按 RRF 分降序 |
| `hybrid_rerank` | `hybrid` 取候选 20 → `/rerank` → 前 10 | 按 relevance_score 降序 |

**参数**：`RAG_HYBRID_TOP_K=50`（每路召回）、`RAG_CANDIDATE_K=20`（进重排的候选）、
`RAG_RERANK_TOP_N=10`（最终证据数）、`RRFRanker(k=60)`。

**为什么 RRF 之后还要放大到 20 再重排**：RRF 只用名次，两路各 50 条融合后
真正有区分度的头部可能只有 10 条左右；先取 20 给重排模型一个跨路候选池，
再收 10，比「融合后直接取 10」召回更高。

**去重**：Milvus 两路命中的是同一个 `id`（同一块），RRF 天然按 id 合并；
回 MySQL 取原文按 id 去重，保留 RRF 名次最高的那次。

**并发**：dense 嵌入与 BM25 文本检索互不依赖，用 `asyncio.gather` 并行；
`hybrid_search` 是单次 Milvus 调用（服务端融合），不需要客户端并行。

---

## 6. 重排（`app/knowledge/rerank.py`）

```python
class Reranker(Protocol):
    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]: ...

@dataclass(frozen=True)
class RerankHit:
    index: int          # 候选列表中的下标
    score: float        # relevance_score
```

**实现**：`SiliconFlowReranker`，`POST {RERANK_BASE_URL}/rerank`，
body `{model, query, documents, top_n, return_documents: False}`，
`documents` 用「问法 + 答案」文本（与 BM25 的 text 同源，避免重排看的是另一份文本）。

**降级**：没配 Key、超时、非 2xx → 记 `warning` 并**返回 RRF 序的前 N 条**，
`rerank_degraded=True` 一路带回报告。重排是提质手段，不是检索的硬依赖；
把重排故障升级成「检索失败」会让一次网络抖动表现为「客服什么都不知道」。

**为什么不重排 50 条**：重排成本与候选数线性相关，而 RRF 头部 20 条已覆盖绝大多数
相关块；用评估集回归确认 20 → 10 的 Recall@10 与 50 → 10 无显著差异（Task 里跑一次对照）。

---

## 7. 元数据过滤

**前置过滤**：过滤表达式在**发起检索之前**交给 Milvus（作为每个 `AnnSearchRequest` 的 `expr`），
不是拿回结果再筛。后过滤会把 Top-50 先截断再丢掉一半，等于自残召回。

> 实测坑（pymilvus 2.6.17）：`hybrid_search` 的顶层 `filter` 参数**被静默忽略**，
> 顶层 `expr` 会直接抛 `TypeError`。所以过滤必须挂到**两个** `AnnSearchRequest.expr` 上 ——
> 两路都过滤，融合后的候选才都满足条件。这条已写进 Task 3/Task 8 的测试断言。

- `query_faq` 内部支持**可选** `category` 参数（`query_faq(keyword, category=...)`），
  传了就按品类前置过滤；**但它不进工具 Schema** —— 实测模型会把型号题猜成
  `category="商品参数"`，而语料里的品类是「商品型号手册 / 巡检仪」这种真实路径，
  猜错就把检索过滤成空，用户收到的是一句误拒。过滤能力由评估脚本与集成测试调用，
  不交给模型自由填字符串（这条在执行期按实测修正，见 dev-notes 阶段 6）。
- 评估脚本支持 `--category` 做单桶过滤验证；真机过滤由
  `tests/test_knowledge_milvus.py::test_per_request_expr_filters_both_paths` 钉住。
- 表达式只允许等值：`category == "..."`，`content_type == "..."`。
  不做 `LIKE` / `in` —— 这一章要证明的是「先过滤再检索」这个结构，不是查询语言。

**为什么不做自动品类识别**：那是意图分类，属于下一章的路由话题；本章只把
「能按品类过滤」这条通路打通并测到。

---

## 8. 证据编排与引用（`app/knowledge/citations.py`）

### 8.1 编号

```python
@dataclass(frozen=True)
class Evidence:
    n: int                 # 相关度序号 1..N，**用户看到的角标就是它**
    chunk_id: int
    section_path: str
    category: str
    quote: str             # chunk 的 answer 原文（证据快照，供抽屉展示与审计）
    score: float           # 重排分或 RRF 分
    source: str            # vector / bm25 / both / rerank
```

编号只由**相关度**决定，不随呈现顺序变化 —— 否则同一块在不同轮次里角标不同，
用户点开 `[3]` 看到的和上一次不是同一条。

### 8.2 首尾编排（需求 2）

塞进 prompt 的顺序：奇数位升序 + 偶数位降序：

```
k=10 → [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
k=5  → [1, 3, 5, 4, 2]
```

通式：`[n | n 为奇数且升序] + [n | n 为偶数且降序]`。k 为奇数时最后落单的奇数位
（示例里的 5）落在第二个位置，最弱的证据仍在正中。

最相关的两条落在首尾（模型中段注意力最弱），最不相关的压中间。
编号不变，所以正文里的 `[n]` 仍指向正确证据。

**为什么不是简单倒序或随机**：首尾两个位置都能吃到注意力，把 top1 与 top2 都放在
首尾，等于用两个位置覆盖最强的两条；中间放弱证据，即使被忽略损失也最小。

### 8.3 引用解析与校验

- 正则 `\[(\d{1,2})\]` 提取正文角标；
- 合法集合 = 本轮 `Evidence.n` 的集合；越界编号**不删正文**，只丢弃映射并记日志
  （模型多写一个角标不该让整段回答消失）；
- 输出 `{citations: [...], dropped: [...], cited_count, coverage}`；
- `coverage = 被引用证据数 / 证据总数`，进评估报告（低覆盖意味着证据没用上）。

### 8.4 落库与传输

- SSE `done` 事件新增 `citations` 字段；
- `messages.citations` 存同一份快照（**快照而不是指针**：chunk 日后被重切，
  审计时仍能看到当时依据；`faith_cases.citations` 用的是同一份结构）；
- `/api/chat/sessions/{id}/messages` 回读时带上 `citations`，刷新后角标仍可点。

---

## 9. 低置信度：自评、拒答、入池

### 9.1 两道闸

| 闸 | 触发条件 | `source` | 谁判 |
|---|---|---|---|
| 检索层 | 0 命中，或最高分 < `RAG_MIN_RERANK_SCORE`（默认 0.3） | `retrieval_low_conf` | 确定性规则 |
| 生成层 | 模型自评 `sufficient=false` | `self_check` | judge 模型 |

这两道闸**只在线上链路（`hybrid_rerank`）生效**：评估脚本要量的就是「没有闸时四策略
各自的表现」，所以评估跑检索时不做拒答、也不写池。反过来说，`RAG_MIN_RERANK_SCORE`
只对带重排分的管线有意义；纯 BM25 / 纯向量策略没有重排分，不参与该闸的标定。

`RAG_MIN_RERANK_SCORE` 用评估集的 8 条负样本 + 40 条正样本标定：取「正样本最低分」
与「负样本最高分」之间的中点；**若两侧分数重叠**（正样本最低分 < 负样本最高分），
改取正样本 5 分位与负样本 95 分位的中点，并把重叠区间写进报告 —— 重叠本身就是要
报出来的事实，不是要抹平的瑕疵。分数区间以 Task 0 实测为准（若接口返回 logits 而不是
0~1 的 relevance，阈值按实测重新标定）。

### 9.2 自评 judge

- 入参：原话 + 编号证据（相关度序）+ 要求严格 JSON
  `{"sufficient": bool, "missing": str, "reason": str}`；
- `max_tokens` 收紧、超时 5s；
- **解析失败时 fail-open**（当 `sufficient=true`）+ 记 `self_check_degraded`。
  理由：JSON 解析失败是我方 prompt/模型的问题，fail-closed 会把每一次格式抖动
  变成「客服什么都不知道」，并污染低置信度池；而证据已经过了分数阈值这道确定性闸。

### 9.3 拒答的短路实现

自评不足或检索闸触发时，编排层**不发起第二次模型调用**，直接流式吐出拒答话术。

```
query_faq 返回 {refused: true, source, reason, message}
  → chat_tools 看到 refused → 落库 user 轮 + 拒答正文 → 写池 → emit citations: []
```

**为什么短路而不是「告诉模型请拒答」**：验收 4 是硬标准，不能让「模型今天愿不愿意拒答」
决定它过不过。短路是结构性的：证据不足这条路根本走不到生成。

### 9.4 拒答话术

固定模板，明确三件事：知识库里没有这条 → 不猜 → 给下一步（转人工/建单/补充信息）。
话术里**不出现**引用角标（没有证据可引）。

### 9.5 入池

`LowConfidenceRepository.add(conversation_id, raw_question, source, reason)`。
`conversation_id` 经 `conversations.upsert_by_session` 取得（幂等），
避免「池子里有一行但会话 FK 为空」。

### 9.6 残留风险（明确承认）

模型若**根本没调** `query_faq` 就凭印象回答政策题，后端拿不到证据、拦不住。
对策只有两条：System Prompt 强约束 + 评估集把「负样本未拒答率」当成一个指标统计出来，
而不是假装它不存在。

---

## 10. Prompt 与负面知识

**新增行为约束**（进 `BEHAVIOR_RULES_WITH_TOOLS`，逐字被测试钉住）：

- `不承诺到账时间与送达时点`：退款/赔付到账、发货、送达、审核完成的时间一律不给
  确定承诺，只说「以审核/实际为准」。
- `无证据不答`：涉及平台规则、商品参数、政策条款，必须基于本轮证据回答；
  证据里没有就不说。

**引用规范段**（新增一节，位置在「你有查询工具」之后）：

- 正文用 `[n]` 标注，`n` 只能取本轮证据的编号；
- 一句话里有多个依据就并列 `[1][3]`；
- 没有依据的句子不要写；
- 不要输出证据全文，用编号指路。

**负面知识清单**（禁止承诺）：到账时间、发货时间、送达时间、赔付金额结论、
库存保证、审核通过率。清单以「禁止承诺 X」的句式写进 System Prompt，
评估集用负样本验（问「退款多久到账」必须拒答或转人工，不能给「3 个工作日」以外的
承诺 —— 而 3 个工作日本身是知识库里的规则，允许引用 `[n]` 后陈述）。

> 注意区分：**引用知识库里的规则数字**（允许，必须带 `[n]`）与
> **对本笔订单做承诺**（禁止）。这是负面知识清单要解决的真问题。

---

## 11. 评估体系

### 11.1 评估集 `evals/ch04_cases.jsonl`

48 条 = 四桶 × 12（每桶 10 条可答 + 2 条「库里没有」）：

| 桶 | 覆盖 | 难度梯度 |
|---|---|---|
| `A_policy` | 退货/运费/发票/赔付/时效条款 | easy 直问、medium 换说法、hard 条款边界 |
| `B_model` | 型号参数（新语料的独有串） | easy 报型号、medium 报场景问型号、hard 两型号对比 |
| `C_colloquial` | 口语模糊问法 | medium/hard |
| `E_multi` | 跨节多跳（包邮门槛 + 破损赔付） | hard 为主 |

字段：

```json
{"id":"B01","bucket":"B_model","difficulty":"easy",
 "query":"XH-3000A 的续航多久",
 "behavior":"answer","gold_anchors":["XH-3000A"],"must_contain":["18 小时"]}
{"id":"E12","bucket":"E_multi","difficulty":"hard",
 "query":"杭州买 80 块还差多少包邮，路上碎了能赔多少",
 "behavior":"answer","gold_anchors":["杭州","赔付"],"must_contain":["99","500"]}
{"id":"A11","bucket":"A_policy","difficulty":"hard",
 "query":"你们支持货到付款吗",
 "behavior":"refuse","gold_anchors":[]}
```

**gold 用锚串标注**：跑评估时在 `knowledge_chunks` 的 `questions` / `answer` 上做子串匹配，
把锚串解析成 chunk_id 集合；gold 集合 = 各锚串命中块的**并集**。判定规则：

- 某个锚串命中 **0 块** → 直接报错退出（标注写错了，或语料里根本没这段）；
- 某个锚串命中 **> 3 块** → 打印警告（锚串太泛，建议加长），不中断；
- 每个锚串命中的块数写进报告，方便回头修标注。

锚串而不是硬写 chunk_id：重切 chunk 后锚串仍然有效，不会一改切分就全错。

### 11.2 指标

| 段 | 指标 | 口径 |
|---|---|---|
| 检索 | `Recall@1/3/5/10`、`MRR@10` | 只算 40 条正样本；命中 = 任一 gold chunk 出现在前 K |
| 生成 | `Faithfulness` | claim 级：答案拆句 → 每条判「有无证据支持」；无引用句一律 unsupported |
| 生成 | `拒答准确率` | 8 条负样本里被正确拒答的比例 |
| 生成 | `引用有效率` | 正文 `[n]` 能映射到证据的比例 |
| 生成 | `未检索就答率` | 负样本里既没拒答、又没走检索的比例（幻觉代理指标） |

### 11.3 四策略与成本

- 检索段：4 策略 × 40 条，全跑（便宜）。
- 生成段：默认 `hybrid` + `hybrid_rerank` 两策略全量（48 × 2 次生成 + 裁判），
  `--all-strategies` 可四策略全跑。
- 报告里没跑的格子写「—」并注明「未评测」，**不填 0 冒充**。

### 11.4 输出

- `evals/reports/ch04-<timestamp>.json`：逐条明细 + 分桶聚合；
- `dev-notes/ch04-eval-report.md`：三张表（策略 × 桶 的 Recall@5 / MRR / Faithfulness）
  + 阈值标定说明 + 降级/失败计数；
- 裁判判出的编造条目 upsert 进 `faith_cases`。

### 11.5 退出码

`0` = `hybrid_rerank` 整体 Recall@5 ≥ `vector` 且 `B_model` 桶 Recall@5 ≥ `bm25`；
否则 `1`。这条是验收 1 的机器判定。

---

## 12. 前端（Vibe Coding，不套 TDD / review）

- **引用角标**：正文里的 `[n]` 渲染成可点徽章；点击打开右侧证据抽屉，显示
  `section_path`（`商品FAQ / 商品与配送 / 运费怎么算`）+ 该 chunk 原文（`quote`）。
  数据来自 `done.citations`；刷新后由 `/messages` 的 `citations` 还原。
- **满意度**：每条 bot 回答左下角 👍 / 👎。点击后：所选点亮、另一侧禁用、
  旁边显示「已反馈」，`localStorage`（键含 session 与消息序号）一次性锁定，刷新不重置。
- **不做**：反馈上报后端、聚合统计、引用高亮定位（都属于数据飞轮/下一章）。
- 工作方式：我按描述先做一版，你在页面上看效果、直接说改哪里；不写单测、不走 review。

---

## 13. 配置新增（`app/config.py` + `.env.example`）

| 键 | 默认 | 说明 |
|---|---|---|
| `RERANK_BASE_URL` | `https://api.siliconflow.cn/v1` | 重排服务根地址（实际请求 `{base}/rerank`） |
| `RERANK_API_KEY` | 空 | 空则回落到 `EMBEDDING_API_KEY`（同一家账户） |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 用户指定 |
| `RERANK_TIMEOUT` | `10` | 秒 |
| `RAG_HYBRID_TOP_K` | `50` | 每路召回数 |
| `RAG_CANDIDATE_K` | `20` | 进重排的候选数 |
| `RAG_RERANK_TOP_N` | `10` | 最终证据数 |
| `RAG_MIN_RERANK_SCORE` | `0.3` | 检索层低置信阈值（用评估集标定） |
| `RAG_QUERY_REWRITE` | `true` | 关掉则只用原话检索（排障用） |
| `RAG_REWRITE_TIMEOUT` | `3` | 秒 |

---

## 14. 测试策略

**单测（全部注入假件，离线可跑）**

| 模块 | 钉住什么 |
|---|---|
| `test_knowledge_query` | 静态表命中不调模型；口语才调改写；改写失败回落原话；缓存命中 |
| `test_knowledge_hybrid` | 四策略的请求参数（`anns_field`、`limit`、`filter` 表达式、BM25 传**文本**而非向量）；RRF 去重与排序；dense/BM25 并行 |
| `test_knowledge_rerank` | 请求体字段；无 Key/超时/5xx 降级为 RRF 序且带 `rerank_degraded` |
| `test_knowledge_citations` | 编号按相关度；首尾编排序列 `[1,3,5,7,9,10,8,6,4,2]`；`[n]` 越界丢弃；coverage 计算 |
| `test_knowledge_sufficiency` | JSON 解析；不足→拒答；解析失败 fail-open；超时降级 |
| `test_tool_knowledge` | 出参向后兼容（原有键仍在）+ 新证据字段；refused 短路 |
| `test_chat_citations` | 证据捕获、拒答短路不发起第二次模型调用、入池参数、`messages.citations` 落库 |
| `test_db_models` | 两张新表 + `messages.citations` 逐列对齐 |
| `test_prompts`（扩展） | 新增约束条款与引用规范逐字存在 |

**真机集成**：真 Milvus 建集合 + 型号 BM25 命中 + 过滤；真 MySQL 建表/列/仓库；
真 `/rerank` 调一次。集成测试沿用 ch03 的 skip 策略（没起服务就跳过，不伪装通过）。

**非代码任务的替代验证**（用户要求）：提示词与评估集不写单测，改为
「改一版 → 拿 8 条标注样例跑子集 → 看条款是否生效」，结果记进 dev-notes。

---

## 15. 验收步骤（交付时逐条跑，输出进 dev-notes）

```powershell
# 0) 依赖与配置（一次性）
cd MemHelp-code
uv sync
# .env 需有 EMBEDDING_API_KEY（已有）；RERANK_* 可留空走默认

# 1) 建库（新增型号语料 + 重建带 text/sparse 的集合）
uv run python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild

# 2) 验收 2：型号题 BM25 命中（逐条打印）
uv run python evals/run_ch04_evals.py --only B_model --strategies bm25 -v

# 3) 验收 1：四策略对比报告
uv run python evals/run_ch04_evals.py --strategies vector,bm25,hybrid,hybrid_rerank

# 4) 起服务，页面验收 3（点引用看原文）与 4（拒答 + 入池）
uv run uvicorn app.main:app --port 8000
# 问「XH-3000A 续航多久」→ 点角标 → 抽屉显示章节路径与原文
# 问「你们支持货到付款吗」→ 明确拒答
uv run python scripts/show_low_confidence.py    # 池子里应有这一行
```

---

## 16. 交付物

| 类别 | 路径 |
|---|---|
| 设计 | `docs/superpowers/specs/2026-10-17-ch04-hybrid-retrieval-design.md` |
| 计划 | `docs/superpowers/plans/2026-10-17-ch04-hybrid-retrieval-plan.md` |
| 留痕 | `dev-notes/ch04.md`（逐阶段追记）+ `dev-notes/ch04-eval-report.md` |
| 检索 | `app/knowledge/query.py`、`hybrid.py`、`rerank.py`、`citations.py`、`sufficiency.py` |
| 编排 | `app/knowledge/pipeline.py`（`RagPipeline`：理解→检索→重排→两道闸→证据） |
| 工具契约 | `app/tools/knowledge.py`、`registry.py`、`schemas.py`（`query_faq` 出参升级；`category` 只保留程序可达，不进 Schema） |
| 数据 | `app/db/models.py`、`repositories.py`、`sql/ch04-ddl.sql`（含 `messages.citations`） |
| 语料 | `knowledge/商品型号手册.md`（新增） |
| 评估 | `evals/ch04_cases.jsonl`、`evals/run_ch04_evals.py`、`evals/reports/` |
| 前端 | `web/app.js`、`web/style.css`、`web/index.html` |

---

## 17. 风险与已知限制

| 风险 | 影响 | 处置 |
|---|---|---|
| chinese analyzer 对 `XH-3000A` 的切分可能丢连字符 | 验收 2 直接不成立 | Task 0 用临时集合实测；失败就停下来问用户，不自行改分词方案 |
| `AnnSearchRequest` / rerank 响应字段名与文档有出入 | 首次真机调用报错 | Task 0 探针钉住签名，不凭记忆写 |
| 重排服务额度或超时 | 结果质量下降 | 降级为 RRF 序 + 报告标注 `rerank_degraded` |
| LLM 改写把语义改跑偏 | 召回变差且难归因 | 只对静态表未命中的口语问法改写；缓存；`C_colloquial` 桶回归 |
| 生成段成本（48×2 次生成 + 裁判） | 评估一次几分钟、花额度 | 默认两策略，`--all-strategies` / `--limit` 可调 |
| 模型未调 `query_faq` 直接作答 | 幻觉无法被后端拦截 | Prompt 强约束 + 把「未检索就答率」当指标统计出来 |
| Milvus 集合未重建就上线 | BM25 静默全空、过滤失效 | `ensure_collection` 的字段自检 + 可执行报错 |
| `sql/` 目录刚迁到 `MemHelp-code/` 且未提交 | 后续改动可能落到错位置 | 本章第一次提交先把迁移落定 |
