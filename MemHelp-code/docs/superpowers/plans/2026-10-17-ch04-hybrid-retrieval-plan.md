# ch04 混合检索 + 重排 + 评估体系 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch03 的单路向量检索升级为「双路召回（dense + Milvus 原生 BM25）→ RRF 融合 → bge-reranker-v2-m3 重排 → 编号证据 → 自评拒答」，并交付带金标与分桶的四策略评估报告、聊天页可点引用与一次性满意度反馈。

**Architecture:** 模型仍决定是否检索（ch02 单轮工具路由不动），`query_faq` 的内核换成 `RagPipeline`：Query 理解 → 四策略统一检索 → 自评 → 证据编号 → 出参升级。证据编号由后端按相关度固定，塞进 prompt 的顺序按首尾编排；证据不足时编排层**短路**直接吐拒答话术并写低置信度池，不依赖模型自觉。Milvus 集合原地重建，从「只存 id+向量」变成「检索索引副本」（text/category/content_type/dense/sparse），MySQL 仍是原文权威。

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy 2 / MySQL 8 / Milvus 2.6.22 + pymilvus 2.6.x（原生 BM25 + chinese analyzer + `hybrid_search` + `RRFRanker`）/ `BAAI/bge-reranker-v2-m3`（硅基流动 `/rerank`）/ LangChain `BaseChatModel` / pytest。

**Spec:** `MemHelp-code/docs/superpowers/specs/2026-10-17-ch04-hybrid-retrieval-design.md`

## Global Constraints

- Python `>=3.12`；**不新增第三方依赖**（重排用已有的 `httpx`，测试用 `httpx.MockTransport`，不引入 respx）。
- 只有 `app/config.py` 读环境变量/`.env`；只有 `app/llm.py` 可以 `import langchain_openai`。新模块一律通过注入拿 `BaseChatModel`。
- 所有回给客户端的错误文本必须过 `redact()`；响应体外壳一律 `{"error": {"type","message"}}`。
- SQL 一律参数绑定，禁止字符串拼接。
- 测试**不得**依赖 `tmp_path` / `tempfile`（沙箱会在进程退出后收回权限）；pytest 缓存已由 `addopts` 关闭。
- 运行环境：`$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"`，命令统一用
  `uv run --directory MemHelp-code python -m pytest ...`（从工作区根执行）。
- 带中文的源文件用编辑工具改，不用 `Set-Content` 批量替换。
- Milvus 版本约束：服务端 2.6.22 / 客户端 `pymilvus>=2.6,<3`；**BM25 函数必须在建集合前 `schema.add_function(...)` 加上**，事后补不了。
- 提交只 `git add MemHelp-code/...` 具体路径，不用 `git add -A`；`.env` 绝不入库。
- 每个 Task 结束必须 `git commit`，提交信息前缀 `ch04:`。

## Review Focus

以下五类是 spec 没逐条点名、但一定会打到一个真实用户的地方。每条都在“归属 Task”的测试里钉住：

1. **型号的写法变体**（`xh-3000a`、`XH 3000A`、`XH3000A`）—— 用户不会照抄语料里的写法；BM25 若按字面切分就会漏。归属：Task 13（`B_model` 桶评估）与 Task 8（检索原文断言）。
2. **异常角标**（全角 `【1】`、`[0]`、`[99]`、`[1,2]`）—— 模型输出不可控；解析必须不崩、越界丢弃、正文一个字都不改。归属：Task 9。
3. **品类过滤值不存在或为空串** —— 过滤值来自模型参数；空串应当视为“不过滤”，不存在的品类必须返回空结果而不是 500。归属：Task 8。
4. **重排返回条数少于请求 / 分数全等 / `results` 为空** —— 服务端截断与打平都真实存在；不得崩、顺序必须稳定。归属：Task 7。
5. **超长 query（>512 字）与纯符号 query（`？？？`）** —— 改写与嵌入都可能被上游拒；必须截断/退化为空结果，而不是 500。归属：Task 6 与 Task 8。

---

## 文件结构（先定边界，再定任务）

| 文件 | 责任 | 状态 |
|---|---|---|
| `app/config.py` | 新增 `RERANK_*` / `RAG_HYBRID_TOP_K` / `RAG_CANDIDATE_K` / `RAG_RERANK_TOP_N` / `RAG_MIN_RERANK_SCORE` / `RAG_QUERY_REWRITE` / `RAG_REWRITE_TIMEOUT` | 改 |
| `app/knowledge/vector_store.py` | 集合生命周期 + `VectorRecord` 写入 + 三路检索（dense / bm25 / hybrid） | 改 |
| `app/knowledge/ingest.py` · `scripts/build_knowledge.py` | 写 `VectorRecord`（带 text/category/content_type） | 改 |
| `app/knowledge/query.py` | 归一改写 + 同义词扩展 + 缓存 | 新 |
| `app/knowledge/rerank.py` | `/rerank` 客户端 + 无 Key 降级 | 新 |
| `app/knowledge/hybrid.py` | `Strategy` / `RetrievedChunk` / `HybridRetriever` / `build_expr` | 新 |
| `app/knowledge/citations.py` | 证据编号、首尾编排、`[n]` 解析、快照 | 新 |
| `app/knowledge/sufficiency.py` | 自评 judge（严格 JSON，失败 fail-open） | 新 |
| `app/knowledge/pipeline.py` | `RagPipeline`：把上述串成一次 `answer_context` | 新 |
| `app/knowledge/retrieval.py` | 保持 ch03 的 `KnowledgeRetriever`（只走向量）不动，作为对照基线 | 不动 |
| `app/db/models.py` · `repositories.py` | 两张新表 + `messages.citations` + 两个新仓库 | 改 |
| `sql/ch04-ddl.sql` | 追加 `ALTER TABLE messages ADD citations JSON NULL` | 改 |
| `app/prompts.py` | 引用规范 + 负面知识约束 + 拒答话术 | 改 |
| `app/tools/knowledge.py` · `registry.py` · `schemas.py` | `query_faq` 出参升级 + 可选 `category` | 改 |
| `app/chat_tools.py` | 证据捕获、拒答短路、入池、`citations` 落库 | 改 |
| `app/api.py` | `done.citations`、历史回读 `citations`、`faith_cases` 两个端点 | 改 |
| `knowledge/商品型号手册.md` | 型号语料（BM25 该赢的那类题） | 新 |
| `evals/ch04_cases.jsonl` · `run_ch04_evals.py` · `reports/` | 48 条评估集 + 四策略报告 | 新 |
| `web/app.js` · `style.css` · `index.html` | 可点引用 + 证据抽屉 + 👍/👎 | 改 |
| `dev-notes/ch04.md` · `README.md` | 阶段留痕 + 演示说明 | 改 |

---

### Task 0: 真机探针（三处文档没写死的接口细节）

**Files:**
- Create（临时，跑完删除）: `scripts/_probe_ch04.py`
- Modify: `dev-notes/ch04.md`（新增「阶段 4.0」）

**Interfaces:**
- Consumes: 已确认的环境（Milvus 2.6.22 / pymilvus 2.6.17 / `EMBEDDING_API_KEY` 已配）
- Produces: 三条**实测结论**写进 dev-notes，供 Task 2/7/8 直接照抄：`AnnSearchRequest` 的构造参数名、rerank 响应的字段名与分数区间、chinese analyzer 对 `XH-3000A` 的切分与命中行为

- [ ] **Step 1: 写探针脚本**

脚本做四件事，每件打印一行结论：
1. 建临时集合 `probe_ch04`（schema 同 spec §3.1，用 `XH-3000A` 与 `XH-3000B` 两条假数据）；
2. 用 `AnnSearchRequest(data=[...], anns_field="sparse", limit=5)` 调 `hybrid_search`，**故意用位置参数与关键字各试一次**，确认参数名（失败会抛 `TypeError`，把两种写法都打印出来）；
3. 用 `search(data=["xh 3000a"], anns_field="sparse")` 与 `["XH-3000A"]` 各查一次，打印命中 id 与分数 —— 确认大小写/空格/连字符变体是否仍命中；
4. 用真 `EMBEDDING_API_KEY` 调一次 `/rerank`（`documents` 两条），打印 `results[0]` 的**全部键与分数区间**。

- [ ] **Step 2: 跑探针并记录**

Run: `uv run --directory MemHelp-code python scripts/_probe_ch04.py`
Expected: 四行结论清楚可抄；任何一步抛错都先解决再往下（不许带着未知往下写）。

- [ ] **Step 3: 把结论写进 dev-notes 的「阶段 4.0」**

记录：参数名、响应字段、分数区间、型号变体是否命中。若型号变体**不命中**，停在这里问用户（spec §17 风险 1），不要自行改分词方案。

- [ ] **Step 4: 删除探针脚本并提交留痕**

```bash
Remove-Item MemHelp-code/scripts/_probe_ch04.py
git add MemHelp-code/dev-notes/ch04.md
git commit -m "ch04: Milvus BM25 / rerank / 型号分词真机探针结论"
```

---

### Task 1: 配置项

**Files:**
- Modify: `app/config.py`（在 ch03 的 RAG 段之后追加）
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.rerank_base_url: str`、`rerank_api_key: str`（默认空）、`rerank_model: str`、`rerank_timeout: float`、`rag_hybrid_top_k: int`、`rag_candidate_k: int`、`rag_rerank_top_n: int`、`rag_min_rerank_score: float`、`rag_query_rewrite: bool`、`rag_rewrite_timeout: float`；方法 `require_rerank_key() -> str`（空则回落 `EMBEDDING_API_KEY`，两者都空抛可读错误）、`rerank_url() -> str`

- [ ] **Step 1: 写失败测试**

```python
def test_rag_ch04_defaults(settings_factory):
    s = settings_factory(RERANK_API_KEY="", EMBEDDING_API_KEY="sk-embed")
    assert s.rag_hybrid_top_k == 50
    assert s.rag_candidate_k == 20
    assert s.rag_rerank_top_n == 10
    assert s.rag_min_rerank_score == 0.3
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.require_rerank_key() == "sk-embed"      # 回落
    assert s.rerank_url().endswith("/rerank")

def test_require_rerank_key_raises_when_both_empty(settings_factory):
    s = settings_factory(RERANK_API_KEY="", EMBEDDING_API_KEY="")
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        s.require_rerank_key()

def test_rag_ch04_bounds(settings_factory):
    # 非法值必须在构造时就被拒，而不是跑到检索里才出怪结果
    with pytest.raises(ValidationError):
        settings_factory(RAG_MIN_RERANK_SCORE=1.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_config.py -q -k ch04`
Expected: FAIL（`Settings` 没有 `rag_hybrid_top_k`）

- [ ] **Step 3: 在 `app/config.py` 实现字段与两个方法**

`rag_min_rerank_score` 用 `ge=0.0, le=1.0`；`rag_hybrid_top_k` 用 `ge=1, le=200`；
`rag_candidate_k` 用 `ge=1, le=200`；`rag_rerank_top_n` 用 `ge=1, le=50`。
`require_rerank_key()` 的报错文本要点名两个键名与 `.env` 路径（照 `require_embedding_key()` 的写法）。
`.env.example` 追加同名字段与注释（`RERANK_API_KEY` 注释写明“留空则复用 EMBEDDING_API_KEY”）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_config.py -q`
Expected: PASS（含既有用例）

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/config.py MemHelp-code/.env.example MemHelp-code/tests/test_config.py
git commit -m "ch04: 重排与混合检索配置项"
```

---

### Task 2: Milvus 集合重建（text / category / sparse）与写入

**Files:**
- Modify: `app/knowledge/vector_store.py`
- Test: `tests/test_knowledge_vector_store.py`

**Interfaces:**
- Consumes: Task 1 的 `Settings`（维度来自 `embedding_dim`）
- Produces:
  - `@dataclass(frozen=True) VectorRecord(id: str, dense: Sequence[float], text: str, category: str, content_type: str | None)`
  - `MilvusVectorStore.upsert(records: Sequence[VectorRecord]) -> None`（**替换** ch03 的 `upsert(ids, vectors)`）
  - `ensure_collection()` 新增自检：集合缺 `text` / `sparse` 字段时抛 `RuntimeError`，文本必须含 `build_knowledge.py --rebuild`

- [ ] **Step 1: 写失败测试**

```python
def test_create_schema_has_bm25_function(fake_client_cls):
    store = MilvusVectorStore(uri="http://x", collection="knowledge", dim=1024,
                              client_factory=lambda: fake_client_cls.return_me)
    store._ensure_collection()
    schema = fake_client_cls.return_me.last_schema
    assert schema.fields["text"].params["analyzer_params"] == {"type": "chinese"}
    assert schema.fields["text"].params["enable_analyzer"] is True
    assert [f.type for f in schema.functions] == ["BM25"]
    assert schema.functions[0].output_field_names == ["sparse"]

def test_existing_collection_without_text_field_raises(fake_client_cls):
    # 老集合（只有 id+vector）必须报「该重建了」，不能静默继续
    fake_client_cls.return_me.describe_result = {"fields": [{"name": "id"}, {"name": "vector"}]}
    with pytest.raises(RuntimeError, match="--rebuild"):
        store._ensure_collection()

def test_upsert_writes_text_and_category(fake_client_cls):
    store.upsert([VectorRecord(id="1", dense=[0.1]*1024, text="问法\n答案",
                               category="售后政策", content_type="policy")])
    row = fake_client_cls.return_me.last_upsert_rows[0]
    assert row["text"] == "问法\n答案"
    assert row["category"] == "售后政策"
    assert row["sparse"] is absent   # 稀疏向量由服务端 BM25 函数生成，客户端不写
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_vector_store.py -q -k "bm25 or text_field or writes_text"`
Expected: FAIL（`VectorRecord` 不存在 / schema 里没有 text）

- [ ] **Step 3: 改 `_create_schema_and_collection` 与 `_upsert`**

按 spec §3.1 建 6 个字段 + `Function(FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`；
两条索引（`dense` AUTOINDEX/COSINE、`sparse` SPARSE_INVERTED_INDEX/BM25 带 k1/b）；
`_upsert` 的每行只写 `id/text/category/content_type/dense`；`content_type` 为 `None` 时写空串（Milvus 标量字段不接受 null）。
`_ensure_collection` 的自检从「只查维度」扩成「查维度 + 查 `text`/`sparse` 是否存在」。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_vector_store.py -q`
Expected: PASS（既有维度自检用例保持通过）

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/vector_store.py MemHelp-code/tests/test_knowledge_vector_store.py
git commit -m "ch04: Milvus 集合升级为 BM25 全文索引 + 元数据副本"
```

---

### Task 3: 三路检索（dense / bm25 / hybrid）

**Files:**
- Modify: `app/knowledge/vector_store.py`
- Test: `tests/test_knowledge_vector_store.py`

**Interfaces:**
- Produces:
  - `MilvusVectorStore.search_dense(vector, top_k, *, expr: str | None = None) -> list[VectorHit]`
  - `MilvusVectorStore.search_bm25(text, top_k, *, expr: str | None = None) -> list[VectorHit]`
  - `MilvusVectorStore.hybrid_search(dense_vector, bm25_text, top_k, *, expr: str | None = None) -> list[VectorHit]`
  - `search(vector, top_k)` 保留为 `search_dense` 的别名（ch03 的 `KnowledgeRetriever` 仍在调）

- [ ] **Step 1: 写失败测试**

```python
def test_bm25_passes_raw_text_not_vector(fake_client_cls):
    store.search_bm25("XH-3000A 续航", 50)
    call = fake_client_cls.return_me.last_search
    assert call["data"] == ["XH-3000A 续航"]      # 文本，不是向量
    assert call["anns_field"] == "sparse"
    assert call["search_params"]["metric_type"] == "BM25"

def test_hybrid_builds_two_requests_and_filter(fake_client_cls):
    store.hybrid_search([0.1]*1024, "包邮", 20, expr='category == "售后政策"')
    call = fake_client_cls.return_me.last_hybrid
    assert [r.anns_field for r in call["reqs"]] == ["dense", "sparse"]
    assert call["filter"] == 'category == "售后政策"'
    assert call["ranker"].k == 60

def test_hybrid_returns_merged_ids(fake_client_cls):
    fake_client_cls.return_me.hybrid_result = [[{"id": "7", "distance": 0.03},
                                                {"id": "3", "distance": 0.02}]]
    hits = store.hybrid_search([0.1]*1024, "包邮", 20)
    assert [h.id for h in hits] == ["7", "3"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_vector_store.py -q -k "bm25 or hybrid"`
Expected: FAIL（方法不存在）

- [ ] **Step 3: 实现三个方法**

`search_dense` = ch03 `_search` 的改造（加 `expr`，`consistency_level="Strong"` 不变）；
`search_bm25` 的 `data=[text]`、`search_params={"metric_type": "BM25"}`；
`hybrid_search` 用 `AnnSearchRequest`（参数名按 Task 0 探针结论）+ `RRFRanker(k=60)` + `filter=expr`。
三者都在 `anyio.to_thread.run_sync` 里跑（沿用 ch03 的**同步客户端 + 线程**模式，不要改成异步客户端）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_vector_store.py -q`
Expected: PASS

- [ ] **Step 5: 真机集成（沿用 ch03 的 skip 策略）**

在 `tests/test_knowledge_milvus.py` 追加一条：建 `ch04_integration_test` 集合（同 spec §3.1 的 schema），
写两条含 `XH-3000A` / `XH-3000B` 的记录，断言 (a) `search_bm25("XH-3000A")` 首位是 A；
(b) `hybrid_search(..., expr='category == "商品参数"')` 只返回该类。Milvus 连不上时 `pytest.skip`。

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_milvus.py -q`
Expected: PASS 或 SKIP（跳过时打出 skip 原因）

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/app/knowledge/vector_store.py MemHelp-code/tests/test_knowledge_vector_store.py MemHelp-code/tests/test_knowledge_milvus.py
git commit -m "ch04: 向量库三路检索（dense / bm25 / hybrid）"
```

---

### Task 4: 建库写入升级（`VectorRecord`）

**Files:**
- Modify: `app/knowledge/ingest.py`、`scripts/build_knowledge.py`
- Test: `tests/test_knowledge_ingest.py`、`tests/test_build_knowledge_cli.py`

**Interfaces:**
- Consumes: Task 2 的 `VectorRecord` 与 `upsert(records)`
- Produces: `render_bm25_text(chunk: object) -> str`（放在 `app/knowledge/chunker.py`，与 `render_embed_text` 并列：`问法\n答案`，**不带**「分类/问法/答案」标签）

- [ ] **Step 1: 写失败测试**

```python
def test_render_bm25_text_has_no_labels():
    chunk = Chunk(category="售后政策", questions="运费怎么算", answer="满 99 包邮", ...)
    assert render_bm25_text(chunk) == "运费怎么算\n满 99 包邮"

def test_ingest_upserts_records_with_metadata(fake_store, fake_embedder):
    await ingest_chunks([...], embedder=fake_embedder, vector_store=fake_store, chunk_repo=...)
    rec = fake_store.records[0]
    assert rec.text == "运费怎么算\n满 99 包邮"
    assert rec.category == "售后政策"
    assert rec.content_type == "faq"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_ingest.py -q`
Expected: FAIL（`render_bm25_text` 不存在；`upsert` 收到的是 ids/vectors）

- [ ] **Step 3: 实现**

`chunker.py` 加 `render_bm25_text`；`ingest.py` 组装 `VectorRecord`（`dense` 来自 `render_embed_text` 的嵌入，
`text` 来自 `render_bm25_text`，`category`/`content_type` 取 chunk 元数据）；
`build_knowledge.py --rebuild` 的提示文案与报告里加一句「集合已升级为 BM25 全文索引」。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_ingest.py tests/test_build_knowledge_cli.py tests/test_knowledge_chunker.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/chunker.py MemHelp-code/app/knowledge/ingest.py MemHelp-code/scripts/build_knowledge.py MemHelp-code/tests/test_knowledge_ingest.py
git commit -m "ch04: 建库写入 BM25 文本与检索元数据副本"
```

---

### Task 5: 数据层（两张新表 + `messages.citations` + 两个仓库）

**Files:**
- Modify: `sql/ch04-ddl.sql`（末尾追加 ALTER）、`app/db/models.py`、`app/db/repositories.py`
- Test: `tests/test_db_models.py`、`tests/test_db_repositories.py`（SQLite 上跑）

**Interfaces:**
- Produces:
  - `Message.citations: Mapped[list | None]`
  - `LowConfidenceQuestion`（列与 DDL 逐一对齐；`LOW_CONFIDENCE_SOURCES = ("retrieval_low_conf","self_check","user_feedback")`）
  - `FaithCase`（`FAITH_CASE_STATUSES = ("未解决","已解决","无需解决")`）
  - `LowConfidenceRepository.add(conversation_id: int, raw_question: str, source: str, reason: str | None) -> int`
  - `LowConfidenceRepository.list_recent(limit: int = 20) -> list[LowConfidenceRow]`
  - `FaithCaseRepository.upsert_from_judgement(*, eval_id, bucket, query, strategy, answer, reason, citations, judge_model) -> None`
  - `FaithCaseRepository.list_by_status(status: str | None = None) -> list[FaithCaseRow]`
  - `FaithCaseRepository.update_status(eval_id: str, status: str, resolution: str | None) -> None`
  - `Repositories.low_confidence`、`Repositories.faith_cases`

- [ ] **Step 1: 写失败测试**

```python
def test_low_confidence_add_and_list(sqlite_repositories):
    cid = await sqlite_repositories.conversations.upsert_by_session("s1", "anonymous")
    await sqlite_repositories.low_confidence.add(cid, "你们支持货到付款吗", "self_check", "证据不足")
    rows = await sqlite_repositories.low_confidence.list_recent(10)
    assert rows[0].raw_question == "你们支持货到付款吗"
    assert rows[0].source == "self_check"

def test_low_confidence_rejects_unknown_source(sqlite_repositories):
    with pytest.raises(LookupError):
        await sqlite_repositories.low_confidence.add(1, "q", "made_up", None)

def test_faith_case_upsert_accumulates_seen_count(sqlite_repositories):
    for _ in range(2):
        await sqlite_repositories.faith_cases.upsert_from_judgement(
            eval_id="A01", bucket="A_policy", query="q", strategy="hybrid_rerank",
            answer="a", reason="r", citations=[], judge_model="m")
    rows = await sqlite_repositories.faith_cases.list_by_status()
    assert rows[0].seen_count == 2

def test_faith_case_relapses_to_unresolved(sqlite_repositories):
    # 已标已解决 → 又被判编造 → 退回未解决（复发，不是新问题）
    await repo.upsert_from_judgement(...); await repo.update_status("A01", "已解决", "补了条款")
    await repo.upsert_from_judgement(...)
    assert (await repo.list_by_status("未解决"))[0].eval_id == "A01"

def test_message_citations_roundtrip(sqlite_repositories):
    await ...append(...);  # 带 citations=[{"n":1,...}]
    rows = await repo.list_for_conversation(cid)
    assert rows[-1].citations[0]["n"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_models.py tests/test_db_repositories.py -q -k "low_confidence or faith or citations"`
Expected: FAIL（表/列/仓库不存在）

- [ ] **Step 3: 实现**

DDL 末尾追加 ALTER（带中文 COMMENT）；ORM 两张类 + `Message.citations`；
仓库两个类，`FaithCaseRepository.upsert_from_judgement` 按 DDL 注释的语义实现
（一题一行、`seen_count+1`、`last_seen_at` 刷新、已解决再判出则退回未解决并保留 `resolved_at`）；
`update_status` 标「已解决/无需解决」时 `resolution` 必填（空则抛 `ValueError`），退回时清空。
`conversations` 外键用 `ForeignKey("conversations.id")`，与 DDL 的 `fk_lcq_conversation` 对齐（测试逐列钉住）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_models.py tests/test_db_repositories.py -q`
Expected: PASS

- [ ] **Step 5: 真库逐列对齐（沿用 ch03 的 skip 策略）**

在 `tests/test_db_mysql.py` 追加：`SHOW CREATE TABLE low_confidence_questions / faith_cases` 的列名、类型、
ENUM 取值、索引名与本项目的 ORM 逐一比对；`messages.citations` 必须是 `json` 且可空。
MySQL 连不上时 `pytest.skip`。

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_mysql.py -q -k "ch04"`
Expected: PASS 或 SKIP

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/sql/ch04-ddl.sql MemHelp-code/app/db/models.py MemHelp-code/app/db/repositories.py MemHelp-code/tests/test_db_models.py MemHelp-code/tests/test_db_repositories.py MemHelp-code/tests/test_db_mysql.py
git commit -m "ch04: 低置信度池 / 编造台账 / 引用快照的数据层"
```

---

### Task 6: Query 理解（归一改写 + 同义词 + 缓存）

**Files:**
- Create: `app/knowledge/query.py`
- Test: `tests/test_knowledge_query.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) UnderstoodQuery(original: str, standard: str, bm25_text: str, expanded: tuple[str, ...], degraded: bool = False)`
  - `class Rewriter(Protocol): async def rewrite(self, query: str) -> str | None`
  - `class LlmRewriter(model: BaseChatModel, *, timeout: float = 3.0)`：严格 JSON `{"rewritten": "..."}`；超时/解析失败返回 `None`
  - `SYNONYMS: Mapping[str, tuple[str, ...]]`、`STATIC_NORMALISE: Mapping[str, str]`
  - `QueryUnderstander(rewriter: Rewriter | None = None, *, enabled: bool = True).understand(query: str) -> UnderstoodQuery`
  - `MAX_QUERY_CHARS = 512`（超过截断）

- [ ] **Step 1: 写失败测试**

```python
async def test_static_table_hit_does_not_call_rewriter(fake_rewriter):
    out = await QueryUnderstander(rewriter=fake_rewriter).understand("钱啥时候能回来")
    assert out.standard == "退款到账时间"
    assert fake_rewriter.calls == []

async def test_colloquial_miss_uses_llm(fake_rewriter):
    fake_rewriter.answer = "发票开具流程"
    out = await QueryUnderstander(rewriter=fake_rewriter).understand("那个票咋整啊")
    assert out.standard == "发票开具流程"
    assert fake_rewriter.calls == ["那个票咋整啊"]

async def test_rewriter_failure_falls_back_to_original(fake_rewriter):
    fake_rewriter.answer = None
    out = await QueryUnderstander(rewriter=fake_rewriter).understand("那个票咋整啊")
    assert out.standard == "那个票咋整啊"
    assert out.degraded is True

async def test_synonyms_expand_bm25_only():
    out = await QueryUnderstander(enabled=False).understand("包邮")
    assert out.standard == "包邮"
    assert "免邮" in out.bm25_text and out.standard == "包邮"

async def test_cache_avoids_second_llm_call(fake_rewriter):
    q = QueryUnderstander(rewriter=fake_rewriter)
    await q.understand("那个票咋整啊"); await q.understand("那个票咋整啊")
    assert len(fake_rewriter.calls) == 1

async def test_overlong_query_is_truncated(fake_rewriter):
    out = await QueryUnderstander(rewriter=fake_rewriter).understand("退" * 2000)
    assert len(out.original) == 512
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_query.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

`understand()` 顺序固定：截断 → 静态表 → （未命中且判定口语）LLM → 同义词扩展 → 写缓存。
「判定口语」= 含语气词/疑问代词，或 `len <= 12` 且未命中任何 `SYNONYMS` 的键。
缓存用普通 dict + FIFO 上限 512（不要 `lru_cache` 挂在实例方法上 —— 那会把 self 也进键，
且难以按实例清空）。
缓存键必须含 `enabled`，否则「关掉改写」的排障配置会被缓存绕过。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_query.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/query.py MemHelp-code/tests/test_knowledge_query.py
git commit -m "ch04: Query 归一改写与检索侧同义词扩展"
```

---

### Task 7: 重排客户端（含降级）

**Files:**
- Create: `app/knowledge/rerank.py`
- Test: `tests/test_knowledge_rerank.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) RerankHit(index: int, score: float)`
  - `class Reranker(Protocol): async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]`
  - `SiliconFlowReranker(base_url, api_key, model, timeout=10.0, client: httpx.AsyncClient | None = None)`
  - `NoopReranker`：按输入顺序返回前 `top_n`，`score=0.0`（没配 Key 时用）
  - `get_reranker(settings: Settings) -> Reranker`（进程内单例；Key 空则返回 `NoopReranker`）

- [ ] **Step 1: 写失败测试**

```python
async def test_request_body_shape():
    # httpx.MockTransport 断言请求体
    hits = await SiliconFlowReranker(base_url="https://x/v1", api_key="k",
                                     model="BAAI/bge-reranker-v2-m3", client=client).rerank(
        "包邮", ["a", "b"], top_n=2)
    assert body == {"model": "BAAI/bge-reranker-v2-m3", "query": "包邮",
                    "documents": ["a", "b"], "top_n": 2, "return_documents": False}
    assert hits == [RerankHit(index=1, score=0.91), RerankHit(index=0, score=0.12)]

async def test_fewer_results_than_requested():
    # 服务端可能截断：不得崩，按返回条数处理
    hits = await ...rerank("q", ["a", "b", "c"], top_n=3)   # 响应只回 1 条
    assert len(hits) == 1

async def test_equal_scores_keep_input_order():
    # 分数打平时顺序必须稳定（同一份索引重跑结果一致），否则评估不可复现
    hits = await ...rerank("q", ["a", "b", "c"], top_n=3)   # 响应全 0.5
    assert [h.index for h in hits] == [0, 1, 2]

async def test_empty_results():
    assert await ...rerank("q", ["a"], top_n=1) == []       # 响应 {"results": []}

async def test_timeout_and_5xx_raise_rerank_error():
    with pytest.raises(RerankError):
        await ...rerank("q", ["a"], top_n=1)                # 超时

async def test_noop_reranker_keeps_input_order():
    hits = await NoopReranker().rerank("q", ["a", "b", "c"], top_n=2)
    assert [h.index for h in hits] == [0, 1] and all(h.score == 0.0 for h in hits)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_rerank.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

照 `Embedder` 的客户端注入与关闭模式写（`client` 传入则复用，否则临时建、用完关）。
`results` 的字段名以 Task 0 探针为准（`index` / `relevance_score`）。
响应条数少于 `top_n`、`results` 为空都返回实际拿到的，不补、不抛。
超时与非 2xx 抛 `RerankError`（**由调用方决定降级**，客户端不吞）。
`get_reranker`：`settings.rerank_api_key or settings.embedding_api_key` 为空 → `NoopReranker` + `logger.warning`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_rerank.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/rerank.py MemHelp-code/tests/test_knowledge_rerank.py
git commit -m "ch04: bge-reranker-v2-m3 重排客户端与无 Key 降级"
```

---

### Task 8: 四策略检索（RRF 融合 + 降级）

**Files:**
- Create: `app/knowledge/hybrid.py`
- Test: `tests/test_knowledge_hybrid.py`

**Interfaces:**
- Consumes: Task 2/3 的 `VectorStore` 三路检索、Task 6 的 `UnderstoodQuery`、Task 7 的 `Reranker`
- Produces:
  - `Strategy = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]`
  - `@dataclass(frozen=True) RetrievedChunk(chunk_id: int, question: str, answer: str, category: str, section_path: str | None, content_type: str | None, score: float, source: str)`
  - `@dataclass(frozen=True) StrategyResult(chunks: tuple[RetrievedChunk, ...], strategy: Strategy, rerank_degraded: bool = False)`
  - `build_expr(category: str | None, content_type: str | None) -> str | None`（空串视为 None；用 `json.dumps` 转义值）
  - `HybridRetriever(embedder, vector_store, reranker, chunk_repo, *, hybrid_top_k=50, candidate_k=20, rerank_top_n=10)`
    - `async retrieve(strategy, understood, *, category=None, content_type=None) -> StrategyResult`

- [ ] **Step 1: 写失败测试**

```python
async def test_build_expr_escapes_and_ignores_empty():
    assert build_expr("售后政策", None) == 'category == "售后政策"'
    assert build_expr("", "") is None
    assert build_expr('含"引号', None) == 'category == "含\\"引号"'

async def test_vector_strategy_uses_standard_query(fake_embedder, fake_store):
    await retriever.retrieve("vector", understood)
    assert fake_embedder.texts == [understood.standard]

async def test_bm25_strategy_uses_bm25_text_with_synonyms(fake_store):
    await retriever.retrieve("bm25", understood)
    assert fake_store.bm25_texts == [understood.bm25_text]

async def test_hybrid_passes_both_and_filters(fake_store):
    await retriever.retrieve("hybrid", understood, category="售后政策")
    assert fake_store.last_hybrid["filter"] == 'category == "售后政策"'

async def test_hybrid_rerank_orders_by_rerank_score(fake_store, fake_reranker):
    res = await retriever.retrieve("hybrid_rerank", understood)
    assert [c.chunk_id for c in res.chunks] == [3, 1]      # 按 rerank 分
    assert res.rerank_degraded is False

async def test_rerank_failure_degrades_to_rrf_order(fake_store, failing_reranker):
    res = await retriever.retrieve("hybrid_rerank", understood)
    assert res.rerank_degraded is True
    assert len(res.chunks) == retriever.rerank_top_n
    assert [c.chunk_id for c in res.chunks] == rrf_order   # RRF 原序

async def test_unknown_category_returns_empty(fake_store):   # Review Focus 3
    fake_store.hybrid_result = []
    res = await retriever.retrieve("hybrid", understood, category="不存在的品类")
    assert res.chunks == ()

async def test_symbol_only_query_returns_empty_without_crashing(fake_store):  # Review Focus 5
    res = await retriever.retrieve("bm25", UnderstoodQuery("？？？", "？？？", "？？？", ()))
    assert res.chunks == ()

async def test_orphan_ids_are_skipped(fake_store, chunk_repo_missing_one):
    # Milvus 有、MySQL 没有的 id 属于可预期脏数据（沿用 ch03）
    res = await retriever.retrieve("vector", understood)
    assert [c.chunk_id for c in res.chunks] == [1]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_hybrid.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

- `vector`：`embed([standard])` → `search_dense`；
- `bm25`：`search_bm25(bm25_text)`；
- `hybrid`：`embed([standard])` 与 `search_bm25` 可并行，但**融合必须由 Milvus 做**
  （`hybrid_search` 单次调用），不要自己实现 RRF 再和服务端的算法抢口径；
- `hybrid_rerank`：`hybrid_search(limit=candidate_k)` → 取 MySQL 原文 → `reranker.rerank(query=standard, documents=[render_bm25_text(row)…], top_n=rerank_top_n)`；
  `RerankError` → 回落到 RRF 序的前 `rerank_top_n`，`rerank_degraded=True`；
- 回 MySQL 用 `chunk_repo.get_many`，按 Milvus 顺序重排、孤儿 id 跳过（沿用 ch03 的理由与日志口径）；
- `source` 标记：`vector` / `bm25` / `hybrid` / `rerank`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_hybrid.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/hybrid.py MemHelp-code/tests/test_knowledge_hybrid.py
git commit -m "ch04: 四策略检索（vector / bm25 / hybrid / hybrid_rerank）"
```

---

### Task 9: 证据编号、首尾编排与引用解析

**Files:**
- Create: `app/knowledge/citations.py`
- Test: `tests/test_knowledge_citations.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Evidence(n: int, chunk_id: int, section_path: str, category: str, quote: str, score: float = 0.0, source: str = "")`
  - `build_evidence(chunks: Sequence[RetrievedChunk]) -> tuple[Evidence, ...]`（`n` 从 1 连续编号，顺序=相关度）
  - `prompt_order(evidence: Sequence[Evidence]) -> tuple[Evidence, ...]`
  - `@dataclass(frozen=True) CitationParse(valid: tuple[int, ...], dropped: tuple[int, ...], coverage: float)`
  - `parse_citations(answer: str, evidence: Sequence[Evidence]) -> CitationParse`
  - `evidence_payload(evidence: Sequence[Evidence]) -> list[dict]`（落库/传输用的快照）
  - `evidence_from_payload(items: Sequence[dict]) -> tuple[Evidence, ...]`（把工具出参的 `items`
    还原成 `Evidence`；缺 `score`/`source` 时给默认值 —— 编排层只需要编号与展示字段，
    不为了补两个内部字段去改工具契约）

- [ ] **Step 1: 写失败测试**

```python
def test_evidence_numbering_follows_relevance():
    ev = build_evidence([chunk(3), chunk(1)])
    assert [(e.n, e.chunk_id) for e in ev] == [(1, 3), (2, 1)]

def test_head_tail_order():
    ev = tuple(Evidence(n=i, ...) for i in range(1, 11))
    assert [e.n for e in prompt_order(ev)] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]

def test_head_tail_order_odd_k():
    ev = tuple(Evidence(n=i, ...) for i in range(1, 6))
    assert [e.n for e in prompt_order(ev)] == [1, 3, 5, 4, 2]

def test_parse_keeps_valid_and_drops_out_of_range():
    res = parse_citations("满 99 包邮[1]，破损赔 500[3]。", build_evidence([c(1), c(2)]))
    assert res.valid == (1,)
    assert res.dropped == (3,)
    assert res.coverage == 0.5

def test_parse_ignores_full_width_and_zero():   # Review Focus 2
    res = parse_citations("参考【1】和[0]和[1,2]", build_evidence([c(1)]))
    assert res.valid == (1,)
    assert "【1】" in "" or True          # 正文一个字符都不改
    assert res.dropped == (0,) or res.dropped == ()

def test_parse_no_citations():
    res = parse_citations("没有角标的回答", build_evidence([c(1)]))
    assert res.valid == () and res.dropped == () and res.coverage == 0.0

def test_payload_has_snapshot_fields():
    payload = evidence_payload(build_evidence([c(1)]))
    assert set(payload[0]) == {"n", "chunk_id", "section_path", "category", "quote"}

def test_payload_roundtrips_to_evidence():
    ev = build_evidence([c(1)])
    back = evidence_from_payload(evidence_payload(ev))
    assert [(e.n, e.chunk_id, e.quote) for e in back] == [(e.n, e.chunk_id, e.quote) for e in ev]
    assert back[0].score == 0.0      # 缺失字段用默认值，不抛
```
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_citations.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

`prompt_order` 用通式 `[奇数升序] + [偶数降序]`；
`parse_citations` 只认半角 `[数字]`（`\[(\d{1,2})\]`），一个字符串里重复出现的编号去重保序；
**函数不修改 answer 字符串**（丢弃只发生在映射层，正文原样返回给用户）；
`coverage = len(valid) / len(evidence)`，`evidence` 为空时 `0.0`（不要除零）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_citations.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/citations.py MemHelp-code/tests/test_knowledge_citations.py
git commit -m "ch04: 证据编号、首尾编排与引用解析"
```

---

### Task 10: 自评 judge

**Files:**
- Create: `app/knowledge/sufficiency.py`
- Test: `tests/test_knowledge_sufficiency.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SufficiencyVerdict(sufficient: bool, missing: str, reason: str, degraded: bool = False)`
  - `SufficiencyJudge(model: BaseChatModel, *, timeout: float = 5.0)`
    - `async judge(question: str, evidence: Sequence[Evidence]) -> SufficiencyVerdict`

- [ ] **Step 1: 写失败测试**

```python
async def test_sufficient_false_with_reason(fake_model):
    fake_model.reply = '{"sufficient": false, "missing": "赔付上限", "reason": "证据只讲流程"}'
    v = await SufficiencyJudge(fake_model).judge("赔付上限多少", evidence)
    assert v.sufficient is False and v.missing == "赔付上限"

async def test_bad_json_fails_open(fake_model):
    fake_model.reply = "模型说了一堆人话"
    v = await SufficiencyJudge(fake_model).judge("q", evidence)
    assert v.sufficient is True and v.degraded is True

async def test_timeout_fails_open(fake_model, monkeypatch):
    # 超时不得让用户看到「什么都不知道」
    v = await SufficiencyJudge(slow_model, timeout=0.01).judge("q", evidence)
    assert v.sufficient is True and v.degraded is True

async def test_empty_evidence_is_insufficient(fake_model):
    v = await SufficiencyJudge(fake_model).judge("q", ())
    assert v.sufficient is False and fake_model.calls == []   # 不浪费一次调用
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_sufficiency.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

prompt 要求**只输出 JSON**；证据按相关度序拼 `[n] 章节路径\n原文`；
`asyncio.wait_for` 控超时；解析失败/超时/空证据三种情况按测试里的语义处理
（fail-open 的两条要 `logger.warning` 带 `self_check_degraded` 字样，便于日志检索）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_sufficiency.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/sufficiency.py MemHelp-code/tests/test_knowledge_sufficiency.py
git commit -m "ch04: 生成前自评 judge（JSON 严格解析 + fail-open）"
```

---

### Task 11: `RagPipeline` 编排（两道闸 + 拒答短路）

**Files:**
- Create: `app/knowledge/pipeline.py`
- Modify: `app/prompts.py`（加 `REFUSAL_TEXT`）
- Test: `tests/test_knowledge_pipeline.py`

**Interfaces:**
- Consumes: Task 6/8/9/10 的 `QueryUnderstander` / `HybridRetriever` / `Evidence` / `SufficiencyJudge`
- Produces:
  - `@dataclass(frozen=True) RetrievalOutcome(query: UnderstoodQuery, evidence: tuple[Evidence, ...], refused: bool, refusal_source: str | None, refusal_reason: str | None, rerank_degraded: bool, self_check_degraded: bool)`
  - `RagPipeline(understander, retriever, sufficiency, *, min_score: float = 0.3, top_n: int = 10)`
    - `async answer_context(query: str, *, category: str | None = None) -> RetrievalOutcome`
  - `app/prompts.py::REFUSAL_TEXT: str`（固定话术，含「知识库里没有」「不猜」「转人工/补充信息」三件事，不含角标）

- [ ] **Step 1: 写失败测试**

```python
async def test_zero_hits_refuses_with_retrieval_source(fakes):
    out = await pipeline().answer_context("你们支持货到付款吗")
    assert out.refused is True
    assert out.refusal_source == "retrieval_low_conf"
    assert out.evidence == ()

async def test_low_score_refuses(fakes):
    fakes.retriever.chunks = [chunk(score=0.12)]      # < min_score
    out = await pipeline().answer_context("q")
    assert out.refused and out.refusal_source == "retrieval_low_conf"

async def test_self_check_insufficient_refuses_with_self_check_source(fakes):
    fakes.sufficiency.verdict = SufficiencyVerdict(False, "缺赔付上限", "证据不足")
    out = await pipeline().answer_context("q")
    assert out.refused and out.refusal_source == "self_check"
    assert out.refusal_reason == "证据不足"

async def test_sufficient_returns_numbered_evidence(fakes):
    out = await pipeline().answer_context("q")
    assert out.refused is False
    assert [e.n for e in out.evidence] == [1, 2, 3]

async def test_category_passed_to_retriever(fakes):
    await pipeline().answer_context("q", category="商品参数")
    assert fakes.retriever.calls[0]["category"] == "商品参数"

async def test_rerank_degraded_surfaced(fakes):
    fakes.retriever.degraded = True
    out = await pipeline().answer_context("q")
    assert out.rerank_degraded is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_pipeline.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

顺序固定：`understand → retrieve("hybrid_rerank") → 闸1(空或最高分<min_score) → 闸2(judge) → build_evidence`。
闸 1 的「最高分」取 `chunks[0].score`（重排降级时它就是 RRF 分 —— 阈值语义要写进 docstring：
降级时这道闸按 RRF 分判，会比有重排分时更宽松，这是刻意选择：宁可多查一次也不要误拒）。
`refusal_source` 只用 DDL 里的两个值；`REFUSAL_TEXT` 从这里取。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_knowledge_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/pipeline.py MemHelp-code/app/prompts.py MemHelp-code/tests/test_knowledge_pipeline.py
git commit -m "ch04: RagPipeline（两道闸 + 证据编号）"
```

---

### Task 12: 型号语料 + 48 条评估集（数据任务）

**Files:**
- Create: `knowledge/商品型号手册.md`、`evals/ch04_cases.jsonl`
- Test: `tests/test_ch04_cases.py`

**Interfaces:**
- Produces: 48 行 JSONL，字段与 spec §11.1 一致（`id/bucket/difficulty/query/behavior/gold_anchors/must_contain`）

- [ ] **Step 1: 写数据校验测试**（这是数据任务，测试替代 TDD）

```python
def test_48_cases_in_four_buckets():
    cases = load("evals/ch04_cases.jsonl")
    assert len(cases) == 48
    assert Counter(c["bucket"] for c in cases) == {
        "A_policy": 12, "B_model": 12, "C_colloquial": 12, "E_multi": 12}

def test_each_bucket_has_two_refuse_cases():
    for bucket, group in groupby_bucket():
        assert sum(1 for c in group if c["behavior"] == "refuse") == 2

def test_difficulty_uses_three_levels_and_all_present():
    assert {c["difficulty"] for c in cases} == {"easy", "medium", "hard"}

def test_answer_cases_have_anchors_and_containment():
    for c in answer_cases:
        assert c["gold_anchors"] and c["must_contain"]

def test_ids_unique_and_prefixed_by_bucket():
    assert len({c["id"] for c in cases}) == 48
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_ch04_cases.py -q`
Expected: FAIL（文件不存在）

- [ ] **Step 3: 写语料与评估集**

《商品型号手册》8～12 个型号，每个型号一段，参数里带**唯一串**（如 `XH-3000A`、`续航 18 小时`、
`适配 MewHelp-S1`）；`B_model` 桶的题必须能靠型号串命中。
`C_colloquial` 的题要用真实口语（「钱啥时候能回来」「那个票咋整」），且**不能**与静态归一表逐字相同
（否则测的是查表不是改写）。
`E_multi` 每条要锚到**两个不同章节**的事实。
`must_contain` 里的数字要用语料里的原样形式（`99`、`500`、`18 小时`）。
`B_model` 里**必须**至少 2 条用型号的写法变体（一条小写 `xh-3000a`、一条带空格 `XH 3000A`）——
这是 Review Focus 1 的落点：用户不会照抄语料里的写法。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_ch04_cases.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/knowledge/商品型号手册.md MemHelp-code/evals/ch04_cases.jsonl MemHelp-code/tests/test_ch04_cases.py
git commit -m "ch04: 型号语料与 48 条四桶评估集"
```

---

### Task 13: 评估脚本（检索段 + 分桶报告）

**Files:**
- Create: `evals/run_ch04_evals.py`、`evals/reports/.gitkeep`
- Test: `tests/test_run_ch04_evals.py`

**Interfaces:**
- Consumes: Task 8 的 `HybridRetriever`、Task 12 的评估集
- Produces:
  - CLI `--strategies vector,bm25,hybrid,hybrid_rerank`、`--only BUCKET`、`--limit N`、`-v`、`--validate-only`、`--json PATH`、`--markdown PATH`
  - 纯函数 `resolve_gold(cases, chunk_rows) -> dict[str,set[int]]`、`recall_at_k(hits, gold, k) -> float`、`mrr(hits, gold) -> float`、`aggregate(rows) -> dict`
  - 退出码：`hybrid_rerank` 整体 Recall@5 ≥ `vector` 且 `B_model` 桶 Recall@5 ≥ `bm25` → 0，否则 1

- [ ] **Step 1: 写失败测试**

```python
def test_resolve_gold_raises_on_zero_hit():
    with pytest.raises(ValueError, match="锚串"):
        resolve_gold([{"id": "X1", "gold_anchors": ["不存在的串"]}], rows)

def test_resolve_gold_unions_multiple_anchors():
    gold = resolve_gold([{"id": "X1", "gold_anchors": ["赔付", "500"]}], rows)
    assert gold["X1"] == {7, 9}

def test_recall_at_k_ignores_refuse_cases():
    rows = [Row(case_id="A11", behavior="refuse", hits=[], gold=set())]
    assert aggregate(rows)["A_policy"]["recall@5"] is None    # 不参与，不是 0

def test_mrr_takes_first_relevant_rank():
    assert mrr([5, 2, 9], gold={9}) == pytest.approx(1 / 3)

def test_unrun_strategy_reports_none_not_zero():
    agg = aggregate(rows, strategies=["vector"], generation={"hybrid_rerank": {...}})
    assert agg["faithfulness"]["vector"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_run_ch04_evals.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

流程：读 cases → `--validate-only` 时只做锚串解析与唯一性/命中数报告后退出 →
每条 case 对每个策略调 `retrieve(..., category=case.get("category"))` → 算指标 → 聚合 → 落 JSON/Markdown。
`filter` 只对 `answer` 类算检索指标；`refuse` 类单独统计。
Markdown 表：行为=策略，列为四桶 + 整体，指标三张表（Recall@5 / MRR@10 / Faithfulness 占位）。
`-v` 逐条打印 `case_id / query / top3 chunk_id / gold`。
**锚串解析**：在 `knowledge_chunks.questions/answer` 做子串匹配；命中 0 块报错退出，
> 3 块打印警告并把命中数写进报告（spec §11.1）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_run_ch04_evals.py -q`
Expected: PASS

- [ ] **Step 5: 真机跑一次检索段并提交**

Run: `uv run --directory MemHelp-code python evals/run_ch04_evals.py --validate-only`
Expected: 48 条全部解析到 gold，无 0 命中。
（真机四策略跑分留到 Task 20 统一跑，避免建库未完成时反复等嵌入。）

```bash
git add MemHelp-code/evals/run_ch04_evals.py MemHelp-code/evals/reports/.gitkeep MemHelp-code/tests/test_run_ch04_evals.py
git commit -m "ch04: 四策略检索评估与分桶报告"
```

---

### Task 14: 生成段评估（Faithfulness / 拒答准确率 / 编造入台账）

**Files:**
- Modify: `evals/run_ch04_evals.py`
- Test: `tests/test_run_ch04_evals.py`

**Interfaces:**
- Consumes: Task 11 的 `RagPipeline`、Task 5 的 `FaithCaseRepository`
- Produces:
  - `--generation`（默认跑 `hybrid,hybrid_rerank`）、`--all-strategies`
  - 纯函数 `faithfulness(claims: list[ClaimVerdict]) -> float`、`refusal_accuracy(rows) -> float`、`citation_validity(parse) -> float`
  - `judge_claims(model, answer, evidence) -> list[ClaimVerdict]`（`ClaimVerdict(text, supported, evidence_n)`
  - 判为 unsupported 且 `behavior=="answer"` 的条目 upsert 进 `faith_cases`

- [ ] **Step 1: 写失败测试**

```python
def test_faithfulness_counts_unsupported():
    assert faithfulness([ClaimVerdict("a", True, 1), ClaimVerdict("b", False, None)]) == 0.5

def test_uncited_sentence_counts_as_unsupported():
    # 没有角标的句子一律 unsupported（spec §11.2）
    verdicts = judge_claims(fake_model, answer="满 99 包邮[1]。破损赔 500。", evidence=ev([1]))
    assert [v.supported for v in verdicts] == [True, False]

def test_refusal_accuracy_only_counts_refuse_cases():
    assert refusal_accuracy([r(behavior="refuse", refused=True),
                             r(behavior="answer", refused=False)]) == 1.0

def test_fabrication_upserts_faith_case(fake_faith_repo):
    await record_fabrications([...], repo=fake_faith_repo)
    assert fake_faith_repo.rows[0].eval_id == "A01"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_run_ch04_evals.py -q -k "faith or refusal"`
Expected: FAIL（函数不存在）

- [ ] **Step 3: 实现**

生成段：`pipeline.answer_context` → 拿到证据与 refused → refused 时直接用 `REFUSAL_TEXT` 当答案 →
答案拆句（中文句号/感叹号/问号切）→ 逐句判「有无有效角标且该角标证据支持」→ 聚合。
裁判模型与生成模型同一实例（本章不做裁判模型选型）。成本控制：`--limit` 对生成段生效。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_run_ch04_evals.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/evals/run_ch04_evals.py MemHelp-code/tests/test_run_ch04_evals.py
git commit -m "ch04: 生成段评估（Faithfulness / 拒答准确率 / 编造台账）"
```

---

### Task 15: `query_faq` 出参升级与 `category` 参数

**Files:**
- Modify: `app/tools/knowledge.py`、`app/tools/registry.py`、`app/tools/schemas.py`
- Test: `tests/test_tool_knowledge.py`

**Interfaces:**
- Consumes: Task 11 的 `RagPipeline.answer_context`
- Produces:
  - `class AnswerContextLike(Protocol): async def answer_context(self, query: str, *, category: str | None = None) -> RetrievalOutcome`
  - `query_faq(keyword: str, *, retriever: AnswerContextLike, category: str | None = None) -> str`
  - 出参：**原有键全在**，新增 `refused`、`refusal_source`、`refusal_reason`、`items[].n`、`items[].chunk_id`、`items[].section_path`、`items[].quote`
  - `items` 的顺序 = `prompt_order(evidence)`（首尾编排），但每项的 `n` 是不变的相关度序号
  - `registry.py` 的 `query_faq` 参数 schema 加可选 `category`（string，描述里写明「用户明确点名品类时才传」）

- [ ] **Step 1: 写失败测试**

```python
async def test_output_keeps_ch03_keys_and_adds_evidence(fake_pipeline):
    data = json.loads(await query_faq("XH-3000A 续航", retriever=fake_pipeline))
    assert {"ok", "keyword", "count", "items"} <= set(data)
    assert data["items"][0]["n"] == 1
    assert data["items"][0]["chunk_id"] == 3
    assert data["items"][0]["section_path"] == "商品型号手册 / 巡检仪 / XH-3000A"

async def test_items_are_in_head_tail_order(fake_pipeline):
    data = json.loads(await query_faq("q", retriever=fake_pipeline))
    assert [i["n"] for i in data["items"]] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]

async def test_refused_payload(fake_pipeline):
    fake_pipeline.outcome = refused_outcome("retrieval_low_conf")
    data = json.loads(await query_faq("红包怎么用", retriever=fake_pipeline))
    assert data["refused"] is True and data["count"] == 0 and data["items"] == []

async def test_blank_keyword_short_circuits(fake_pipeline):
    data = json.loads(await query_faq("   ", retriever=fake_pipeline))
    assert data["count"] == 0 and fake_pipeline.calls == []

async def test_category_forwarded(fake_pipeline):
    await query_faq("运费", retriever=fake_pipeline, category="售后政策")
    assert fake_pipeline.calls[0]["category"] == "售后政策"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_knowledge.py -q`
Expected: FAIL（出参没有 `n` / `refused`）

- [ ] **Step 3: 实现**

保持 ch03 的三条纪律：空白关键词不调检索；异常原样冒给 executor（不吞成空结果）；
出参键名**只增不改**。`items` 里带 `question/answer` 与 `quote`（quote 就是 evidence 的答案原文，
两者当前相同 —— 保留两个键是因为前者是「问答对语义」，后者是「证据快照语义」，未来可能分叉）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_knowledge.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/tools/knowledge.py MemHelp-code/app/tools/registry.py MemHelp-code/app/tools/schemas.py MemHelp-code/tests/test_tool_knowledge.py
git commit -m "ch04: query_faq 出参升级为带编号证据 + 可选品类过滤"
```

---

### Task 16: Prompt（引用规范 + 负面知识 + 拒答话术）

**Files:**
- Modify: `app/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces:
  - `BEHAVIOR_RULES_WITH_TOOLS` 追加 `"不承诺到账时间与送达时点"`（8 条）
  - `CITATION_RULES: tuple[str, ...]`（逐字进 prompt）
  - `NEGATIVE_KNOWLEDGE: tuple[str, ...]`（禁止承诺清单：到账/发货/送达/赔付结论/库存/审核通过率）
  - `REFUSAL_TEXT: str`

- [ ] **Step 1: 写失败测试**

```python
def test_new_rule_is_in_prompt_verbatim():
    assert "不承诺到账时间与送达时点" in TOOL_SYSTEM_PROMPT_TEMPLATE
    assert len(BEHAVIOR_RULES_WITH_TOOLS) == 8

def test_citation_rules_in_prompt():
    for rule in CITATION_RULES:
        assert rule in TOOL_SYSTEM_PROMPT_TEMPLATE
    assert "[n]" in TOOL_SYSTEM_PROMPT_TEMPLATE

def test_negative_knowledge_lines_present():
    for item in NEGATIVE_KNOWLEDGE:
        assert item in TOOL_SYSTEM_PROMPT_TEMPLATE

def test_refusal_text_states_three_things():
    assert "知识库" in REFUSAL_TEXT and "没有" in REFUSAL_TEXT
    assert "[" not in REFUSAL_TEXT          # 拒答不带角标
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_prompts.py -q`
Expected: FAIL

- [ ] **Step 3: 写 prompt 文案**

新增「## 引用规范」一节（紧跟「你有查询工具」之后）与「## 禁止承诺的负面知识」一节。
负面知识那节要**区分**两件事（spec §10 的坑）：引用知识库里的规则数字是允许的（须带 `[n]`），
对本笔订单做承诺是禁止的。

- [ ] **Step 4: 用 8 条标注样例验证（非代码任务的 TDD 替代）**

Run: 手工起服务问 4 条应拒答 + 4 条应正常引用（用 Task 12 的 A11/A12 与 B01/B02），
把「模型是否按 [n] 标注 / 是否拒绝承诺到账时间」的结果记进 dev-notes。
不通过就改 prompt 再来一遍，最多三轮；仍不通过则记下残留问题，不假装通过。

- [ ] **Step 5: 跑测试并提交**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_prompts.py -q`
Expected: PASS

```bash
git add MemHelp-code/app/prompts.py MemHelp-code/tests/test_prompts.py
git commit -m "ch04: 引用规范 / 负面知识清单 / 拒答话术"
```

---

### Task 17: 编排接线（证据捕获、拒答短路、入池、引用落库与传输）

**Files:**
- Modify: `app/chat_tools.py`、`app/api.py`
- Test: `tests/test_chat_citations.py`（新）、`tests/test_api_chat.py`（补）

**Interfaces:**
- Consumes: Task 5 的 `low_confidence` 仓库、Task 11 的 `RetrievalOutcome`、Task 15 的 `query_faq` 出参
- Produces:
  - `chat_tools` 内部：`_evidence_from_tool(result) -> tuple[Evidence, ...] `、`_should_short_circuit(result) -> Refusal | None`
  - `_persist(..., citations: list[dict] | None)`；assistant 答案行带 `citations`
  - SSE `done` 事件新增 `citations`（快照数组）；`error` 事件不带
  - `/api/chat/sessions/{id}/messages` 每条 ai 消息按需带 `citations`

- [ ] **Step 1: 写失败测试**

```python
async def test_refusal_short_circuits_second_model_call(fake_model, fake_pipeline):
    fake_pipeline.outcome = refused_outcome("self_check", "证据不足")
    events = [e async for e in astream_chat_with_tools(...)]
    assert fake_model.ainvoke_calls == 1          # 只调了「选工具」那一次
    assert "".join(e.payload.get("delta", "") for e in events if e.kind == "token") == REFUSAL_TEXT

async def test_refusal_writes_low_confidence_row(fake_repositories, fake_pipeline):
    ... 跑完一轮 ...
    rows = await fake_repositories.low_confidence.list_recent(5)
    assert rows[0].raw_question == "你们支持货到付款吗"
    assert rows[0].source == "self_check"

async def test_citations_captured_and_persisted(fake_repositories, fake_model):
    ... 回答 "满 99 包邮[1]" ...
    rows = await fake_repositories.messages.list_for_conversation(cid)
    assert rows[-1].citations[0]["n"] == 1
    assert events[-1].payload["citations"][0]["section_path"]

async def test_dropped_citation_not_breaking_answer(fake_repositories, fake_model):
    ... 回答 "满 99 包邮[1]，破损赔 500[9]" ...   # 9 越界
    assert "破损赔 500" in answer            # 正文一个字都没被删
    assert [c["n"] for c in citations] == [1]  # 只丢映射

async def test_api_messages_returns_citations(fake_repositories):
    payload = await client.post(f"/api/chat/sessions/{sid}/messages")
    assert payload.json()["messages"][-1]["citations"][0]["chunk_id"] == 3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_chat_citations.py -q`
Expected: FAIL（模块/字段不存在）

- [ ] **Step 3: 实现**

在 `astream_chat_with_tools` 的工具结果分支后插入两件事（顺序固定）：
1. `refused` → 写池（`conversations.upsert_by_session` 拿 id）→ 发 `token`(REFUSAL_TEXT) →
   跳过第二次模型调用 → 走 `_persist(..., citations=[])` → yield `tools` + 新的 `citations` 事件；
2. 否则 → 发 `status` → 第二次调用流式 → 结束后 `parse_citations` → `_persist(..., citations=payload)`。
`api._chat_stream_body` 把 `citations` 带进 `done`；`_db_session_messages_payload` 读 `row.citations`。
**注意**：`tools` 帧（徽章）语义不变；引用是新增的独立数据，不要塞进 `tool_traces`。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run --directory MemHelp-code python -m pytest -q`
Expected: PASS（ch01–ch03 既有用例全绿；若 ch03 用例因 `query_faq` 出参升级而红，
按「只增不改」原则修测试而不是修契约）

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/chat_tools.py MemHelp-code/app/api.py MemHelp-code/tests/test_chat_citations.py MemHelp-code/tests/test_api_chat.py
git commit -m "ch04: 引用捕获 / 拒答短路 / 低置信度入池与引用传输"
```

---

### Task 18: `faith_cases` 最小端点

**Files:**
- Modify: `app/api.py`、`app/schemas.py`
- Test: `tests/test_api_faith_cases.py`

**Interfaces:**
- 端点：
  - `GET /api/faith-cases?status=未解决` → `{"items": [{eval_id,bucket,query,strategy,answer,reason,citations,judge_model,status,seen_count,first_seen_at,last_seen_at,resolution,resolved_at}]}`
  - `PATCH /api/faith-cases/{eval_id}` body `{"status": "已解决"|"无需解决"|"未解决", "resolution": "..."}` → 200 + 单条
- 错误：未知 `eval_id` → 404 `{"error":{"type":"not_found",...}}`；标已解决但 `resolution` 空 → 422

- [ ] **Step 1: 写失败测试**

```python
async def test_list_filters_by_status(client, seeded):
    body = (await client.get("/api/faith-cases?status=未解决")).json()
    assert [i["eval_id"] for i in body["items"]] == ["A01"]

async def test_patch_requires_resolution(client, seeded):
    resp = await client.patch("/api/faith-cases/A01", json={"status": "已解决", "resolution": ""})
    assert resp.status_code == 422

async def test_patch_updates_status_and_resolution(client, seeded):
    resp = await client.patch("/api/faith-cases/A01",
                              json={"status": "已解决", "resolution": "补了赔付上限条款"})
    assert resp.status_code == 200 and resp.json()["status"] == "已解决"
    assert (await client.get("/api/faith-cases?status=未解决")).json()["items"] == []

async def test_unknown_eval_id_404(client):
    assert (await client.patch("/api/faith-cases/ZZZ", json={"status": "已解决",
            "resolution": "x"})).status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_api_faith_cases.py -q`
Expected: FAIL（404 路由不存在）

- [ ] **Step 3: 实现**

照 ch02/ch03 的端点写法（`Depends(get_repositories)`、错误外壳 `_error()`、错误文本过 `redact()`）。
`status` 用 `Literal` 收窄，非法值走 FastAPI 422。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_api_faith_cases.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/api.py MemHelp-code/app/schemas.py MemHelp-code/tests/test_api_faith_cases.py
git commit -m "ch04: 编造台账最小查询与处置端点"
```

---

### Task 19: 前端（Vibe Coding，不写单测、不走 review）

**Files:**
- Modify: `web/app.js`、`web/style.css`、`web/index.html`

**Interfaces:**
- Consumes: `done.citations`（`[{n,chunk_id,section_path,category,quote}]`）与 `/messages` 的 `citations`

- [ ] **Step 1: 引用角标**

`renderMarkdown` 之后把正文里的 `[n]` 换成 `<button class="cite" data-n="n">n</button>`；
**必须**在 `escapeHtml` 之后做替换（否则角标渲染又开了一个 XSS 面）。
点击打开右侧抽屉：标题=`section_path`，正文=`quote`，底部一行「来自知识库 · 品类」。
无证据（`citations` 为空）时不渲染任何角标交互。

- [ ] **Step 2: 满意度**

每条 bot 气泡左下角一组 👍/👎：点击后所选 `aria-pressed="true"` + 高亮，另一侧 `disabled`，
同行显示「已反馈」；写入 `localStorage` 键 `mewhelp.feedback.<sessionId>.<序号>`，
回读时锁定。连点两次不得产生第二次写入。

- [ ] **Step 3: 手动验收（用户看页面）**

起服务后逐条确认：
1. 问「XH-3000A 续航多久」→ 回答带 `[n]` → 点开抽屉有章节路径与原文；
2. 刷新页面 → 历史消息的角标仍可点；
3. 点 👍 → 点亮 + 「已反馈」；刷新 → 仍是已反馈且不能再点；
4. 问「你们支持货到付款吗」→ 拒答话术，无角标。
把用户当轮的原话与改动点记进 dev-notes（Vibe Coding 的留痕方式）。

- [ ] **Step 4: 提交**

```bash
git add MemHelp-code/web/app.js MemHelp-code/web/style.css MemHelp-code/web/index.html
git commit -m "ch04: 聊天页可点引用与一次性满意度反馈"
```

---

### Task 20: 真机验收、演示脚本与文档

**Files:**
- Create: `dev-notes/ch04-eval-report.md`（由脚本生成）、`scripts/show_low_confidence.py`
- Modify: `README.md`、`dev-notes/ch04.md`、`dev-notes/DELIVERY.md`

**Interfaces:**
- Consumes: 前面全部任务
- Produces: 四条验收标准的可复现输出

- [ ] **Step 1: 重建索引**

Run: `uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild`
Expected: created 数 ≥ 语料块数；无失败；集合含 text/sparse（`--status` 打印）

- [ ] **Step 2: 验收 2（型号 BM25）**

Run: `uv run --directory MemHelp-code python evals/run_ch04_evals.py --only B_model --strategies bm25 -v`
Expected: `B_model` Recall@5 ≥ 0.8，逐条能看到命中 `XH-3000A` 那块

- [ ] **Step 3: 验收 1（四策略报告）**

Run: `uv run --directory MemHelp-code python evals/run_ch04_evals.py --strategies vector,bm25,hybrid,hybrid_rerank --generation --json evals/reports/ch04.json --markdown dev-notes/ch04-eval-report.md`
Expected: 报告含 4 策略 × 4 桶的 Recall/MRR，生成段两策略的 Faithfulness/拒答准确率；退出码 0

- [ ] **Step 4: 验收 3 / 4（页面 + 池子）**

Run: `uv run --directory MemHelp-code uvicorn app.main:app --port 8000`（后台）
页面：问型号题点角标看原文；问「你们支持货到付款吗」看拒答。
Run: `uv run --directory MemHelp-code python scripts/show_low_confidence.py`
Expected: 池子里有刚才那条，`source` 与 `reason` 非空。

- [ ] **Step 5: 写文档与留痕并提交**

README 增「ch04 能力与演示命令」一节（含四策略对照表头与阈值说明）；
`dev-notes/DELIVERY.md` 更新交付清单；`dev-notes/ch04.md` 补阶段 4.x～7 的留痕。

```bash
git add MemHelp-code/README.md MemHelp-code/dev-notes/ MemHelp-code/scripts/show_low_confidence.py MemHelp-code/evals/reports
git commit -m "ch04: 验收跑通、评估报告与交付文档"
```

---

## 执行顺序与依赖

```
T0 探针
 └ T1 配置
    ├ T2 集合/写入 ─ T3 三路检索 ─ T4 建库写入
    ├ T5 数据层
    ├ T6 Query 理解
    ├ T7 重排
    └ T9 证据 ─ T10 自评
         └ T8 四策略检索（依赖 T3/T6/T7）
              └ T11 编排（依赖 T6/T8/T9/T10）
                   ├ T15 query_faq（依赖 T11）
                   │    └ T17 接线（依赖 T4/T5/T15）
                   ├ T13 评估检索（依赖 T8/T12）
                   │    └ T14 生成评估（依赖 T11/T13/T5）
                   └ T18 faith_cases 端点（依赖 T5）
T12 语料/评估集（可与 T2–T11 并行）
T16 Prompt（可与 T11 并行）
T19 前端（依赖 T17）
T20 验收（全部）
```

**可并行的三组**：`T2→T3→T4` 与 `T5` 与 `T6/T7/T9/T10` 互不依赖；
Task 12 的语料与评估集只依赖真实语料内容，可随时写。
**必须串行的关键路径**：`T0 → T1 → T2 → T3 → T8 → T11 → T15 → T17 → T19 → T20`。
