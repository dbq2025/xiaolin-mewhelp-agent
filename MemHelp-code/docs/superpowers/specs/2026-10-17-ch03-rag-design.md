# MewHelp 第三章「RAG 基础 · 向量语义检索」设计文档

- 日期：2026-10-17
- 状态：待用户评审
- 上游：沿用 ch01/ch02 单端点（`https://api.deepseek.com/v1`，`deepseek-flash`）
- 前置：ch02「Function Calling 工具链」已验收（`main`，HEAD `82cb528`）

## 0. 流程分类与范围

**分类：architectural**（新增离线建库子系统、新持久化面 Milvus、替换在线检索实现，
并重划「原文权威源」边界）。已在对话中当面向用户宣布，用户未推翻。

**本章流程变体（用户指定）**

| 对象 | 走的流程 |
|---|---|
| 切分器、双写状态机、检索链路、挖矿流水线、`query_faq` 改造 | brainstorm → spec → plan → 执行；**有可单测代码的走 TDD** |
| 三份 Markdown 语料、抽取 Prompt、评估集 | 不走 TDD，改用**标注样例 / 评估集跑一遍**验证 |

**本章不做（用户明确）**

- 关键词召回、混合检索（hybrid）、重排（rerank）——只跑 dense 向量单路
- 关键词检索降级兜底（Milvus 或嵌入不可用时**不回落** `LIKE`，由 executor 分类成 `ok=false`）
- 知识库的后台管理界面、人工审核流（`qa_extraction_staging` 的 `approved/rejected` 不在本章用）
- 文档增量监听（改一个文件重建整篇即可）

### 验收标准与判定方式

| # | 验收标准 | 判定方式 | 通过条件 |
|---|---|---|---|
| 1 | 「邮费是多少」这类换说法问题召回运费说明并答对 | ① `evals/run_rag_evals.py` 的 `paraphrase-fee` 用例；② 浏览器问同一句，工具徽章 `query_faq · 成功` 且答案含包邮/运费口径 | 向量检索命中且答案正确；**同一语料下关键词基线不命中**（对照表两列都要出） |
| 2 | 故意中断建库再重跑，漏向量化的块被捡起补齐 | `--rebuild --max-chunks 5` 建库（只补前 5 块）→ `--status` 看 pending → 不带 `--max-chunks` 重跑 → `--status` | 中断后 `pending = 总块数 - 5`；重跑后 `pending=0`、`done 数 == Milvus 行数 == 总块数`、Milvus 无重复主键 |

> 验收 1 的「对照」是硬要求：只证明向量能召回不算数，必须同时给出**同一语料**
> 下关键词检索的命中情况，否则无法证明升级真的解决了 ch02 的漏召回。

## 1. 已确认的决策（对话中逐个敲定）

| 决策点 | 结论 | 理由 |
|---|---|---|
| 嵌入运行形态 | **进程内 `sentence-transformers`，只跑 CPU** | 上游无 `/embeddings`（实测 404）；单进程失败面最小。GPU（RTX 5060 / sm_120）对 torch 版本过于挑剔，本章不赌 |
| 语料来源 | **我写三份演示 Markdown**，与现有 `faq` 表口径一致 | 切分规则必须有真实形态的语料才能验收（多级标题 / 超长条款 / 大表格 / 关键条款） |
| 定时任务形态 | **CLI 脚本 + 外部调度器** | 无新依赖、可观察、可重跑；文档给 Windows 计划任务与 cron 的注册写法 |
| 契约不变的范围 | **改 `description`，参数名与出参 JSON 一字不动** | `description` 是写给模型的提示词，不是接口形状；不改它，模型会继续截成名词，向量检索的优势被耗掉一半 |
| 去重口径 | **字面归一化 + 向量余弦阈值（默认 0.92）** | 字面相同零误杀；近义靠向量；丢弃行留档可追溯 |
| 演示历史会话 | **造约 30 条 `demo-ch03-*` 会话入库** | 现有 5 个会话 / 7 条用户消息，切多批做「跨批去重」看不出效果 |

## 2. 现状勘查与环境实测

### 2.1 仓库现状

- `query_faq` 现实现：`app/tools/knowledge.py::query_faq(keyword, *, faq_repo)` →
  `FaqRepository.search()` 的 `LIKE %kw%`（`\`/`%`/`_` 已在仓库层转义）。
- 工具接线：`app/tools/registry.py::get_tools(repositories)` 用闭包把当次请求的仓库绑进工具；
  `app/chat_tools.py` 调 `get_tools(repositories)`；`app/api.py::create_app` 可注入
  `model` / `store` / `settings` / `repositories` / `db_ping`，测试靠它离线运行。
- 仓库层：所有 SQL 只在 `app/db/repositories.py`，同步调用经 `anyio.to_thread.run_sync`。
- 数据库：`knowledge_chunks`、`qa_extraction_staging` **已由用户建好**（实测 `SHOW CREATE TABLE`）。

### 2.2 环境实测（全部为探针结果，非记忆）

| 项 | 实测 |
|---|---|
| Milvus | 已运行；`:19530` gRPC、`:9091` 健康 200；`milvus_build_info` → **2.6.22**；集合列表为空 |
| pymilvus | **必须 2.6.X**（官方兼容表：Milvus 2.6.* ↔ pymilvus 2.6.X）。`uv run --with pymilvus` 默认拉 3.0.2：能连上，但与官方兼容表不符、API 面按 3.x 演进，**不采用** |
| 实测通过的能力 | `create_schema` + `add_field` + `prepare_index_params` + `create_collection`；同主键 `upsert` 两次仍 1 行；`search(consistency_level='Strong')` 写完立刻可读；`query(filter='', output_fields=['count(*)'])` 可数行 |
| 实测坑 | `get_collection_stats().row_count` 不可靠（写完仍报 0），**不用它**做校验 |
| MySQL | 8.4.11；`knowledge_chunks` = 用户 DDL 逐列一致；`qa_extraction_staging.status` 真库是 5 值超集（多 `approved`/`rejected`） |
| 上游 embeddings | `POST /v1/embeddings` → **404**，DeepSeek 不提供 |
| BGE-M3 | `pytorch_model.bin` 2271 MB；ModelScope 4.32 MB/s vs HF 1.04 MB/s |
| torch | 2.14.1 Windows 轮子 **124 MB（纯 CPU）**，不会拖 CUDA 组合包 |
| GPU | RTX 5060 Laptop 8GB（本章不用） |
| 宿主机 HTTPS | PowerShell `Invoke-WebRequest` 全挂（schannel），探针一律走 Python `httpx` |

## 3. 数据层设计

### 3.1 Milvus 集合 `knowledge`

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `VARCHAR(64)`，主键，`auto_id=False` | **= `str(knowledge_chunks.id)`**，确定性主键 |
| `vector` | `FLOAT_VECTOR(dim=1024)` | BGE-M3 dense 输出，已 L2 归一化 |

- 索引：`AUTOINDEX`，度量 `COSINE`。
- **只有两列**：`section_path` / `content_type` / `is_key_clause` / `prev` / `next`
  一个都不进 Milvus —— 用户要求「只存不进向量」，且原文权威源是 MySQL。
- 确定性主键的意义：`upsert` 天然幂等，重跑不会产生第二条向量，也不需要
  「先查 vector_id 再决定 insert/update」的往返（这是 Context7 文档推荐的 RAG 重摄入做法）。

### 3.2 MySQL ORM 与真库对齐

`app/db/models.py` 新增两个 ORM 类，**逐列对齐用户 DDL**（与 ch02 同一条纪律）：

- `KnowledgeChunk`（表 `knowledge_chunks`）：`category` / `questions` / `answer`（进向量），
  `section_path` / `content_type` / `is_key_clause` / `prev_chunk_id` / `next_chunk_id`
  （元数据不进向量），`vector_id` / `vectorize_status`（双写状态）。
  `prev/next` 是自引用外键（`ON DELETE SET NULL`）。
- `QaStaging`（表 `qa_extraction_staging`）：`batch_no` / `source_ref` / `question` /
  `answer` / `status`。
  **真库 ENUM 是 5 值超集**，ORM 的 `SAEnum` 必须包含 `approved` / `rejected`，
  否则读到后续章节写入的行会 `LookupError`。本章只写 `extracted` / `kept` / `discarded`。
  这条按 ch02 惯例记入 `tests/test_db_mysql.py::KNOWN_SCHEMA_DIVERGENCES`。

### 3.3 幂等与续跑（验收 2 的实现基础）

| 层 | 幂等手段 |
|---|---|
| MySQL 建块 | 自然键 `(content_type, section_path, questions)` 先查后插；`answer` 变了就 UPDATE，`vectorize_status` 重置为 `pending` |
| Milvus 写向量 | 确定性主键 `str(id)` + `upsert`（同 id 覆盖，不产生第二行） |
| 续跑 | 只取 `vectorize_status='pending' OR vector_id IS NULL` 的块补向量 |
| 回填 | 每批 `UPDATE ... SET vector_id=str(id), vectorize_status='done'` |

**三种中断点的收敛性**（spec 层面的承诺，测试要逐条钉住）：

1. MySQL 已提交、Milvus 未写 → 块仍是 `pending`，重跑补齐。
2. Milvus 已写、回填未做 → 块仍是 `pending`，重跑用同一主键覆盖，**不产生重复**。
3. 两库都写成功、进程在循环下一页时被杀 → 该块已 `done`，重跑跳过。

## 4. Markdown 结构感知切分

### 4.1 数据结构

```python
@dataclass(frozen=True)
class Chunk:
    category: str        # 上级标题路径，进向量
    questions: str       # 当前标题（FAQ 文档里标题就是真实问法），进向量
    answer: str          # 本节正文，进向量
    section_path: str    # 含当前标题的完整路径，元数据
    content_type: str    # faq / policy / manual
    is_key_clause: bool  # 元数据
    ordinal: int         # 文档内序号，用于回填 prev/next
```

```python
def split_markdown(text: str, *, content_type: str, source: str,
                   max_chars: int = 600, overlap: int = 80) -> list[Chunk]
```

纯函数、无 IO —— 这是本章最该被 TDD 钉死的一块。

### 4.2 切分规则

1. **结构**：按 ATX 标题（`^#{1,6}\s+`）建栈。
   - `section_path` = 祖先标题 + 当前标题，`" / "` 连接；
   - `category` = 祖先标题（**不含**当前标题）；顶层时退化为文档一级标题；
   - `questions` = 当前标题文本（去掉 `#` 与前导空白）。
   - 于是「商品 FAQ 填真实问法」与「政策手册填章节标题」是**同一条规则** ——
     FAQ 文档把标题写成问句即可，不需要第二套解析。
2. **超长递归**：本节正文 > `max_chars` 时，按
   「空行分段 → 句子拼装 → 硬切」三级递归降级。拼装用贪心：能塞进 `max_chars` 就继续塞。
3. **重叠裁句**：因长度被迫切开的相邻块，后一块前缀复述前一块尾部 ≤ `overlap` 字；
   **回退到最近的句末标点**（`。！？；.!?;`）或换行，宁可少重叠，绝不从半句开始。
   不同 section 之间**不加重叠**（章节边界比连贯性重要）。
4. **表格**：识别连续 `|` 行（`| a | b |` + `|---|` 分隔行）。
   - **表格不参与第 2 条的「按字符递归切」** —— 否则切出来的后半截会丢表头，
     正是需求里点名要避免的。表格只按**行**切。
   - 判定顺序：整表字符 ≤ `max_chars` 且数据行 ≤ 20 → 整表作为一个原子块参与拼装；
     否则按行切：**每块都复制表头行 + 分隔行**，块首另加一行
     `（接上表：{最近的上文标题}）`，单行绝不跨块。
5. **关键条款**：只认显式标记 —— 标题含「（关键条款）」或正文含 `<!-- key-clause -->`。
   不靠「赔付/期限」这类关键词猜：猜错了会污染元数据，而元数据是给人看的。
6. **空节**：只有标题没有正文的节跳过（不产出空 answer 的块）。

### 4.3 向量化文本

```
分类：{category}
问法：{questions}
答案：{answer}
```

固定前缀让「分类 / 问法 / 答案」三格在语义上可区分（BGE-M3 无需 instruction 前缀）。
拼接函数单一实现，建库与检索**必须共用**它，否则「建库时嵌入的文本」与「检索时
假想的文本」会悄悄分叉。

## 5. 离线建库编排（`app/knowledge/ingest.py`）

```python
@dataclass
class IngestReport:
    docs: int
    chunks_total: int
    chunks_created: int
    chunks_updated: int
    chunks_skipped: int
    vectors_written: int

async def ingest_markdown(paths, *, chunk_repo, vector_store, embedder, settings) -> IngestReport
async def ingest_faq_table(*, faq_repo, chunk_repo, vector_store, embedder, settings) -> IngestReport
async def vectorize_pending(*, chunk_repo, vector_store, embedder, settings, limit=None) -> int
```

流程：

```
读文件 → split_markdown → 逐块 chunk_repo.upsert_by_natural_key(status=pending)
       → 整篇插完后回填 prev/next（外键要求 id 先存在）
       → vectorize_pending：取 pending → 拼文本 → embed（批 32）→ Milvus upsert
       → 回填 vector_id / done
```

- `prev/next` 回填：同文档同 `section_path` 顺序，按 `ordinal` 前后相连；跨文档不连。
- 存量 `faq` 表也作为语料源：`content_type='faq'`，`questions=question`、`answer=answer`、
  `category=category`、`section_path='平台FAQ / {category}'`。这样「商品 FAQ 填真实问法」
  对既有数据同样成立。
- `--rebuild`：**drop 并重建 Milvus 集合**（清空向量）+ `TRUNCATE knowledge_chunks`，再全量重建。
  两库同时清空，避免「表清了集合没清」留下孤儿向量。**默认不 rebuild**，只做增量补 pending。
- 存量 `faq` 表读取需要 `FaqRepository.list_all()`（新增只读方法），返回纯数据
  `[{question, answer, category}]`。

## 6. 在线检索与 `query_faq` 改造

### 6.1 检索（`app/knowledge/retrieval.py`）

```python
@dataclass(frozen=True)
class KnowledgeHit:
    question: str      # 块的 questions
    answer: str
    category: str
    score: float       # 余弦相似度，仅用于日志/调试，不进工具出参

class KnowledgeRetriever(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[KnowledgeHit]: ...
```

流程：`embed(query) → Milvus search(Strong, top_k) → 按 id 批量取 MySQL rows →
按分数序回填 → 丢弃 MySQL 已删的行（记 warning 计数）`。

- 不设相似度阈值：Top-K 少召回是正常业务结果（`count` 可以小于 K）。
- 嵌入与 Milvus 调用都是同步库 → 用 `anyio.to_thread.run_sync` 送进工作线程，
  与仓库层同一条纪律（否则阻塞事件循环）。

### 6.2 `query_faq` 契约

| 项 | 改前 | 改后 |
|---|---|---|
| 参数名 / 类型 | `keyword: str`（必填非空） | **不变** |
| 出参 | `{ok, keyword, count, items:[{question, answer, category}]}` | **不变** |
| `description` | 「取最核心的名词，不要整句照抄」 | 「传用户问题的完整原话，不要截成名词」 |
| 内部实现 | `FaqRepository.search` 的 `LIKE` | `retriever.search(query, top_k)` |
| 异常语义 | 仓库异常原样冒出 → executor 分类 `ok=false` | **不变**（不降级回关键词） |

`items[].question` 返回块的 `questions` 原文（多问法换行连接，原样给模型）。
原 `MAX_FAQ_ITEMS=5` 不再是独立的业务常数：取数由 `RAG_TOP_K`（默认 5）决定，
`query_faq` 只把它作为 `retriever.search` 的 `top_k` 传入。

### 6.3 接线与冷启动

- `app/knowledge/embedder.py` 提供 `BgeM3Embedder`（懒加载单例）+ `Embedder` Protocol。
- `create_app` 新增两个可注入参数：`retriever` 与 `rag_warmup: bool = True`。
  - `rag_warmup=True` → `lifespan` 里预热：加载模型 + `ensure_collection`。
    预热失败**只告警不崩**（Milvus 挂了应用仍可起，查询时由 executor 分类成 `ok=false`）。
  - 测试的 `conftest` 注入 `rag_warmup=False`，整套测试**不加载 2.3GB 模型**。
- 冷启动成本：BGE-M3 首载约 20–40s，而 `TOOL_TIMEOUT_SECONDS=3`。所以
  **必须预热**，否则演示的第一个问句必然超时——这是本章最容易被当成「检索坏了」的坑。
- `get_tools(repositories, retriever)` 加一个参数；注册表自检逻辑不变。

## 7. 对话挖知识（`app/knowledge/mining.py` + `scripts/mine_conversations.py`）

### 7.1 流水线

```
选会话（未挖过）→ 按 batch_size=8 分组
  → 每批一次 LLM 调用（structured output: list[{question, answer}]）
  → 写 qa_extraction_staging（batch_no, source_ref=session_id, status=extracted）
全部批次结束后：
  → 读全部 extracted → 字面归一化去重 → 向量余弦 ≥0.92 去重
  → kept 入 knowledge_chunks（content_type='mined', category='对话挖掘'）
  → discarded 保留在暂存表（status='discarded'）
```

- **分批的原因**（用户原话「分批防串味」）：一次喂太多会话，模型会把 A 会话的答案
  接到 B 会话的问题上；按批调用且每批标明会话边界，串味概率显著下降。
- **整体去重的原因**：跨批去重是这一节的价值所在 —— 逐批去重会让「第 2 批里和第 1 批
  同义的问法」重复入库。
- 幂等：`source_ref` 已在暂存表的会话默认跳过；`--force` 才重挖。
- `--dedupe-only`：不调 LLM，只对暂存表里现有 `extracted` 行重跑一次全量去重并入库，
  用于「抽取已跑完、阈值调过了要重去重」的场景。
- `--status`：打印批号、`extracted/kept/discarded` 行数与已入 `knowledge_chunks` 的 `mined` 块数。
- LLM 失败：**整批不落库**（不做半批写入），记录批号并以退出码 1 结束；
  重跑会重新处理该批。
- 提示词要求模型只抽「用户真实问过的问题 + 客服给出的可复用答案」，
  寒暄、物流单号、订单号这类一次性信息丢弃。

### 7.2 去重细节

- 归一化：`NFKC` → 去所有空白与标点 → `casefold`。
- 字面完全相同 → 后者 `discarded`。
- 语义：候选问题向量与**已保留**问题逐个算余弦，`≥ QA_DEDUPE_THRESHOLD` 判同义：
  后者 `discarded`，并把它的问法并入保留行的 `questions`（换行连接，**最多 5 条**，
  防止 `questions` 无限膨胀）。
- 阈值可由 `.env` 覆盖；本章默认 0.92，用评估集回归验证。

## 8. 配置新增（`app/config.py` + `.env.example`）

```
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=knowledge
EMBEDDING_MODEL_PATH=models/bge-m3
EMBEDDING_DIM=1024
RAG_TOP_K=5
RAG_WARMUP=true
CHUNK_MAX_CHARS=600
CHUNK_OVERLAP=80
QA_BATCH_SIZE=8
QA_DEDUPE_THRESHOLD=0.92
```

- `EMBEDDING_MODEL_PATH` 相对路径以 `PROJECT_ROOT` 解析（与 `.env` 同一条纪律，
  不依赖 CWD）；目录不存在时给出明确报错，指向 `scripts/download_model.py`。
- 依赖新增：`sentence-transformers`、`torch`（CPU）、`pymilvus>=2.6,<3`。
  **不允许装 3.x**（实测服务端 2.6.22，官方兼容表要求 2.6.X）。
- `models/` 加进 `.gitignore`（2.3GB 绝不入库）。

## 9. 测试策略

| 层 | 方式 | 钉住什么 |
|---|---|---|
| `chunker.py` | **TDD**，纯离线 | 标题栈、section_path/category/questions 映射、超长递归、重叠回退到句末、表头复制、关键条款标记、空节跳过 |
| `embedder.py` | TDD，假后端 + 真模型（skip） | 懒加载只发生一次、归一化、维度 1024、批量顺序对齐 |
| `vector_store.py` | TDD，假客户端 + 真 Milvus（skip） | 建集合参数（dim/主键/COSINE）、幂等 upsert、Strong 读己写、count 口径 |
| `ingest.py` | TDD，SQLite + 假嵌入 + 假向量库 | 自然键幂等、三种中断点收敛、prev/next 回填、pending 只补漏、rebuild |
| `retrieval.py` | TDD，假嵌入 + 假向量库 + SQLite | 分数序保持、MySQL 缺行被丢弃、返回契约字段 |
| `tools/knowledge.py` | TDD 改写（ch02 用例迁移） | **出参 JSON 形状一字不变**、`ok/count/items` 语义、异常原样冒出 |
| `mining.py` | TDD，假 LLM + SQLite | 分批边界、整批失败不落库、跨批去重、kept 入库、幂等跳过 |
| `api.py` / `chat_tools.py` | 扩展既有 HTTP 测试 | `retriever` 注入生效、`rag_warmup=False` 不加载模型 |
| **真实 Milvus** | `tests/test_knowledge_milvus.py`（`MEMHELP_TEST_MILVUS=1`） | 集合 schema、upsert 幂等、写完立刻搜得到 |
| **真实 MySQL** | 扩展 `tests/test_db_mysql.py`（`MEMHELP_TEST_MYSQL=1`） | 两张新表逐列对齐、5 值 ENUM 可读写 |
| **真实嵌入** | `MEMHELP_TEST_EMBED=1` | BGE-M3 真加载、1024 维、同义句余弦显著高于无关句 |
| **评估集** | `evals/rag_cases.jsonl` + `evals/run_rag_evals.py` | 换说法召回率：关键词基线 vs 向量，一表两列 |

评估集格式（jsonl，每行一条）：

```json
{"id":"paraphrase-fee","query":"邮费是多少","expect_any":["包邮","运费"],"must_contain":["99"],"paraphrase":true}
```

判定：`answer` 命中 `expect_any` 任一且包含所有 `must_contain` 即算命中。
`--baseline keyword` 走 `knowledge_chunks` 上的 `LIKE`（**同一语料**，只换检索方式），
`--baseline vector` 走本章检索。评估规则沿用 ch01/ch02：**未达标先改切分/描述/阈值，
不许改评估标准**。

## 10. 验收步骤（交付时逐条跑）

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"

# 0) 依赖 + 模型（一次性）
& $uv sync
& $uv run --directory MemHelp-code python scripts/download_model.py

# 1) 建库（文档 + faq 表）
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status

# 2) 验收 2：故意中断再重跑
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild --max-chunks 5
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status     # pending = 总块数-5，done = 5
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status     # pending=0，done == Milvus

# 3) 验收 1：评估集对照（关键词语料基线 vs 向量，默认两者都跑）
& $uv run --directory MemHelp-code python evals/run_rag_evals.py

# 4) 挖知识（分批 + 整体去重）
& $uv run --directory MemHelp-code python scripts/mine_conversations.py --limit 30
& $uv run --directory MemHelp-code python scripts/mine_conversations.py --dedupe-only

# 5) 端到端
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
# 浏览器 http://127.0.0.1:8000 问「邮费是多少」
```

## 11. 交付物

| 交付物 | 路径 |
|---|---|
| 建表 DDL | 真库已有；本章补 `sql/ch03-ddl.sql` 作为可重建来源（含 5 值 ENUM 说明）。**`sql/` 全部首次提交**（ch02 三份 + ch03 两份），用户已裁定 |
| 演示语料 | `MemHelp-code/knowledge/{退货政策,商品FAQ,售后手册}.md` |
| 演示历史会话 | `sql/ch03-seed-history.sql`（`demo-ch03-*` 前缀） |
| 后端 | `MemHelp-code/app/knowledge/**`、`app/db/{models,repositories}.py`、`app/tools/{knowledge,registry}.py`、`app/{api,config}.py` |
| 脚本 | `scripts/{build_knowledge,mine_conversations,download_model}.py` |
| 评估集 | `evals/rag_cases.jsonl`、`evals/run_rag_evals.py` |
| 测试 | `tests/test_knowledge_*.py`、`tests/test_tool_knowledge.py`、`tests/test_db_*.py` |
| 开发留痕 | `MemHelp-code/dev-notes/ch03.md` |

## 12. 风险与已知限制

| 风险 | 应对 |
|---|---|
| pymilvus 版本漂移（默认装 3.0.2） | 引脚 `>=2.6,<3`；集成测试先断言 `get_server_version()` 与集合 schema |
| 冷启动 20–40s 打穿 3s 工具超时 | `rag_warmup` 在 lifespan 预热；演示命令先起服务再问 |
| CPU 编码慢（约 100–300ms/句） | 建库批 32；检索只编码 1 句，可接受 |
| `get_collection_stats` 不可靠 | 校验一律用 `query(count(*))` |
| 真库 ENUM 超集 | ORM 含 5 值 + `KNOWN_SCHEMA_DIVERGENCES` 记录 |
| `sql/` 目前未跟踪 | **已裁定**：`sql/` 全部首次提交（ch02 三份 + 本章两份），随本章代码一起进 git |
| 切分参数（600/80）没有理论最优 | 用评估集回归；参数可配，改参不改进评估标准 |
| 去重阈值 0.92 可能误杀 | 丢弃行留档可人工复核；阈值可配、有回归用例 |
| 模型目录 2.3GB | 加 `.gitignore`；`download_model.py` 支持断点续传 |
