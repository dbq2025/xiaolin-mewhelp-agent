# MewHelp

小林coding《AI 智能客服实战》实战项目 —— 电商智能客服。

技术栈：Python + FastAPI + LangChain。

## 目录约定（重要）

**本项目全部代码都在 `MemHelp-code/` 下**（就是本目录）。完整规则见工作区根的
[`../agent.md`](../agent.md)；下面是要点：

- 仓库根 = 工作区根 `F:\DeepSeek-Harness\memhelp-code-myself`（`.git` 在代码目录的**上一级**）
- 依赖与虚拟环境也留在工作区根（`.venv/`），`uv` 会自动在本目录找到 `pyproject.toml`
- 后续章节（ch02…）继续写在本目录里，不要另起仓库或移到工作区根

## 环境备注（本机）

- 运行时长命令统一用 `uv`（当前不在 PATH 的话用 `C:\Users\dbq\.local\bin\uv.exe`）
- 本机没有 `make`，课程里的 Makefile 目标需手工拆成对应命令执行
- **沙箱下必须设 `UV_CACHE_DIR`**：全局 `C:\Users\dbq\AppData\Local\uv\cache` 无写权限，
  缓存要指到工作区内，否则 `uv` 报 `failed to open file ... 拒绝访问 (os error 5)`
- 其它沙箱坑（锁死的临时目录、PowerShell 编码/执行策略、宿主 TLS）见 `../agent.md` 第 4 节

## 第一章「纯对话」运行说明

> 下面的命令**从工作区根**运行（也可以先 `cd MemHelp-code` 再跑，`uv` 都认）。
> 从根跑 pytest 时会用工作区根的 `pytest.ini`；从本目录跑用本目录的
> `pyproject.toml`——**两处测试配置必须保持一致**（原因见 `../agent.md` 第 3 节）。

### 1. 装依赖

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv sync
```

### 2. 配 `.env`

复制 `MemHelp-code/.env.example` 为 `MemHelp-code/.env` 并填真实值（`.env` 已在
`.gitignore`，**绝不入库**）：

```dotenv
OPENAI_BASE_URL=https://your-gateway.example.com/v1
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_MODEL=deepseek-chat
CONTEXT_TOKEN_BUDGET=2000
REQUEST_TIMEOUT=60
```

一个 OpenAI 兼容端点就够了，换 GPT / Claude / DeepSeek / Ollama 只改 `OPENAI_MODEL`（必要时改 `OPENAI_BASE_URL`），**代码不用动**。

`.env` 的路径在 `app/config.py` 里用绝对路径解析，**与当前工作目录无关**（早期用相对路径，
从仓库根启动会报 3 个 `Field required`）。

### 3. 起服务

`app` 包在本目录下，所以**不要**从工作区根直接 `uvicorn app.main:app`（会
`ModuleNotFoundError`）。用 `--directory` 或先 `cd` 进来：

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --reload --port 8000
# 或
cd MemHelp-code; & $uv run uvicorn app.main:app --reload --port 8000
```

### 4. 跑离线测试（不需要 `.env`，不发任何网络请求）

```powershell
# 从工作区根（用 ../pytest.ini）
& $uv run python -m pytest -v --color=no

# 或从本目录（用 ./pyproject.toml）—— 结果一致
cd MemHelp-code; & $uv run python -m pytest -v --color=no
```

**「不发网络请求」有实证**：把上游指向黑洞 `http://127.0.0.1:1/v1`（环境变量优先级
高于 `.env`）后仍然全绿 —— 测试确实没有真实网络调用。

### 5. 跑评估集（需要真实 `.env`）

```powershell
& $uv run --directory MemHelp-code python evals/run_evals.py
# 不想联网？带 --self-test 可以只验证评分逻辑
& $uv run --directory MemHelp-code python evals/run_evals.py --self-test
```

退出码：`0` 达标 / `1` 未达标 / `2` 配置缺失或上游故障（会提示「先修 `.env`，不要改 Prompt」）。

### 6. 聊天页面

起服务后浏览器打开 **http://127.0.0.1:8000/**（会自动跳到 `/chat/`）：

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
# 然后浏览器访问 http://127.0.0.1:8000/
```

页面特性：

- 消息气泡排布（客服在左、用户在右），进入即有欢迎语与三个示例问题可点
- **逐字渲染**：`fetch` 读 SSE 流，`token` 事件的 `delta` 逐个追加；等待时显示三点动画
- **多轮对话**：`session_id` 存在 `sessionStorage`，刷新页面继续同一会话；「清空」开新会话
- 顶栏状态灯反映 `/healthz`（绿色=已就绪并显示模型名，红色=配置缺失）
- 勾选「统计」可在每轮回复下显示上下文裁剪情况（保留/裁掉几条）
- Enter 发送、Shift+Enter 换行；移动端可用；用户往上翻时不会被强行拉回底部

**技术说明**：这里没用 `EventSource` —— 后端是 `POST` + SSE，而 `EventSource` 只支持
GET 且不能带请求体。所以用 `fetch` 拿 `ReadableStream` 自己按 SSE 规范切帧
（`event:` / `data:` / 空行分隔）。前端只有三个静态文件（`web/index.html`、
`style.css`、`app.js`），**无构建步骤、无 npm 依赖**，由 FastAPI 的 `StaticFiles` 直接托管。

### 7. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 跳转到 `/chat/` |
| GET | `/chat/` | 聊天页面（静态） |
| POST | `/api/chat/completions` | 对话，`stream=true` 走 SSE 逐 token，`false` 返回完整 JSON |
| POST | `/api/chat/sessions/{session_id}/messages` | 查看该会话已存历史 |
| POST | `/api/extract` | 售后描述 → 结构化 JSON |
| GET | `/healthz` | 健康检查（不回显密钥） |

### 8. 演示命令（三条验收标准）

> ⚠️ **沙箱提示**：在 DSH 沙箱里，宿主 shell 的 HTTPS 凭证初始化会被拒绝
> （`schannel: SEC_E_NO_CREDENTIALS`），所以**由我代为执行**的 `curl` 打到需要
> 出网的地址会失败。但 Agent 进程自己的 TLS 是正常的，所以下面这些打**本地**
> 服务的命令不受影响；真要验证上游连通性，请你在普通终端里跑。

**验收 1 · 流式回复**（`-N` 关掉缓冲，才能看到逐块滚出来）

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"session_id\":\"acc-1\",\"message\":\"我上周买的耳机还没发货\",\"stream\":true}"
```

**验收 2 · 两轮接住上下文**

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/chat/completions -H "Content-Type: application/json" -d "{\"session_id\":\"acc-2\",\"message\":\"我的订单号是 20240115001\",\"stream\":false}"
curl.exe -X POST http://127.0.0.1:8000/api/chat/completions -H "Content-Type: application/json" -d "{\"session_id\":\"acc-2\",\"message\":\"我刚才说的订单号是多少？\",\"stream\":false}"
curl.exe -X POST http://127.0.0.1:8000/api/chat/sessions/acc-2/messages
```

**验收 3 · 结构化 JSON**（第二条刻意不提订单号，用来验「零编造」）

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/extract -H "Content-Type: application/json" -d "{\"text\":\"订单 20240115001 昨天就该到，现在还没收到，我要退款\"}"
curl.exe -X POST http://127.0.0.1:8000/api/extract -H "Content-Type: application/json" -d "{\"text\":\"我的快递一直没动\"}"
```

### 9. 开发留痕

- 设计文档：`docs/superpowers/specs/2026-02-14-ch01-pure-chat-design.md`
- 实施计划：`docs/superpowers/plans/2026-02-14-ch01-pure-chat-plan.md`
- 开发过程追记：`dev-notes/ch01.md`
- 客服 Prompt 行为约束检查清单：`evals/service_prompt_cases.md`

---

## 第二章「Function Calling 工具链」运行说明

本章给客服装上五个工具（`query_order` / `query_product` / `query_logistics` /
`query_faq` / `create_ticket`），由模型自己决定调哪个。**只做单轮**：模型调一次工具就收敛。

### 1. 装依赖

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv sync     # 本章新增 sqlalchemy / pymysql / anyio
```

### 2. 准备数据库（两种接法，选一种）

**A) 复用已有的 `mewhelp` 库**（本机现状：`mewhelp-mysql` 容器，宿主 3306，`root`/`root`）

这类库通常是**别的章节建好的**，`conversations` 表里**没有 `session_id`** 列，
而本章的 ORM 靠它定位会话 —— 不迁移会直接报 `Unknown column 'session_id'`。
所以必须跑一次**幂等**迁移（可反复执行）：

```powershell
& $uv run --directory MemHelp-code python scripts/migrate_existing.py `
    --host 127.0.0.1 --port 3306 --user root --password root --database mewhelp
```

它只做三件事，且都不碰其它表：给 `conversations` 加 `session_id`（`NOT NULL` + 唯一键）、
把历史行的空会话键回填成 `demo-seed-<id>`、加唯一键。想先看会执行什么，加 `--dry-run`。

**B) 用本章自带的 compose 起一个干净的库**

```powershell
docker compose -f MemHelp-code/docker-compose.yml up -d --wait
```

新库会自动执行 `sql/ch02-ddl.sql`（建表）与 `sql/ch02-seed.sql`（灌 10 条 FAQ + 演示会话 + 2 条工单）。
它映射在宿主 **3307**（刻意避开常被占用的 3306），账号 `mewhelp`/`mewhelp`，没有 root。

> ⚠️ **改了 `sql/` 里的脚本想让新库重跑 initdb，必须先 `down -v`** ——
> MySQL 只在数据目录为空时才执行 `/docker-entrypoint-initdb.d` 下的脚本。
> 对既有库，改结构请追加一份迁移脚本，不要改 DDL（DDL 只对新库生效）。

### 3. 配 `.env`

在第一章的基础上追加 MySQL 与工具配置（`.env.example` 里有带注释的模板）：

```dotenv
# A) 复用既有库（默认）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=mewhelp

# 工具基础设施：单次执行超时（秒）与「仅对瞬时错误」的重试次数
TOOL_TIMEOUT_SECONDS=3.0
TOOL_MAX_RETRIES=1
```

> 参数错的工具调用与业务性「查不到」**不重试** —— 重试也是同样结果，只会让用户多等一个超时。

### 4. 起服务与跑测试

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000

# 全量（不连真库；MySQL 那部分自动跳过）
& $uv run --directory MemHelp-code python -m pytest -q

# 真实 MySQL 集成（会真的连库、建/删探针行）
$env:MEMHELP_TEST_MYSQL = "1"
& $uv run --directory MemHelp-code python -m pytest tests/test_db_mysql.py -v
```

`/healthz` 现在还会报 `"database": "ok" | "unavailable"`；数据库不可用时整体返回 **503**
（一个连不上库的实例接不了任何真实请求）。

> 怎么判断 /healthz 连的是哪个库：它**不回显**连接串（连接串含明文密码）。
> 看 `MYSQL_PORT` 即可 —— 3306 = 复用既有库，3307 = compose 起的新库。

### 5. 接口变化（第二章）

| 方法 | 路径 | 变化 |
|---|---|---|
| POST | `/api/chat/completions` | SSE 新增 `status` 帧（工具执行**之前**推，供页面显示「正在查询物流…」）；`done` 帧新增 `tools` 数组（本轮工具轨迹，含成败与耗时） |
| POST | `/api/chat/sessions/{session_id}/messages` | 改从 `messages` **表**回读（不再读内存）；每轮若调过工具则多一个 `tools` 字段 —— 这是聊天页徽章**刷新后仍在**的依据 |
| GET | `/healthz` | 新增 `database` 字段；库不可用返回 503 |

`status` 与 `done.tools` 的**分工**：前者只负责让用户立刻看到「在查数据」，
不承担持久化职责；后者是徽章的**权威来源**。

> **行为变更（需知情）**：为了拿到确定的 `tool_calls`，模型第 1 次调用改成了**非流式**。
> 于是「模型不调工具」这条路上，回答是**一次性**出现的；真正逐块的流式发生在
> 「调了工具、回灌之后」那一段。代价是首 token 多一个 RTT（记为后续可优化项）。

### 6. 浏览器验收（三条验收标准）

先起服务，再打开 <http://127.0.0.1:8000/chat/>：

1. 问 **「订单 1001 的物流到哪了」** —— 气泡上方先出现「正在查询物流…」，
   气泡下方出现 `🔧 查询物流 · 成功 · 1ms` 徽章，回答基于工具返回的真实数据。**刷新页面徽章仍在。**
2. 问 **「退货政策是什么」** —— 出现 `🔧 检索常见问题` 徽章，回答引用 FAQ 内容。
3. 问 **「邮费是多少」** —— 见下面的「实测结论」，**与最初预期不同**。

脚本化的可自动判定部分（含落库核对），会写一份输出文件：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo-ch02.ps1 `
    -Out dev-notes/ch02-acceptance-output.txt
```

### 7. 工具选择评估集

「模型自己决定调什么工具」是**模型的判断行为**，断言替代不了，所以用标注样例跑真实上游：

```powershell
& $uv run --directory MemHelp-code python evals/run_tool_evals.py
& $uv run --directory MemHelp-code python evals/run_tool_evals.py --only acc1,acc2
& $uv run --directory MemHelp-code python evals/run_tool_evals.py --json dev-notes/tool-evals.json
```

当前：**14/14 通过**（明细见 `dev-notes/tool-evals.json`）。
**未达标时按既定规则：先改 Prompt / 工具描述，不许改评估标准** ——
把标准改软只是把问题藏起来，而暴露它正是评估集存在的意义。

### 8. ★ 验收标准 3 的实测结论（与原预期不同）

原预期：「邮费是多少」关键词查表查不出来，模型应承认没查到。**实测分两层，结论不同：**

| 层 | 结果 |
|---|---|
| 数据层 | `query_faq("邮费")` **稳定**返回 `count=0` —— 基线成立（语料含「运费」不含「邮费」，两者不是子串关系） |
| 用户可见层 | 模型把用户的「邮费」**归一化成「运费」**再传参，命中「运费怎么算」，答出「满 99 包邮 / 未满 10 元」 |

所以「用户问邮费会查不出来」这条**预期前提不成立**：召回失败在
「组织工具参数」这一步被模型补偿掉了。**这不是工具或 SQL 的缺陷。**

已确认原因在**模型侧的同义归一化**，属已知问题、留待后续章节修复，本章不处理。
因此评估集 `acc3` **只断言工具选择**，不断言用户可见结果；
数据层基线由 `tests/test_db_mysql.py` 的 `faq_search("邮费") == []` **硬钉**。

### 9. 本章留痕

- 设计文档：`docs/superpowers/specs/2026-02-15-ch02-tools-design.md`
- 实施计划：`docs/superpowers/plans/2026-02-15-ch02-tools-plan.md`
- 开发过程追记：`dev-notes/ch02.md`
- 验收输出：`dev-notes/ch02-acceptance-output.txt`
- 评估明细：`dev-notes/tool-evals.json`

**明确未做（用户指定）**：多轮自动循环的 Agent Loop、向量检索 / RAG。

---

# 第三章「RAG 基础 · 向量语义检索」

> 一句话：把 `query_faq` 的内部实现从「keyword 查 `LIKE`」换成「BGE-M3 + Milvus
> 的 dense 语义检索」，**入参名与出参 JSON 一字不变**；补齐离线建库（结构感知切分 +
> 对话挖知识）与 MySQL/Milvus 双写。

## 1. 依赖与配置

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv sync
```

本机需要：MySQL（3306，复用既有库）、Milvus（19530）。**嵌入走云端**，不本地部署模型。

在 `MemHelp-code/.env` 里补三行（`EMBEDDING_API_KEY` 是硅基流动的账户 Key）：

```dotenv
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_MODEL=BAAI/bge-m3
```

其余阈值（`RAG_TOP_K` / `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP` / `QA_DEDUPE_THRESHOLD` …）
见 `.env.example`，不改也能跑。

## 2. 离线建库

```powershell
# 全量（三份 Markdown + 存量 faq 表）
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status
```

`--status` 会并排打印两侧计数，例如
`knowledge_chunks: pending=0 done=22 | milvus=22 → 一致`。

### 验收 2：故意中断再重跑

```powershell
# 只补前 5 块的向量，模拟「跑到一半挂了」
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild --max-chunks 5
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status     # pending = 总数-5
# 重跑，只补漏下的
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --status     # pending=0 且 done==milvus
```

为什么能补齐：MySQL 记 `pending`，Milvus 用**确定性主键** `str(chunk_id)` upsert ——
重写覆盖同一行，不产生重复向量。

## 3. 对话挖知识（定时任务）

> 与验收 1 的先后**不影响结论**：评估脚本的关键词基线只跑文档语料
> （排除 `content_type='mined'`），挖矿产物的问法里带的「邮费」不会污染对照。

```powershell
# 先把演示历史会话灌进去（30 个 demo-ch03-* 会话，脚本幂等）
& $uv run --directory MemHelp-code python scripts/seed_demo_history.py

# 挖一批（默认 8 个会话一批，抽取 → 暂存 → 整体去重 → 入库 → 补向量）
& $uv run --directory MemHelp-code python scripts/mine_conversations.py --limit 30
& $uv run --directory MemHelp-code python scripts/mine_conversations.py --status
& $uv run --directory MemHelp-code python scripts/mine_conversations.py --dedupe-only
```

注册成计划任务（每天 03:00）：

```powershell
schtasks /create /tn "MewHelpMineKnowledge" /sc daily /st 03:00 `
  /tr "uv run --directory MemHelp-code python scripts/mine_conversations.py --limit 200"
```

## 4. 验收 1：换说法召回（关键词 vs 向量）

```powershell
& $uv run --directory MemHelp-code python evals/run_rag_evals.py
```

输出三列对照：`kw-raw`（整句 LIKE）、`kw-noun`（ch02 时代模型截出的名词）、
`vector`（本章检索）。**为什么分两列**：ch02 的 README 记录过模型会把「邮费」归一化成
「运费」再传参，只报整句会夸大向量优势、只报名词会掩盖「关键词依赖先猜对词」。

关键词基线**只跑文档语料**（排除挖矿产物）。若验收用例的关键词列命中，脚本直接
返回 1 —— 那说明文档里混进了「邮费」这类换说法用词，对照已经失效。

## 5. 端到端

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000/chat/>，问 **「邮费是多少」**：
应出现 `🔧 检索常见问题` 徽章，回答含「满 99 元包邮 / 未满 10 元」。

> 启动时会预热（发一次真实嵌入请求 + 确认 Milvus 集合）。预热失败只告警不崩 ——
> 但检索会因此 `ok=false`，看到「暂时查不到」时先看启动日志里的 warning。

## 6. 本章留痕

- 设计文档：`docs/superpowers/specs/2026-10-17-ch03-rag-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-17-ch03-rag-plan.md`
- 开发过程追记：`dev-notes/ch03.md`
- 建表 DDL / 演示会话：`../sql/ch03-ddl.sql`、`../sql/ch03-seed-history.sql`

**本章不做（用户指定）**：关键词召回、混合检索、重排；只跑 dense 向量单路。

---

# 第四章「混合检索 + 重排 + 评估体系」

## 1. 本章新增了什么

| 能力 | 落点 |
|---|---|
| 双路召回：dense + **Milvus 原生 BM25**（`text` 字段 + 内置 chinese analyzer） | `app/knowledge/hybrid.py`、`vector_store.py` |
| RRF 融合（由 Milvus `hybrid_search` 做）+ `bge-reranker-v2-m3` 精排 Top-10 | 同上 + `app/knowledge/rerank.py` |
| 品类元数据**先过滤再检索** | `build_expr()` → 每个 `AnnSearchRequest.expr` |
| Query 理解：口语归一 + 检索侧同义词 + 型号归一 | `app/knowledge/query.py` |
| 带编号引用、可回跳章节路径、刷新后仍可点 | `app/knowledge/citations.py`、`messages.citations` |
| 低置信度显式拒答 + 入池 | `app/knowledge/{pipeline,sufficiency}.py`、`low_confidence_questions` |
| 四策略评估（Recall@K / MRR / Faithfulness，按桶） | `evals/run_ch04_evals.py` |
| 前端可点引用 + 👍/👎 一次性反馈 | `web/app.js` |

> **口径变更**：ch03 说「Milvus 不存第二份真相」，ch04 起 Milvus 是**检索索引副本**
> （`text` / `category` / `content_type` 是派生数据）。MySQL 仍是原文权威；
> 改了原文必须重建索引，否则 BM25 与过滤会按旧副本工作。

## 2. 配置（`.env` 新增）

```dotenv
RERANK_BASE_URL=https://api.siliconflow.cn/v1
RERANK_API_KEY=            # 留空则复用 EMBEDDING_API_KEY；两者都空 → 重排降级为 RRF 序
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RAG_HYBRID_TOP_K=50        # 每一路各召回多少条
RAG_CANDIDATE_K=20         # 融合后进重排的候选数
RAG_RERANK_TOP_N=10        # 最终证据条数
RAG_MIN_RERANK_SCORE=0.3   # 检索层低置信阈值（用评估集标定）
```

## 3. 重建索引（schema 升级）

ch03 的集合只有 `id + vector`，**BM25 的 `text`/`sparse` 字段只能在建集合时声明**，
所以升级必须重建集合。两条路，别选错：

```powershell
# ✅ 保留原文（含 ch03 挖矿产物 mined），只重建向量 —— 升级 schema 用这条
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --reindex

# ⚠️ 清空 knowledge_chunks 再重灌（会丢掉 mined）—— 只在确定不要旧数据时用
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild
```

新增语料单独入库（`--reindex` 不会 ingest 新文件）：

```powershell
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge/商品型号手册.md
```

## 4. 评估

```powershell
# 金标锚串能否解析（0 命中会直接报错，>3 块会告警）
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py --validate-only

# 验收 2：型号题必须被 BM25 命中（逐条打印）
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py --only B_model --strategies bm25 -v

# 验收 1：四策略对比 + 生成段（Faithfulness / 拒答准确率）
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py `
    --strategies vector,bm25,hybrid,hybrid_rerank `
    --generation hybrid,hybrid_rerank --limit 24 `
    --json evals/reports/ch04.json --markdown dev-notes/ch04-eval-report.md
```

退出码 0 = `hybrid_rerank` 整体 Recall@5 ≥ `vector`，且 `B_model` 桶 ≥ `bm25`。

**读报告前先知道两件事**：

1. 生成段用的是**评估时提示词**（与线上共享引用规则，但不是同一条链路）；
2. gold 锚串在真库里可能命中多块（同一事实被 mined 问答对重复陈述），
   Recall 因此偏宽松 —— 这是数据分布，不是标注错。

## 5. 端到端与验收 3 / 4

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000/chat/>：

- 问 **「XH-3000A 续航多久」** → 回答带 `[n]` 角标 → **点角标**，右侧抽屉显示
  章节路径与 chunk 原文；刷新页面后角标仍可点（引用存在 `messages.citations`）。
- 每条回答左下角 **👍 / 👎**：点一下点亮、显示「已反馈」、一次性锁定；刷新后保持。
- 问 **「你们支持货到付款吗」** → **明确拒答**（不编造），并落进低置信度池：

```powershell
& $uv run --directory MemHelp-code python scripts/show_low_confidence.py
# 或直接查库：
# SELECT id, source, LEFT(raw_question,30), LEFT(reason,40), created_at
#   FROM low_confidence_questions ORDER BY id DESC LIMIT 5;
```

## 6. 评估看板 `/rag-eval`

起服务后打开 <http://localhost:8000/rag-eval/>（不带斜杠会 307 到带斜杠）。
页面**只读** `evals/reports/ch04.json`，不在浏览器里重算指标；展示：

- 顶部元信息：评估集题数、知识库块数、重排模型与阈值、生成段策略、报告生成时间；
- 四张 KPI：最佳整体 MRR、口语桶 MRR 提升（混合+重排 vs 纯 BM25）、端到端答案覆盖、意外问题拒答；
- 柱状图三个口径：**MRR@10 / Recall@5 / 证据覆盖（Faithfulness）**，按四桶 + 整体分组；
- 四策略明细表、锚串警告、逐条明细（检索 / 生成）；点一行打开抽屉，能看完整答案、
  claim 是否被证据支持、引用编号与 Top-10 命中；
- 没跑的策略显示 `—` 而不是 0；报告缺失显示「先跑一次评估」的空状态，报告损坏显示错误态。

右上「重跑评估」会调后台任务（子进程），页面每 2 秒轮询状态，跑完自动刷新报告：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/rag-eval` | 归一后的看板数据；没有报告返回 404 `report_not_found` |
| POST | `/api/rag-eval/rerun` | 启动后台评估，202；已有任务在跑返回 409 |
| GET | `/api/rag-eval/run` | 重跑状态与日志尾部（供页面轮询） |

> 重跑会覆盖 `evals/reports/ch04.json` 与 `dev-notes/ch04-eval-report.md`，需要
> `.env`、MySQL / Milvus、嵌入与重排服务都可用；通常几分钟。

## 7. 本章留痕

- 设计文档：`docs/superpowers/specs/2026-10-17-ch04-hybrid-retrieval-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-17-ch04-hybrid-retrieval-plan.md`
- 开发过程追记：`dev-notes/ch04.md`
- 评估报告：`dev-notes/ch04-eval-report.md`、`evals/reports/ch04.json`
- 建表 DDL：`sql/ch04-ddl.sql`（两张新表 + `messages.citations`）

**本章不做（用户指定）**：指代消解、多轮改写；LangGraph / Langfuse；
`faith_cases` 管理页面；👎 的后端落库。

---

# 第五章「Workflow 确定性编排 + 主力 Agent」

> 一句话：把「模型自己决定一切」拆成**代码可判定的骨架**——指代消解 → 意图识别 →
> 七类意图写死分流 → 知识类强制检索+置信度闸 → 主力 Agent 在自己的节点里做 ReAct →
> 日志记录。投诉不进 Agent，转人工/建工单交用户在前端自己点。

## 1. 本章新增

| 能力 | 落点 |
|---|---|
| 手写 Agent 循环（祛魅热身） | `scripts/bare_agent_loop.py`（不进线上） |
| LangGraph 图 + State + SQLite checkpointer | `app/agent/graph.py`、`app/agent/checkpoint.py` |
| 意图识别（七类 JSON + 启发式 + 寒暄快速通道） | `app/agent/intent.py` |
| 置信度闸（检索后、Agent 前） | `app/agent/gate.py`、`RagPipeline.retrieve_only()` |
| ReAct Agent（流式、单工具步、步数/token 刹车） | `app/agent/react.py` |
| 投诉/闲聊固定出口 | `app/agent/prompts.py` |
| 建议动作控制面工具 | `app/agent/tools.py::suggest_actions` |
| 建议动作落库 + 前端双按钮 | `messages.actions`、`web/app.js` |
| 建工单端点（用户点按钮才写 tickets） | `POST /api/actions/create-ticket` |
| 每轮结构化日志 | `logs/ch05-turns.jsonl` |
| 意图标注集 + 评估 | `evals/ch05_intent_cases.jsonl`、`evals/run_ch05_intent_evals.py` |

## 2. 配置（`.env` 新增）

```dotenv
# LangGraph checkpointer（相对路径按 MemHelp-code/ 解析）
CHECKPOINT_DB_PATH=data/checkpoints.sqlite
# ReAct 最多几轮工具 / 单轮累计 token 预算
AGENT_MAX_STEPS=6
AGENT_TOKEN_BUDGET=6000
# 每轮 JSONL 日志；留空只打 logger
TURN_LOG_PATH=logs/ch05-turns.jsonl
```

`data/` 与 `logs/` 已在 `.gitignore`，不会入库。

## 3. 真库迁移（只加一列）

`messages` 表新增 `actions JSON NULL`（刷新后按钮还在靠它）；`tickets` 表新增
`idempotency_key VARCHAR(64)` + 唯一键（同一个按钮重复提交返回同一张工单）。
幂等脚本：

```powershell
& $uv run --directory MemHelp-code python scripts/apply_ch05_migration.py
```

等价的 SQL 在 `sql/ch05-ddl.sql`（带 `SET NAMES utf8mb4` 与
`information_schema` 幂等判断）。

## 4. 起服务

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8010
```

浏览器打开 <http://127.0.0.1:8010/chat/>。

启动日志里能看到 `ch05 Agent 图已编译`；checkpointer 文件默认在
`MemHelp-code/data/checkpoints.sqlite`。**测试**注入 `InMemorySaver`，不落盘。

## 5. 祛魅热身（先看裸循环）

```powershell
# 不联网，看「调模型 → 有工具调用就执行并回灌 → 没有就收敛」
& $uv run --directory MemHelp-code python scripts/bare_agent_loop.py --fake

# 真实模型（需要 .env）
& $uv run --directory MemHelp-code python scripts/bare_agent_loop.py --message "订单 1001 到哪了"
```

输出形如：

```
[step 1] -> query_order({"order_id": "1001"})
=== 最终答案 ===
订单 1001 已发货，正在派送…
工具调用：['query_order']；步数：1
```

## 6. 图与分流（写死在代码里）

| 意图 | 出口 | 行为 |
|---|---|---|
| 商品咨询 / 退款退货 | knowledge | 强制检索 → 置信度闸 → Agent；不预检索业务数据 |
| 物流 / 订单 / 售后 | business | 直接 Agent，自己调工具 |
| 投诉 | complaint | 不进 Agent；固定安抚 + 两个建议动作 |
| 闲聊 | chitchat | 固定话术；明显寒暄走零模型调用快速通道 |

七类 → 四出口的映射只有一处：`app/agent/state.py::INTENT_TO_ROUTE`。
置信度闸失败会回 `REFUSAL_TEXT` 并写入 ch04 已有的
`low_confidence_questions`（不建新飞轮表）。

## 7. 意图评估（纯 Prompt 任务的验证）

```powershell
# 离线验证聚合逻辑
& $uv run --directory MemHelp-code python evals/run_ch05_intent_evals.py --self-test
# 真实模型跑 56 条标注样例
& $uv run --directory MemHelp-code python evals/run_ch05_intent_evals.py `
    --json dev-notes/ch05-intent-evals.json `
    --markdown dev-notes/ch05-intent-eval-report.md
```

退出码 0 = 整体 ≥0.85、route ≥0.85、每类 ≥0.60。不达标先改 Prompt/示例，
**不改标签**。

## 8. 五条验收演示

先起服务（8000 被占用就用 8010）：

```powershell
& $uv run --directory MemHelp-code python scripts/demo-ch05.py `
    --base-url http://127.0.0.1:8010 `
    --out dev-notes/ch05-acceptance-output.txt
```

脚本会逐条打印 PASS/FAIL，并写入 `dev-notes/ch05-acceptance-output.txt`。
它覆盖：政策问题强制检索、订单物流工具调用、投诉两动作+建单写库、闲聊固定话术、
复杂问题 `agent_steps>=2`。

**前端按钮（必须浏览器验）**：打开聊天页问「我要投诉」，会出现两个独立按钮：

- 点「转人工」：页面显示「已转接人工客服」，并出现「您好，我是客服小猫，请问有什么
  可以帮您的？」；**不发任何后端请求**；
- 点「建工单」：才调 `POST /api/actions/create-ticket` 写 `tickets` 表；
  前端带一个 `idempotency_key`，同一个按钮重复提交（超时重试/刷新后再点）
  返回同一张工单；
- 两个按钮互不绑定；都不点就继续发消息，正常走下一轮；
- 刷新后按钮仍在（`messages.actions`），点击状态用 localStorage 一次性锁定。

本章已用 headless Edge 真实点击验证，证据：`dev-notes/ch05-acceptance-output.txt`
的「前端 CDP 验收」段 + `dev-notes/ch05-ui.png`。

## 9. 接口变化

| 方法 | 路径 | 变化 |
|---|---|---|
| POST | `/api/chat/completions` | 聊天入口切图；`done` 新增 `intent` / `route` / `actions` / `agent_steps` / `stop_reason` / `gate` |
| POST | `/api/actions/create-ticket` | 新增；用户点「建工单」才调用，走第 2 章 `create_ticket` 工具 |
| POST | `/api/chat/sessions/{id}/messages` | assistant 行新增 `actions`（仅非空时出现） |

SSE 既有事件名不变；`tools` / `citations` 事件仍在有内容时发。`done.actions` 形状：

```json
[{"action": "transfer_human", "ticket_type": "投诉"},
 {"action": "create_ticket", "ticket_type": "投诉"}]
```

## 10. 日志

每轮一行 JSONL（`TURN_LOG_PATH`），关键字段：

```json
{"session_id":"…","intent":"refund_return","route":"knowledge",
 "trace":["load_history","begin_turn","resolve_coref","classify_intent",
          "route_intent","knowledge_retrieve","confidence_gate","agent"],
 "retrieval_attempted":true,"retrieval_score":0.929,"gate_passed":true,
 "agent_steps":0,"llm_calls":1,"tools":[],"token_usage":{...},
 "suggested_actions":[],"stop_reason":"final","latency_ms":…}
```

验收 1 直接看 `trace` 是否含 `knowledge_retrieve`。日志写入失败只告警，不影响回答。

## 11. 本章留痕

- 设计文档：`docs/superpowers/specs/2026-10-18-ch05-workflow-agent-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-18-ch05-workflow-agent-plan.md`
- 开发过程追记：`dev-notes/ch05.md`
- 意图评估：`dev-notes/ch05-intent-eval-report.md`、`dev-notes/ch05-intent-evals.json`
- 验收输出：`dev-notes/ch05-acceptance-output.txt`、`dev-notes/ch05-ui.png`
- 建表迁移：`sql/ch05-ddl.sql`、`scripts/apply_ch05_migration.py`

**本章不做（用户指定）**：指代消解/意图识别的正式版、上下文管理策略升级、MCP、
飞轮入库；正式的置信度检查留给可观测章节。

---

# 第六章「正式版分流器」运行说明

## 1. 本章新增

- `resolve_coref` 从透传升级为 **LLM 指代消解 + Query 改写**：有指代/省略/口语标记才调模型；
  完整问句/寒暄原样透传；补全结果里新出现的订单号/型号必须来自原话+历史，否则回退。
- 意图识别升级为 **七类业务意图 + `other`**（八选一），强制 JSON、9 组边界 few-shot；
  默认质量档，`INTENT_COST_MODE=true` 时走「小模型先判 → 低置信度大模型重判」。
- `refund_return` / `after_sales` 统一走 **refund 确定性子流程**：
  订单槽位 → 确定性订单快照 → Query 扩写（强制 JSON `queries`）→
  多查询强制检索政策并去重合并 → 只把「这一单能不能退/售后」交给主力 Agent。
- 缺订单号时用 LangGraph `interrupt()` 暂停，前端聊天流里弹**订单卡片**；
  点选后 `POST /api/chat/refund/select` 从 checkpoint 续跑。
- 退款单是内联简单表单：原因固定六类下拉，提交复用 `tickets`（售后工单），
  不新增退款表。
- Query 扩写只在检索侧现查现用，知识库仍只有一份。

## 2. 配置（`.env` 新增，均有默认值）

```dotenv
INTENT_MODEL=
INTENT_CHEAP_MODEL=
INTENT_COST_MODE=false
INTENT_CONFIDENCE_THRESHOLD=0.6
INTENT_TIMEOUT=5
COREF_ENABLED=true
COREF_TIMEOUT=5
COREF_HISTORY_MESSAGES=8
QUERY_EXPANSION_ENABLED=true
QUERY_EXPANSION_TIMEOUT=5
QUERY_EXPANSION_MAX_QUERIES=4
REFUND_ORDER_COUNT=3
```

- `INTENT_MODEL` 留空 = 用 `OPENAI_MODEL`；要更强模型只改这一项。
- 成本档必须同时设 `INTENT_COST_MODE=true` 和 `INTENT_CHEAP_MODEL`，否则不生效。

## 3. 测试与评估

```powershell
# 离线测试（不连网、不连 MySQL/Milvus）
& $uv run --directory MemHelp-code python -m pytest -q --color=no

# 多轮指代/意图/扩写评估（真实模型；输出报告）
& $uv run --directory MemHelp-code python evals/run_ch06_evals.py
```

报告：`dev-notes/ch06-eval-report.md` / `dev-notes/ch06-evals.json`。
当前模型 `deepseek-flash` 的实测（22 轮）：意图 1.000、指代 1.000、other 3/3、
JSON 失败 0、扩写 8/8。

## 4. 功能演示

先起服务（真实 `.env`；MySQL/Milvus 可用）：

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
```

```powershell
# 四条验收一次跑完，输出到文件
& $uv run --directory MemHelp-code python scripts/demo-ch06.py `
    --case all --base-url http://127.0.0.1:8000 `
    --out dev-notes/ch06-acceptance-output.txt

# 单独跑
& $uv run --directory MemHelp-code python scripts/demo-ch06.py --case multiturn
& $uv run --directory MemHelp-code python scripts/demo-ch06.py --case other
& $uv run --directory MemHelp-code python scripts/demo-ch06.py --case refund
& $uv run --directory MemHelp-code python scripts/demo-ch06.py --case picker
```

## 5. 浏览器验收（订单卡片 / 退款单）

前置：Chrome/Edge 以 `--remote-debugging-port=9222` 启动；服务在 8000。

```powershell
& $uv run --directory MemHelp-code python scripts/ch06_ui_check.py
```

脚本会：
1. 多轮问「订单 20240115001 的物流到哪了」→「那它能退吗」，断言出现
   **「已补全：订单 20240115001 能退吗」**；
2. 新会话问「这个能退吗」，断言订单卡片出现；
3. 点第一张卡，断言退款单出现；
4. 选「商品质量问题」提交，断言显示工单号。

截图：`dev-notes/ch06-coref.png`、`dev-notes/ch06-order-picker.png`、
`dev-notes/ch06-refund-form.png`。

## 6. 接口变化

| 方法 | 路径 | 变化 |
|---|---|---|
| POST | `/api/chat/completions` | SSE 新增 `coref` / `intent` / `order_picker` / `awaiting_input`；`done` 新增 `resolved_message` / `coref_changed` / `query_expansion` / `refund_form` / `trace` |
| POST | `/api/chat/refund/select` | 新增；订单点选后从 checkpoint 续跑，无 pending 返回 409 |
| POST | `/api/actions/create-refund` | 新增；固定原因提交退款申请，复用 `create_ticket` 写 `tickets` |

暂停轮只发 `awaiting_input`，**不发 `done`**；续跑完成后才落 MySQL。

## 7. 本章留痕

- 设计文档：`docs/superpowers/specs/2026-10-18-ch06-intent-router-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-18-ch06-intent-router-plan.md`
- 开发过程追记：`dev-notes/ch06.md`
- 评估报告：`dev-notes/ch06-eval-report.md`、`dev-notes/ch06-evals.json`
- 验收输出：`dev-notes/ch06-acceptance-output.txt`
- 截图：`dev-notes/ch06-coref.png`、`dev-notes/ch06-order-picker.png`、
  `dev-notes/ch06-refund-form.png`

**本章不做（用户指定）**：微调小模型/BERT 做意图分类、跨会话记忆；真实订单表/退款单表；
入库侧多份拆存；刷新页面后恢复未完成的订单选择。


