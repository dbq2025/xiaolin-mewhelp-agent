# 第一章「纯对话」交付总结

> 自动生成的交付总结。**所有数字都来自实测**，命令与原始输出附在文末路径里。

## 1. 交付物

| 项 | 位置 |
|---|---|
| 源码 | `app/`（10 个模块） |
| 离线测试 | `tests/`（14 个文件，143 条） |
| 评估集与脚本 | `evals/`（12 条标注样例 + 7 条行为约束清单 + `run_evals.py`） |
| 演示脚本 | `scripts/demo.ps1` |
| 设计文档 | `docs/superpowers/specs/2026-02-14-ch01-pure-chat-design.md` |
| 实施计划 | `docs/superpowers/plans/2026-02-14-ch01-pure-chat-plan.md` |
| 开发留痕 | `dev-notes/ch01.md` |
| 验收原始输出 | `dev-notes/acceptance-output.txt` |

## 2. 功能演示命令

> 代码在 `MemHelp-code/`，`app` 包也在那里，所以除 pytest 外的命令都要用
> `--directory MemHelp-code`（或先 `cd` 进去），否则 `ModuleNotFoundError: app`。
> 详见工作区根的 `agent.md`。

```powershell
# 0) 装依赖（本机 uv 不在 PATH）
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv sync

# 1) 配 .env（参考 MemHelp-code/.env.example；.env 不入库）
#    OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL

# 2) 起服务（从工作区根，用 --directory）
& $uv run --directory MemHelp-code uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3) 另一个终端：一键跑完三条验收并存档
cd MemHelp-code
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 -Out dev-notes/acceptance-output.txt
```

`-ExecutionPolicy Bypass` 是必需的（本机 `CurrentUser` 策略为 `RemoteSigned`）。

## 3. 测试结果

### 离线单元测试（不需要 `.env`，不触网）

```
uv run python -m pytest --color=no
145 passed
```

**「不触网」的证据**：把上游指向黑洞 `http://127.0.0.1:1/v1`（环境变量优先级高于
`.env`）后仍是 `143 passed` —— 测试确实没有真实网络调用。

### 评估集（真实模型 `deepseek-v4-pro`）

```
总体字段准确率：91.7%（33/36）        订单号编造次数：0        → 达标，退出码 0
  order_no           12/12  100.0%
  request_type        9/12   75.0%
  expected_solution  12/12  100.0%
  detail(关键词)      12/12  100.0%
```

三条失败全在 `request_type`，是**标注口径歧义**（用户口头说「要退款」时判 `refund`
完全合理，我原本标的是 `logistics`），非模型错误。已记入留痕，下一章应把
`RequestType`（诉求类型）与 `ExpectedSolution`（期望方案）的边界写进 Prompt。

### 三条验收标准（真实上游）

| # | 验收标准 | 实测结果 |
|---|---|---|
| 1 | curl 调对话接口能看到流式回复 | ✅ `event: token` 逐块滚出，总耗时 5.4s |
| 2 | 连续问两轮，第二轮能接住第一轮的上下文 | ✅ 第二轮答「您刚才说的订单号是 **20240115001**」；`kept_messages` 0→2 |
| 3 | 发一段售后描述，能拿到结构化 JSON | ✅ 3a `order_no=20240115001`；3b 未提订单号 → **`order_no=null`**（零编造） |

## 4. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/completions` | 对话。`stream=true` → SSE（`meta`/`token`/`done`/`error`）；`false` → 完整 JSON |
| POST | `/api/chat/sessions/{session_id}/messages` | 查看该会话已存历史 |
| POST | `/api/extract` | 售后描述 → 结构化 JSON |
| GET | `/healthz` | 健康检查（不回显密钥；配置缺失时返回 503） |

## 5. 架构要点

```
api.py       HTTP 语义：校验 / 错误码 / SSE 编码
  ├─ chat.py        对话用例编排（流式 + 非流式 + 历史落库）
  │    ├─ prompts.py    Prompt 模板（客服 System Prompt + 7 条行为约束 + 提取 Prompt）
  │    ├─ context.py    token 预算裁剪（唯一知道预算的地方）
  │    └─ memory.py     会话存储（唯一知道历史存哪的地方，纯 Python）
  ├─ extract.py     结构化提取（with_structured_output）
  └─ errors.py      上游错误分类 + 脱敏（唯一判断故障种类的地方）
llm.py       模型工厂（唯一知道 base_url/key/model 的地方）
config.py    .env 单一来源
sse.py       SSE 帧编码
```

三条隔离约束，各有测试钉住：
1. 只有 `app/llm.py` 导入 `langchain_openai`（业务代码只用 `langchain_core` 抽象）
2. 只有 `app/config.py` 读配置
3. 只有 `app/errors.py` 判断上游故障种类，且所有回给客户端的错误文本都过脱敏

## 6. 已知限制与顺延项

- **`.env` 里的模型名**：用户初填的 `deepseek-v4.1-flash` 上游不存在（400 回了合法值
  清单），已改为 `deepseek-v4-pro`。
- **网关不支持 `tool_choice=<函数名>`**：`app/llm.py` 的 `GatewayChatOpenAI` 把它改写成
  `"auto"`；`with_structured_output` 这个 API 本身保留。
- **顺延到下一章**（评审 Minor，已记入 `dev-notes/ch01.md`）：
  - `schemas.py::SessionId` 是死代码
  - `memory.py` 无上限/TTL、同会话并发无序列化（规格已排除，属已知取舍）
  - 422 未带原始模型输出（需要 `include_raw=True` 并重构返回契约）
  - 评估集 `c12` 只覆盖一种订单号格式
- **未做**（本章范围外）：工具调用、Agent 循环、LangGraph、RAG、记忆持久化、前端页面、鉴权与限流。

## 7. 原始输出留档

- 评估运行记录：`.superpowers/sdd/2026-02-14-ch01-pure-chat-plan/eval-run-*.txt`
- 验收演示原始输出：`dev-notes/acceptance-output.txt`
- 逐阶段开发留痕（含所有 Ruling 与翻车记录）：`dev-notes/ch01.md`

---

# 第四章「混合检索 + 重排 + 评估体系」交付总结

> 数字全部来自本机实测（Milvus 2.6.22 / pymilvus 2.6.17 / 硅基流动嵌入与重排 / 真实 chat 模型）。

## 1. 交付物

| 项 | 位置 |
|---|---|
| 设计 | `docs/superpowers/specs/2026-10-17-ch04-hybrid-retrieval-design.md` |
| 计划 | `docs/superpowers/plans/2026-10-17-ch04-hybrid-retrieval-plan.md` |
| 留痕 | `dev-notes/ch04.md`（逐阶段，含 Ruling 与翻车） |
| 评估报告 | `dev-notes/ch04-eval-report.md` + `evals/reports/ch04.json` |
| 检索 | `app/knowledge/{query,hybrid,rerank,citations,sufficiency,pipeline}.py` |
| 数据 | `sql/ch04-ddl.sql`、`app/db/{models,repositories}.py` |
| 评估 | `evals/ch04_cases.jsonl`（48 条）、`evals/run_ch04_evals.py` |
| 语料 | `knowledge/商品型号手册.md`（新增） |
| 前端 | `web/{app.js,style.css}`（引用抽屉 + 👍/👎）、`web/rag-eval/`（评估看板） |
| 评估接口 | `app/rag_eval.py`；`GET /api/rag-eval`、`POST /api/rag-eval/rerun`、`GET /api/rag-eval/run` |

## 2. 演示命令

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"; $uv = "C:\Users\dbq\.local\bin\uv.exe"

# 0) schema 升级：drop 集合后**保留原文**重建向量（58 条，含 30 条 mined）
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --reindex
#    新语料单独入库（--reindex 不 ingest 新文件）
& $uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge/商品型号手册.md

# 1) 金标锚串自检（0 命中即报错）
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py --validate-only

# 2) 验收 2：型号题 BM25 命中（逐条打印）
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py --only B_model --strategies bm25 -v

# 3) 验收 1：四策略对比 + 生成段
& $uv run --directory MemHelp-code python evals/run_ch04_evals.py `
    --strategies vector,bm25,hybrid,hybrid_rerank `
    --generation hybrid,hybrid_rerank --limit 4 `
    --json evals/reports/ch04.json --markdown dev-notes/ch04-eval-report.md

# 4) 端到端
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
#   问「XH-3000A 的续航多久」→ 点角标看原文；问「你们支持货到付款吗」→ 拒答
& $uv run --directory MemHelp-code python scripts/show_low_confidence.py

# 5) 评估看板：四策略对照 / 证据覆盖 / 逐条明细，页面可重跑
#   浏览器打开 http://localhost:8000/rag-eval/
```

## 3. 测试结果

```
uv run --directory MemHelp-code python -m pytest -q
694 passed, 39 skipped
```

真机集成（默认 skip，需显式开启）：

```
$env:MEMHELP_TEST_MILVUS=1  → tests/test_knowledge_milvus.py 5 passed
$env:MEMHELP_TEST_MYSQL=1   → tests/test_db_mysql.py 33 passed
```

## 4. 评估数字（48 条 / 四桶 × 12）

检索段（Recall@1 / Recall@5 / MRR@10，overall；`evals/reports/ch04.json` 为准）：

| 策略 | Recall@1 | Recall@5 | MRR@10 | B_model Recall@5 |
|---|---|---|---|---|
| vector | 0.900 | 1.000 | 0.942 | 1.000 |
| bm25 | 0.825 | 0.925 | 0.880 | 1.000 |
| hybrid | 0.875 | 1.000 | 0.926 | 1.000 |
| **hybrid_rerank** | **0.900** | **1.000** | **0.950** | **1.000** |

生成段（每桶 3 条可答 + 1 条拒答，共 16 条/策略）：

| 策略 | Faithfulness | 拒答准确率 | 引用有效率 |
|---|---|---|---|
| hybrid | 0.972 | 1.000 | 1.000 |
| hybrid_rerank | 0.937 | 1.000 | 1.000 |

> 读报告要注意三件事：① 生成段用**评估时提示词**（与线上共享引用规则，但不是同一条链路），
> 且每桶只抽 4 条，`hybrid` 与 `hybrid_rerank` 的 Faithfulness 差 0.035 **是抽样噪声**，
> 不能据此说「重排让忠实度变差」；② gold 锚串在真库里常命中多块（同一事实被 mined
> 问答对重复陈述），Recall@5 因此饱和，区分度看 **MRR@10 与 Recall@1**；
> ③ 重排降级计数 `rerank_degraded` 全为 0（本次运行重排服务正常）。

## 5. 四条验收标准（实测）

| # | 标准 | 实测 |
|---|---|---|
| 1 | 四策略对比报告能跑出数字 | `dev-notes/ch04-eval-report.md` 三张表 + `evals/reports/ch04.json`；脚本退出码 **0** |
| 2 | 带具体型号的问题，BM25 能命中 | `--only B_model --strategies bm25` → **Recall@5 1.000**（该命令的退出码现在独立校验 ≥0.8）；小写 `xh-3000a`、带空格 `XH 3000B`、紧贴中文 `xh3000a续航` 三种写法都归一后命中 |
| 3 | 引用编号能定位回原文，页面可点 | 回答含 `[1]…[4]`；`messages.citations` 存快照（例：`chunk_id=317`，`section_path=商品型号手册 / 巡检仪 / XH-3000A 与 XH-3000B 的区别`）；前端点角标开抽屉 |
| 4 | 问库里没有的内容 → 明确拒答 + 入池 | 「你们支持货到付款吗」→ 拒答话术；`low_confidence_questions` 新增一行 `source=retrieval_low_conf`、`reason=最高重排分 0.036 低于阈值 0.3` |

## 6. 已知限制

- 「模型根本没调 `query_faq`」这种情况后端拦不住（spec §9.6 已写明），只能靠 Prompt 约束
  与评估集的「未检索就答率」观察。
- 重排服务不可用时降级为 RRF 序（`rerank_degraded=true` 进报告），此时检索层阈值闸不生效。
- `faith_cases` 只提供最小查询/处置端点，没有管理页面。
- 👎 只做纯前端采集（localStorage），不上报后端。
