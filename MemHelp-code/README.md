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
