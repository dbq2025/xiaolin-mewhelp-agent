# MewHelp

小林coding《AI 智能客服实战》实战项目 —— 电商智能客服。

技术栈：Python + FastAPI + LangChain。

## 环境备注（本机）

- 运行时长命令统一用 `uv`（当前不在 PATH 的话用 `C:\Users\dbq\.local\bin\uv.exe`）
- 本机没有 `make`，课程里的 Makefile 目标需手工拆成对应命令执行
- 工作区根目录已 `git init`（分支 `main`），后续每章用 `git worktree` 开隔离区
- **沙箱下必须设 `UV_CACHE_DIR`**：全局 `C:\Users\dbq\AppData\Local\uv\cache` 无写权限，
  缓存要指到工作区内，否则 `uv` 报 `failed to open file ... 拒绝访问 (os error 5)`

## 第一章「纯对话」运行说明

### 1. 装依赖

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv sync
```

### 2. 配 `.env`

复制 `.env.example` 为 `.env` 并填真实值（`.env` 已在 `.gitignore`）：

```dotenv
OPENAI_BASE_URL=https://your-gateway.example.com/v1
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_MODEL=deepseek-chat
CONTEXT_TOKEN_BUDGET=2000
REQUEST_TIMEOUT=60
```

一个 OpenAI 兼容端点就够了，换 GPT / Claude / DeepSeek / Ollama 只改 `OPENAI_MODEL`（必要时改 `OPENAI_BASE_URL`），**代码不用动**。

### 3. 起服务

```powershell
& $uv run uvicorn app.main:app --reload --port 8000
```

### 4. 跑离线测试（不需要 `.env`，不发任何网络请求）

```powershell
& $uv run python -m pytest -v --color=no
```

### 5. 跑评估集（需要真实 `.env`）

```powershell
& $uv run python evals/run_evals.py
```

### 6. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/completions` | 对话，`stream=true` 走 SSE 逐 token，`false` 返回完整 JSON |
| POST | `/api/chat/sessions/{session_id}/messages` | 查看该会话已存历史 |
| POST | `/api/extract` | 售后描述 → 结构化 JSON |
| GET | `/healthz` | 健康检查（不回显密钥） |

### 7. 演示命令（三条验收标准）

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

### 8. 开发留痕

- 设计文档：`docs/superpowers/specs/2026-02-14-ch01-pure-chat-design.md`
- 实施计划：`docs/superpowers/plans/2026-02-14-ch01-pure-chat-plan.md`
- 开发过程追记：`dev-notes/ch01.md`
- 客服 Prompt 行为约束检查清单：`evals/service_prompt_cases.md`
