# Agent 工作区规则（MemHelp）

本文件是本工作区的**长期约定**，所有在本目录工作的 agent / 开发者都必须遵守。

---

## 1. 代码位置（唯一约定）

**MemHelp 项目全部开发过程中产生的所有代码，一律写入 `MemHelp-code/`。**

```
MemHelp-code/
├─ app/            # 应用代码
├─ web/            # 聊天页面（静态资源，无构建步骤、无 npm 依赖，由 StaticFiles 托管）
├─ tests/          # 测试
├─ evals/          # 评估集与脚本
├─ docs/           # 设计文档、实施计划（docs/superpowers/**）
├─ dev-notes/      # 分阶段开发留痕
├─ scripts/        # 演示/运维脚本
├─ pyproject.toml  # 依赖与项目配置
├─ uv.lock         # 依赖锁定
├─ .env.example    # 配置模板（入库）
├─ .env            # 真实配置（**绝不入库**）
├─ .gitignore
└─ README.md       # 运行说明
```

### 硬性规则

1. **新代码只放 `MemHelp-code/` 下**，不要在仓库根新建 `app/`、`src/`、`tests/` 等。
2. **后续章节（ch02、ch03…）也在这里**。建议按章建子目录或按章加提交，但不要另起仓库、不要移到工作区根。
3. **`docs/` 与 `dev-notes/` 是交付物的一部分**，不是临时垃圾 —— 设计决策与留痕要跟着代码走。
4. **不要把 `.venv` / `.uv-cache` 搬进 `MemHelp-code/`**（见第 4 节）。
5. 如果新增顶层目录（例如 `migrations/`），在本文档更新目录树，别让它只存在于代码里。

---

## 2. 版本控制

**仓库根 = 工作区根 `F:\DeepSeek-Harness\memhelp-code-myself`（`.git` 在这一层）。**

### 为什么仓库不放在 `MemHelp-code/` 里面

这是刻意的选择，理由：

- 工作区里除了代码还有工具目录（`.venv`、`.uv-cache`、`.idea`），仓库放在根上能用一个 `.gitignore` 统一排除它们；
- 如果 `MemHelp-code/` 里再建一个仓库，就会形成**嵌套双仓库**，容易提交到错的地方；
- 保留完整提交历史（ch01 的 28 个提交），不做历史重写。

### 当前状态

- 分支：`feat/ch01-pure-chat`
- 远端：`origin` → `https://github.com/dbq2025/xiaolin-mewhelp-agent.git`（远端只有初始提交 `e06e271`；**ch01 的全部提交都还没推**）
- 未跟踪（且必须保持不跟踪）：`MemHelp-code/.env`（含真实密钥）

### 常用命令（**在工作区根执行**）

```powershell
git status
git log --oneline -10
git add MemHelp-code/          # 只加代码目录，别用 git add -A 图省事
git commit -m "ch01: ..."
```

### 提交前必须确认

```powershell
# 1) 密钥没进暂存区
git diff --cached --name-only | Select-String "\.env$"     # 应无输出（.env.example 除外）

# 2) 没有把工具目录提交进去
git diff --cached --name-only | Select-String "\.venv|\.uv-cache|\.idea"   # 应无输出
```

### `.gitignore` 的分工

| 文件 | 作用 |
|---|---|
| 根 `.gitignore`（现在在 `MemHelp-code/.gitignore`） | 排除 `.env`、`.venv`、`__pycache__`、`.uv-cache`、`pytest-cache-files-*/` 等 |
| `/pytest.ini`（仓库根） | 从根跑 pytest 时的配置（见第 3 节） |

---

## 3. 运行方式

依赖与虚拟环境**仍在工作区根**（`.venv/`），代码在 `MemHelp-code/`。`uv` 会自动在 `MemHelp-code/` 下找到 `pyproject.toml`。

**Python 导入路径的坑**：`app` 包在 `MemHelp-code/` 下，所以从工作区根直接
`uv run python -c "import app"` 或 `uvicorn app.main:app` 会 `ModuleNotFoundError`。
三种可用写法（选一种，别混）：

```powershell
# 一次性设置
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"      # uv 不在 PATH

# 装依赖
& $uv sync

# ── 方式 1（推荐）：进代码目录，配置与路径都唯一
cd MemHelp-code
& $uv run python -m pytest -q --color=no
& $uv run uvicorn app.main:app --port 8000
& $uv run python evals/run_evals.py

# ── 方式 2：留在工作区根，用 --directory 让 uv 在代码目录里执行
& $uv run --directory MemHelp-code python -m pytest -q --color=no
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
& $uv run --directory MemHelp-code python evals/run_evals.py

# ── 方式 3：只有 pytest 支持从根直接跑（根级 pytest.ini 配了 pythonpath/testpaths）
& $uv run python -m pytest -q --color=no
```

### 测试配置有两份，改一处必须改另一处

| 位置 | 何时生效 |
|---|---|
| `MemHelp-code/pyproject.toml` 的 `[tool.pytest.ini_options]` | 从 `MemHelp-code/` 或其子目录运行 pytest |
| `/pytest.ini`（仓库根） | 从工作区根运行 pytest |

**踩过的坑**：从仓库根跑 pytest 时，pytest 把 rootdir 定在仓库根，**读不到** `MemHelp-code/pyproject.toml` 里的配置，`asyncio_mode` 退化成 `STRICT`，异步测试整片失败（实测 25 failed）。所以加了根级 `pytest.ini`。两处必须保持一致。

---

## 4. 沙箱注意事项（DSH 环境特有）

1. **`UV_CACHE_DIR` 必须设到工作区内**。全局 uv 缓存（`C:\Users\dbq\AppData\Local\uv\cache`）无写权限，否则报 `os error 5`。
2. **测试进程新建的目录，会在退出后被收回权限**，之后连列举和删除都被拒绝（`PermissionError: [WinError 5]`），而且**连 ACL 诊断工具都判定为非权限问题**。表现：工作区根偶尔出现删不掉的 `pytest-cache-files-*/`。
   - 应对：测试不依赖 `tmp_path` / `tempfile.TemporaryDirectory`，需要临时文件就写进已存在且可写的目录；pytest 缓存用 `-p no:cacheprovider` 关掉。
   - 清理：需要把会话切到**完全权限**后 `Remove-Item -Recurse -Force` 才删得掉。
3. **宿主 shell 的 HTTPS 会失败**（`schannel: SEC_E_NO_CREDENTIALS`），但 agent 进程自己的 TLS 正常（`uv` 能下载依赖）。打本机服务的 `curl` 不受影响。
4. **PowerShell 5.1 按 GBK 读 `.ps1`**：脚本里有中文就必须存成 **UTF-8 with BOM**，否则乱码会撑破字符串字面量。
5. **执行策略是 `RemoteSigned`**：直接 `& .\scripts\demo.ps1` 会被拒，要用
   `powershell -ExecutionPolicy Bypass -File scripts/demo.ps1`。
6. **改带中文的源文件用编辑工具，不要用 `Set-Content` 批量替换** —— 会把编码写坏。

---

## 5. 项目约定（沿用第一章）

- **上游单一开口**：只有 `app/llm.py` 可以 `import langchain_openai`（约束作用于 `app/` 包；`tests/test_smoke.py` 的冒烟检查是刻意例外）。
- **配置单一来源**：只有 `app/config.py` 读配置。`.env` 路径用**绝对路径**（`app/config.py::ENV_FILE`），不依赖当前工作目录。
- **错误分类单一来源**：只有 `app/errors.py` 判断上游故障种类，且**所有回给客户端的错误文本都必须过 `redact()`**（脱敏），响应体一律 `{"error": {"type", "message"}}` 外壳。
- **不落库半截回答**：上游出错时只保留用户那一轮，避免污染下一轮上下文。
- **每条约束都要有测试钉住**，包括「只有 llm.py 导入 langchain_openai」「密钥不入响应体」这类跨文件约束。

---

## 6. 留痕要求

每个阶段（设计定稿、计划评审、每个任务完成、code review、收尾）都要往 `MemHelp-code/dev-notes/chXX.md` 追记，**不许收尾时一次性补记**。每段记四样：

1. 关键原话（用户这一步说了什么）
2. 关键产出（spec / plan 路径、评审结论）
3. 被拒绝或纠偏的内容
4. 翻车与返工（含根因与修法）
