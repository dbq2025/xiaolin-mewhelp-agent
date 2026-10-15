# 目录结构与版本控制说明

> 本文件说明**代码放在哪**、**Git 仓库边界在哪**、**日常怎么用**。
> 硬性规则（agent 必须遵守的）在 [`agent.md`](agent.md)。

---

## 1. 目录结构

```
F:\DeepSeek-Harness\memhelp-code-myself\        ← 工作区根 = Git 仓库根
│
├─ .git\                    Git 仓库（唯一，含 ch01 全部提交历史）
│
├─ agent.md                 ← 工作区规则：代码位置、提交约定、沙箱注意事项
├─ pytest.ini               ← 从工作区根跑 pytest 时的配置
├─ .gitignore               ← 工作区根级忽略规则（工具目录、沙箱产物）
│
├─ .venv\                   Python 虚拟环境（uv 管理）
├─ .uv-cache\               uv 缓存
├─ .idea\                   编辑器配置
│
└─ MemHelp-code\            ← ★ 全部项目代码都在这里
   │
   ├─ app\                  应用代码（10 个模块）
   │  ├─ api.py               HTTP 层：校验 / 错误码 / SSE 编码
   │  ├─ chat.py              对话用例编排（流式 + 非流式 + 历史落库）
   │  ├─ prompts.py           Prompt 模板（客服 System Prompt + 提取 Prompt）
   │  ├─ context.py           token 预算裁剪（唯一知道预算的地方）
   │  ├─ memory.py            会话存储（唯一知道历史存哪的地方，纯 Python）
   │  ├─ extract.py           结构化提取（with_structured_output）
   │  ├─ errors.py            上游错误分类 + 脱敏（唯一判断故障种类的地方）
   │  ├─ llm.py               模型工厂（唯一知道 base_url/key/model 的地方）
   │  ├─ config.py            .env 单一来源（绝对路径解析）
   │  ├─ sse.py               SSE 帧编码
   │  ├─ schemas.py           请求/响应/结构化输出的数据结构
   │  └─ main.py              应用入口
   │
   ├─ tests\                离线测试（145 条，不发网络请求）
   ├─ evals\                评估集与脚本（12 条标注样例 + 7 条行为约束清单）
   ├─ docs\superpowers\     设计文档与实施计划
   ├─ dev-notes\            分阶段开发留痕、交付总结、验收原始输出
   ├─ scripts\              演示脚本
   │
   ├─ pyproject.toml        依赖与项目配置（含 pytest 配置）
   ├─ uv.lock               依赖锁定
   ├─ .env.example          配置模板（入库）
   ├─ .env                  真实配置（**绝不入库**）
   ├─ .gitignore            代码目录级忽略规则
   └─ README.md             运行说明
```

### 为什么 `.venv` / `.uv-cache` 不在 `MemHelp-code/` 下

- `.venv` 是 `uv` 管理的，`uv` 会**向上查找**项目；放在工作区根可以同时服务当前与将来的章节。
- `.uv-cache` 必须在工作区内（沙箱限制，见 `agent.md` 第 4 节）。
- 两者都被忽略，不影响代码集中管理。

> 注意：用 `uv run --directory MemHelp-code ...` 时，`uv` 可能在 `MemHelp-code/`
> 下**另建**一份 `.venv` / `.uv-cache`。同样已在忽略范围内。

---

## 2. 版本控制

### 仓库边界

**仓库根 = 工作区根**（`.git` 在 `MemHelp-code/` 的上一级）。

这是刻意的选择，而不是遗漏：

| 方案 | 结论 |
|---|---|
| 仓库在工作区根，代码在 `MemHelp-code/` | ✅ **采用**。一个 `.gitignore` 管住工具目录；无嵌套双仓库；保留 ch01 完整历史 |
| 在 `MemHelp-code/` 里再建一个仓库 | ❌ 会形成嵌套双仓库，容易提交到错的地方 |

### 当前状态

| 项 | 值 |
|---|---|
| 分支 | `feat/ch01-pure-chat` |
| 远端 | `origin` → `https://github.com/dbq2025/xiaolin-mewhelp-agent.git` |
| 远端内容 | 只有初始提交 `e06e271`（**ch01 的提交都还没推**） |
| 被跟踪文件 | 50 个：`MemHelp-code/**` 47 个 + `agent.md` + `pytest.ini` + 根 `.gitignore` |
| 未跟踪（必须保持） | `MemHelp-code/.env`（含真实密钥）、`.venv/`、`.uv-cache/`、`.idea/` |

### 日常命令（在工作区根执行）

```powershell
git status                        # 查看状态（应无告警）
git log --oneline -10             # 查看历史
git diff MemHelp-code/            # 查看改动

git add MemHelp-code/app/chat.py                       # 精确添加
git add MemHelp-code/                                   # 添加整个代码目录
git commit -m "ch02: 加了什么"
```

**不要用 `git add -A` 图省事** —— 它会把工作区根的新东西一起卷进来（本次就误提交过 `.idea`）。

### 提交前自检（两条）

```powershell
# 1) 密钥没进暂存区（.env.example 除外）
git diff --cached --name-only | Select-String "\.env$"

# 2) 工具目录没进暂存区
git diff --cached --name-only | Select-String "^\.venv/|^\.uv-cache/|^\.idea/"
```

两条都应**无输出**。`agent.md` 第 2 节也记了这两条。

### 忽略规则的分工

| 文件 | 负责范围 |
|---|---|
| 工作区根 `.gitignore` | `.venv/` `.uv-cache/` `.idea/` `pytest-cache-files-*/` 等**根级**内容 |
| `MemHelp-code/.gitignore` | 代码目录**内部**：`.env`、`__pycache__/`、`.pytest_cache/`、`pytest-cache-files-*/` 等 |

**踩过的坑**：`.gitignore` 原在根，搬迁时随代码移入 `MemHelp-code/`，于是它的规则
**管不到工作区根**了——`.idea/` 在空档被 `git add` 卷了进来（已撤回）。所以现在
**两处 `.gitignore` 都要有**，各自管自己那一层。

---

## 3. 运行方式

`app` 包在 `MemHelp-code/` 下，所以从工作区根直接 `python -c "import app"` 或
`uvicorn app.main:app` 会 `ModuleNotFoundError`。三种可用写法：

```powershell
# 一次性设置
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"

# ── 方式 1（推荐）：进代码目录，配置与路径都唯一
cd MemHelp-code
& $uv run python -m pytest -q --color=no
& $uv run uvicorn app.main:app --port 8000
& $uv run python evals/run_evals.py

# ── 方式 2：留在工作区根，用 --directory
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8000
& $uv run --directory MemHelp-code python evals/run_evals.py

# ── 方式 3：只有 pytest 支持从根直接跑
& $uv run python -m pytest -q --color=no
```

### 测试配置有两份，改一处必须改另一处

| 位置 | 何时生效 |
|---|---|
| `MemHelp-code/pyproject.toml` 的 `[tool.pytest.ini_options]` | 从 `MemHelp-code/` 或其子目录运行 pytest |
| `pytest.ini`（工作区根） | 从工作区根运行 pytest |

**踩过的坑**：从仓库根跑 pytest 时，pytest 把 rootdir 定在仓库根，**读不到**
`MemHelp-code/pyproject.toml` 的配置，`asyncio_mode` 退化为 `STRICT`，异步测试整片
失败（实测 25 failed）。所以加了根级 `pytest.ini`。两处必须同步。

---

## 4. 搬迁后的验证结果

| 检查项 | 结果 |
|---|---|
| 从工作区根跑测试 | **145 passed** |
| 从 `MemHelp-code/` 跑测试 | **145 passed** |
| 评估脚本离线自测 | 通过（4 种场景正确） |
| 从根用 `--directory` 起服务 | 通过，`/healthz` 报 `ok`，能读到 `MemHelp-code/.env` |
| 真实上游对话 | 通过（客服人设、上下文、结构化输出都正常） |
| 全部 48 个文本文件编码 | UTF-8 干净、无异常控制字符、无乱码特征 |
| `git status` | 干净、无告警 |
