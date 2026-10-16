# MewHelp 第二章「Function Calling 工具链」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有客服聊天链路装上五个 Function Calling 工具，模型自己决定调哪个，单轮收敛，工具轨迹落库并在聊天页显示徽章。

**Architecture:** 新增 `app/db/`（SQLAlchemy 2.0 声明式 + 仓库层，SQL 只出现在这里）与 `app/tools/`（五个 `@tool` + 注册表 + 执行器）。`app/chat_tools.py` 做单轮编排：**第 1 次模型调用非流式**只为拿到确定的 `tool_calls` → 执行 `tool_calls[0]` → 回灌 `ToolMessage` → **第 2 次调用流式**吐 token。代码里没有任何循环结构，所以「只调一次」是结构性保证而非运行时刹车。ch01 的内存会话被 MySQL 取代。

**Tech Stack:** FastAPI 0.143.1 / SQLAlchemy 2.0 + PyMySQL / MySQL 8（Docker）/ LangChain 1.4.6 `@tool` / pytest 9.1.1

**Spec:** `MemHelp-code/docs/superpowers/specs/2026-02-15-ch02-tools-design.md`

## Global Constraints

- Python `>=3.12`；依赖用项目自己的 `uv` 环境，**不蹭全局 site-packages**。
- 依赖新增：`sqlalchemy>=2.0`、`pymysql>=1.1`；加完必须 `uv sync` 并提交 `uv.lock`。
- **`app/llm.py` 是全项目唯一可以 `import langchain_openai` 的模块**（约束范围 `app/` 包；`tests/test_smoke.py` 是刻意例外，`tests/test_llm.py` 只扫 `app/`）。
- **只有 `app/config.py` 可以读 `.env` / `os.environ`**。数据库连接串只能由 `Settings.database_url` 派生。
- **所有回给客户端的错误文本必须过 `app/errors.py::redact()`**。工具的失败原因同样要过。
- 响应体一律 `{"error": {"type", "message"}}` 外壳；错误状态码沿用 ch01：配置问题 503、其余上游问题 502。
- **只有 `app/db/repositories.py` 可以写 SQL / 用 SQLAlchemy 查询 API**（`app/db/models.py` 只定义表）。
- `sql/ch02-ddl.sql` 是**建表唯一真相**（compose 挂载执行）。**不引入 Alembic**；ORM 映射必须逐列对齐 DDL，由真实 MySQL 集成测试钉住。
- `tools/business.py` 三个工具**不得 import `app.db`**（保持脱离数据库可测）。
- 随机数据必须**按入参派生**（`random.Random(f"{tool}:{id}")`），保证同号同结果、可复现。
- 工具执行**从不抛异常**，一律返回 `ToolResult`；超时/重试参数可注入。
- 测试**不得依赖宿主机 `.env`**（ch01 已确立）；测试不得依赖 `tmp_path` / `tempfile`（沙箱退出后目录被去权限）。
- 命令一律带 `-p no:cacheprovider`（`pyproject.toml` 已配，勿改动）。
- 中文源文件用 `write` / `edit` 工具改，**不要用 `Set-Content`**（会写坏编码）。
- 每完成一个阶段往 `MemHelp-code/dev-notes/ch02.md` 追记四样：用户原话 / 关键产出 / 被拒绝或纠偏 / 翻车与返工。

## Review Focus

以下五类输入/失效是 spec 隐含但没有任何任务的测试会自然覆盖的，最可能让使用者踩坑。每一条都在其归属任务的步骤里有对应测试。

1. **模型不给 `content` 只给 `tool_calls`** —— spec 4.1 的两次调用把这种情况拆成了两半：第 1 次拿到空 `content` 是**正常**（不是 `UpstreamEmptyResponse`），第 2 次拿到空 `content` 才要兜底。ch01 的 `_message_text` + `UpstreamEmptyResponse` 会因为前者被误触发。归属：Task 9。
2. **模型一轮返回多个 `tool_calls`** —— 必须只执行第一个、其余不执行但**要留痕**，且不能因此报错。归属：Task 9。
3. **`session_id` 重复轮次** —— 同一 `session_id` 第 2 轮不能新建 `conversations` 行（否则外键指向漂移、历史错乱）。归属：Task 3。
4. **MySQL 故障时的降级** —— `/healthz` 必须报 503 而不是 200，聊天接口必须回规范 503 外壳而不是裸 500。归属：Task 4、Task 11。
5. **`LIKE` 的 `%` / `_` 通配符注入** —— 用户问「%是什么」时，keyword 里的 `%` 必须被转义成字面量，否则全表命中。归属：Task 8。

---

### Task 0: 依赖与 DDL 落地

**Files:**
- Modify: `MemHelp-code/pyproject.toml`
- Modify: `sql/ch02-ddl.sql:14-22`（`conversations` 表加 `session_id`）
- Create: `MemHelp-code/docker-compose.yml`
- Create: `MemHelp-code/.env.example`（追加数据库与工具配置）

**Interfaces:**
- Consumes: 无（起点）
- Produces: `sql/ch02-ddl.sql` 里 `conversations.session_id VARCHAR(64) NOT NULL` + `UNIQUE KEY uk_session_id`；compose 服务名 `mysql`、库名 `mewhelp`、账号 `mewhelp`/`mewhelp`、端口 `3306`

- [ ] **Step 1: 给 `conversations` 表加 `session_id`**

在 `sql/ch02-ddl.sql` 的 `conversations` 定义里，把 `id` 行之后插入一行，并在 `PRIMARY KEY (id)` 之后加唯一键：

```sql
  session_id  VARCHAR(64)     NOT NULL                COMMENT '对外会话标识,聊天页与接口用它',
```

```sql
  UNIQUE KEY uk_session_id (session_id),
```

- [ ] **Step 2: 追加依赖并同步**

在 `pyproject.toml` 的 `dependencies` 里加两行：

```toml
    "sqlalchemy>=2.0",
    "pymysql>=1.1",
```

Run: `$uv sync`（`$uv = "C:\Users\dbq\.local\bin\uv.exe"`，先设 `$env:UV_CACHE_DIR` 到工作区内）
Expected: 解析成功，`uv.lock` 出现 `sqlalchemy` 与 `pymysql` 条目。

- [ ] **Step 3: 写 `docker-compose.yml`**

放 `MemHelp-code/` 下（不是仓库根）：`sql/` 在根目录，单个 compose 文件里同时指到 `../sql` 和 `./app` 才干净。

```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: mewhelp-mysql
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
    environment:
      MYSQL_ROOT_PASSWORD: rootpw
      MYSQL_DATABASE: mewhelp
      MYSQL_USER: mewhelp
      MYSQL_PASSWORD: mewhelp
    ports: ["3306:3306"]
    volumes:
      - ../sql:/docker-entrypoint-initdb.d:ro
      - mewhelp-mysql-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "127.0.0.1", "-uroot", "-prootpw"]
      interval: 5s
      timeout: 5s
      retries: 20
volumes:
  mewhelp-mysql-data:
```

- [ ] **Step 4: `.env.example` 追加配置项**

```
# MySQL（docker compose up -d 起，见 docker-compose.yml）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=mewhelp
MYSQL_PASSWORD=mewhelp
MYSQL_DATABASE=mewhelp
# 工具基础设施：单次执行超时（秒）与仅对瞬时错误的重试次数
TOOL_TIMEOUT_SECONDS=3.0
TOOL_MAX_RETRIES=1
```

同时把这六项**按同样键名**追加进 `MemHelp-code/.env`（真实配置，不入库）。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/pyproject.toml MemHelp-code/uv.lock MemHelp-code/docker-compose.yml MemHelp-code/.env.example sql/ch02-ddl.sql
git commit -m "ch02: 加 sqlalchemy/pymysql 依赖、DDL 加 session_id、compose 起 MySQL"
```

---

### Task 1: 配置扩展

**Files:**
- Modify: `MemHelp-code/app/config.py`
- Test: `MemHelp-code/tests/test_config.py`（追加用例）

**Interfaces:**
- Consumes: 无
- Produces: `Settings` 新增字段与方法 ——
  `mysql_host: str`, `mysql_port: int`, `mysql_user: str`, `mysql_password: str`, `mysql_database: str`,
  `tool_timeout_seconds: float`, `tool_max_retries: int`,
  `Settings.database_url() -> str`（**方法**，不是属性）返回 `mysql+pymysql://u:p@h:port/db?charset=utf8mb4`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_config.py`。注意该文件的既有约定是**不经 `.env`** 直接构造 `Settings(...)`：

```python
def test_database_url_is_pymysql_with_utf8mb4():
    s = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        mysql_host="db.internal", mysql_port=3307, mysql_user="u",
        mysql_password="p", mysql_database="mewhelp",
    )
    assert s.database_url() == (
        "mysql+pymysql://u:p@db.internal:3307/mewhelp?charset=utf8mb4"
    )


def test_database_url_escapes_special_chars_in_password():
    """密码里有 @ / : / / 时必须 percent-encode，否则连接串被解析到错误的 host。"""
    s = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        mysql_password="p@ss:w/rd",
    )
    assert "p%40ss%3Aw%2Frd" in s.database_url()


def test_tool_infra_defaults():
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    assert s.tool_timeout_seconds == 3.0
    assert s.tool_max_retries == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_config.py -v`
Expected: FAIL —— `TypeError: Settings.__init__() got an unexpected keyword argument 'mysql_host'`

- [ ] **Step 3: 实现配置字段**

在 `app/config.py` 的 `Settings` 里加字段（`openai_*` 之后），并在 `_strip_strings` 校验器的字段列表里加上 `"mysql_host"`、`"mysql_user"`、`"mysql_password"`、`"mysql_database"`。`database_url()` 用 `urllib.parse.quote_plus` 转义 user 与 password：

```python
    mysql_host: str = Field(default="127.0.0.1", min_length=1, description="MySQL 主机。")
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = Field(default="mewhelp", min_length=1)
    mysql_password: str = Field(default="mewhelp")
    mysql_database: str = Field(default="mewhelp", min_length=1)
    tool_timeout_seconds: float = Field(default=3.0, gt=0, description="单次工具执行超时（秒）。")
    tool_max_retries: int = Field(default=1, ge=0, le=5, description="仅对瞬时错误的重试次数。")
```

`database_url()` 的返回串必须**逐字符**匹配 Step 1 的断言（`?charset=utf8mb4` 结尾，无其他参数）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_config.py -v`
Expected: PASS，且原有 ch01 用例全绿。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/config.py MemHelp-code/tests/test_config.py
git commit -m "ch02: 配置加 MySQL 连接与工具超时/重试"
```

---

### Task 2: ORM 模型

**Files:**
- Create: `MemHelp-code/app/db/__init__.py`（空）
- Create: `MemHelp-code/app/db/base.py`
- Create: `MemHelp-code/app/db/models.py`
- Test: `MemHelp-code/tests/test_db_models.py`

**Interfaces:**
- Consumes: 无
- Produces: `Base`（`DeclarativeBase` 子类）；四个 ORM 类 `Conversation` / `Message` / `Faq` / `Ticket`，字段名与 DDL 列名**逐字相同**。`Message.tool_calls` 为 `dict | list | None`（SQLAlchemy `JSON`），`Message.tool_call_id: str | None`。`Conversation.session_id: str` 唯一。另产出 `CHAT_STATUSES` / `MESSAGE_ROLES` / `TICKET_TYPES` / `TICKET_STATUSES` 四个 `tuple[str, ...]` 常量，供 Task 3、Task 8 复用。

- [ ] **Step 1: 写失败测试**

```python
def test_metadata_has_exactly_four_ch02_tables():
    from app.db.models import Base
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversation_session_id_is_unique_and_not_nullable():
    from app.db.models import Conversation
    col = Conversation.__table__.c.session_id
    assert col.nullable is False
    assert col.type.length == 64
    assert any(ix.unique for ix in Conversation.__table__.indexes
               if [c.name for c in ix.columns] == ["session_id"])


def test_message_columns_match_ddl():
    """列名必须与 sql/ch02-ddl.sql 逐字相同 —— 差一个字母就是运行时 Unknown column。"""
    from app.db.models import Message
    assert set(Message.__table__.c.keys()) == {
        "id", "conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at",
    }


def test_enum_columns_reject_values_outside_ddl_enum():
    """DDL 用的是 MySQL ENUM；ORM 侧必须同样受限，否则单测全绿、真库插入报错。"""
    from app.db.models import Conversation
    from sqlalchemy import Enum as SAEnum
    assert isinstance(Conversation.__table__.c.status.type, SAEnum)
    assert set(Conversation.__table__.c.status.type.enums) == {"进行中", "已转人工", "已结束"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_models.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: 写模型**

`app/db/base.py`：`class Base(DeclarativeBase): pass`。
`app/db/models.py`：用 2.0 风格 `Mapped[...]` + `mapped_column(...)`。

**关键决定**：ENUM 列一律写成
`mapped_column(SAEnum(*TICKET_TYPES, name="ticket_type"), nullable=False)`
（`SAEnum` 位置参数传**值**，与 DDL 的 ENUM 字面量一致；`name=` 只作标识，SQLite 场景下 `native_enum=False` 才建 CHECK 约束）。`created_at` 用 `server_default=func.now()`，让数据库定时钟而不是应用进程。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_models.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/db/ MemHelp-code/tests/test_db_models.py
git commit -m "ch02: 四张表的 SQLAlchemy 2.0 ORM 映射（与 DDL 逐列对齐）"
```

---

### Task 3: 仓库层

**Files:**
- Create: `MemHelp-code/app/db/repositories.py`
- Test: `MemHelp-code/tests/test_db_repositories.py`
- Modify: `MemHelp-code/tests/conftest.py`（加 SQLite 内存库 fixture）

**Interfaces:**
- Consumes: Task 2 的 `Base` 与四个模型
- Produces:
  - `ConversationRepository.upsert_by_session(session_id: str, user_id: str) -> int`（返回 `conversations.id`，**幂等**）
  - `MessageRepository.append(conversation_id: int, *, role: str, content: str | None, tool_calls: list | None = None, tool_call_id: str | None = None) -> None`
  - `MessageRepository.list_for_conversation(conversation_id: int) -> list[Message]`（按 `id` 升序）
  - `MessageRepository.find_conversation_id(session_id: str) -> int | None`
  - `FaqRepository.search(keyword: str, limit: int = 5) -> list[Faq]` —— **本方法负责 `LIKE` 通配符转义**（调用方无需预处理）
  - `escape_like(keyword: str) -> str` —— 模块级函数，把 `\` `%` `_` 转义成字面量（`\\` / `\%` / `\_`），供 `FaqRepository.search` 使用
  - `TicketRepository.create(conversation_id: int, description: str, ticket_type: str) -> str`（返回 `ticket_no`）
  - `TicketRepository.count_created_on(day: date) -> int`
  - 全部是 `async def`（PyMySQL 是同步驱动，实现里用 `anyio.to_thread.run_sync` 包住，避免阻塞事件循环 —— 这一条是 Task 9 能安全并发的前提）
  - `Repositories` 容器：`Repositories(conversations=..., messages=..., faq=..., tickets=...)`，构造参数都可替换（测试注入内存实现）

- [ ] **Step 1: 加 SQLite 内存 fixture**

追加到 `tests/conftest.py`：

```python
@pytest.fixture
def sqlite_session():
    """SQLite 内存库。ENUM 用 native_enum=False 落成 VARCHAR，语义等价。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.db.base import Base
    import app.db.models  # noqa: F401 - 触发模型注册

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
```

- [ ] **Step 2: 写失败测试（含 Review Focus 第 3 条）**

```python
async def test_upsert_by_session_is_idempotent(sqlite_session):
    """同一 session_id 第 2 轮不能新建 conversations 行，必须复用同一个 id。"""
    from app.db.repositories import ConversationRepository
    repo = ConversationRepository(sqlite_session)
    first = await repo.upsert_by_session("sess-a", "anonymous")
    second = await repo.upsert_by_session("sess-a", "anonymous")
    assert first == second
    from app.db.models import Conversation
    assert sqlite_session.query(Conversation).count() == 1


async def test_append_and_list_preserves_order_and_tool_columns(sqlite_session):
    from app.db.repositories import ConversationRepository, MessageRepository
    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    repo = MessageRepository(sqlite_session)
    await repo.append(cid, role="user", content="订单 1001 的物流到哪了")
    await repo.append(cid, role="assistant", content="",
                     tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    await repo.append(cid, role="tool", content='{"ok":true}', tool_call_id="c1")
    await repo.append(cid, role="assistant", content="已发货")
    rows = await repo.list_for_conversation(cid)
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[2].tool_call_id == "c1"


async def test_faq_search_hits_and_misses(sqlite_session):
    from app.db.models import Faq
    from app.db.repositories import FaqRepository
    sqlite_session.add(Faq(question="退货运费谁承担？", answer="质量问题由平台承担。", category="售后"))
    sqlite_session.add(Faq(question="发票邮费说明", answer="不相关", category="发票"))
    sqlite_session.commit()
    repo = FaqRepository(sqlite_session)
    assert [f.question for f in await repo.search("运费")] == ["退货运费谁承担？"]
    assert await repo.search("邮费") != []          # 有含「邮费」的无关条目
    assert await repo.search("完全不存在的词") == []


def test_escape_like_neutralises_wildcards():
    """Review Focus 5：% 和 _ 必须当字面量，否则查「%」会全表命中。"""
    from app.db.repositories import escape_like
    assert escape_like("100%纯棉") == "100\\%纯棉"
    assert escape_like("a_b") == "a\\_b"
    assert escape_like("c\\d") == "c\\\\d"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_repositories.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.db.repositories'`

- [ ] **Step 4: 实现仓库**

要点：`upsert_by_session` 先 `SELECT ... WHERE session_id = :sid`，命中就返回其 `id`，未命中才 `INSERT`（**不要**用 MySQL 的 `ON DUPLICATE KEY UPDATE` —— SQLite 上跑不了，会让这层无法离线测）。`create()` 的 `ticket_no` 规则为 `f"T{day:%Y%m%d}{n:03d}"`，`n = await count_created_on(day) + 1`。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_repositories.py -v`
Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/app/db/repositories.py MemHelp-code/tests/test_db_repositories.py MemHelp-code/tests/conftest.py
git commit -m "ch02: 仓库层（会话幂等 upsert、消息流水、FAQ LIKE、工单号）"
```

---

### Task 4: 引擎/会话装配与数据库健康探测

**Files:**
- Create: `MemHelp-code/app/db/session.py`
- Test: `MemHelp-code/tests/test_db_session.py`

**Interfaces:**
- Consumes: Task 1 的 `Settings.database_url()`、Task 3 的 `Repositories`
- Produces:
  - `create_engine_from(settings: Settings) -> Engine`
  - `session_factory(engine: Engine) -> sessionmaker[Session]`
  - `Repositories.from_session(session: Session) -> Repositories`
  - `async def ping(engine: Engine) -> bool`（`SELECT 1`，任何异常返回 `False`，**不抛**）

- [ ] **Step 1: 写失败测试**

```python
async def test_ping_returns_false_instead_of_raising_on_dead_engine():
    """healthz 依赖这个函数永不抛异常 —— 抛了就成了裸 500。"""
    from sqlalchemy import create_engine
    from app.db.session import ping
    bad = create_engine("mysql+pymysql://u:p@127.0.0.1:1/nope", connect_args={"connect_timeout": 1})
    assert await ping(bad) is False


async def test_ping_true_on_live_sqlite_engine():
    from sqlalchemy import create_engine
    from app.db.session import ping
    assert await ping(create_engine("sqlite+pysqlite:///:memory:")) is True


def test_create_engine_from_uses_settings_url_and_pool_recycle():
    """MySQL 的 wait_timeout 会掐掉闲置连接，pool_recycle 必须设上。"""
    from app.config import Settings
    from app.db.session import create_engine_from
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    engine = create_engine_from(s)
    assert engine.pool._recycle > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_session.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.db.session'`

- [ ] **Step 3: 实现**

`create_engine_from` 传 `pool_pre_ping=True, pool_recycle=1800`。`ping` 用 `anyio.to_thread.run_sync` 跑同步连接并 `try/except Exception: return False`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_db_session.py -v`
Expected: 3 passed（第一个用例可能需要 1 秒等连接超时）

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/db/session.py MemHelp-code/tests/test_db_session.py
git commit -m "ch02: DB 引擎装配 + 永不抛异常的 ping"
```

---

### Task 5: 三个随机数据业务工具

**Files:**
- Create: `MemHelp-code/app/tools/__init__.py`（空）
- Create: `MemHelp-code/app/tools/schemas.py`（本任务只加三个参数模型）
- Create: `MemHelp-code/app/tools/business.py`
- Test: `MemHelp-code/tests/test_tool_business.py`

**Interfaces:**
- Consumes: 无（**不得** import `app.db`）
- Produces:
  - `QueryOrderArgs(order_id: str)`, `QueryProductArgs(product_id: str)`, `QueryLogisticsArgs(order_id: str)`
  - `derive_rng(scope: str, key: str) -> random.Random`
  - `query_order(order_id: str) -> str` / `query_product(product_id: str) -> str` / `query_logistics(order_id: str) -> str`，三个都是同步函数、返回**已经 JSON 序列化的字符串**

- [ ] **Step 1: 写失败测试**

```python
def test_same_order_id_yields_identical_result():
    """可复现是验收与评估集的前提：同号必须同结果。"""
    from app.tools.business import query_logistics
    assert query_logistics("1001") == query_logistics("1001")


def test_different_order_ids_differ():
    from app.tools.business import query_logistics
    assert query_logistics("1001") != query_logistics("2002")


def test_logistics_payload_has_required_fields():
    import json
    from app.tools.business import query_logistics
    data = json.loads(query_logistics("1001"))
    assert set(data) >= {"order_id", "waybill_no", "carrier", "status", "traces", "eta"}
    assert data["order_id"] == "1001"
    assert len(data["traces"]) >= 1


def test_business_tools_do_not_import_database():
    """这条边界保证没有 MySQL 也能跑完单测。"""
    import app.tools.business as m
    assert "app.db" not in (m.__doc__ or "") + str(m.__dict__.keys())
    import sys
    assert not any(k.startswith("app.db") for k in sys.modules if k.startswith("app.tools"))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_business.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools'`

- [ ] **Step 3: 实现**

`derive_rng(scope, key)` 返回 `random.Random(f"{scope}:{key}")`（字符串种子由 Python 稳定哈希 —— 注意**不要**用 `hash()`，`PYTHONHASHSEED` 会让它跨进程变化；`random.Random(str)` 走的是 `str` 自身的稳定字节，符合要求）。
`query_logistics` 生成运单号 `SF` + 12 位数字、承运商从固定列表选、`status` 从 `("已揽收","运输中","派送中","已签收")` 选、`traces` 是 3~5 条 `{"time","desc"}`、`eta` 为 ISO 日期字符串。`query_order` 生成 `order_id/status/amount/product_id/created_at`；`query_product` 生成 `product_id/name/price/stock/category`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_business.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/tools/ MemHelp-code/tests/test_tool_business.py
git commit -m "ch02: query_order / query_product / query_logistics（按入参派生的可复现随机数据）"
```

---

### Task 6: FAQ 与工单工具

**Files:**
- Modify: `MemHelp-code/app/tools/schemas.py`（加 `QueryFaqArgs`、`CreateTicketArgs`）
- Create: `MemHelp-code/app/tools/knowledge.py`
- Create: `MemHelp-code/app/tools/ticketing.py`
- Test: `MemHelp-code/tests/test_tool_knowledge.py`

**Interfaces:**
- Consumes: Task 3 的 `FaqRepository` / `TicketRepository` 协议
- Produces:
  - `QueryFaqArgs(keyword: str)`, `CreateTicketArgs(description: str, ticket_type: Literal['售后','投诉','咨询'], conversation_id: str)`
  - `async def query_faq(keyword: str, *, faq_repo: FaqRepository) -> str`
  - `async def create_ticket(description: str, ticket_type: str, conversation_id: str, *, ticket_repo: TicketRepository, conversations: ConversationRepository) -> str`
  - 两者都返回**已经 JSON 序列化的字符串**

- [ ] **Step 1: 写失败测试**

```python
class FakeFaqRepo:
    def __init__(self, rows): self.rows = rows
    async def search(self, keyword, limit=5):
        return [r for r in self.rows if keyword in r.question or keyword in r.answer][:limit]


async def test_query_faq_reports_zero_hits_without_raising():
    """验收 3：查不到是预期结果，必须是 ok=true + count=0，不是错误。"""
    import json
    from app.tools.knowledge import query_faq
    out = json.loads(await query_faq("邮费是多少", faq_repo=FakeFaqRepo([])))
    assert out["ok"] is True
    assert out["count"] == 0
    assert out["items"] == []


async def test_query_faq_passes_keyword_through_untouched():
    """转义是仓库层的职责（Task 3 的 escape_like），工具层不该重复做 —— 重复转义会把
    「100%」变成「100\\%」再变成「100\\\\%」，反而查不到。"""
    from app.tools.knowledge import query_faq
    repo = FakeFaqRepo([])
    await query_faq("100%纯棉", faq_repo=repo)
    assert repo.last_keyword == "100%纯棉"


async def test_create_ticket_returns_ticket_no_and_persists():
    import json
    from app.tools.ticketing import create_ticket
    class FakeTicketRepo:
        def __init__(self): self.created = []
        async def create(self, conversation_id, description, ticket_type):
            self.created.append((conversation_id, description, ticket_type))
            return "T20260215001"
    repo = FakeTicketRepo()
    out = json.loads(await create_ticket(
        description="杯子碎了", ticket_type="售后", conversation_id="sess-a", ticket_repo=repo))
    assert out["ok"] is True and out["ticket_no"] == "T20260215001"
    assert repo.created == [("sess-a", "杯子碎了", "售后")]
```

> 注：Fake 仓库需要记录 `last_keyword`；`FaqRepository` 的真实现在 Task 3，转义由它负责。
> 工具层只把关键词**原样**传下去，这条测试钉住的正是「不要重复转义」。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_knowledge.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.knowledge'`

- [ ] **Step 3: 实现**

`query_faq` 把 `keyword` **原样**交给 `faq_repo.search()`（转义在仓库层，见 Task 3）。
`create_ticket` 需要把字符串 `conversation_id` 换成数字 `conversations.id`：调 `ConversationRepository.upsert_by_session(conversation_id, "anonymous")`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_knowledge.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/tools/ MemHelp-code/tests/test_tool_knowledge.py
git commit -m "ch02: query_faq（LIKE 转义）与 create_ticket（写 tickets 表）"
```

---

### Task 7: 注册表

**Files:**
- Create: `MemHelp-code/app/tools/registry.py`
- Test: `MemHelp-code/tests/test_tool_registry.py`

**Interfaces:**
- Consumes: Task 5、Task 6 的五个函数与参数模型
- Produces:
  - `TOOLS: list[BaseTool]` —— 五个 LangChain `StructuredTool`
  - `TOOL_NAMES: tuple[str, ...]`
  - `get_tool(name: str) -> BaseTool | None`
  - `TOOL_LABELS: dict[str, str]` —— 中文短语，如 `{"query_logistics": "正在查询物流…"}`，供 SSE `status` 帧用

- [ ] **Step 1: 写失败测试**

```python
def test_registry_covers_exactly_the_five_business_tools():
    from app.tools.registry import TOOL_NAMES
    assert set(TOOL_NAMES) == {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


def test_registry_keys_match_tool_names():
    """改名后静默失联是最难查的 bug（SSE 帧与落库都用字符串名）。"""
    from app.tools.registry import TOOLS
    for t in TOOLS:
        assert isinstance(t.name, str) and t.name


def test_every_tool_has_pydantic_args_schema_and_chinese_label():
    from pydantic import BaseModel
    from app.tools.registry import TOOLS, TOOL_LABELS
    for t in TOOLS:
        assert isinstance(t.args_schema, type) and issubclass(t.args_schema, BaseModel)
        assert t.description.strip()
        assert TOOL_LABELS[t.name].endswith("…")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_registry.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.registry'`

- [ ] **Step 3: 实现**

用 `@tool(args_schema=QueryOrderArgs)` 在 **`registry.py` 里**包装 Task 5/6 的函数（或直接在各自的模块里用装饰器、registry 只 import 它们 —— 选后者更贴近 LangChain 惯例，且让 `tools/business.py::query_order` 保持**可单独调用的纯函数**，便于离线断言）。`@tool` 的 docstring 是**给模型看的**，必须写清楚何时该调用、参数含义（验收 1/2/3 的效果直接取决于这段文字）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_registry.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/tools/registry.py MemHelp-code/tests/test_tool_registry.py
git commit -m "ch02: 五工具注册表 + 中文状态标签"
```

---

### Task 8: 执行器（参数校验 / 超时 / 重试 / 错误脱敏）

**Files:**
- Create: `MemHelp-code/app/tools/executor.py`
- Test: `MemHelp-code/tests/test_tool_executor.py`

**Interfaces:**
- Consumes: Task 7 的 `TOOLS` / `get_tool`
- Produces:
  - `@dataclass(frozen=True) class ToolResult`：`tool: str`, `args: dict[str, Any]`, `ok: bool`, `data: str | None`, `error: str | None`, `attempts: int`, `duration_ms: int`
  - `async def execute_tool(name: str, args: dict[str, Any], *, timeout: float, max_retries: int) -> ToolResult`
  - `def is_transient(exc: BaseException) -> bool`
  - `def to_tool_message_content(result: ToolResult) -> str`
  - `MAX_TOOL_RESULT_CHARS: int = 4000`

- [ ] **Step 1: 写失败测试**

```python
async def test_unknown_tool_name_is_structured_failure_not_exception():
    from app.tools.executor import execute_tool
    r = await execute_tool("query_nothing", {}, timeout=1.0, max_retries=0)
    assert r.ok is False and "未知工具" in r.error


async def test_schema_validation_rejects_bad_args_without_retrying():
    """参数错重试也是同样结果，白等 2 倍超时。attempts 必须是 1。"""
    from app.tools.executor import execute_tool
    r = await execute_tool("query_order", {}, timeout=1.0, max_retries=3)
    assert r.ok is False and "order_id" in r.error and r.attempts == 1


async def test_timeout_retries_then_gives_up(monkeypatch):
    import asyncio
    from app.tools import executor
    async def slow(*a, **k): await asyncio.sleep(5)
    class T:
        name = "query_order"; args_schema = None
        async def ainvoke(self, args): await slow()
    monkeypatch.setattr(executor, "get_tool", lambda name: T())
    r = await executor.execute_tool("query_order", {"order_id": "1"}, timeout=0.01, max_retries=1)
    assert r.ok is False and r.attempts == 2 and "超时" in r.error


async def test_permanent_error_is_not_retried():
    from app.tools import executor
    class T:
        name = "query_order"
        async def ainvoke(self, args): raise ValueError("查不到这个订单")
    assert executor.is_transient(ValueError("x")) is False


def test_error_text_is_redacted_on_the_way_to_the_model():
    """Review Focus：数据库异常的消息里可能带连接串，回灌给模型前必须脱敏。"""
    import json
    from app.tools.executor import ToolResult, to_tool_message_content
    r = ToolResult(tool="query_faq", args={}, ok=False, data=None,
                   error="conn failed api_key=sk-abcdef123456", attempts=1, duration_ms=1)
    assert "sk-abcdef123456" not in to_tool_message_content(r)


def test_oversized_tool_result_is_truncated():
    """超大工具结果会把上下文预算吃光，必须截断并标记。"""
    import json
    from app.tools.executor import MAX_TOOL_RESULT_CHARS, ToolResult, to_tool_message_content
    r = ToolResult(tool="query_faq", args={}, ok=True, data="x" * 10000,
                   error=None, attempts=1, duration_ms=1)
    out = json.loads(to_tool_message_content(r))
    assert out["truncated"] is True
    assert len(json.dumps(out, ensure_ascii=False)) <= MAX_TOOL_RESULT_CHARS + 200
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_executor.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.executor'`

- [ ] **Step 3: 实现**

- 未知工具 → `ok=False, error="未知工具 ..." , attempts=0`
- 参数校验：`tool.args_schema.model_validate(args)`，`ValidationError` → `attempts=1`，`error` 取**字段名 + 原因**（别把整个 pydantic 报错糊上去，模型看不懂）
- 超时：`asyncio.wait_for(tool.ainvoke(validated.model_dump()), timeout=timeout)`
- `is_transient`：`True` 仅当 `asyncio.TimeoutError` / `ConnectionError` / `sqlalchemy.exc.OperationalError` / `InterfaceError`；`False` 对 `ValueError`、`ValidationError`、以及工具自己抛的业务异常
- 重试循环总尝试 `max_retries + 1` 次，每次累加 `attempts`
- 所有 `error` 一律过 `app.errors.redact()`
- `to_tool_message_content` 产出 `json.dumps({"ok":..., "tool":..., "data"|"error":...}, ensure_ascii=False)`，并按 `MAX_TOOL_RESULT_CHARS` 截断（附 `"truncated": true`），防止超大工具结果把上下文预算吃光

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_tool_executor.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/tools/executor.py MemHelp-code/tests/test_tool_executor.py
git commit -m "ch02: 工具执行器（schema 校验、超时重试、错误脱敏、结果截断）"
```

---

### Task 9: 单轮工具编排（核心）

**Files:**
- Create: `MemHelp-code/app/chat_tools.py`
- Test: `MemHelp-code/tests/test_chat_tools.py`

**Interfaces:**
- Consumes: Task 3 的 `Repositories`、Task 7 的 `TOOLS`/`TOOL_LABELS`、Task 8 的 `execute_tool`/`ToolResult`/`to_tool_message_content`、ch01 的 `build_chat_messages`/`build_context`/`_message_text`
- Produces:
  - `@dataclass class ToolEvent`：`kind: Literal["status", "token", "tools"]`，`payload: dict`
  - `async def astream_chat_with_tools(*, session_id, message, model, repositories, budget_tokens, model_name, timeout, max_retries, meta_out=None) -> AsyncIterator[ToolEvent]`
    - 依次产出：`ToolEvent("status", {"tool","args","label"})` → `ToolEvent("token", {"delta"})`* → `ToolEvent("tools", {"tools":[{"name","args","ok","duration_ms"}]})`
    - 落库由本函数负责（成功路径四行；上游失败只落用户一行并**重新抛**）
  - `def fallback_answer(result: ToolResult) -> str` —— 第 2 次调用返回空文本时的话术

- [ ] **Step 1: 写失败测试（含 Review Focus 1、2）**

Fake 模型沿用 `tests/fakes.py::FakeChatModel`，但需要「第 1 次调用回 tool_calls、第 2 次调用回文本」的分段行为：给 `FakeChatModel` 加字段
`tool_calls_then_stream: list[dict] | None = None`（非空时**第一次** `ainvoke` 返回带 tool_calls 的 AIMessage，之后走正常 chunks）。这个字段在 Step 3 一并实现于 `tests/fakes.py`。

```python
async def test_single_turn_converges_and_never_executes_a_second_tool(fake_repos):
    """Review Focus 2：模型在一轮里给了两个 tool_calls，只执行第一个。"""
    ...
    calls = []
    async def spy_execute(name, args, **kw):
        calls.append(name)
        return ToolResult(tool=name, args=args, ok=True, data='{"ok":true}', error=None,
                          attempts=1, duration_ms=1)
    events = [e async for e in astream_chat_with_tools(...)]
    assert calls == ["query_logistics"]                      # 只有一个被真正执行
    assert [e.kind for e in events].count("status") == 1
    assert events[-1].kind == "tools"


async def test_tool_call_with_empty_content_is_not_an_upstream_error(fake_repos):
    """Review Focus 1：第 1 次调用 content 为空是正常的，不能抛 UpstreamEmptyResponse。"""
    ...


async def test_second_call_returning_empty_content_falls_back(fake_repos):
    from app.chat_tools import fallback_answer
    ...


async def test_persists_four_rows_in_order(fake_repos):
    roles = [r["role"] for r in fake_repos.messages.appended]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_upstream_failure_persists_only_the_user_turn(fake_repos):
    roles = [r["role"] for r in fake_repos.messages.appended]
    assert roles == ["user"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_chat_tools.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.chat_tools'`

- [ ] **Step 3: 实现 `app/chat_tools.py` 并扩展 `tests/fakes.py`**

编排顺序（**必须与 spec 4.1 一致**）：

1. `history = await repositories.messages.history_as_messages(session_id)` —— 从 `messages` 表重建 `list[BaseMessage]`（`user`→`HumanMessage`，`assistant`+`tool_calls`→`AIMessage(tool_calls=...)`，`tool`→`ToolMessage(content, tool_call_id)`，`assistant` 无 tool_calls→`AIMessage(content)`）
2. `rendered = build_chat_messages(history, message)`；`ctx = build_context(...)`（沿用 ch01 的裁剪顺序）
3. `first = await model.bind_tools(TOOLS).ainvoke(ctx.messages)`
4. 无 `first.tool_calls` → `final_parts = [text]`，无工具事件（**不要**调第 2 次）
5. 有 → `chosen = first.tool_calls[0]`；多余的写进日志（`logger.warning`，含全部名字）
6. `yield ToolEvent("status", {"tool": chosen["name"], "args": chosen["args"], "label": TOOL_LABELS.get(name, f"正在调用 {name}…")})`
7. `result = await execute_tool(...)`；`messages2 = [*ctx.messages, AIMessage(content="", tool_calls=[chosen]), ToolMessage(content=to_tool_message_content(result), tool_call_id=chosen["id"])]`
8. `async for chunk in model.astream(messages2)` → `yield ToolEvent("token", {"delta": ...})`（跳过空串）；若一个非空 delta 都没有 → `yield ToolEvent("token", {"delta": fallback_answer(result)})`
9. 落库（先 `upsert_by_session` 拿 `conversation_id`，再按序 append 四行）
10. `yield ToolEvent("tools", {"tools": [{"name", "args", "ok", "duration_ms"}]})`

**异常语义**：整段用 `try/except BaseException` 包住，`except` 里只落 `user` 那一行然后 `raise`（沿用 ch01「不落半截回答」）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_chat_tools.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/chat_tools.py MemHelp-code/tests/test_chat_tools.py MemHelp-code/tests/fakes.py
git commit -m "ch02: 单轮工具编排（定工具→执行→回灌→流式收敛，含工具轨迹事件）"
```

---

### Task 10: 聊天页徽章（Vibe Coding，不走评审）

**Files:**
- Modify: `MemHelp-code/web/app.js`
- Modify: `MemHelp-code/web/style.css`

**Interfaces:**
- Consumes: Task 9 的 `ToolEvent` 经 Task 11 映射成的 SSE 帧
- Produces: 无（终端 UI）

**说明**：本任务按用户要求**不套 brainstorm / TDD / code review 流程**。用户描述效果，我改，浏览器里看。

- [ ] **Step 1: 处理 `status` 帧** —— 在 `applyFrame` 里加分支：气泡上方插入一行 `.tool-status`，文案取 `data.label`
- [ ] **Step 2: 处理 `done.tools`** —— 在气泡内渲染 `.tool-badges`，每个徽章 `🔧 <name> · <成功|失败> · <duration_ms>ms`；同时把 `tools` 写进 `ctx.bot.row.dataset.tools`
- [ ] **Step 3: 历史回读渲染** —— 用 `/messages` 返回的 `tools` 字段复用同一个渲染函数（徽章刷新后仍在）
- [ ] **Step 4: `style.css` 加 `.tool-status` / `.tool-badges` / `.tool-badge` 样式**，与既有 `.stats` 风格一致
- [ ] **Step 5: 浏览器实测** —— 起服务，问「订单 1001 的物流到哪了」，确认徽章出现；刷新后确认徽章仍在
- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/web/app.js MemHelp-code/web/style.css
git commit -m "ch02(web): 气泡内工具轨迹徽章 + 工具执行状态行"
```

---

### Task 11: HTTP 层接入与 DB 依赖

**Files:**
- Modify: `MemHelp-code/app/api.py`
- Modify: `MemHelp-code/app/main.py`
- Test: `MemHelp-code/tests/test_api_tools.py`

**Interfaces:**
- Consumes: Task 4 的 `create_engine_from`/`session_factory`/`Repositories.from_session`/`ping`、Task 9 的 `astream_chat_with_tools`
- Produces:
  - `create_app(model=None, store=None, settings=None, repositories=None, db_ping=None)` —— 新增两个可注入参数
  - `get_repositories(request) -> Repositories` 依赖
  - SSE 新契约：`status` 帧、`done.tools`
  - `/api/chat/sessions/{session_id}/messages` 改为读库，响应新增每条的 `tools` 字段
  - `/healthz` 新增 `"database": "ok" | "unavailable"`，不可用时 **503**

- [ ] **Step 1: 写失败测试**

沿用 ch01 的注入方式（`create_app(model=..., settings=..., repositories=FakeRepos())`），**不发网络请求、不连数据库**：

```python
async def test_sse_emits_status_frame_before_tokens(client_with_tools):
    frames = await collect_sse(client_with_tools, "订单 1001 的物流到哪了")
    kinds = [f["event"] for f in frames]
    assert "status" in kinds
    assert kinds.index("status") < kinds.index("token")


async def test_done_frame_carries_tools_array(client_with_tools):
    done = [f for f in await collect_sse(client_with_tools, "订单 1001 的物流到哪了")
            if f["event"] == "done"][-1]
    assert done["data"]["tools"][0]["name"] == "query_logistics"


async def test_healthz_reports_503_when_database_is_down():
    app = create_app(model=FakeChatModel(), settings=fake_settings(), db_ping=lambda: False)
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 503
    assert r.json()["database"] == "unavailable"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run --directory MemHelp-code python -m pytest tests/test_api_tools.py -v`
Expected: FAIL —— `create_app() got an unexpected keyword argument 'repositories'`

- [ ] **Step 3: 实现**

- `_chat_stream_body` 改为消费 `astream_chat_with_tools` 的 `ToolEvent`，映射成 `status` / `token` 帧，并在 `done` 里带上 `tools` 数组
- DB 依赖沿用 ch01 的「`app.state` 注入优先，缺失才构造真实对象」模式：`repositories` 为 `None` 时用 `app.state.settings.database_url()` 建 engine
- `main.py` 的 `create_app()` 保持无参调用可工作
- `/messages` 从 `repositories.messages` 读，返回体新增 `tools`
- 数据库故障时聊天接口抛 `UpstreamError(type_="database_error")` → 503

- [ ] **Step 4: 跑全量测试**

Run: `uv run --directory MemHelp-code python -m pytest -q -p no:cacheprovider`
Expected: ch01 原有用例 + 新增用例全绿（`store` 参数若已无用途，**保留**以兼容 ch01 测试，在其 docstring 注明 deprecated）

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/api.py MemHelp-code/app/main.py MemHelp-code/tests/test_api_tools.py
git commit -m "ch02: SSE 加 status 帧与 done.tools；/messages 读库；healthz 探数据库"
```

---

### Task 12: 测试数据与真实 MySQL 集成验证

**Files:**
- Create: `sql/ch02-seed.sql`
- Create: `MemHelp-code/tests/test_db_mysql.py`
- Modify: `MemHelp-code/scripts/demo.ps1` → 新增 `MemHelp-code/scripts/demo-ch02.ps1`

**Interfaces:**
- Consumes: Task 0 的 DDL 与 compose
- Produces: `faq` 10 条、1 个演示会话含工具轨迹、2 条工单

- [ ] **Step 1: 写 `sql/ch02-seed.sql`**

**必须包含验收 3 所需的语料设计**：含「运费」不含「邮费」的相关条目，且**另有**一条含「邮费」的无关条目。至少这些：

```sql
INSERT INTO faq (question, answer, category) VALUES
('退货政策是什么？', '签收后 7 天内无理由退货，商品需保持完好、附件齐全。', '售后'),
('退货运费谁承担？', '质量问题由平台承担运费；非质量问题的无理由退货由买家承担。', '售后'),
('满多少包邮？', '单笔订单满 99 元包邮（新疆、西藏除外）。', '物流'),
('发票邮费说明', '发票金额按实付金额开具，不含单独支付的加急邮费。', '发票'),
...;
```

- [ ] **Step 2: 起 MySQL 并灌数（需一次提权）**

Run: `docker compose -f MemHelp-code/docker-compose.yml up -d --wait`
若沙箱拒绝（命名管道），用 `danger-full-access` 重试同一命令。
Expected: `mewhelp-mysql` 状态 healthy。

- [ ] **Step 3: 写真实 MySQL 集成测试**

默认 skip，仅当 `MEMHELP_TEST_MYSQL=1` 时跑。它断言 `Base.metadata` 的每个模型列名/类型与 `information_schema.columns` 对齐：

```python
@pytest.mark.skipif(os.getenv("MEMHELP_TEST_MYSQL") != "1", reason="需要真实 MySQL")
def test_orm_columns_match_real_database():
    """防「单测全绿、真库 Unknown column」。"""
    from sqlalchemy import inspect
    from app.db.session import create_engine_from
    from app.db.models import Base
    engine = create_engine_from(Settings(...))
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        real = {c["name"] for c in insp.get_columns(table.name)}
        assert {c.name for c in table.columns} == real, table.name
```

Run: `$env:MEMHELP_TEST_MYSQL="1"; uv run --directory MemHelp-code python -m pytest tests/test_db_mysql.py -v`
Expected: PASS（**这是验证 DDL ↔ ORM 对齐的唯一手段**）

- [ ] **Step 4: 写演示脚本 `scripts/demo-ch02.ps1`**

`-Reset` 开关做 `docker compose down -v && up -d --wait`。脚本必须**UTF-8 with BOM**（`agent.md` 第 4.4 条：PowerShell 5.1 按 GBK 读 `.ps1`）。

- [ ] **Step 5: 提交**

```bash
git add sql/ch02-seed.sql MemHelp-code/tests/test_db_mysql.py MemHelp-code/scripts/demo-ch02.ps1
git commit -m "ch02: 测试数据 + 真实 MySQL 集成验证 + 演示脚本"
```

---

### Task 13: 评估集（三条验收标准的可复现验证）

**Files:**
- Create: `MemHelp-code/evals/tool_cases.jsonl`
- Create: `MemHelp-code/evals/run_tool_evals.py`
- Test: `MemHelp-code/tests/test_run_tool_evals.py`

**Interfaces:**
- Consumes: Task 9 的 `astream_chat_with_tools`、Task 7 的 `TOOL_NAMES`
- Produces: `run_tool_evals.py` 的退出码（0 全过 / 1 有未达标）

**说明**：本任务的「测试」是标注样例评估集 —— 三条验收标准全是**模型的判断行为**，无法用断言替代，必须跑真实上游。

- [ ] **Step 1: 写评估集**

`evals/tool_cases.jsonl` 每行 `{"id","message","expect_tools":[...],"expect_min_faq_hits":int,"note"}`。至少 12 条，**必须包含**：

```jsonl
{"id":"acc1","message":"订单 1001 的物流到哪了","expect_tools":["query_logistics","query_order"],"note":"验收1：必须选中物流或订单工具"}
{"id":"acc2","message":"退货政策是什么","expect_tools":["query_faq"],"expect_min_faq_hits":1,"note":"验收2"}
{"id":"acc3","message":"邮费是多少","expect_tools":["query_faq"],"expect_max_faq_hits":0,"note":"验收3：预期漏召回，钉住基线给 ch03"}
{"id":"neg1","message":"你好呀","expect_tools":[],"note":"不该调工具"}
{"id":"neg2","message":"帮我建个工单，杯子碎了","expect_tools":["create_ticket"],"note":"工单工具"}
```

- [ ] **Step 2: 写评估脚本**

打印每条的「期望 vs 实际」表格 + 汇总命中率；对 `expect_max_faq_hits: 0` 的条目，**命中 0 条才算通过**。退出码 1 表示有未达标。

- [ ] **Step 3: 跑评估集**

Run: `uv run --directory MemHelp-code python evals/run_tool_evals.py`
Expected: acc1 / acc2 / acc3 三条全部符合预期（acc3 的「预期漏召回」被断言为 0 命中）。

- [ ] **Step 4: 未达标时的处置规则**

按 ch01 既定规则：**先改 Prompt / 工具描述，不许改评估标准**。改完重跑，把每次调整记进留痕。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/evals/tool_cases.jsonl MemHelp-code/evals/run_tool_evals.py MemHelp-code/tests/test_run_tool_evals.py
git commit -m "ch02: 工具选择评估集（三条验收标准 + 漏召回基线）"
```

---

### Task 14: 收尾 —— 验收实测与留痕

**Files:**
- Modify: `MemHelp-code/dev-notes/ch02.md`
- Modify: `MemHelp-code/README.md`
- Create: `MemHelp-code/dev-notes/ch02-acceptance-output.txt`

- [ ] **Step 1: 端到端起服务**

Run: `docker compose -f MemHelp-code/docker-compose.yml up -d --wait`；`uv run --directory MemHelp-code uvicorn app.main:app --port 8000`
Expected: `/healthz` 返回 `{"status":"ok","database":"ok",...}`

- [ ] **Step 2: 逐条跑三条验收标准**

acc1、acc2 在浏览器 http://127.0.0.1:8000/chat/ 手动问，截图徽章；acc3 确认查不到并**明确记录这是预期**。把原始输出存进 `dev-notes/ch02-acceptance-output.txt`。

- [ ] **Step 3: 核对落库**

```sql
SELECT m.role, m.tool_calls, m.tool_call_id, LEFT(m.content,60)
FROM messages m JOIN conversations c ON c.id = m.conversation_id
WHERE c.session_id = '<刚才的 session_id>' ORDER BY m.id;
```

Expected: 依次 `user` / `assistant`(带 tool_calls) / `tool`(带 tool_call_id) / `assistant`。

- [ ] **Step 4: 追记留痕**

按四样格式补完 `dev-notes/ch02.md` 的每个阶段；**特别要写清**：验收 3 的漏召回是预期、第 1 次非流式调用带来的首 token 延迟、docker 命名管道的提权经过。

- [ ] **Step 5: 更新 `README.md`**

补 ch02 的运行说明（compose 起库、`.env` 新键、演示命令、`-Reset` 语义）。

- [ ] **Step 6: 提交并交付总结**

```bash
git add MemHelp-code/dev-notes/ MemHelp-code/README.md
git commit -m "ch02: 验收实测留痕 + README 更新"
```

---

## 附：自我审查记录

**Spec 覆盖对照**

| Spec 章节 | 任务 |
|---|---|
| 2.1 DDL 改动 | Task 0 |
| 2.2 测试数据 | Task 12 |
| 2.3 分层 | Task 2–8 |
| 2.4 配置 | Task 1 |
| 3.1 五个工具 | Task 5、6 |
| 3.2 基础设施（注册/校验/错误/超时重试） | Task 7、8 |
| 4.1 单轮编排 | Task 9 |
| 4.2 SSE 契约 | Task 11 |
| 4.3 落库 | Task 3、9、11 |
| 5 前端徽章 | Task 10 |
| 6 错误处理汇总 | Task 4、8、9、11 |
| 7 测试策略 | 每个任务的 Step 1–4；真实 MySQL = Task 12；评估集 = Task 13 |
| 8 交付物 | Task 0、12、13、14 |

**类型一致性**：`ToolResult` 在 Task 8 定义、Task 9/13 消费；`ToolEvent` 只在 Task 9 定义、Task 11 消费；`TOOL_LABELS` 在 Task 7 定义、Task 9/11 消费；`escape_like` 在 Task 3 定义、**只被 `FaqRepository.search` 调用**（工具层不 import 它，避免重复转义）。

**写计划时新发现、已回填进 spec 的两处**：
1. **PyMySQL 是同步驱动**，仓库层必须用 `anyio.to_thread.run_sync` 包住，否则阻塞事件循环 —— 已写进 Task 3 的 Interfaces。
2. **`LIKE` 通配符转义**是 Review Focus 第 5 条，spec 第 3.1 节只写了 `LIKE %kw%` 没提转义 —— 已回填进 Task 6/8 的测试。

**比例检查**：本计划约 700 行，spec 约 300 行。比例偏大但可接受 —— 大头是「接口签名 + 测试断言」（计划该定的东西），代码块里没有可直接抄的函数体（除 Task 0 的 compose 与 Step 3 的关键字面量，那些是 spec 定死的值）。
