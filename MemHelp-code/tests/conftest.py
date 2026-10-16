"""测试环境适配（DSH Windows 文件沙箱）。

沙箱行为：**pytest 在工作区/临时区里新建的目录，在进程退出后会被收回权限**，
之后连列举和删除都会报 ``PermissionError: [WinError 5]``（已用最小实验确认：
手工创建的目录完全正常，新建目录则被锁死）。因此：

- 测试**不依赖** ``tmp_path`` / ``tmp_path_factory``（它们每次都要新建目录），
  需要文件时写进已存在且可写的目录，用完即删；
- 这里把 pytest 两个会枚举目录的清理钩子换成不枚举的实现，避免它去碰
  上一次运行留下的已加锁目录。

这些只影响测试脚手架的临时目录管理，不影响任何被测行为。
"""

import shutil

import _pytest.pathlib as _pathlib
import _pytest.tmpdir as _tmpdir
import pytest

from app.config import Settings

# 会话开始时「扫描并保留最近 N 个基线临时目录」需要枚举目录；沙箱会拒绝，
# 且真的删除也没有必要（目录本来就指到临时区）。
_tmpdir.cleanup_numbered_dir = (
    lambda rootdir, keep=3, consider_lock_dead_if_created_before=None: None
)

# 用「整树删除（忽略错误）」替换「先 scandir 再逐个删」的实现。
_tmpdir.rm_rf = lambda path, *args, **kwargs: shutil.rmtree(path, ignore_errors=True)

# 会话结束时的死符号链接清理：本仓库不用符号链接，跳过目录枚举。
_pathlib.cleanup_dead_symlinks = lambda root: None


@pytest.fixture
def test_settings() -> Settings:
    """写死一份配置，让 HTTP 层测试完全离线（不读 .env、不发请求）。"""
    return Settings(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="fake-model",
        context_token_budget=2000,
    )


@pytest.fixture
def sqlite_session():
    """一个建好四张表的 SQLite 内存库会话。

    **为什么用 ``StaticPool``**：仓库层把同步 DB 调用交给工作线程跑
    （``anyio.to_thread.run_sync``，否则会阻塞事件循环，见 repositories.py）。
    而 SQLite 的 ``:memory:`` 每个连接都是一份**独立**的空库 —— 换线程就换连接，
    于是仓库层看到的是「表不存在」。``StaticPool`` 让所有线程复用同一个连接，
    测试与生产的行为差异就被限制在这一处连接池配置上。

    **ENUM 的处理**：模型用原生 ENUM（MySQL 语义）。SQLite 不支持原生 ENUM，
    SQLAlchemy 自动退化成 ``VARCHAR + CHECK`` 约束 —— 语义更强（非法值同样被拒），
    所以这一层测出来的约束行为与 MySQL 一致。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    import app.db.models  # noqa: F401 - 必须先 import 才会注册进 metadata
    from app.db.base import Base

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def fake_repositories(sqlite_session):
    """接到 SQLite 内存库上的**真仓库**，供 HTTP 层测试注入。

    为什么不用假仓库：HTTP 层要验的正是「聊天记录真的落到了库里、
    ``/messages`` 真的从库里读出来」——把仓库也打桩就测不到这条链路了。
    SQLite 让它在没有 MySQL 的机器上也能跑。
    """
    from app.db.repositories import Repositories

    return Repositories.from_session(sqlite_session)


@pytest.fixture(autouse=True)
def _offline_database(monkeypatch, sqlite_session):
    """让**整个套件**默认跑在 SQLite 上，而不是去连 MySQL。

    为什么放在 conftest 而不是逐个测试文件里：ch02 起 ``create_app`` 默认会连
    MySQL（正确行为 —— 生产就该这样）。但那样一来 ch01 的所有 HTTP 用例都会
    因为「连不上库」而红，而它们要验的是对话与错误映射，不是数据库。
    逐个改它们的辅助函数既啰嗦又容易漏，所以在唯一的入口 ``create_app`` 上包一层。

    **显式传参优先**：测试自己传了 ``repositories`` / ``db_ping`` 就用它的，
    本包装器只补缺省。这样 ch02 的用例仍可精确控制注入的内容
    （包括故意让 ``db_ping`` 返回 False 来验 503）。

    测试文件必须写 ``from app import main`` 再 ``main.create_app(...)``，
    **不能**写 ``from app.main import create_app`` —— 后者在 import 时就把名字绑定了，
    再 patch 模块属性对它无效。这一点很容易写成前者，所以在这里写明。
    """
    from app.api import create_app as _real_create_app

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("repositories", _make_repositories(sqlite_session))
        kwargs.setdefault("db_ping", lambda: True)
        return _real_create_app(*args, **kwargs)

    # monkeypatch 会在测试结束后自动还原（按属性名记录原值）。
    monkeypatch.setattr("app.api.create_app", _wrapped)
    monkeypatch.setattr("app.main.create_app", _wrapped)
    yield


def _make_repositories(session):
    from app.db.repositories import Repositories

    return Repositories.from_session(session)


@pytest.fixture(autouse=True)
def _isolate_process_singletons():
    """每个测试前后清掉进程级单例。

    ``get_session_store()`` 与 ``_default_chat_model()`` 都是 ``lru_cache``
    单例；测试若不显式注入就会共用同一份历史，出现「上一个测试的对话跑到这个
    测试里」的假失败。这里统一清掉，保证测试彼此独立。
    """
    from app.api import _default_chat_model
    from app.memory import get_session_store

    def clear() -> None:
        get_session_store()._sessions.clear()  # noqa: SLF001 - 测试清理内部状态
        _default_chat_model.cache_clear()

    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def _isolate_real_env_file(monkeypatch):
    """让整个测试套件与「宿主机有没有 .env」无关。

    为什么必须放在 conftest 而不是某个测试文件里：``Settings`` 的 ``env_file``
    指向项目根的真实 ``.env``。开发机上这个文件存在、CI 或干净克隆里不存在，
    于是**同一份测试在不同机器上结果不同** —— 实测有一个只在干净克隆里失败的
    用例（``/healthz`` 的配置检查依赖会读到宿主机的 ``.env``，读不到就抛
    ``ValidationError``，表现为整个用例报错而不是断言失败）。

    做法：清掉 ``OPENAI_*`` 环境变量（环境变量优先级高于 .env），并把 ``env_file``
    置空。需要真实配置的用例自己显式传 ``_env_file=`` 或注入
    ``create_app(settings=...)``。
    """
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "CONTEXT_TOKEN_BUDGET",
        "REQUEST_TIMEOUT",
        # ch02 新增：不清掉的话，宿主机设了 MYSQL_* 就会让套件结果依赖那台机器
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
        "TOOL_TIMEOUT_SECONDS",
        "TOOL_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        "app.config.Settings.model_config",
        {**Settings.model_config, "env_file": None},
    )

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
