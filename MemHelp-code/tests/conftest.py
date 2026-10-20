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
        # ch05：测试不往仓库写 JSONL；需要日志的用例显式传路径。
        turn_log_path="",
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
        # ch03：测试**绝不**加载 2.3GB 的嵌入模型，也不连 Milvus。
        kwargs.setdefault("retriever", EmptyRetriever())
        kwargs.setdefault("rag_warmup", False)
        # ch05：默认内存 checkpointer，不往仓库写 data/checkpoints.sqlite。
        from app.agent.checkpoint import memory_checkpointer

        kwargs.setdefault("checkpointer", memory_checkpointer())
        return _real_create_app(*args, **kwargs)

    # monkeypatch 会在测试结束后自动还原（按属性名记录原值）。
    monkeypatch.setattr("app.api.create_app", _wrapped)
    monkeypatch.setattr("app.main.create_app", _wrapped)
    yield


def _make_repositories(session):
    from app.db.repositories import Repositories

    return Repositories.from_session(session)


class EmptyRetriever:
    """测试默认注入的检索器：永远返回空列表。

    ch01/ch02 的对话与工具用例不关心检索；给它们一个**不连 Milvus、不加载模型**
    的空实现，既能让 ``query_faq`` 的接线成立，又不会拖慢套件。
    需要真实命中的用例自己显式注入 ``fake_retriever`` 或真检索器。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.context_calls: list[tuple[str, str | None]] = []

    async def search(self, query: str, top_k: int = 5) -> list:
        self.calls.append((query, top_k))
        return []

    async def answer_context(self, query: str, *, category: str | None = None):
        """ch04 的在线检索协议。

        刻意返回 ``refused=False`` + 空证据：这些用例要验的是对话与工具接线，
        不是拒答逻辑。若在这里返回 refused，每一轮都会往低置信度池写一行，
        把「没命中」和「拒答」两件事混进同一个测试关注点。
        """
        from tests.fakes import retrieval_outcome

        self.context_calls.append((query, category))
        return retrieval_outcome(query, (), refused=False)

    async def retrieve_only(self, query: str, *, category: str | None = None):
        """ch05 的知识类预检索协议：返回空证据、不拒答。

        这样这些用例走「知识类 → 闸失败 → 兜底」而不是「Agent 被调用」；
        需要真实证据的用例自己注入带 `retrieve_only` 的假检索器。
        """
        self.context_calls.append((query, category))
        return retrieval_outcome(query, (), refused=False)

    async def warmup(self) -> None:
        return None


@pytest.fixture
def empty_retriever() -> EmptyRetriever:
    return EmptyRetriever()


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
        # ch03：嵌入器是进程单例，不清掉会让「上一个测试加载的模型」跨用例存活
        from app.knowledge.embedder import reset_embedders

        reset_embedders()

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
        # ch03 新增：不清掉的话，宿主机设了 MILVUS_* / RAG_* 就会让套件结果依赖那台机器
        "MILVUS_URI",
        "MILVUS_COLLECTION",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "EMBEDDING_TIMEOUT",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_MAX_RETRIES",
        "RAG_TOP_K",
        "RAG_WARMUP",
        "CHUNK_MAX_CHARS",
        "CHUNK_OVERLAP",
        "QA_BATCH_SIZE",
        "QA_DEDUPE_THRESHOLD",
        # ch05 新增：不清掉的话，宿主机设了 CHECKPOINT_DB_PATH / AGENT_* 就会
        # 让 test_settings_have_ch05_defaults 之类的用例依赖那台机器。
        "CHECKPOINT_DB_PATH",
        "AGENT_MAX_STEPS",
        "AGENT_TOKEN_BUDGET",
        "TURN_LOG_PATH",
        # ch06 新增：宿主机设了 INTENT_* / COREF_* / QUERY_EXPANSION_* /
        # REFUND_ORDER_COUNT 时，配置默认值与图测试不能依赖那台机器。
        "INTENT_MODEL",
        "INTENT_CHEAP_MODEL",
        "INTENT_COST_MODE",
        "INTENT_CONFIDENCE_THRESHOLD",
        "INTENT_TIMEOUT",
        "COREF_ENABLED",
        "COREF_TIMEOUT",
        "COREF_HISTORY_MESSAGES",
        "QUERY_EXPANSION_ENABLED",
        "QUERY_EXPANSION_TIMEOUT",
        "QUERY_EXPANSION_MAX_QUERIES",
        "REFUND_ORDER_COUNT",
        # ch07 新增：宿主机设了这些键时，「默认窗口下不触发压缩」之类的用例
        # 会变成依赖那台机器。别名两个名字都要清。
        "MODEL_CONTEXT_WINDOW",
        "MAX_OUTPUT_TOKENS",
        "MAX_USER_INPUT_TOKENS",
        "TOOL_RESULT_MAX_TOKENS",
        "MAX_AGENT_STEPS",
        "RERANK_TOP_K",
        "SAFETY_MARGIN_RATIO",
        "SYSTEM_PROMPT_RESERVE_TOKENS",
        "EVIDENCE_TOKENS_PER_ITEM",
        "SUMMARY_RESERVE_TOKENS",
        "HISTORY_DESIRED_TURNS",
        "HISTORY_TOKENS_PER_TURN",
        "LAYER1_RATIO",
        "LAYER2_RATIO",
        "ASSISTANT_PREVIEW_CHARS",
        "SUMMARY_ENABLED",
        "SUMMARY_MODEL",
        "SUMMARY_TIMEOUT_SECONDS",
        "SUMMARY_MAX_TOKENS",
        "CJK_CHARS_PER_TOKEN",
        "OTHER_CHARS_PER_TOKEN",
        "MESSAGE_OVERHEAD_TOKENS",
        "APP_LOG_PATH",
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
