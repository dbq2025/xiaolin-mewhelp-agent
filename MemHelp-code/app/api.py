"""HTTP 层 —— 只负责 HTTP 语义：校验、错误码、SSE 编码。

用例逻辑在 ``app.chat`` / ``app.extract``；本模块不做业务判断。

## SSE 为什么用 ``StreamingResponse`` 而不是 ``fastapi.sse``

FastAPI 0.141.1 有原生 ``EventSourceResponse`` / ``ServerSentEvent``，但它要求
路由函数**本身**是异步生成器，且只认 ``ServerSentEvent``：yield 普通
``Response`` 会被当作数据 JSON 编码，yield 字符串会被编码成 ``data: "..."``。
因此「同一路径用 ``stream`` 开关切换 SSE 与普通 JSON」用它做不到。

拆成两条同路径路由也不行：Starlette 匹配到第一条路径+方法相同的路由就结束，
guard 里抛 ``HTTPException`` 只会直接返回 422，**不会回落到第二条**（四种组合都
实测过，见 ``dev-notes/ch01.md``）。

所以：``/api/chat/completions`` 用一条普通异步生成器路由 + ``StreamingResponse``，
``stream=true`` 时逐帧 yield SSE 文本，``stream=false`` 时 yield 一个 JSON 对象。
帧编码抽到 ``app.sse`` 并由 ``tests/test_sse.py`` 单独钉住。
"""

import hashlib
import inspect
import json
import logging
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack, asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import Engine
from sqlalchemy.exc import InterfaceError, OperationalError, PendingRollbackError

from app.agent.checkpoint import open_checkpointer
from app.agent.graph import build_agent_graph
from app.agent.persist import persist_user_only
from app.agent.runner import (
    astream_resume,
    astream_turn,
    complete_turn,
    pending_interrupt,
    pending_payload,
)
from app.agent.summarizer import ContextSummarizer
from app.config import PROJECT_ROOT, Settings, get_settings
from app.context.budget import compute_budget, format_budget_report
from app.db.repositories import Repositories
from app.db.session import create_engine_from, ping, session_factory
from app.errors import (
    UpstreamFault,
    classify_upstream_error_type,
    redact,
    upstream_error_message,
)
from app.extract import ExtractionError, extract_request
from app.knowledge.embedder import get_embedder
from app.knowledge.hybrid import HybridRetriever
from app.flywheel.service import FlywheelService
from app.knowledge.pipeline import RagPipeline
from app.knowledge.query import LlmRewriter, QueryUnderstander
from app.knowledge.rerank import get_reranker
from app.knowledge.sufficiency import SufficiencyJudge
from app.knowledge.retrieval import warmup as warmup_rag
from app.knowledge.vector_store import MilvusVectorStore
from app.observability import tracing
from app.llm import get_chat_model
from app.memory import InMemorySessionStore, get_session_store, new_session_id
from app.rag_eval import (
    EvalAlreadyRunning,
    EvalRunner,
    EvalStartError,
    ReportInvalid,
    ReportNotFound,
    load_dashboard,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ContextStats,
    ConversationListResponse,
    ConversationSummaryOut,
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ErrorResponse,
    ExtractRequest,
    FaithCaseUpdateRequest,
    RefundSelectRequest,
    SessionMessagesResponse,
    TicketConfirmRequest,
)
from app.sse import SSE_MEDIA_TYPE, sse_frame
from app.tools.executor import execute_tool
from app.tools.knowledge import AnswerContextLike
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)


class UpstreamError(UpstreamFault):
    """上游（模型服务）不可用、拒绝请求，或本地配置缺失。

    类型与判定规则在 ``app.errors``；这里保留一个别名，让本模块的历史调用点
    继续可读。
    """

    def __init__(self, message: str, type_: str = "upstream_error") -> None:
        super().__init__(message, type_=type_)


# ---------------------------------------------------------------------------
# 错误响应助手
# ---------------------------------------------------------------------------


def _error(status_code: int, type_: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"type": type_, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _jsonable_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """把校验错误转成可 JSON 序列化的形式。

    pydantic 的错误上下文里可能带异常对象（例如 ``ValueError`` 实例），
    直接塞进 JSONResponse 会炸，所以只取稳定字段。

    ``input`` 也要脱敏：用户完全可能把密钥粘进某个体字段，而那会被回显在这里。
    """
    return [
        {
            "type": str(item.get("type", "")),
            "loc": [str(p) for p in item.get("loc", ())],
            "msg": redact(item.get("msg", "")),
            "input": redact(repr(item.get("input"))[:200]),
        }
        for item in exc.errors()
    ]


def _classify_upstream_error(exc: Exception) -> UpstreamError:
    """把上游异常包装成带类型的 ``UpstreamError``。

    分类规则集中在 ``app.errors``，这里只负责拼装消息。
    """
    type_ = classify_upstream_error_type(exc)
    return UpstreamError(upstream_error_message(exc, type_), type_=type_)


#: 数据库故障的异常类型。用元组导入而不是靠字符串名匹配 ——
#: SQLAlchemy 的异常层次是稳定的，而 ``classify_upstream_error_type`` 那套
#: 关键词表是为 **OpenAI SDK** 的异常名写的，判数据库错误会给出**错误答案**
#: （实测：``OperationalError(2003, "Can't connect...")`` 被它归成
#: ``upstream_unavailable``，于是提示运维「请检查 OPENAI_BASE_URL 和网络」，
#: 把排查方向指向了一个没坏的依赖）。
_DB_FAULT_TYPES: tuple[type[BaseException], ...] = (
    OperationalError,
    InterfaceError,
    PendingRollbackError,
)


def _is_database_fault(exc: BaseException) -> bool:
    """这个异常是不是数据库故障（沿异常链判断，兼容被包装的情况）。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _DB_FAULT_TYPES):
            return True
        current = current.__cause__ or current.__context__
    return False


def _as_upstream_error(exc: Exception) -> UpstreamFault:
    """把异常映射成本项目的故障类型。

    已经是本地定义的故障就直接沿用它的 ``type``，否则先判**数据库故障**，
    最后才落到上游分类。

    **数据库必须先判**：它是本服务自己的依赖，与「模型服务怎么了」是两类问题，
    对应不同的状态码与不同的排查方向。混在一起会让一条 502 + 「检查 OPENAI_*」
    的错误信息指向一个没坏的东西。

    数据库故障一律 503（服务没准备好）：库连不上时这个实例接不了任何真实请求。
    """
    if isinstance(exc, UpstreamFault):
        return exc
    if _is_database_fault(exc):
        # 不把驱动原文给客户端：它含库地址、端口、账号，甚至 [SQL: ...] 语句。
        # 这些细节留在服务端日志里（logger.exception），响应只给可判读的结论。
        logger.exception("数据库故障", exc_info=exc)
        return UpstreamError("数据库暂时不可用，请稍后重试。", type_="database_error")
    return _classify_upstream_error(exc)


# ---------------------------------------------------------------------------
# 依赖：全部经 app.state 注入，测试可完全离线
# ---------------------------------------------------------------------------


def get_store(request: Request) -> InMemorySessionStore:
    store = getattr(request.app.state, "session_store", None)
    return store if store is not None else get_session_store()


def _ensure_database(app: FastAPI, settings: Settings) -> tuple[object, Engine]:
    """取（或首次创建）**进程内共享**的 ``(sessionmaker, engine)``。

    与 ``app.state._database`` 同一份缓存：``/healthz``、每条聊天请求、后台摘要任务
    都必须拿到**同一个**引擎。三条路径各自建一个 = 三个连接池，而且都以为自己是
    唯一的那个（Review I3 的教训：修了一处漏了另一处）。

    放在模块级而不是嵌在 ``_database_for`` 里，是因为后台摘要任务没有 ``Request``
    可用 —— 它只有 ``app``。共用一个函数，这类「两处各建一个」就不会复发。
    """
    cached = getattr(app.state, "_database", None)
    if cached is not None:
        return cached
    engine = create_engine_from(settings)
    factory = session_factory(engine)
    database = (factory, engine)
    app.state._database = database  # noqa: SLF001 - 进程级共享
    return database


def _database_for(request: Request) -> tuple[object, Engine]:
    """取（或首次创建）**进程内共享**的 ``(sessionmaker, engine)``。

    ``create_engine`` 会建连接池，所以引擎**必须**进程内只建一次 ——
    每次新建等于把连接管理交给 GC，持续流量下表现为「MySQL 连接数缓慢涨到上限」，
    而且每次都要付一次完整的连接握手。

    **为什么把 engine 一起返回**：``/healthz`` 需要直接拿引擎去 ``SELECT 1``，
    而它不该自己去拆 ``sessionmaker`` 的内部结构（那依赖 SQLAlchemy 的实现细节）。
    显式存下引擎，两条路径都用同一份。

    **为什么单独抽出来**：``get_repositories`` 与 ``_ping_database`` 都要用它。
    最初只有前者用了缓存，后者每次 ``/healthz`` 都新建一个引擎且从不 dispose ——
    而 ``/healthz`` 不是只在启动时跑，聊天页**每次加载**都会探它（Review I3）。
    两条路径共用一个函数，这类「修了一处漏了另一处」就不会复发。

    :raises UpstreamError: 配置缺失/非法，或数据库配置本身不合法。
        转成规范的 ``configuration_error``（503），而不是把 pydantic 的报错
        裸抛到 ASGI 层。
    """
    cached = getattr(request.app.state, "_database", None)
    if cached is not None:
        return cached

    try:
        current = get_app_settings(request)
    except UpstreamFault:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            redact(f"配置缺失或非法，请检查 .env：{exc}"), type_="configuration_error"
        ) from exc

    try:
        return _ensure_database(request.app, current)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            redact(f"数据库配置非法：{exc}"), type_="configuration_error"
        ) from exc


async def get_repositories(request: Request):
    """取当次请求的数据仓库。**异步生成器依赖**，请求结束后关闭 Session。

    两段式：``app.state.repositories`` 有就直接用（测试注入，不关它 —— 那是调用方的）；
    否则为本次请求新建一个 ``Session``，并在 ``finally`` 里关掉。

    **为什么必须是生成器而不是普通函数**：普通函数每请求新建一个 ``Session``
    却没人关它，连接要等 GC 才回到池里 —— 持续流量下表现为「MySQL 连接数缓慢涨到
    上限」，而症状看起来像连接池配置问题，很难归因。
    （这一条是自查发现的：所有 HTTP 用例都注入了仓库，于是真实那条路径从未被跑到；
    ``tests/test_api_tools.py::test_real_dependency_path_closes_its_session`` 现在钉住它。）

    **为什么 Session 必须每请求一个**：它不是线程安全的，跨请求共享会让并发请求
    互相写进对方的会话（表现为间歇性的「聊天记录串台」）。仓库只是它的薄包装。

    **为什么生成器依赖不会在流式响应中途被关闭**：FastAPI 在**响应发送完毕之后**
    才执行 yield 之后的清理，所以 SSE 流期间 Session 一直可用。
    这一点由 ``tests/test_api_tools.py`` 的流式用例实际跑到（它们会落库）。
    """
    injected = getattr(request.app.state, "repositories", None)
    if injected is not None:
        # 注入的仓库归调用方管（测试夹具），这里不关它
        yield injected
        return

    session = _database_for(request)[0]()
    try:
        yield Repositories.from_session(session)
    finally:
        session.close()


@lru_cache(maxsize=4)
def _vector_store_for(uri: str, collection: str, dim: int) -> MilvusVectorStore:
    """进程内复用 Milvus 客户端。

    ``MilvusClient`` 自带 gRPC 连接池，每请求重建等于把连接管理交给 GC ——
    持续流量下表现为「Milvus 连接数缓慢上涨」，和 ch02 里 MySQL 引擎那条经验同源。
    """
    return MilvusVectorStore(uri, collection, dim)


#: ch04：Query 理解的缓存挂在 ``QueryUnderstander`` 实例上，所以实例必须**进程内复用**。
#: 每请求新建一个的话，spec §4 要的 512 条缓存永远不会命中 —— 每一句口语问法都要多一次
#: 改写模型调用（真机评审实测）。
_UNDERSTANDERS: dict[tuple[bool, float], QueryUnderstander] = {}


def _understander_for(settings: Settings, model: object) -> QueryUnderstander:
    """按「是否启用改写 + 超时」取一个进程内复用的理解器。"""
    key = (settings.rag_query_rewrite, settings.rag_rewrite_timeout)
    understander = _UNDERSTANDERS.get(key)
    if understander is None:
        understander = QueryUnderstander(
            LlmRewriter(model, timeout=settings.rag_rewrite_timeout),
            enabled=settings.rag_query_rewrite,
        )
        _UNDERSTANDERS[key] = understander
    return understander


def get_retriever(
    request: Request, repositories: Repositories = Depends(get_repositories)
) -> AnswerContextLike:
    """当次请求的在线检索管线（`RagPipeline`）。

    两段式（与 ``get_repositories`` 同一个套路）：``app.state.retriever`` 有就用
    （测试/演示注入）；否则用进程级单例的嵌入器 / Milvus 客户端 / 重排器，配上
    **当请求的** chunk 仓库 —— 仓库绑着本次的 DB Session，不能跨请求共享。

    ch04 起返回的是 ``RagPipeline``（理解 → 双路召回 → RRF → 重排 → 两道闸），
    ch03 的 ``KnowledgeRetriever`` 只保留给评估脚本做「纯向量」基线。
    """
    injected = getattr(request.app.state, "retriever", None)
    if injected is not None:
        return injected

    settings = get_app_settings(request)
    store = _vector_store_for(
        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
    )
    model = getattr(request.app.state, "chat_model", None) or get_chat_model(settings)
    understander = _understander_for(settings, model)
    retriever = HybridRetriever(
        embedder=get_embedder(settings),
        vector_store=store,
        reranker=get_reranker(settings),
        chunk_repo=repositories.knowledge,
        hybrid_top_k=settings.rag_hybrid_top_k,
        candidate_k=settings.rag_candidate_k,
        rerank_top_n=settings.rag_rerank_top_n,
    )
    return RagPipeline(
        understander=understander,
        retriever=retriever,
        sufficiency=SufficiencyJudge(model, timeout=settings.rag_rewrite_timeout),
        min_score=settings.rag_min_rerank_score,
        top_n=settings.rag_rerank_top_n,
    )


def get_agent_graph(request: Request):
    """取进程内编译好的 LangGraph 图。

    图在 lifespan 里编译（生产用 SQLite saver，测试注入 InMemorySaver）。
    没有图时返回 503 而不是 500：这是「实例还没准备好」，不是代码缺陷。
    """
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise UpstreamError(
            "Agent 图尚未初始化：请检查 CHECKPOINT_DB_PATH 与启动日志。",
            type_="configuration_error",
        )
    return graph


async def _ping_database(request: Request) -> bool:
    """探一次数据库。**永不抛异常**（``ping()`` 自己也吞掉一切）。

    没注入 ``db_ping`` 时用真实探测。探测前若连配置都读不到，直接判定不可用 ——
    ``/healthz`` 的职责是「能不能服务」，配置缺失同样是不能服务，
    只是原因不同（由 ``status: misconfigured`` 表达）。

    **引擎复用**（Review I3）：走 ``_database_for`` 拿**缓存的**引擎，不再每次新建。
    修复前每次 ``/healthz`` 都新建一个连接池且从不 dispose，而聊天页每次加载
    都会探它 —— 那是一条实打实的连接泄漏 + 每次一次完整的连接握手。

    **驱动异常的原文只进服务端日志，绝不进响应体**。``/healthz`` 通常是未认证端点，
    而 ``(2003, "Can't connect to MySQL server on '127.0.0.1' (111)")``
    与 ``using password: YES`` 这类文本会泄露库的内部地址、端口与账号
    —— ``redact()`` 只遮密钥形态，遮不住这些。
    """
    injected = getattr(request.app.state, "db_ping", None)
    if injected is not None:
        return bool(await _maybe_await(injected()))

    try:
        _, engine = _database_for(request)
    except Exception:  # noqa: BLE001 - 配置缺失已在 healthz 主流程里单独报告
        return False

    alive = await ping(engine)
    if not alive:
        # 原因留给运维看日志；响应体只给一句通用说明。
        current = getattr(request.app.state, "settings", None)
        logger.warning(
            "数据库探测失败：无法连接 %s:%s",
            getattr(current, "mysql_host", "?"),
            getattr(current, "mysql_port", "?"),
        )
    return alive


async def _maybe_await(value: Any) -> Any:
    """注入的 ``db_ping`` 可能是同步函数也可能是协程函数。"""
    if inspect.isawaitable(value):
        return await value
    return value


@lru_cache
def _default_chat_model() -> BaseChatModel:
    """默认模型只构造一次，避免每个请求都新建客户端。"""
    return get_chat_model()


def get_model(request: Request) -> BaseChatModel:
    model = getattr(request.app.state, "chat_model", None)
    if model is not None:
        return model
    try:
        return _default_chat_model()
    except Exception as exc:  # noqa: BLE001 - 配置缺失是预期内的运维状态
        # 与 get_app_settings 同样的要求：配置缺失必须变成规范的 503 外壳，
        # 而不是把 pydantic 的 ValidationError 裸抛到 ASGI 层 —— 那样客户端拿到
        # 的是 "Exception in ASGI application"，根本没法解析。
        raise UpstreamError(
            redact(f"配置缺失或非法，请检查 .env：{exc}"),
            type_="configuration_error",
        ) from exc


#: 意图模型客户端进程内复用：与 engine/embedder/Milvus 的教训同源，
#: 每请求新建会让连接池随请求数增长。键含 base_url/key/model/timeout。
_INTENT_MODEL_CACHE: dict[tuple[str, str, str, float], BaseChatModel] = {}

#: 同一 session 的退款续跑互斥：两个 select 并发会让同一 thread 跑两遍。
_RESUME_INFLIGHT: set[str] = set()


def _cached_intent_model(settings: Settings, model_name: str) -> BaseChatModel:
    key = (
        settings.openai_base_url,
        settings.openai_api_key,
        model_name,
        settings.request_timeout,
    )
    model = _INTENT_MODEL_CACHE.get(key)
    if model is None:
        model = get_chat_model(settings, model_name=model_name)
        _INTENT_MODEL_CACHE[key] = model
    return model


def get_intent_model(request: Request) -> BaseChatModel:
    """意图主模型；`INTENT_MODEL` 为空时复用聊天主模型。"""
    injected = getattr(request.app.state, "intent_model", None)
    if injected is not None:
        return injected
    settings = get_app_settings(request)
    if not settings.intent_model:
        return get_model(request)
    try:
        return _cached_intent_model(settings, settings.intent_model)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            redact(f"意图模型配置非法，请检查 INTENT_MODEL：{exc}"),
            type_="configuration_error",
        ) from exc


def get_intent_cheap_model(request: Request) -> BaseChatModel | None:
    """成本档小模型；没配返回 None，成本档自动不启用。"""
    injected = getattr(request.app.state, "intent_cheap_model", None)
    if injected is not None:
        return injected
    settings = get_app_settings(request)
    if not settings.intent_cheap_model:
        return None
    try:
        return _cached_intent_model(settings, settings.intent_cheap_model)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            redact(f"意图小模型配置非法，请检查 INTENT_CHEAP_MODEL：{exc}"),
            type_="configuration_error",
        ) from exc


def get_app_settings(request: Request) -> Settings:
    injected = getattr(request.app.state, "settings", None)
    if injected is not None:
        return injected
    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001 - 配置缺失是可预期的运维状态
        # 必须走 UpstreamError 而不是 HTTPException：后者的响应体是
        # FastAPI 默认的 {"detail": ...}，会绕过本项目统一的 {"error": {...}}
        # 外壳，导致同一个接口在配置缺失时给出不同形状的错误体。
        raise UpstreamError(
            redact(f"配置缺失或非法，请检查 .env：{exc}"),
            type_="configuration_error",
        ) from exc


def _resolve_session_id(requested: str | None) -> str:
    return requested or new_session_id()


# ---------------------------------------------------------------------------
# 业务无关的响应体构造
# ---------------------------------------------------------------------------


def _session_messages_payload(
    session_id: str, store: InMemorySessionStore
) -> SessionMessagesResponse:
    def role_of(message) -> str:
        if isinstance(message, HumanMessage):
            return "human"
        if isinstance(message, AIMessage):
            return "ai"
        return "other"

    def content_of(message) -> str:
        if isinstance(message.content, str):
            return message.content
        return json.dumps(message.content, ensure_ascii=False)

    history = store.get(session_id)
    return SessionMessagesResponse(
        session_id=session_id,
        messages=[{"role": role_of(m), "content": content_of(m)} for m in history],
    )


#: 落库角色 → 对外角色名。沿用 ch01 的命名（``human`` / ``ai``），
#: 而不是把库里的 ``user`` / ``assistant`` 直接透出去 —— 前端已经按前者写了。
_ROLE_TO_API: dict[str, str] = {"user": "human", "assistant": "ai", "tool": "tool"}


async def _db_session_messages_payload(
    session_id: str, repositories: Repositories
) -> SessionMessagesResponse:
    """从 ``messages`` 表回读历史。

    ``tools`` 字段**只在那一轮真的调过工具时出现**（省略而不是填 null）：
    前端只需 ``if (msg.tools)`` 一处判断，不必区分 ``null`` / ``[]`` / 缺字段。

    工具轨迹从 ``assistant`` 行的 ``tool_calls`` JSON 还原，
    成败与耗时从紧随其后的 ``tool`` 行的正文里解析 —— 这两行是我们自己写的，
    格式固定（见 ``chat_tools._persist``），所以解析是可靠的。
    """
    conversation_id = await repositories.messages.find_conversation_id(session_id)
    if conversation_id is None:
        return SessionMessagesResponse(session_id=session_id, messages=[])

    rows = await repositories.messages.list_for_conversation(conversation_id)
    payload: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        item: dict[str, Any] = {
            "role": _ROLE_TO_API.get(row.role, row.role),
            "content": row.content or "",
        }

        traces = getattr(row, "tool_traces", None)
        if row.role == "assistant" and (row.tool_calls or traces is not None):
            # 优先用 ``tool_traces``（我们自己写的展示用数据，含耗时与成败）。
            # 它是 ch02 迁移新增的列，老行可能没有 —— 那样退化成从 ``tool_calls``
            # 与紧随其后的 ``tool`` 行推断，至少能还原 name/args/ok。
            #
            # ch05：`tool_traces == []` 是**有意义**的（只有 suggest_actions 的控制面
            # 调用，不该显示徽章），不能再用 truthiness 回退去猜。最终答案行也可能带
            # 整轮的工具轨迹（ch05 把徽章从申请单行挪到了答案行）。
            item["tools"] = traces if traces is not None else _tool_traces(row.tool_calls, rows, index)

        # ch04：引用快照。与 ``tools`` 同理 —— 只在有引用时出现，
        # 前端 `if (msg.citations)` 一处判断就够。
        citations = getattr(row, "citations", None)
        if row.role == "assistant" and citations:
            item["citations"] = citations

        # ch05：建议动作快照。刷新后按钮还能渲染，就靠它。
        actions = getattr(row, "actions", None)
        if row.role == "assistant" and actions:
            item["actions"] = actions

        payload.append(item)

    return SessionMessagesResponse(session_id=session_id, messages=payload)


def _faith_case_payload(row: Any) -> dict[str, Any]:
    """把台账行转成对外 JSON。

    时间统一 ISO 字符串：让前端直接 ``new Date(...)``，不必依赖某个时区格式。
    """
    return {
        "eval_id": row.eval_id,
        "bucket": row.bucket,
        "query": row.query,
        "strategy": row.strategy,
        "answer": row.answer,
        "reason": row.reason,
        "citations": row.citations,
        "judge_model": row.judge_model,
        "status": row.status,
        "seen_count": row.seen_count,
        "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "resolution": row.resolution,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }


def _tool_traces(
    tool_calls: Any, rows: list[Any], assistant_index: int
) -> list[dict[str, Any]]:
    """把一行 ``tool_calls`` 与它后面那条 ``tool`` 行的结果拼成徽章数据。"""
    calls = tool_calls if isinstance(tool_calls, list) else []
    traces: list[dict[str, Any]] = []

    for offset, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        trace: dict[str, Any] = {
            "name": call.get("name", ""),
            "args": call.get("args", {}),
            "ok": True,
        }
        follow = assistant_index + 1 + offset
        if follow < len(rows) and rows[follow].role == "tool":
            try:
                parsed = json.loads(rows[follow].content or "{}")
            except (ValueError, TypeError):
                parsed = {}
            trace["ok"] = bool(parsed.get("ok", True))
        traces.append(trace)

    return traces


def _sse_event(event: dict[str, Any]) -> str | None:
    """把 runner 事件映射成 SSE 帧；未知事件丢弃。"""
    kind = event.get("type")
    if kind == "status":
        return sse_frame("status", event)
    if kind == "coref":
        return sse_frame(
            "coref",
            {
                "original": event.get("original") or "",
                "resolved": event.get("resolved") or "",
                "changed": bool(event.get("changed")),
            },
        )
    if kind == "intent":
        return sse_frame(
            "intent",
            {
                "intent": event.get("intent"),
                "confidence": event.get("confidence"),
                "source": event.get("source"),
            },
        )
    if kind == "order_picker":
        return sse_frame(
            "order_picker",
            {
                "turn_id": event.get("turn_id"),
                "orders": event.get("orders") or [],
            },
        )
    if kind == "awaiting_input":
        return sse_frame("awaiting_input", event.get("payload") or {})
    if kind == "token":
        return sse_frame("token", {"delta": event.get("delta") or ""})
    if kind == "tools":
        return sse_frame("tools", {"tools": event.get("tools") or []})
    if kind == "tool_result":
        return sse_frame(
            "tool_result",
            {
                "call_id": event.get("call_id") or "",
                "tool": event.get("tool") or {},
            },
        )
    if kind == "citations":
        return sse_frame("citations", {"citations": event.get("citations") or []})
    if kind == "done":
        return sse_frame("done", event.get("payload") or {})
    return None


async def _chat_stream_body(
    req: ChatRequest,
    settings: Settings,
    model: BaseChatModel,
    store: InMemorySessionStore,
    repositories: Repositories,
    session_id: str,
    graph: Any,
    retriever: AnswerContextLike | None = None,
    intent_model: BaseChatModel | None = None,
    intent_cheap_model: BaseChatModel | None = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
) -> AsyncIterable[str]:
    """SSE 响应体：``meta`` → ``status``? → ``token``* → ``tools``? →
    ``citations``? → ``done``。

    中途失败补 ``error`` 再收尾（沿用 ch01/ch02）。失败时只落用户轮 ——
    不能让半截回答进历史，否则下一轮模型会以为它已经答过了。
    """
    yield sse_frame("meta", {"session_id": session_id, "model": settings.openai_model})

    try:
        async for event in astream_turn(
            graph,
            session_id=session_id,
            message=req.message,
            model=model,
            intent_model=intent_model,
            intent_cheap_model=intent_cheap_model,
            repositories=repositories,
            retriever=retriever,
            settings=settings,
            summarizer=summarizer,
            mcp_provider=mcp_provider,
            flywheel=FlywheelService(repositories, model, settings),
        ):
            frame = _sse_event(event)
            if frame is not None:
                yield frame
    except Exception as exc:  # noqa: BLE001 - 必须转成事件，不能让连接裸断
        err = _as_upstream_error(exc)
        try:
            await persist_user_only(repositories, session_id, req.message)
        except Exception:  # noqa: BLE001 - 落库失败不该盖住原始故障
            logger.exception("上游失败后落库用户轮次也失败了（已忽略，保留原始异常）")
        yield sse_frame(
            "error", {"type": err.type, "message": redact(str(err)), "session_id": session_id}
        )
        return


async def _delete_thread(graph: Any, session_id: str) -> None:
    """续跑失败后清掉未收敛的 checkpoint，避免 session 永久 409。"""
    saver = getattr(graph, "checkpointer", None)
    delete = getattr(saver, "adelete_thread", None)
    if delete is None:
        logger.warning("checkpointer 不支持 adelete_thread，无法清理 session=%s", session_id)
        return
    try:
        await delete(session_id)
    except Exception:  # noqa: BLE001 - 清理失败只告警，错误帧已经要发了
        logger.exception("清理未收敛 checkpoint 失败（session=%s）", session_id)


async def _refund_resume_body(
    req: RefundSelectRequest,
    settings: Settings,
    model: BaseChatModel,
    intent_model: BaseChatModel | None,
    intent_cheap_model: BaseChatModel | None,
    repositories: Repositories,
    graph: Any,
    retriever: AnswerContextLike | None = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
    *,
    release_session: str | None = None,
    recover_message: str = "",
    persist_on_error: bool = False,
) -> AsyncIterable[str]:
    """订单点选后的续跑流；不重复落用户轮（暂停轮仍只有 checkpoint）。"""
    yield sse_frame("meta", {"session_id": req.session_id, "model": settings.openai_model})
    try:
        async for event in astream_resume(
            graph,
            session_id=req.session_id,
            order_id=req.order_id,
            model=model,
            intent_model=intent_model,
            intent_cheap_model=intent_cheap_model,
            repositories=repositories,
            retriever=retriever,
            settings=settings,
            summarizer=summarizer,
            mcp_provider=mcp_provider,
            flywheel=FlywheelService(repositories, model, settings),
        ):
            frame = _sse_event(event)
            if frame is not None:
                yield frame
    except Exception as exc:  # noqa: BLE001 - 必须转成事件，不能让连接裸断
        err = _as_upstream_error(exc)
        if persist_on_error and recover_message:
            try:
                await persist_user_only(
                    repositories, req.session_id, recover_message
                )
            except Exception:  # noqa: BLE001 - 落库失败不该盖住原始故障
                logger.exception("续跑失败后落用户轮次也失败（已忽略）")
        await _delete_thread(graph, req.session_id)
        yield sse_frame(
            "error",
            {
                "type": err.type,
                "message": redact(str(err)),
                "session_id": req.session_id,
                "recovered": True,
            },
        )
    finally:
        if release_session is not None:
            _RESUME_INFLIGHT.discard(release_session)


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


async def _ticket_resume_body(
    req: TicketConfirmRequest,
    settings: Settings,
    model: BaseChatModel,
    intent_model: BaseChatModel | None,
    intent_cheap_model: BaseChatModel | None,
    repositories: Repositories,
    graph: Any,
    retriever: AnswerContextLike | None = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
    *,
    release_session: str | None = None,
) -> AsyncIterable[str]:
    """工单预览的确认/取消续跑；不落用户轮，结果由图内节点写审计。"""
    yield sse_frame("meta", {"session_id": req.session_id, "model": settings.openai_model})
    try:
        async for event in astream_resume(
            graph,
            session_id=req.session_id,
            resume_value={"confirmed": bool(req.confirmed)},
            model=model,
            intent_model=intent_model,
            intent_cheap_model=intent_cheap_model,
            repositories=repositories,
            retriever=retriever,
            settings=settings,
            summarizer=summarizer,
            mcp_provider=mcp_provider,
            flywheel=FlywheelService(repositories, model, settings),
        ):
            frame = _sse_event(event)
            if frame is not None:
                yield frame
    except Exception as exc:  # noqa: BLE001 - 必须转成事件，不能让连接裸断
        err = _as_upstream_error(exc)
        await _delete_thread(graph, req.session_id)
        yield sse_frame(
            "error",
            {
                "type": err.type,
                "message": redact(str(err)),
                "session_id": req.session_id,
                "recovered": True,
            },
        )
    finally:
        if release_session is not None:
            _RESUME_INFLIGHT.discard(release_session)


def _check_context_budget(settings: Settings) -> None:
    """启动自检：把倒推出来的预算打一行，装不下就打 ERROR（**不阻断**）。

    为什么要在启动时算一遍：预算公式里有七个固定开销项，任何一项配错
    （比如 ``MAX_OUTPUT_TOKENS`` 比窗口还大、``MAX_AGENT_STEPS`` 填了 12）
    都会让历史预算归零，而症状是「聊到第三轮就忘事」—— 那时候已经很难
    反推到某一个环境变量上。启动日志是唯一能把它一次说清的位置。

    ``compute_budget`` 本身不抛；这里的 ``try`` 只兜「配置对象缺属性」这类
    意外，因为启动阶段不该为一句日志而失败。
    """
    try:
        report = compute_budget(settings)
    except Exception:  # noqa: BLE001 - 自检失败不该阻断启动
        logger.warning("ch07 上下文预算自检失败（已忽略）", exc_info=True)
        return
    if report.ok:
        logger.info("ch07 上下文预算 %s", format_budget_report(report))
    else:
        logger.error("%s", report.reason)


def _attach_summarizer(app: FastAPI, settings: Settings) -> None:
    """把后台摘要器挂到 ``app.state.summarizer``，供图依赖读取。

    **每次开一个新 Session**（``repositories_factory``）：摘要任务跑在独立的
    ``asyncio.Task`` 里，请求那个 Session 在响应结束时就被关了 —— 借用它会得到
    ``This session is provisioning a new connection; concurrent operations are not permitted``
    这类难查的错。

    没有数据库配置（``configuration_error``）时**不挂**：宁可让层 2 只记账不压缩，
    也不要为了摘要去启动一个连不上的连接池。
    """
    if getattr(app.state, "summarizer", None) is not None:
        return
    if not getattr(settings, "summary_enabled", True):
        logger.info("ch07 后台摘要已关闭（SUMMARY_ENABLED=false）")
        return

    factory_holder: dict[str, Any] = {}

    def _factory() -> Repositories:
        factory = factory_holder.get("factory")
        if factory is None:
            raise RuntimeError("摘要任务拿不到数据库会话工厂：启动时未能建立连接池")
        return Repositories.from_session(factory())

    try:
        _factory_of, _engine = _ensure_database(app, settings)
        factory_holder["factory"] = _factory_of
    except Exception:  # noqa: BLE001 - 摘要装不上就只记账，聊天不受影响
        logger.warning("ch07 后台摘要器未启用（数据库配置不可用）", exc_info=True)
        return

    try:
        summary_model = (
            get_chat_model(settings, settings.summary_model)
            if str(getattr(settings, "summary_model", "") or "")
            else app.state.chat_model
        )
        if summary_model is None:
            summary_model = get_chat_model(settings)
        app.state.summarizer = ContextSummarizer(
            model=summary_model,
            repositories_factory=_factory,
            settings=settings,
        )
        logger.info("ch07 后台摘要器已就绪（model=%s）", getattr(settings, "summary_model", "") or settings.openai_model)
    except Exception:  # noqa: BLE001 - 同上，失败只降级
        logger.warning("ch07 后台摘要器初始化失败（层 2 将只记账不压缩）", exc_info=True)


def _attach_mcp_provider(app: FastAPI, settings: Settings) -> None:
    """按配置建 MCP 工具发现器，挂到 ``app.state.mcp_provider``。

    **建了不连**：构造只解析配置，真正的发现发生在每轮对话前的
    ``ToolRegistry.refresh_mcp()``。MCP Server 没起时启动不受影响，聊天退回
    只用内置工具 —— 外部系统可用性不该决定客服系统能不能启动。
    """
    if getattr(app.state, "mcp_provider", None) is not None:
        return
    if not getattr(settings, "mcp_enabled", True):
        logger.info("ch08 MCP 接入已关闭（MCP_ENABLED=false）")
        return
    try:
        from app.tools.mcp_client import (
            McpToolProvider,
            connections_from_settings,
            readonly_servers_from_settings,
        )

        connections = connections_from_settings(settings)
        if not connections:
            return
        app.state.mcp_provider = McpToolProvider(
            connections,
            read_only_servers=readonly_servers_from_settings(settings),
            timeout=float(getattr(settings, "mcp_timeout_seconds", 3.0)),
        )
        logger.info("ch08 MCP 工具发现器已就绪（servers=%s）", ",".join(sorted(connections)))
    except Exception:  # noqa: BLE001 - 接不上 MCP 不影响内置工具与聊天
        logger.warning("ch08 MCP 工具发现器初始化失败（仅使用内置工具）", exc_info=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动预热（ch03）：发一次真实嵌入请求 + 确保 Milvus 集合存在。

    **失败只告警不崩**：Milvus 没起时应用仍要能启动（聊天与另外四个工具不依赖它），
    查询时再由 executor 把异常分类成 ``ok=false``。启动即崩会把「向量库没起」
    升级成「客服系统整体不可用」，那是错误的严重度。

    ``rag_warmup`` 为 ``None`` 时读配置（``RAG_WARMUP``）；显式传 ``True/False``
    时以调用方为准 —— 测试靠这个参数把整套用例与网络隔离。

    被注入的检索器（测试/演示）若自带 ``warmup``，优先预热它 —— 否则会去按配置
    构造一个谁都不用的默认检索器。
    """
    stack = AsyncExitStack()
    try:
        # ch07：预算自检 + 后台摘要器。
        #
        # 自检放在最前面：它只读配置，不依赖图与数据库，而且是「这个实例到底
        # 能不能装下上下文」的唯一一处喊话机会。**绝不阻断启动** —— 装不下是
        # 容量问题，不是缺陷，把服务关掉只会让人连日志都看不到。
        current_settings = None
        try:
            current_settings = app.state.settings or get_settings()
        except Exception:  # noqa: BLE001 - 配置缺失时聊天会 503，不该启动即崩
            current_settings = None
        if current_settings is not None:
            _check_context_budget(current_settings)
            _attach_summarizer(app, current_settings)
            _attach_mcp_provider(app, current_settings)
            # ch09：必须在编译图之前配置，build_agent_graph 才会把 callback 挂上。
            tracing.configure(current_settings)

        # ch05：编译图。生产开 SQLite checkpointer；测试注入 InMemorySaver。
        if getattr(app.state, "agent_graph", None) is None:
            current = current_settings
            if current is not None:
                try:
                    if getattr(app.state, "checkpointer", None) is not None:
                        app.state.agent_graph = build_agent_graph(
                            checkpointer=app.state.checkpointer
                        )
                    else:
                        saver = await stack.enter_async_context(
                            open_checkpointer(current.checkpoint_db_path)
                        )
                        app.state.agent_graph = build_agent_graph(checkpointer=saver)
                    logger.info("ch05 Agent 图已编译（checkpointer=%s）", current.checkpoint_db_path)
                except Exception:  # noqa: BLE001 - 图起不来时聊天 503，其它接口仍可用
                    logger.warning("ch05 Agent 图初始化失败，聊天接口将返回 503", exc_info=True)

        warmup_enabled = getattr(app.state, "rag_warmup", None)
        if warmup_enabled is None:
            try:
                warmup_enabled = (app.state.settings or get_settings()).rag_warmup
            except Exception:  # noqa: BLE001 - 配置缺失时不该在启动阶段崩
                warmup_enabled = False

        if warmup_enabled:
            try:
                injected = getattr(app.state, "retriever", None)
                if injected is not None and hasattr(injected, "warmup"):
                    await injected.warmup()
                else:
                    settings = app.state.settings or get_settings()
                    store = _vector_store_for(
                        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
                    )
                    await warmup_rag(get_embedder(settings), store)
                logger.info("RAG 预热完成")
            except Exception:  # noqa: BLE001 - 预热失败不阻断启动
                logger.warning("RAG 预热失败：检索将在首个查询时按需初始化", exc_info=True)

        yield
    finally:
        # ch09：先把 trace 刷出去，再关摘要器/嵌入客户端（它们可能还在打点）。
        try:
            tracing.shutdown()
        except Exception:  # noqa: BLE001 - 退出路径不该再抛
            logger.debug("关闭 Langfuse tracing 失败（已忽略）", exc_info=True)
        # ch07：先等后台摘要任务收敛。放在最前面 —— 它们还要写库，
        # 而后面几步会关掉需要的东西（checkpointer、嵌入客户端）。
        try:
            summarizer = getattr(app.state, "summarizer", None)
            if summarizer is not None and hasattr(summarizer, "aclose"):
                await summarizer.aclose()
        except Exception:  # noqa: BLE001 - 退出路径不该再抛
            logger.debug("关闭后台摘要器失败（已忽略）", exc_info=True)
        # 评估重跑是子进程；应用退出时先收掉它，避免留下孤儿进程。
        try:
            runner = getattr(app.state, "rag_eval_runner", None)
            if runner is not None and hasattr(runner, "aclose"):
                await runner.aclose()
        except Exception:  # noqa: BLE001 - 退出路径不该再抛
            logger.debug("关闭评估重跑任务失败（已忽略）", exc_info=True)
        # 关掉进程单例持有的 httpx 连接（CLI 自己会关；Web 进程在退出时关一次）。
        try:
            settings = app.state.settings or get_settings()
            await get_embedder(settings).aclose()
        except Exception:  # noqa: BLE001 - 退出路径不该再抛
            logger.debug("关闭嵌入客户端失败（已忽略）", exc_info=True)
        # 关闭 SQLite checkpointer（AsyncSqliteSaver 的上下文退出）。
        try:
            await stack.aclose()
        except Exception:  # noqa: BLE001 - 退出路径不该再抛
            logger.debug("关闭 Agent checkpointer 失败（已忽略）", exc_info=True)


def create_app(
    model: BaseChatModel | None = None,
    store: InMemorySessionStore | None = None,
    settings: Settings | None = None,
    repositories: Repositories | None = None,
    db_ping: Any = None,
    retriever: AnswerContextLike | None = None,
    rag_warmup: bool | None = None,
    rag_eval_runner: Any = None,
    checkpointer: Any = None,
    agent_graph: Any = None,
    intent_model: BaseChatModel | None = None,
    intent_cheap_model: BaseChatModel | None = None,
) -> FastAPI:
    """构造应用。

    ``model`` / ``store`` / ``settings`` / ``repositories`` / ``db_ping`` 都可注入：
    测试里传假模型、假存储、写死的配置，以及接到 SQLite 上的真仓库，就能在
    **没有 .env、不连 MySQL、不发任何网络请求**的前提下跑完整 HTTP 测试。

    ``store`` 从 ch02 起**只在非流式路径与历史回读的降级实现里**还被用到 ——
    正式的聊天记录读写已经改走 ``repositories``（见 ``dev-notes/ch02.md``）。
    保留这个参数是为了不破坏 ch01 的调用方与测试。

    ``db_ping`` 是一个**探测函数**（同步或异步皆可），不是探测结果：
    健康检查每次执行都要拿当下的状态，缓存一个布尔值会让数据库恢复之后
    仍然一直报不健康。

    ``retriever``：向量检索器，``query_faq`` 依赖它。``None`` 时按配置构造
    （进程级单例的嵌入器与 Milvus 客户端）。

    ``rag_warmup``：启动时预热嵌入服务与 Milvus 集合。``None`` 表示读配置
    （``RAG_WARMUP``，生产默认开启）；显式 ``True/False`` 覆盖配置。
    预热会**发一次真实嵌入请求**，所以密钥/模型名配错能在启动阶段就喊出来。
    测试必须传 ``False``（``conftest`` 已统一处理），否则整套测试会去连外网。

    ``rag_eval_runner``：评估看板的后台重跑器。默认是 ``EvalRunner``；测试可注入
    一个假 runner，避免真的拉起评估子进程。
    """
    app = FastAPI(title="MewHelp 智能客服 API", version="0.3.0", lifespan=_lifespan)
    app.state.chat_model = model
    app.state.session_store = store
    # 显式赋值（即使为 None）：这是「测试用注入配置、不读宿主机 .env」的实际生效点。
    app.state.settings = settings
    app.state.repositories = repositories
    app.state.db_ping = db_ping
    app.state.retriever = retriever
    app.state.intent_model = intent_model
    app.state.intent_cheap_model = intent_cheap_model
    app.state.rag_warmup = rag_warmup
    # 评估重跑是进程内长任务；注入点让测试不用真的拉起评估子进程。
    app.state.rag_eval_runner = rag_eval_runner or EvalRunner()
    # ch05：图与 checkpointer。测试注入 InMemorySaver；生产在 lifespan 里开 SQLite。
    app.state.checkpointer = checkpointer
    app.state.agent_graph = agent_graph
    # ch07：后台摘要器由 lifespan 建（测试可注入）。显式置 None 而不是不赋值，
    # 是为了让 `getattr(state, "summarizer", None)` 与「属性根本不存在」可区分。
    app.state.summarizer = None
    # ch08：MCP 工具发现器。lifespan 按配置建；测试可用 ``None`` 表示不接 MCP。
    app.state.mcp_provider = None
    # 注入 checkpointer 时**立即编译**：TestClient 不进入 context manager 也不会跑
    # lifespan，而测试仍然要能聊天。生产路径（checkpointer=None）仍由 lifespan 管
    # AsyncSqliteSaver 的开关。
    if app.state.agent_graph is None and checkpointer is not None:
        app.state.agent_graph = build_agent_graph(checkpointer=checkpointer)

    # -- 异常处理 ---------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        return _error(
            422, "validation_error", "请求参数不合法。", _jsonable_validation_errors(exc)
        )

    @app.exception_handler(UpstreamFault)
    async def on_upstream_fault(request: Request, exc: UpstreamFault):
        # 注册在父类上，让 UpstreamEmptyResponse 之类的子类也被接住。
        #
        # 503 = 「服务没准备好，运维该去看配置或依赖」：本地的配置缺失/非法，
        # 以及**数据库不可用**（库连不上时这个实例接不了任何真实请求）。
        # 其余上游问题一律 502（我们的服务在，是它依赖的上游出错）。
        # **database_error 不加进来就会回落成 502**，把「我方依赖挂了」
        # 与「模型服务挂了」混为一谈，两者排查方向不同。
        needs_attention = exc.type in ("configuration_error", "database_error")
        status = 503 if needs_attention else 502
        return _error(status, exc.type, redact(str(exc)))

    @app.exception_handler(ExtractionError)
    async def on_extraction_error(request: Request, exc: ExtractionError):
        return _error(422, "extraction_failed", redact(str(exc)))

    # -- 健康检查 ---------------------------------------------------------

    @app.get("/healthz")
    async def healthz(request: Request):
        """健康检查。

        配置缺失返回 **503**（而不是 200）—— 否则编排器/负载均衡会认为实例健康，
        持续把流量打进来，而每个请求都会 503。缺配置的实例就该被判为未就绪。

        数据库不可用同样返回 **503**：聊天与工具都依赖它，一个连不上库的实例
        接不了任何真实请求。
        """
        try:
            current = app.state.settings if app.state.settings is not None else get_settings()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=503,
                content={
                    "status": "misconfigured",
                    "model": None,
                    "base_url_configured": False,
                    "context_token_budget": None,
                    "database": "unknown",
                    "hint": "请在项目根目录创建 .env（可参考 .env.example）。",
                    # 脱敏：pydantic 的校验错误会带出 input_value，里面有密钥前几位
                    "detail": redact(exc),
                },
            )

        database_ok = await _ping_database(request)
        body = {
            "status": "ok" if database_ok else "degraded",
            "model": current.openai_model,
            "base_url_configured": bool(current.openai_base_url),
            "context_token_budget": current.context_token_budget,
            "database": "ok" if database_ok else "unavailable",
        }
        if not database_ok:
            body["hint"] = (
                "数据库不可用：确认 docker compose up -d 已起，且 .env 的 MYSQL_* 与 "
                "docker-compose.yml 一致。"
            )
            return JSONResponse(status_code=503, content=body)
        return body

    # -- 对话（一个路径，stream 开关切换两种响应形态）----------------------

    @app.post(
        "/api/chat/completions",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "stream=true 时返回 text/event-stream（meta / token / done / error）；"
                    "stream=false 时返回 ChatResponse JSON。"
                )
            }
        },
    )
    async def chat_completions(
        req: ChatRequest,
        request: Request,
        settings_: Settings = Depends(get_app_settings),
        model_: BaseChatModel = Depends(get_model),
        intent_model_: BaseChatModel = Depends(get_intent_model),
        intent_cheap_model_: BaseChatModel | None = Depends(get_intent_cheap_model),
        store_: InMemorySessionStore = Depends(get_store),
        repositories_: Repositories = Depends(get_repositories),
        retriever_: AnswerContextLike = Depends(get_retriever),
        graph_: Any = Depends(get_agent_graph),
    ):
        """一个路径两种响应形态。

        ``stream=true`` 返回 SSE 流；``stream=false`` 在这里就把完整回复算出来，
        用普通 ``JSONResponse`` 返回 —— 这样上游错误发生在「响应开始之前」，
        能被上面的 ``UpstreamError`` 处理器映射成 502/503，而不是变成一个
        已经开始写响应体、只能中断的流。
        """
        session_id = _resolve_session_id(req.session_id)

        if req.stream:
            return StreamingResponse(
                _chat_stream_body(
                    req,
                    settings_,
                    model_,
                    store_,
                    repositories_,
                    session_id,
                    graph_,
                    retriever_,
                    intent_model_,
                    intent_cheap_model_,
                    getattr(request.app.state, "summarizer", None),
                    getattr(request.app.state, "mcp_provider", None),
                ),
                media_type=SSE_MEDIA_TYPE,
            )

        try:
            text, done = await complete_turn(
                graph_,
                session_id=session_id,
                message=req.message,
                model=model_,
                intent_model=intent_model_,
                intent_cheap_model=intent_cheap_model_,
                repositories=repositories_,
                retriever=retriever_,
                settings=settings_,
                summarizer=getattr(request.app.state, "summarizer", None),
                mcp_provider=getattr(request.app.state, "mcp_provider", None),
                flywheel=FlywheelService(repositories_, model_, settings_),
            )
        except Exception as exc:  # noqa: BLE001
            try:
                await persist_user_only(repositories_, session_id, req.message)
            except Exception:  # noqa: BLE001 - 落库失败不该盖住原始故障
                logger.exception("上游失败后落库用户轮次也失败了（已忽略）")
            raise _as_upstream_error(exc) from exc

        awaiting = done.get("awaiting_input") or {}
        if awaiting.get("discarded_message"):
            return _error(
                409,
                "refund_pending",
                str(awaiting.get("message") or "请先选择要办理的订单。"),
            )

        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                session_id=session_id,
                content=text,
                context=ContextStats(
                    kept_messages=int(done.get("kept") or 0),
                    trimmed_messages=int(done.get("trimmed") or 0),
                    # 这个字段 ch01 起就是 ``CONTEXT_TOKEN_BUDGET`` 的回显，
                    # `tests/test_api_tools.py` 与 `test_api_chat.py` 钉住了它；
                    # ch07 倒推出来的历史预算放在 layer1/layer2 两项里表达，
                    # 不去改这个既有字段的含义。
                    budget_tokens=settings_.context_token_budget,
                    layer1_messages=int(done.get("layer1_messages") or 0),
                    layer2_messages=int(done.get("layer2_messages") or 0),
                    summary_messages=int(done.get("summary_messages") or 0),
                    layer1_from_msg_id=done.get("layer1_from_msg_id"),
                    summary_upto_msg_id=done.get("summary_upto_msg_id"),
                    degraded=bool(done.get("degraded")),
                    summary_due=bool(done.get("summary_due")),
                ),
                tools=list(done.get("tools") or []),
                citations=list(done.get("citations") or []),
                actions=list(done.get("actions") or []),
                intent=done.get("intent"),
                route=done.get("route"),
                resolved_message=done.get("resolved_message"),
                coref_changed=bool(done.get("coref_changed")),
                awaiting_input=done.get("awaiting_input"),
                agent_steps=int(done.get("agent_steps") or 0),
                stop_reason=done.get("stop_reason"),
            ).model_dump(),
        )

    # -- 会话历史 ---------------------------------------------------------

    @app.post(
        "/api/chat/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
        responses={200: {"description": "该会话已保存的历史"}},
    )
    async def session_messages(
        session_id: str,
        repositories_: Repositories = Depends(get_repositories),
    ) -> SessionMessagesResponse:
        """从 ``messages`` 表回读历史（含每轮的工具轨迹）。

        ch02 起不再读内存存储 —— 这样徽章在**刷新页面后仍然在**，
        而且重启服务后历史不会凭空消失。
        """
        return await _db_session_messages_payload(session_id, repositories_)

    @app.get("/api/chat/sessions/{session_id}/pending")
    async def session_pending(
        session_id: str,
        graph_: Any = Depends(get_agent_graph),
    ) -> dict[str, Any]:
        """刷新页面后恢复「等待选择/等待确认」的卡片。

        SSE 流断开后前端拿不到那次 ``awaiting_input`` 帧，只能主动问图停在哪；
        没有暂停时返回 ``{"pending": null}``。
        """
        snapshot = await graph_.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        return {"pending": pending_payload(snapshot)}

    # -- ch07：会话侧栏（两个只读接口）------------------------------------

    @app.get(
        "/api/conversations",
        response_model=ConversationListResponse,
        responses={200: {"description": "该用户的会话列表，新在前"}},
    )
    async def list_conversations(
        user_id: str = "anonymous",
        limit: int = 50,
        repositories_: Repositories = Depends(get_repositories),
    ) -> ConversationListResponse:
        """侧栏用：一个用户看自己的全部会话。

        为什么现在固定 ``anonymous``：本章只管「当前会话」，没有账号体系。
        参数仍留着并透传给仓库 —— 等真的接入登录时，改的是调用方传什么，
        而不是这个接口的签名。

        ``limit`` 给个上界是为了不让某天一个演示账号攒下几百个会话时
        把侧栏首屏拖成一次全表扫描。
        """
        rows = await repositories_.conversations.list_for_user(
            user_id=user_id, limit=max(1, min(int(limit), 200))
        )
        return ConversationListResponse(
            conversations=[
                ConversationSummaryOut(
                    id=row.id,
                    session_id=row.session_id,
                    user_id=row.user_id,
                    preview=row.preview,
                    message_count=row.message_count,
                    summarized=row.summary_segments > 0,
                    summary_segments=row.summary_segments,
                    summary_upto_msg_id=row.summary_upto_msg_id,
                    layer1_from_msg_id=row.layer1_from_msg_id,
                    created_at=row.created_at.isoformat() if row.created_at else None,
                    updated_at=row.updated_at.isoformat() if row.updated_at else None,
                )
                for row in rows
            ]
        )

    @app.get(
        "/api/conversations/{conversation_id}/messages",
        response_model=SessionMessagesResponse,
        responses={
            200: {"description": "该会话已保存的历史"},
            404: {"model": ErrorResponse, "description": "会话不存在"},
        },
    )
    async def conversation_messages(
        conversation_id: int,
        repositories_: Repositories = Depends(get_repositories),
    ) -> SessionMessagesResponse | JSONResponse:
        """点侧栏某条会话时回载历史原文。

        **``{conversation_id}`` 是 ``conversations.id``，不是 ``session_id``**。
        两者都是「会话的标识」，混用不会立刻报错（一个数字一个字符串，
        写错了只会 404），但前端拿列表里的 ``id`` 就必须配这个接口。

        实现上先 id → session_id 再复用 ``_db_session_messages_payload``：
        回载逻辑只有一份（含工具轨迹、引用、动作的还原），两个入口不可能给出
        不同的历史。
        """
        session_id = await repositories_.conversations.find_session_id(conversation_id)
        if not session_id:
            return _error(404, "not_found", f"没有 id 为 {conversation_id} 的会话。")
        return await _db_session_messages_payload(session_id, repositories_)

    # -- ch05：用户点「建工单」才走这里 ---------------------------------
    #
    # 转人工是纯前端模拟，不经过后端；建工单是唯一会写 tickets 表的用户动作。
    # Agent 的工具集里没有 create_ticket，所以不存在「模型替你建单」。

    @app.post(
        "/api/actions/create-ticket",
        response_model=CreateTicketResponse,
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    async def create_ticket_action(
        req: CreateTicketRequest,
        settings_: Settings = Depends(get_app_settings),
        repositories_: Repositories = Depends(get_repositories),
    ) -> JSONResponse:
        """用户在前端点「建工单」时调用；复用第 2 章的 `create_ticket` 工具。"""
        description = (req.description or "").strip()
        if not description:
            try:
                description = (
                    await repositories_.messages.last_user_content(req.session_id) or ""
                ).strip()
            except Exception as exc:  # noqa: BLE001 - 必须映射成规范错误外壳
                err = _as_upstream_error(exc)
                status = 503 if err.type == "database_error" else 502
                return _error(status, err.type, redact(str(err)))
        if not description:
            return _error(422, "validation_error", "工单描述为空：请补一句话说明你的诉求。")

        result = await execute_tool(
            "create_ticket",
            {
                "description": description,
                "ticket_type": req.ticket_type,
                "conversation_id": req.session_id,
                "idempotency_key": req.idempotency_key,
            },
            repositories=repositories_,
            registry=build_registry(repositories_, None),
            # 建单不需要检索器；传 None 避免为了一个 tickets 写入去构造 Milvus 客户端。
            retriever=None,
            timeout=settings_.tool_timeout_seconds,
            max_retries=settings_.tool_max_retries,
            # ch05 这条路是「用户点按钮」本身即确认，保持不动；ch08 只补 confirmed。
            confirmed=True,
        )
        if not result.ok:
            if result.fault == "database":
                return _error(503, "database_error", "数据库暂时不可用，请稍后重试。")
            return _error(
                502,
                "tool_error",
                redact(f"建工单失败：{result.error or '工具返回失败'}"),
            )

        data = result.data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                return _error(502, "tool_error", "建工单失败：工具返回了无法解析的结果。")
        if not isinstance(data, dict) or not data.get("ticket_no"):
            return _error(502, "tool_error", "建工单失败：工具没有返回工单号。")

        return JSONResponse(
            status_code=200,
            content=CreateTicketResponse(
                ok=True,
                ticket_no=str(data["ticket_no"]),
                ticket_type=str(data.get("ticket_type") or req.ticket_type),
                status=str(data.get("status") or "待处理"),
                message=str(data.get("message") or f"已为你创建工单 {data['ticket_no']}。"),
            ).model_dump(),
        )

    # -- ch06：订单点选续跑 -------------------------------------------------

    @app.post(
        "/api/chat/refund/select",
        response_class=StreamingResponse,
        responses={
            200: {"description": "SSE：token / tools / citations / done（或再次 awaiting_input）"},
            409: {"model": ErrorResponse},
        },
    )
    async def refund_select(
        req: RefundSelectRequest,
        request: Request,
        settings_: Settings = Depends(get_app_settings),
        model_: BaseChatModel = Depends(get_model),
        intent_model_: BaseChatModel = Depends(get_intent_model),
        intent_cheap_model_: BaseChatModel | None = Depends(get_intent_cheap_model),
        repositories_: Repositories = Depends(get_repositories),
        retriever_: AnswerContextLike = Depends(get_retriever),
        graph_: Any = Depends(get_agent_graph),
    ):
        """用户点选订单卡片后，从退款子流程的 checkpoint 续跑。"""
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            snapshot = await graph_.aget_state(config)
        except Exception as exc:  # noqa: BLE001
            err = _as_upstream_error(exc)
            status = 503 if err.type == "database_error" else 502
            return _error(status, err.type, redact(str(err)))

        pending_nodes = tuple(getattr(snapshot, "next", ()) or ())
        if (
            "refund_await_order" not in pending_nodes
            or pending_interrupt(snapshot) is None
        ):
            return _error(
                409,
                "no_pending_refund",
                "没有待选择的订单：请先发起退款/售后问题。",
            )
        if req.session_id in _RESUME_INFLIGHT:
            return _error(
                409,
                "refund_resume_in_progress",
                "这一轮正在处理订单选择，请稍候。",
            )
        values = dict(getattr(snapshot, "values", {}) or {})
        already_committed = (
            values.get("last_committed_turn_id") == values.get("turn_id")
        )
        persist_on_error = (
            bool(values.get("user_message"))
            and not already_committed
            and "log_turn" not in pending_nodes
        )
        _RESUME_INFLIGHT.add(req.session_id)

        return StreamingResponse(
            _refund_resume_body(
                req,
                settings_,
                model_,
                intent_model_,
                intent_cheap_model_,
                repositories_,
                graph_,
                retriever_,
                mcp_provider=getattr(request.app.state, "mcp_provider", None),
                release_session=req.session_id,
                recover_message=str(values.get("user_message") or ""),
                persist_on_error=persist_on_error,
            ),
            media_type=SSE_MEDIA_TYPE,
        )

    # -- ch08：工单预览卡片的确认 / 取消 -----------------------------------

    @app.post("/api/chat/ticket/confirm")
    async def ticket_confirm(
        req: TicketConfirmRequest,
        request: Request,
        settings_: Settings = Depends(get_app_settings),
        model_: BaseChatModel = Depends(get_model),
        intent_model_: BaseChatModel = Depends(get_intent_model),
        intent_cheap_model_: BaseChatModel | None = Depends(get_intent_cheap_model),
        repositories_: Repositories = Depends(get_repositories),
        retriever_: AnswerContextLike = Depends(get_retriever),
        graph_: Any = Depends(get_agent_graph),
    ):
        """用户点「确认提交」/「取消」后，从工单预览的 checkpoint 续跑。"""
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            snapshot = await graph_.aget_state(config)
        except Exception as exc:  # noqa: BLE001
            err = _as_upstream_error(exc)
            status = 503 if err.type == "database_error" else 502
            return _error(status, err.type, redact(str(err)))

        pending_nodes = tuple(getattr(snapshot, "next", ()) or ())
        pending = pending_interrupt(snapshot)
        if (
            "ticket_await_confirm" not in pending_nodes
            or not pending
            or pending.get("kind") != "ticket_preview"
        ):
            return _error(
                409,
                "no_pending_ticket",
                "没有待确认的工单：请先说明要建什么工单。",
            )
        if req.session_id in _RESUME_INFLIGHT:
            return _error(
                409,
                "ticket_resume_in_progress",
                "这张工单正在处理，请稍候。",
            )
        _RESUME_INFLIGHT.add(req.session_id)

        return StreamingResponse(
            _ticket_resume_body(
                req,
                settings_,
                model_,
                intent_model_,
                intent_cheap_model_,
                repositories_,
                graph_,
                retriever_,
                mcp_provider=getattr(request.app.state, "mcp_provider", None),
                release_session=req.session_id,
            ),
            media_type=SSE_MEDIA_TYPE,
        )

    # -- ch06：用户提交退款单（复用 tickets，不新增退款表）-----------------

    @app.post(
        "/api/actions/create-refund",
        response_model=CreateRefundResponse,
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    async def create_refund_action(
        req: CreateRefundRequest,
        settings_: Settings = Depends(get_app_settings),
        repositories_: Repositories = Depends(get_repositories),
        graph_: Any = Depends(get_agent_graph),
    ) -> JSONResponse:
        """提交退款申请：以售后工单形式落 tickets，原因来自固定六类。"""
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            snapshot = await graph_.aget_state(config)
        except Exception as exc:  # noqa: BLE001
            err = _as_upstream_error(exc)
            status = 503 if err.type == "database_error" else 502
            return _error(status, err.type, redact(str(err)))

        values = dict(getattr(snapshot, "values", {}) or {})
        allowed = {
            str(values.get("refund_order_id") or ""),
            *(
                str(item.get("order_id") or "")
                for item in (values.get("refund_order_candidates") or [])
            ),
            str((values.get("refund_form") or {}).get("order_id") or ""),
        }
        allowed.discard("")
        if req.order_id not in allowed:
            return _error(
                409,
                "refund_order_mismatch",
                "这个订单不在本轮退款流程里：请重新发起退款或从订单卡片选择。",
            )

        description = f"退款申请｜订单 {req.order_id}｜原因：{req.reason}"
        # 幂等键命名空间化并压到 64 字符内；避免不同 session 用同一个 key
        # 互相命中，把别人的工单号当成自己的成功。
        idempotency_key = hashlib.sha256(
            f"{req.session_id}:{req.idempotency_key}".encode("utf-8")
        ).hexdigest()

        try:
            existing = await repositories_.tickets.find_by_idempotency_key(
                idempotency_key
            )
        except Exception as exc:  # noqa: BLE001
            err = _as_upstream_error(exc)
            status = 503 if err.type == "database_error" else 502
            return _error(status, err.type, redact(str(err)))

        if existing is not None:
            ticket_no, stored_description = existing
            if stored_description != description:
                return _error(
                    409,
                    "idempotency_conflict",
                    "这个幂等键已经用于另一笔退款申请；请刷新后重试。",
                )
            return JSONResponse(
                status_code=200,
                content=CreateRefundResponse(
                    ok=True,
                    refund_no=ticket_no,
                    ticket_no=ticket_no,
                    order_id=req.order_id,
                    reason=req.reason,
                    status="待处理",
                    message=f"这笔退款申请已提交，工单号 {ticket_no}。",
                ).model_dump(),
            )

        result = await execute_tool(
            "create_ticket",
            {
                "description": description,
                "ticket_type": "售后",
                "conversation_id": req.session_id,
                "idempotency_key": idempotency_key,
            },
            repositories=repositories_,
            registry=build_registry(repositories_, None),
            retriever=None,
            timeout=settings_.tool_timeout_seconds,
            max_retries=settings_.tool_max_retries,
            confirmed=True,
        )
        if not result.ok:
            if result.fault == "database":
                return _error(503, "database_error", "数据库暂时不可用，请稍后重试。")
            return _error(
                502,
                "tool_error",
                redact(f"提交退款单失败：{result.error or '工具返回失败'}"),
            )

        data = result.data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                return _error(502, "tool_error", "提交退款单失败：工具返回了无法解析的结果。")
        if not isinstance(data, dict) or not data.get("ticket_no"):
            return _error(502, "tool_error", "提交退款单失败：工具没有返回工单号。")
        stored_description = data.get("description")
        if stored_description is not None and stored_description != description:
            return _error(
                409,
                "idempotency_conflict",
                "这个幂等键已经用于另一笔退款申请；请刷新后重试。",
            )

        ticket_no = str(data["ticket_no"])
        return JSONResponse(
            status_code=200,
            content=CreateRefundResponse(
                ok=True,
                refund_no=ticket_no,
                ticket_no=ticket_no,
                order_id=req.order_id,
                reason=req.reason,
                status=str(data.get("status") or "待处理"),
                message=str(
                    data.get("message")
                    or f"已提交退款申请，工单号 {ticket_no}。"
                ),
            ).model_dump(),
        )

    # -- ch04：编造个案台账（最小查询 / 处置面，不做页面）------------------

    @app.get("/api/faith-cases")
    async def list_faith_cases(
        status: str | None = None,
        repositories_: Repositories = Depends(get_repositories),
    ) -> JSONResponse:
        """按处置状态列出编造个案（不传 ``status`` 就是全部）。

        为什么不做分页：台账按「一题一行」增长，一个评估集才几十条；
        现在加页码只会增加一个没人调的参数。
        """
        rows = await repositories_.faith_cases.list_by_status(status)
        return JSONResponse(
            status_code=200,
            content={"items": [_faith_case_payload(row) for row in rows]},
        )

    @app.patch("/api/faith-cases/{eval_id}")
    async def update_faith_case(
        eval_id: str,
        req: FaithCaseUpdateRequest,
        repositories_: Repositories = Depends(get_repositories),
    ) -> JSONResponse:
        """改一条个案的处置状态。

        业务规则在仓库层（``FaithCaseRepository.update_status``）：
        标「已解决/无需解决」必须写处置说明，退回「未解决」会清空它。
        这里只负责把 ``ValueError`` 映射成 422、把「找不到」映射成 404。
        """
        try:
            found = await repositories_.faith_cases.update_status(
                eval_id, req.status, req.resolution
            )
        except ValueError as exc:
            return _error(422, "validation_error", redact(str(exc)))

        row = await repositories_.faith_cases.get(eval_id) if found else None
        if row is None:
            return _error(404, "not_found", f"台账里没有题号 {eval_id}。")
        return JSONResponse(status_code=200, content=_faith_case_payload(row))

    # -- 结构化提取（一次返回完整 JSON，不流式）---------------------------

    @app.post(
        "/api/extract",
        responses={
            200: {"description": "抽取到的售后字段"},
            422: {"model": ErrorResponse, "description": "模型输出无法解析成固定字段"},
        },
    )
    async def extract(
        req: ExtractRequest,
        model_: BaseChatModel = Depends(get_model),
    ) -> JSONResponse:
        try:
            result = await extract_request(req.text, model_)
        except ExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _as_upstream_error(exc) from exc
        return JSONResponse(status_code=200, content=result.model_dump())

    # -- ch04：RAG 评估看板 -------------------------------------------------
    #
    # 页面只消费这里归一过的契约，不直接读 evals/reports/ch04.json：
    # 报告是脚本产物，字段会随评估口径演进；归一放在后端，前端才能稳定。

    @app.get("/api/rag-eval")
    async def rag_eval_dashboard() -> JSONResponse:
        """最新一次评估的结构化看板数据。

        没有报告时返回 **404** 而不是空对象：前端据此展示「先跑一次评估」的
        空状态，而不是把一份全 ``—`` 的看板伪装成「已评估」。
        """
        try:
            payload = load_dashboard()
        except ReportNotFound as exc:
            return _error(404, "report_not_found", redact(str(exc)))
        except ReportInvalid as exc:
            return _error(500, "report_invalid", redact(str(exc)))
        payload["run"] = app.state.rag_eval_runner.status()
        return JSONResponse(status_code=200, content=payload)

    @app.post("/api/rag-eval/rerun")
    async def rag_eval_rerun() -> JSONResponse:
        """启动一次后台评估。

        返回 **202**：任务已接受，结果要靠 ``GET /api/rag-eval/run`` 轮询。
        并发重跑返回 **409** —— 两个任务会同时写同一份 JSON 报告。
        """
        try:
            state = await app.state.rag_eval_runner.start()
        except EvalAlreadyRunning as exc:
            return _error(409, "eval_running", redact(str(exc)))
        except EvalStartError as exc:
            return _error(500, "eval_start_failed", redact(str(exc)))
        return JSONResponse(status_code=202, content={"run": state})

    @app.get("/api/rag-eval/run")
    async def rag_eval_run_status() -> JSONResponse:
        """重跑任务状态（供页面轮询）。"""
        return JSONResponse(status_code=200, content={"run": app.state.rag_eval_runner.status()})

    # -- 聊天页面（静态资源，无构建步骤）----------------------------------

    web_dir = PROJECT_ROOT / "web"
    if web_dir.is_dir():
        # html=True 让 /chat/ 直接返回 index.html
        app.mount("/chat", StaticFiles(directory=web_dir, html=True), name="chat")

        # 评估页单独挂载，避免把 evals/reports 这类运行时产物暴露成静态目录。
        rag_eval_dir = web_dir / "rag-eval"
        if rag_eval_dir.is_dir():

            @app.get("/rag-eval", include_in_schema=False)
            async def rag_eval_redirect() -> RedirectResponse:
                """不带斜杠也能进；与课程里的 ``/rag-eval`` 地址对齐。"""
                return RedirectResponse(url="/rag-eval/")

            app.mount(
                "/rag-eval",
                StaticFiles(directory=rag_eval_dir, html=True),
                name="rag-eval",
            )

        @app.get("/", include_in_schema=False)
        async def root_redirect() -> RedirectResponse:
            """根路径直接跳聊天页，省得记地址。"""
            return RedirectResponse(url="/chat/")

    return app
