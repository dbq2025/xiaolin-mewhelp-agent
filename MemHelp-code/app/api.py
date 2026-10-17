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

import inspect
import json
import logging
from collections.abc import AsyncIterable
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import Engine
from sqlalchemy.exc import InterfaceError, OperationalError, PendingRollbackError

from app.chat import astream_chat, complete_chat
from app.chat_tools import astream_chat_with_tools, complete_chat_with_tools
from app.config import PROJECT_ROOT, Settings, get_settings
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
from app.knowledge.pipeline import RagPipeline
from app.knowledge.query import LlmRewriter, QueryUnderstander
from app.knowledge.rerank import get_reranker
from app.knowledge.sufficiency import SufficiencyJudge
from app.knowledge.retrieval import warmup as warmup_rag
from app.knowledge.vector_store import MilvusVectorStore
from app.llm import get_chat_model
from app.memory import InMemorySessionStore, get_session_store, new_session_id
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ContextStats,
    ErrorResponse,
    ExtractRequest,
    FaithCaseUpdateRequest,
    SessionMessagesResponse,
)
from app.sse import SSE_MEDIA_TYPE, sse_frame
from app.tools.knowledge import AnswerContextLike

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


def _context_stats(context) -> ContextStats:
    """把 ``ContextResult`` 映射成对外的精简统计。"""
    return ContextStats(
        kept_messages=context.kept_messages,
        trimmed_messages=context.trimmed_messages,
        budget_tokens=context.budget_tokens,
        used_tokens=context.used_tokens,
        system_prompt_tokens=context.system_prompt_tokens,
    )


# ---------------------------------------------------------------------------
# 依赖：全部经 app.state 注入，测试可完全离线
# ---------------------------------------------------------------------------


def get_store(request: Request) -> InMemorySessionStore:
    store = getattr(request.app.state, "session_store", None)
    return store if store is not None else get_session_store()


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
        engine = create_engine_from(current)
        factory = session_factory(engine)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            redact(f"数据库配置非法：{exc}"), type_="configuration_error"
        ) from exc

    database = (factory, engine)
    request.app.state._database = database  # noqa: SLF001 - 进程级共享
    return database


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
    understander = QueryUnderstander(
        LlmRewriter(model, timeout=settings.rag_rewrite_timeout),
        enabled=settings.rag_query_rewrite,
    )
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

        if row.role == "assistant" and row.tool_calls:
            # 优先用 ``tool_traces``（我们自己写的展示用数据，含耗时与成败）。
            # 它是 ch02 迁移新增的列，老行可能没有 —— 那样退化成从 ``tool_calls``
            # 与紧随其后的 ``tool`` 行推断，至少能还原 name/args/ok。
            traces = getattr(row, "tool_traces", None)
            item["tools"] = traces or _tool_traces(row.tool_calls, rows, index)

        # ch04：引用快照。与 ``tools`` 同理 —— 只在有引用时出现，
        # 前端 `if (msg.citations)` 一处判断就够。
        citations = getattr(row, "citations", None)
        if row.role == "assistant" and citations:
            item["citations"] = citations

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


async def _chat_stream_body(
    req: ChatRequest,
    settings: Settings,
    model: BaseChatModel,
    store: InMemorySessionStore,
    repositories: Repositories,
    session_id: str,
    retriever: AnswerContextLike | None = None,
) -> AsyncIterable[str]:
    """SSE 响应体：``meta`` → ``status``? → ``token``* → ``tools`` → ``done``。

    中途失败补 ``error`` 再收尾（沿用 ch01）。

    ``status`` 帧只在调了工具时出现；``tools`` 帧是徽章的权威来源
    （含成败与耗时），随后 ``done`` 把它一并带出去，前端只读 ``done.tools`` 即可。
    """
    meta_out: dict = {}
    yield sse_frame("meta", {"session_id": session_id, "model": settings.openai_model})

    tool_summaries: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    try:
        async for event in astream_chat_with_tools(
            session_id=session_id,
            message=req.message,
            model=model,
            repositories=repositories,
            retriever=retriever,
            budget_tokens=settings.context_token_budget,
            model_name=settings.openai_model,
            timeout=settings.tool_timeout_seconds,
            max_retries=settings.tool_max_retries,
            meta_out=meta_out,
        ):
            if event.kind == "status":
                yield sse_frame("status", event.payload)
            elif event.kind == "token":
                yield sse_frame("token", event.payload)
            elif event.kind == "tools":
                tool_summaries = list(event.payload.get("tools") or [])
            elif event.kind == "citations":
                citations = list(event.payload.get("citations") or [])
    except Exception as exc:  # noqa: BLE001 - 必须转成事件，不能让连接裸断
        err = _as_upstream_error(exc)
        yield sse_frame(
            "error", {"type": err.type, "message": redact(str(err)), "session_id": session_id}
        )
        return

    meta = meta_out.get("meta")
    yield sse_frame(
        "done",
        {
            "finish_reason": "stop",
            "session_id": session_id,
            "trimmed": meta.context.trimmed_messages if meta else 0,
            "kept": meta.context.kept_messages if meta else 0,
            "tools": tool_summaries,
            # ch04：引用证据快照。前端靠它把正文里的 [n] 变成可点角标。
            "citations": citations,
        },
    )


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


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
    # 关掉进程单例持有的 httpx 连接（CLI 自己会关；Web 进程在退出时关一次）。
    try:
        settings = app.state.settings or get_settings()
        await get_embedder(settings).aclose()
    except Exception:  # noqa: BLE001 - 退出路径不该再抛
        logger.debug("关闭嵌入客户端失败（已忽略）", exc_info=True)


def create_app(
    model: BaseChatModel | None = None,
    store: InMemorySessionStore | None = None,
    settings: Settings | None = None,
    repositories: Repositories | None = None,
    db_ping: Any = None,
    retriever: AnswerContextLike | None = None,
    rag_warmup: bool | None = None,
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
    """
    app = FastAPI(title="MewHelp 智能客服 API", version="0.3.0", lifespan=_lifespan)
    app.state.chat_model = model
    app.state.session_store = store
    # 显式赋值（即使为 None）：这是「测试用注入配置、不读宿主机 .env」的实际生效点。
    app.state.settings = settings
    app.state.repositories = repositories
    app.state.db_ping = db_ping
    app.state.retriever = retriever
    app.state.rag_warmup = rag_warmup

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
        settings_: Settings = Depends(get_app_settings),
        model_: BaseChatModel = Depends(get_model),
        store_: InMemorySessionStore = Depends(get_store),
        repositories_: Repositories = Depends(get_repositories),
        retriever_: AnswerContextLike = Depends(get_retriever),
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
                    req, settings_, model_, store_, repositories_, session_id, retriever_
                ),
                media_type=SSE_MEDIA_TYPE,
            )

        try:
            text, meta = await complete_chat_with_tools(
                session_id=session_id,
                message=req.message,
                model=model_,
                repositories=repositories_,
                retriever=retriever_,
                budget_tokens=settings_.context_token_budget,
                model_name=settings_.openai_model,
                timeout=settings_.tool_timeout_seconds,
                max_retries=settings_.tool_max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_upstream_error(exc) from exc

        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                session_id=session_id,
                content=text,
                context=_context_stats(meta.context),
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

    # -- 聊天页面（静态资源，无构建步骤）----------------------------------

    web_dir = PROJECT_ROOT / "web"
    if web_dir.is_dir():
        # html=True 让 /chat/ 直接返回 index.html
        app.mount("/chat", StaticFiles(directory=web_dir, html=True), name="chat")

        @app.get("/", include_in_schema=False)
        async def root_redirect() -> RedirectResponse:
            """根路径直接跳聊天页，省得记地址。"""
            return RedirectResponse(url="/chat/")

    return app
