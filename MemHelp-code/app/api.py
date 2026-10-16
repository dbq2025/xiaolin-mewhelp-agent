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
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

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
from app.llm import get_chat_model
from app.memory import InMemorySessionStore, get_session_store, new_session_id
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ContextStats,
    ErrorResponse,
    ExtractRequest,
    SessionMessagesResponse,
)
from app.sse import SSE_MEDIA_TYPE, sse_frame

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


def _as_upstream_error(exc: Exception) -> UpstreamFault:
    """已经是本地定义的故障就直接沿用它的 ``type``，否则去分类。

    用例层（``app.chat``）会抛 ``UpstreamEmptyResponse`` 这类带类型的故障；
    重新分类会把它的类型丢掉，退化成泛泛的 ``upstream_error``。
    """
    if isinstance(exc, UpstreamFault):
        return exc
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


def get_repositories(request: Request) -> Repositories:
    """取当次请求的数据仓库。

    两段式：``app.state.repositories`` 有就直接用（测试注入）；
    否则按 ``app.state.settings`` 建引擎（**引擎缓存**），再为本次请求新建一个
    ``Session``。

    **为什么缓存引擎而不是缓存仓库/Session**：``create_engine`` 建连接池，
    每次新建等于把连接管理交给 GC —— 生产里表现为「MySQL 连接数缓慢涨到上限」。
    但 ``Session`` **必须每请求一个**：它不是线程安全的，跨请求共享会让
    并发请求互相写进对方的会话（表现为间歇性的「聊天记录串台」）。
    仓库只是 Session 的薄包装，每请求新建不花什么钱。

    配置缺失（``.env`` 没配或非法）转成规范的 ``configuration_error``（503），
    而不是把 pydantic 的报错裸抛到 ASGI 层。
    """
    injected = getattr(request.app.state, "repositories", None)
    if injected is not None:
        return injected

    factory = getattr(request.app.state, "_session_factory", None)
    if factory is None:
        try:
            current = get_app_settings(request)
        except UpstreamFault:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                redact(f"配置缺失或非法，请检查 .env：{exc}"), type_="configuration_error"
            ) from exc

        try:
            factory = session_factory(create_engine_from(current))
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                redact(f"数据库配置非法：{exc}"), type_="configuration_error"
            ) from exc
        request.app.state._session_factory = factory  # noqa: SLF001 - 进程级共享

    return Repositories.from_session(factory())


async def _ping_database(request: Request) -> bool:
    """探一次数据库。**永不抛异常**（``ping()`` 自己也吞掉一切）。

    没注入 ``db_ping`` 时用真实探测。探测前若连配置都读不到，直接判定不可用 ——
    ``/healthz`` 的职责是「能不能服务」，配置缺失同样是不能服务，
    只是原因不同（由 ``status: misconfigured`` 表达）。

    **驱动异常的原文只进服务端日志，绝不进响应体**。``/healthz`` 通常是未认证端点，
    而 ``(2003, "Can't connect to MySQL server on '127.0.0.1' (111)")``
    与 ``using password: YES`` 这类文本会泄露库的内部地址、端口与账号
    —— ``redact()`` 只遮密钥形态，遮不住这些。
    """
    injected = getattr(request.app.state, "db_ping", None)
    if injected is not None:
        return bool(await _maybe_await(injected()))

    try:
        current = get_app_settings(request)
    except Exception:  # noqa: BLE001 - 配置缺失已在 healthz 主流程里单独报告
        return False

    try:
        engine = create_engine_from(current)
    except Exception:  # noqa: BLE001
        logger.exception("建数据库引擎失败")
        return False

    alive = await ping(engine)
    if not alive:
        # 原因留给运维看日志；响应体只给一句通用说明。
        logger.warning("数据库探测失败：无法连接 %s:%s", current.mysql_host, current.mysql_port)
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
            item["tools"] = _tool_traces(row.tool_calls, rows, index)

        payload.append(item)

    return SessionMessagesResponse(session_id=session_id, messages=payload)


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
) -> AsyncIterable[str]:
    """SSE 响应体：``meta`` → ``status``? → ``token``* → ``tools`` → ``done``。

    中途失败补 ``error`` 再收尾（沿用 ch01）。

    ``status`` 帧只在调了工具时出现；``tools`` 帧是徽章的权威来源
    （含成败与耗时），随后 ``done`` 把它一并带出去，前端只读 ``done.tools`` 即可。
    """
    meta_out: dict = {}
    yield sse_frame("meta", {"session_id": session_id, "model": settings.openai_model})

    tool_summaries: list[dict[str, Any]] = []
    try:
        async for event in astream_chat_with_tools(
            session_id=session_id,
            message=req.message,
            model=model,
            repositories=repositories,
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
        },
    )


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


def create_app(
    model: BaseChatModel | None = None,
    store: InMemorySessionStore | None = None,
    settings: Settings | None = None,
    repositories: Repositories | None = None,
    db_ping: Any = None,
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
    """
    app = FastAPI(title="MewHelp 智能客服 API", version="0.2.0")
    app.state.chat_model = model
    app.state.session_store = store
    # 显式赋值（即使为 None）：这是「测试用注入配置、不读宿主机 .env」的实际生效点。
    app.state.settings = settings
    app.state.repositories = repositories
    app.state.db_ping = db_ping

    # -- 异常处理 ---------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        return _error(
            422, "validation_error", "请求参数不合法。", _jsonable_validation_errors(exc)
        )

    @app.exception_handler(UpstreamFault)
    async def on_upstream_fault(request: Request, exc: UpstreamFault):
        # 注册在父类上，让 UpstreamEmptyResponse 之类的子类也被接住。
        # 只有本地的配置缺失/非法用 503（服务没准备好，运维该去修 .env）；
        # 其余上游问题一律 502（我们的服务在，是它依赖的上游出错）。
        status = 503 if exc.type == "configuration_error" else 502
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
                    req, settings_, model_, store_, repositories_, session_id
                ),
                media_type=SSE_MEDIA_TYPE,
            )

        try:
            text, meta = await complete_chat_with_tools(
                session_id=session_id,
                message=req.message,
                model=model_,
                repositories=repositories_,
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
