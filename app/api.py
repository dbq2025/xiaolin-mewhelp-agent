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

import json
from collections.abc import AsyncIterable
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.chat import astream_chat, complete_chat
from app.config import Settings, get_settings
from app.errors import (
    classify_upstream_error_type,
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


class UpstreamError(RuntimeError):
    """上游（模型服务）不可用、拒绝请求，或本地配置缺失。"""

    def __init__(self, message: str, type_: str = "upstream_error") -> None:
        super().__init__(message)
        self.type = type_


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
    """
    return [
        {
            "type": str(item.get("type", "")),
            "loc": [str(p) for p in item.get("loc", ())],
            "msg": str(item.get("msg", "")),
            "input": repr(item.get("input"))[:200],
        }
        for item in exc.errors()
    ]


def _classify_upstream_error(exc: Exception) -> UpstreamError:
    """把上游异常包装成带类型的 ``UpstreamError``。

    分类规则集中在 ``app.errors``，这里只负责拼装消息。
    """
    type_ = classify_upstream_error_type(exc)
    return UpstreamError(upstream_error_message(exc, type_), type_=type_)


def _as_upstream_error(exc: Exception) -> UpstreamError:
    return exc if isinstance(exc, UpstreamError) else _classify_upstream_error(exc)


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


@lru_cache
def _default_chat_model() -> BaseChatModel:
    """默认模型只构造一次，避免每个请求都新建客户端。"""
    return get_chat_model()


def get_model(request: Request) -> BaseChatModel:
    model = getattr(request.app.state, "chat_model", None)
    return model if model is not None else _default_chat_model()


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
            f"配置缺失或非法，请检查 .env：{exc}",
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


async def _chat_stream_body(
    req: ChatRequest,
    settings: Settings,
    model: BaseChatModel,
    store: InMemorySessionStore,
    session_id: str,
) -> AsyncIterable[str]:
    """SSE 响应体：``meta`` → ``token``* → ``done``，中途失败补 ``error`` 再收尾。"""
    meta_out: dict = {}
    yield sse_frame("meta", {"session_id": session_id, "model": settings.openai_model})

    try:
        async for delta in astream_chat(
            session_id=session_id,
            message=req.message,
            store=store,
            model=model,
            budget_tokens=settings.context_token_budget,
            model_name=settings.openai_model,
            meta_out=meta_out,
        ):
            yield sse_frame("token", {"delta": delta})
    except Exception as exc:  # noqa: BLE001 - 必须转成事件，不能让连接裸断
        err = _as_upstream_error(exc)
        yield sse_frame(
            "error", {"type": err.type, "message": str(err), "session_id": session_id}
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
        },
    )


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


def create_app(
    model: BaseChatModel | None = None,
    store: InMemorySessionStore | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """构造应用。

    ``model`` / ``store`` / ``settings`` 都可注入：测试里传假模型、独立存储和
    写死的配置，就能在**没有 .env、不发任何网络请求**的前提下跑完整 HTTP 测试。
    """
    app = FastAPI(title="MewHelp 智能客服 API", version="0.1.0")
    app.state.chat_model = model
    app.state.session_store = store
    app.state.settings = settings

    # -- 异常处理 ---------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        return _error(
            422, "validation_error", "请求参数不合法。", _jsonable_validation_errors(exc)
        )

    @app.exception_handler(UpstreamError)
    async def on_upstream_error(request: Request, exc: UpstreamError):
        # 配置问题用 503（服务不可用，运维该去修 .env），
        # 上游问题用 502（服务在，是它依赖的上游出错）。
        status = 503 if exc.type == "configuration_error" else 502
        return _error(status, exc.type, str(exc))

    @app.exception_handler(ExtractionError)
    async def on_extraction_error(request: Request, exc: ExtractionError):
        return _error(422, "extraction_failed", str(exc))

    # -- 健康检查 ---------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        try:
            current = app.state.settings if app.state.settings is not None else get_settings()
        except Exception:  # noqa: BLE001
            return {
                "status": "misconfigured",
                "model": None,
                "base_url_configured": False,
                "context_token_budget": None,
                "hint": "请在项目根目录创建 .env（可参考 .env.example）。",
            }
        return {
            "status": "ok",
            "model": current.openai_model,
            "base_url_configured": bool(current.openai_base_url),
            "context_token_budget": current.context_token_budget,
        }

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
                _chat_stream_body(req, settings_, model_, store_, session_id),
                media_type=SSE_MEDIA_TYPE,
            )

        try:
            text, meta = await complete_chat(
                session_id=session_id,
                message=req.message,
                store=store_,
                model=model_,
                budget_tokens=settings_.context_token_budget,
                model_name=settings_.openai_model,
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
        store_: InMemorySessionStore = Depends(get_store),
    ) -> SessionMessagesResponse:
        return _session_messages_payload(session_id, store_)

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

    return app
