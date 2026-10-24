"""ch09 观测层：Langfuse 的**唯一**接入点。

设计约束：

- **聊天不能被观测拖垮**。密钥缺失、网络不通、SDK 抛错都只记 warning，
  然后整条链路降级为 no-op —— 观测是附加能力，不是请求的前置条件。
- **只有本模块 import langfuse**。其余模块只依赖这里的四个函数，
  换 SDK / 换后端时改一个文件。
- **编译期挂一次回调**：``build_agent_graph`` 调到 ``callback_handler()``，
  把 handler 交给 ``.with_config({"callbacks": [...]})``；节点内模型调用
  由 LangGraph 传播，不需要每个调用点重复传 config。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_client: Any | None = None
_handler: Any | None = None
_enabled: bool = False


class TurnSpan:
    """一轮对话的根 span。

    ``finish(payload)`` 把这一轮的结论（意图、路由、闸结果、token、耗时）
    写到根 span 的 metadata/output 上，Langfuse 里点开 trace 就能直接读到，
    不用逐个子 span 拼。
    """

    def __init__(self, observation: Any | None) -> None:
        self._observation = observation
        self._finished = False

    @property
    def enabled(self) -> bool:
        return self._observation is not None

    def finish(self, payload: dict[str, Any] | None = None) -> None:
        if self._observation is None or self._finished:
            return
        self._finished = True
        data = dict(payload or {})
        metadata = {
            "intent": data.get("intent"),
            "route": data.get("route"),
            "gate": data.get("gate"),
            "token_usage": data.get("token_usage"),
            "stop_reason": data.get("stop_reason"),
            "message_id": data.get("message_id"),
        }
        try:
            self._observation.update(
                output=data,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        except Exception:  # pragma: no cover - 观测失败不影响主流程
            logger.warning("Langfuse 根 span 收尾失败，已忽略", exc_info=True)


def configure(settings: Any) -> None:
    """按配置初始化客户端与回调 handler（幂等）。"""
    global _client, _handler, _enabled

    if _client is not None or _handler is not None:
        return
    if not getattr(settings, "langfuse_enabled", True):
        logger.info("tracing 已由配置关闭")
        return
    public_key = getattr(settings, "langfuse_public_key", "")
    secret_key = getattr(settings, "langfuse_secret_key", "")
    base_url = getattr(settings, "langfuse_base_url", "")
    if not (public_key and secret_key and base_url):
        logger.warning("Langfuse 连接项不完整，tracing 降级为 no-op")
        return

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=getattr(settings, "langfuse_environment", None) or None,
            tracing_enabled=True,
        )
        _handler = CallbackHandler()
        _enabled = True
        logger.info("Langfuse tracing 已启用：%s", base_url)
    except Exception:
        _client = None
        _handler = None
        _enabled = False
        logger.warning("Langfuse 初始化失败，tracing 降级为 no-op", exc_info=True)


def callback_handler() -> Any | None:
    """给图编译期用的 LangChain 回调；未启用时返回 ``None``。"""
    return _handler


def _attributes(**kwargs: Any):
    """trace 级属性（tags/metadata/session）的唯一出口，测试可替换。"""
    if not _enabled:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(**kwargs)


@contextmanager
def turn_span(session_id: str, question: str) -> Iterator[TurnSpan]:
    """一轮对话的根 span；未启用时 yield 一个 no-op 的 ``TurnSpan``。"""
    if not _enabled or _client is None:
        yield TurnSpan(None)
        return

    try:
        manager = _client.start_as_current_observation(
            name="chat-turn",
            as_type="span",
            input={"question": question},
            metadata={"session_id": session_id},
        )
    except Exception:  # pragma: no cover
        logger.warning("Langfuse 根 span 创建失败，本轮不记录", exc_info=True)
        yield TurnSpan(None)
        return

    with manager as observation:
        with _attributes(session_id=session_id, trace_name="chat-turn"):
            yield TurnSpan(observation)


@contextmanager
def tag_intent(intent: str, confidence: float = 0.0) -> Iterator[None]:
    """给当前 trace 打意图标签，供按意图聚合 token。"""
    if not _enabled or not intent:
        yield
        return
    with _attributes(
        tags=[f"intent:{intent}"],
        metadata={"intent": intent, "intent_confidence": confidence},
    ):
        yield


@contextmanager
def observation(name: str, as_type: str = "span", input: Any | None = None):
    """一个手动子 span（检索、工具、自评各用一次）；未启用时 yield ``None``。"""
    if not _enabled or _client is None:
        yield None
        return
    with _client.start_as_current_observation(
        name=name, as_type=as_type, input=input
    ) as span:
        yield span


def shutdown() -> None:
    """进程退出前 flush，避免最后几轮 trace 还在内存里。"""
    global _client, _handler, _enabled
    client, _client = _client, None
    _handler = None
    _enabled = False
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # pragma: no cover - 退出路径不抛
        logger.warning("Langfuse flush 失败，已忽略", exc_info=True)


def reset_for_tests() -> None:
    """测试专用：清空模块级状态，避免用例之间串味。"""
    global _client, _handler, _enabled
    _client = None
    _handler = None
    _enabled = False
