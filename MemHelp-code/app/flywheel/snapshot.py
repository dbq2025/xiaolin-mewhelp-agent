"""👎 事后落池：从 checkpoint 历史里回捞那一轮的检索快照。

用户点差评时这一轮可能已经过去很久，``state["retrieval_snapshot"]`` 只在
那一轮的 checkpoint 里；按 ``user_message`` 找回它。真没走检索（闲聊、
业务数据类）就返回 ``None``，由调用方退回消息 ``citations`` 或留空。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["snapshot_for_turn"]


async def snapshot_for_turn(
    graph: Any,
    *,
    session_id: str,
    user_message: str,
    message_id: int | None = None,
) -> list[dict[str, Any]] | None:
    """按 thread + 当轮问题回捞 ``retrieval_snapshot``；找不到返回 ``None``。

    :param message_id: 当轮**回答**的消息 id。同一会话里重复问同一句话时，
        只按问题文本会拿到更早一轮的快照；给了 id 就优先精确匹配，
        找不到再退回问题文本（兼容 ch09 之前没有 ``message_id`` 的旧 checkpoint）。
    """
    config = {"configurable": {"thread_id": session_id}}
    fallback: list[dict[str, Any]] | None = None
    try:
        async for snapshot in graph.aget_state_history(config):
            values = dict(getattr(snapshot, "values", {}) or {})
            if str(values.get("user_message") or "") != user_message:
                continue
            chunks = values.get("retrieval_snapshot") or []
            if message_id is not None:
                state_message_id = values.get("message_id")
                if state_message_id is not None and int(state_message_id) == message_id:
                    return [dict(item) for item in chunks]
                if state_message_id is not None:
                    continue
            if chunks:
                if fallback is None:
                    fallback = [dict(item) for item in chunks]
                continue
            # 这一轮确实没走检索（闲聊、业务数据类）：继续往前找，历史里更早
            # 的同问轮次不能拿来冒充当轮；全部轮次都没有快照就返回 None。
            continue
    except Exception:  # noqa: BLE001 - 回捞失败不该挡住反馈落池
        logger.warning(
            "按 checkpoint 回捞检索快照失败（session=%s）", session_id, exc_info=True
        )
        return None
    return fallback
