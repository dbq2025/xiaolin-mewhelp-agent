"""图执行器 —— HTTP 层与脚本共用的唯一入口。

## 为什么不让 API 直接调 `graph.astream`

API 只该关心 SSE 帧；「input 怎么拼、config 里放什么、done 取哪些字段」属于
编排细节。抽一层后，验收脚本、未来的 CLI、HTTP 三条入口看到的是同一份
`done` 摘要，不会各自拼一套。

## 自定义事件与暂停

Agent / coref / 退款节点通过 `get_stream_writer()` 写自定义事件；这里原样转发。
退款子流程缺订单号时会 `interrupt()` 暂停：`astream` 结束但 `snapshot.next` 非空，
此时 runner 发 `awaiting_input`，**不发 `done`**，避免前端把「等待选择」当成答完。
用户点选后走 `astream_resume`，用 `Command(resume=...)` 从同一 thread 续跑。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command

__all__ = [
    "astream_resume",
    "astream_turn",
    "complete_turn",
    "done_payload",
    "pending_interrupt",
]

_AWAITING_TEXT = "请选择要办理的订单。"


def _config(
    *,
    session_id: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    intent_model: Any = None,
    intent_cheap_model: Any = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": session_id,
            "model": model,
            "intent_model": intent_model,
            "intent_cheap_model": intent_cheap_model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
            # ch07：后台摘要器。不传就是 None，图里层 2 只记账不压缩。
            "summarizer": summarizer,
            # ch08：MCP 工具发现器。None 表示本实例不接 MCP。
            "mcp_provider": mcp_provider,
        }
    }


def _unwrap(chunk: Any) -> dict[str, Any] | None:
    """兼容 `stream_mode="custom"` 的两种返回形态（裸 payload / v2 包装）。"""
    if not isinstance(chunk, dict):
        return None
    if chunk.get("type") == "custom" and isinstance(chunk.get("data"), dict):
        return chunk["data"]
    return chunk


def pending_interrupt(snapshot: Any) -> dict[str, Any] | None:
    """取当前快照里的第一个中断载荷；没有暂停返回 None。"""
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt_item in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt_item, "value", interrupt_item)
            if isinstance(value, dict):
                return value
    return None


def _awaiting_payload(
    values: dict[str, Any],
    interrupt_value: dict[str, Any],
    *,
    discarded_message: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": interrupt_value.get("kind") or "order_picker",
        "turn_id": values.get("turn_id"),
        "orders": list(interrupt_value.get("orders") or []),
        "message": _AWAITING_TEXT,
    }
    if payload["kind"] == "ticket_preview":
        # 建单预览：前端渲染「确认提交 / 取消」，message 也换成对应提示。
        payload["call_id"] = interrupt_value.get("call_id")
        payload["preview"] = dict(interrupt_value.get("preview") or {})
        payload["message"] = "请确认是否提交这张工单。"
    if discarded_message:
        payload["discarded_message"] = True
        payload["message"] = (
            "请先确认或取消刚才的工单；刚才的消息没有被处理。"
            if payload["kind"] == "ticket_preview"
            else "请先选择要办理的订单；刚才的消息没有被处理。"
        )
    if interrupt_value.get("error"):
        payload["error"] = interrupt_value["error"]
    return payload


def pending_payload(snapshot: Any) -> dict[str, Any] | None:
    """把「图正停在哪」压成前端可直接渲染的 awaiting 载荷。

    刷新页面后前端没有 SSE 上下文，只能靠这个接口把预览卡片重新画出来；
    没有暂停时返回 ``None``。
    """
    interrupt_value = pending_interrupt(snapshot)
    if interrupt_value is None:
        return None
    values = dict(getattr(snapshot, "values", {}) or {})
    return _awaiting_payload(values, interrupt_value)


def done_payload(values: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    """把最终 state 压成 SSE `done` / 非流式响应用的摘要。"""
    context = values.get("context") or {}
    stop_reason = values.get("stop_reason")
    finish_reason = {
        "max_steps": "length",
        "token_budget": "length",
    }.get(str(stop_reason), "stop")
    actions = [
        {"action": action, "ticket_type": values.get("ticket_type") or "售后"}
        for action in (values.get("suggested_actions") or [])
    ]
    return {
        "finish_reason": finish_reason,
        "session_id": session_id,
        "trimmed": int(context.get("trimmed_messages") or 0),
        "kept": int(context.get("kept_messages") or 0),
        # ch07：三层结构的可观测字段。全部带默认值，ch01–ch06 的调用方
        # 不读这些键也不受影响。
        "layer1_messages": int(context.get("layer1_messages") or 0),
        "layer2_messages": int(context.get("layer2_messages") or 0),
        "summary_messages": int(context.get("summary_messages") or 0),
        "layer1_from_msg_id": context.get("layer1_from_msg_id"),
        "summary_upto_msg_id": context.get("summary_upto_msg_id"),
        "degraded": bool(context.get("degraded")),
        "summary_due": bool(context.get("summary_due")),
        "tools": list(values.get("tool_traces") or []),
        "citations": list(values.get("citations") or []),
        "actions": actions,
        "intent": values.get("intent"),
        "route": values.get("route"),
        "resolved_message": values.get("resolved_message"),
        "coref_changed": bool(values.get("coref_changed")),
        "query_expansion": {
            "queries": list(values.get("expanded_queries") or []),
            "degraded": bool(values.get("query_expansion_degraded")),
        },
        "refund_form": values.get("refund_form") or None,
        "trace": list(values.get("trace") or []),
        "agent_steps": int(values.get("agent_steps") or 0),
        "stop_reason": values.get("stop_reason"),
        "gate": {
            "passed": values.get("gate_passed"),
            "reason": values.get("gate_reason") or "",
            "score": values.get("retrieval_score") or 0.0,
        },
    }


async def _stream_events(
    graph: Any, payload: Any, config: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    async for chunk in graph.astream(payload, config, stream_mode="custom"):
        event = _unwrap(chunk)
        if event is not None:
            yield event


async def _finish_or_pause(
    graph: Any,
    config: dict[str, Any],
    *,
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values or {})
    pending = pending_interrupt(snapshot)
    if pending is not None:
        # 暂停前若已经产生工具/引用，先发出去，别让 UI 丢记录。
        if values.get("tool_traces"):
            yield {"type": "tools", "tools": list(values["tool_traces"])}
        if values.get("citations"):
            yield {"type": "citations", "citations": list(values["citations"])}
        yield {"type": "awaiting_input", "payload": _awaiting_payload(values, pending)}
        return
    if getattr(snapshot, "next", ()):
        yield {
            "type": "awaiting_input",
            "payload": {
                "kind": "incomplete",
                "turn_id": values.get("turn_id"),
                "orders": [],
                "message": "上一轮没有收敛，请重新发起或清空会话后再试。",
            },
        }
        return

    done = done_payload(values, session_id=session_id)
    if done["tools"]:
        yield {"type": "tools", "tools": done["tools"]}
    if done["citations"]:
        yield {"type": "citations", "citations": done["citations"]}
    yield {"type": "done", "payload": done}


async def astream_turn(
    graph: Any,
    *,
    session_id: str,
    message: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    intent_model: Any = None,
    intent_cheap_model: Any = None,
    turn_id: str | None = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """跑一轮图，逐条产出 token/status/tools/citations/done 事件。"""
    config = _config(
        session_id=session_id,
        model=model,
        intent_model=intent_model,
        intent_cheap_model=intent_cheap_model,
        repositories=repositories,
        retriever=retriever,
        settings=settings,
        summarizer=summarizer,
        mcp_provider=mcp_provider,
    )

    # 同一 thread 还在等订单选择时，不要把新消息塞进暂停中的图：
    # 重新发一次选择器，模型调用次数保持为 0。
    snapshot = await graph.aget_state(config)
    pending = pending_interrupt(snapshot)
    if pending is not None:
        yield {
            "type": "awaiting_input",
            "payload": _awaiting_payload(
                dict(snapshot.values or {}),
                pending,
                discarded_message=True,
            ),
        }
        return
    if getattr(snapshot, "next", ()):
        yield {
            "type": "awaiting_input",
            "payload": {
                "kind": "incomplete",
                "turn_id": (snapshot.values or {}).get("turn_id"),
                "orders": [],
                "message": "上一轮没有收敛，请重新发起或清空会话后再试。",
                "discarded_message": True,
            },
        }
        return

    payload = {
        "session_id": session_id,
        "turn_id": turn_id or uuid.uuid4().hex,
        "user_message": message,
    }

    async for event in _stream_events(graph, payload, config):
        yield event

    async for event in _finish_or_pause(graph, config, session_id=session_id):
        yield event


async def astream_resume(
    graph: Any,
    *,
    session_id: str,
    order_id: str = "",
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    intent_model: Any = None,
    intent_cheap_model: Any = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
    resume_value: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """从中断点续跑；不重新跑 begin_turn。

    ``resume_value`` 优先：建工单确认传 ``{"confirmed": bool}``；不传时按
    ch06 的订单选择器约定传 ``{"order_id": order_id}``。
    """
    config = _config(
        session_id=session_id,
        model=model,
        intent_model=intent_model,
        intent_cheap_model=intent_cheap_model,
        repositories=repositories,
        retriever=retriever,
        settings=settings,
        summarizer=summarizer,
        mcp_provider=mcp_provider,
    )
    resumed = Command(
        resume=resume_value if resume_value is not None else {"order_id": order_id}
    )
    async for event in _stream_events(graph, resumed, config):
        yield event
    async for event in _finish_or_pause(graph, config, session_id=session_id):
        yield event


async def complete_turn(
    graph: Any,
    *,
    session_id: str,
    message: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    intent_model: Any = None,
    intent_cheap_model: Any = None,
    turn_id: str | None = None,
    summarizer: Any = None,
    mcp_provider: Any = None,
) -> tuple[str, dict[str, Any]]:
    """非流式：收齐 token 事件，返回完整回复与 done 摘要。"""
    parts: list[str] = []
    done: dict[str, Any] | None = None
    awaiting: dict[str, Any] | None = None
    async for event in astream_turn(
        graph,
        session_id=session_id,
        message=message,
        model=model,
        repositories=repositories,
        retriever=retriever,
        settings=settings,
        intent_model=intent_model,
        intent_cheap_model=intent_cheap_model,
        turn_id=turn_id,
        summarizer=summarizer,
        mcp_provider=mcp_provider,
    ):
        if event.get("type") == "token":
            parts.append(str(event.get("delta") or ""))
        elif event.get("type") == "awaiting_input":
            awaiting = event.get("payload") or {}
        elif event.get("type") == "done":
            done = event.get("payload")

    if awaiting is not None and done is None:
        done = {
            "finish_reason": "stop",
            "session_id": session_id,
            "stop_reason": "awaiting_input",
            "awaiting_input": awaiting,
            "tools": [],
            "citations": [],
            "actions": [],
        }
        return _AWAITING_TEXT, done

    if done is None:  # pragma: no cover - 图正常结束一定会发 done
        raise RuntimeError("图执行结束但没有产出 done 事件")
    return "".join(parts), done
