"""图执行器 —— HTTP 层与脚本共用的唯一入口。

## 为什么不让 API 直接调 `graph.astream`

API 只该关心 SSE 帧；「input 怎么拼、config 里放什么、done 取哪些字段」属于
编排细节。抽一层后，验收脚本、未来的 CLI、HTTP 三条入口看到的是同一份
`done` 摘要，不会各自拼一套。

## 自定义事件

Agent 节点通过 `get_stream_writer()` 写 `{"type": "token" | "status"}`；
这里原样转发。图跑完后用 `aget_state` 取最终 state，再补 `tools` / `citations`
/ `done` —— 不在流里反复推整份 state（messages 会很大）。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

__all__ = ["astream_turn", "complete_turn", "done_payload"]


def _config(
    *,
    session_id: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": session_id,
            "model": model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
        }
    }


def _unwrap(chunk: Any) -> dict[str, Any] | None:
    """兼容 `stream_mode="custom"` 的两种返回形态（裸 payload / v2 包装）。"""
    if not isinstance(chunk, dict):
        return None
    if chunk.get("type") == "custom" and isinstance(chunk.get("data"), dict):
        return chunk["data"]
    return chunk


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
        "tools": list(values.get("tool_traces") or []),
        "citations": list(values.get("citations") or []),
        "actions": actions,
        "intent": values.get("intent"),
        "route": values.get("route"),
        "agent_steps": int(values.get("agent_steps") or 0),
        "stop_reason": values.get("stop_reason"),
        "gate": {
            "passed": values.get("gate_passed"),
            "reason": values.get("gate_reason") or "",
            "score": values.get("retrieval_score") or 0.0,
        },
    }


async def astream_turn(
    graph: Any,
    *,
    session_id: str,
    message: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    turn_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """跑一轮图，逐条产出 token/status/tools/citations/done 事件。"""
    config = _config(
        session_id=session_id,
        model=model,
        repositories=repositories,
        retriever=retriever,
        settings=settings,
    )
    payload = {
        "session_id": session_id,
        "turn_id": turn_id or uuid.uuid4().hex,
        "user_message": message,
    }

    async for chunk in graph.astream(payload, config, stream_mode="custom"):
        event = _unwrap(chunk)
        if event is not None:
            yield event

    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values or {})
    done = done_payload(values, session_id=session_id)

    if done["tools"]:
        yield {"type": "tools", "tools": done["tools"]}
    if done["citations"]:
        yield {"type": "citations", "citations": done["citations"]}
    yield {"type": "done", "payload": done}


async def complete_turn(
    graph: Any,
    *,
    session_id: str,
    message: str,
    model: Any,
    repositories: Any,
    retriever: Any,
    settings: Any,
    turn_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """非流式：收齐 token 事件，返回完整回复与 done 摘要。"""
    parts: list[str] = []
    done: dict[str, Any] | None = None
    async for event in astream_turn(
        graph,
        session_id=session_id,
        message=message,
        model=model,
        repositories=repositories,
        retriever=retriever,
        settings=settings,
        turn_id=turn_id,
    ):
        if event.get("type") == "token":
            parts.append(str(event.get("delta") or ""))
        elif event.get("type") == "done":
            done = event.get("payload")

    if done is None:  # pragma: no cover - 图正常结束一定会发 done
        raise RuntimeError("图执行结束但没有产出 done 事件")
    return "".join(parts), done
