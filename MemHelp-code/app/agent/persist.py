"""一轮对话的 MySQL 落库。

## 为什么单独一层

`log_turn` 节点与失败路径（API 只落用户轮）都要写 ``messages``，但它们的行形状
不同：成功路径可能有多组 tool_calls/tool 行，失败路径只有一行。把「消息对象 →
数据库行」的映射集中在这里，避免两处各拼一遍、字段慢慢漂移。

## 单事务

一轮的所有行经 `append_many` 一次 commit。ReAct 可能产生
``assistant(tool_calls) → tool → assistant(tool_calls) → tool → assistant``；
逐行 commit 会在中途失败时留下「有申请单、没结果」的半截历史，下一轮回灌时
上游直接拒绝。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.db.repositories import Repositories

__all__ = ["persist_turn", "persist_user_only"]


def _tool_calls_payload(message: AIMessage) -> list[dict[str, Any]]:
    """模型历史只认 ``{name, args, id}`` 三个键（多一个就会让 AIMessage 构造失败）。"""
    return [
        {
            "name": call.get("name", ""),
            "args": call.get("args", {}),
            "id": call.get("id", ""),
        }
        for call in message.tool_calls
    ]


def _rows_from_pending(
    pending: Sequence[BaseMessage],
    *,
    citations: Sequence[dict[str, Any]] | None,
    actions: Sequence[dict[str, Any]] | None,
    tool_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces_by_id = {
        str(record.get("tool_call_id")): list(record.get("traces") or [])
        for record in tool_records
    }
    rows: list[dict[str, Any]] = []
    last_answer_index: int | None = None

    for message in pending:
        if isinstance(message, ToolMessage):
            rows.append(
                {
                    "role": "tool",
                    "content": message.content or "",
                    "tool_call_id": message.tool_call_id,
                }
            )
            continue

        if not isinstance(message, AIMessage):
            # 理论上不会出现在 pending 里（用户消息在 begin_turn 已提交）；
            # 真出现了也不写，避免重复用户行。
            continue

        if message.tool_calls:
            traces: list[dict[str, Any]] = []
            for call in message.tool_calls:
                traces.extend(traces_by_id.get(str(call.get("id")), []))
            rows.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": _tool_calls_payload(message),
                    # 空列表而不是 None：`suggest_actions` 有 tool_call 但没有徽章，
                    # 用 [] 明确告诉回读层「这一轮没有要展示的工具」，不要回退去猜。
                    "tool_traces": traces,
                }
            )
            last_answer_index = None
        else:
            rows.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                }
            )
            last_answer_index = len(rows) - 1

    # 只把引用/动作挂在**最后一条纯文本 assistant** 上；挂到工具申请单行会让
    # 「这一轮到底引了什么」变得含糊。
    if last_answer_index is not None:
        rows[last_answer_index]["citations"] = list(citations) if citations else None
        rows[last_answer_index]["actions"] = list(actions) if actions else None

    return rows


def _action_rows(
    suggested_actions: Sequence[str], ticket_type: str
) -> list[dict[str, Any]]:
    return [
        {"action": action, "ticket_type": ticket_type}
        for action in suggested_actions
    ]


async def persist_turn(
    *,
    repositories: Repositories,
    session_id: str,
    user_message: str,
    pending_messages: Sequence[BaseMessage],
    citations: Sequence[dict[str, Any]] | None = None,
    suggested_actions: Sequence[str] = (),
    ticket_type: str = "售后",
    tool_records: Sequence[dict[str, Any]] = (),
) -> None:
    """把一轮成功对话写进 MySQL（user + pending 里的全部行）。"""
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")
    action_rows = _action_rows(suggested_actions, ticket_type)
    rows = [
        {"role": "user", "content": user_message},
        *_rows_from_pending(
            pending_messages,
            citations=citations,
            actions=action_rows,
            tool_records=tool_records,
        ),
    ]
    await repositories.messages.append_many(conversation_id, rows)


async def persist_user_only(
    repositories: Repositories, session_id: str, message: str
) -> None:
    """失败路径：只保留用户这一轮，不落半截回答。"""
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")
    await repositories.messages.append_many(
        conversation_id, [{"role": "user", "content": message}]
    )
