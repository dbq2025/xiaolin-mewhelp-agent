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
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.db.repositories import Repositories

__all__ = ["PersistedTurn", "persist_turn", "persist_user_only"]


@dataclass(frozen=True)
class PersistedTurn:
    """一轮落库后拿到的两个锚点；👎 反馈用 ``answer_message_id`` 定位回答。"""

    user_message_id: int
    answer_message_id: int


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
    # 一轮的工具轨迹统一挂到**最终答案行**：中间的工具调用申请单只用于模型历史，
    # 不能同时变成一条带徽章的 bot 气泡。否则刷新后 bot 序号比实时多，
    # localStorage 的 action/feedback 键会整体错位。
    all_traces = [
        trace
        for record in tool_records
        for trace in (record.get("traces") or [])
    ]
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
            rows.append(
                {
                    "role": "assistant",
                    # 中间步骤的「我查一下」不单独落成气泡；最终答案才是用户看到的那条。
                    "content": "",
                    "tool_calls": _tool_calls_payload(message),
                    # 空列表而不是 None：明确告诉回读层「这一行不展示徽章」，
                    # 否则它会回退到从 tool_calls 猜，又画出一条工具气泡。
                    "tool_traces": [],
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

    # 只把引用/动作/工具轨迹挂在**最后一条纯文本 assistant** 上；
    # 挂到工具申请单行会让「这一轮到底引了什么」变得含糊。
    if last_answer_index is not None:
        rows[last_answer_index]["citations"] = list(citations) if citations else None
        rows[last_answer_index]["actions"] = list(actions) if actions else None
        rows[last_answer_index]["tool_traces"] = all_traces

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
) -> PersistedTurn:
    """把一轮成功对话写进 MySQL（user + pending 里的全部行），返回消息 id。"""
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
    ids = await repositories.messages.append_many(conversation_id, rows)
    answer_index = next(
        (
            index
            for index in range(len(rows) - 1, 0, -1)
            if rows[index]["role"] == "assistant"
        ),
        0,
    )
    return PersistedTurn(
        user_message_id=ids[0],
        answer_message_id=ids[answer_index] if ids else 0,
    )


async def persist_user_only(
    repositories: Repositories, session_id: str, message: str
) -> None:
    """失败路径：只保留用户这一轮，不落半截回答。"""
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")
    await repositories.messages.append_many(
        conversation_id, [{"role": "user", "content": message}]
    )
