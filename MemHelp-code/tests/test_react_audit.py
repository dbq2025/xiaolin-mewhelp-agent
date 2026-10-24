"""review I3：ReAct 自己拦下的调用也要落审计。

``create_ticket`` 的参数校验、以及模型编出不存在的工具名，都不会走到
执行引擎；但「每次工具调用一条审计」是本章的硬约束，所以这两条路径
必须自己补写。
"""

from __future__ import annotations

from app.agent.react import run_react_agent
from langchain_core.messages import HumanMessage
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


def _messages():
    return [HumanMessage(content="帮我建个工单")]


async def test_intercepted_create_ticket_writes_validation_audit(
    fake_repositories, empty_retriever
):
    model = ScriptedChatModel(
        script=[
            tool_turn(
                "create_ticket",
                {"ticket_type": "投诉", "conversation_id": "s-x"},
                call_id="call_bad",
            ),
            text_turn("请问具体是什么问题？"),
        ]
    )
    await run_react_agent(
        model=model,
        messages=_messages(),
        repositories=fake_repositories,
        retriever=empty_retriever,
        include_faq=True,
    )

    rows = await fake_repositories.audit.list_recent(5)
    blocked = [row for row in rows if row.tool_name == "create_ticket"]
    assert blocked, "被校验拦下的建单没有审计"
    assert blocked[0].status == "校验拦下"
    assert blocked[0].tool_call_id == "call_bad"
    assert "description" in (blocked[0].error_message or "")
    assert blocked[0].arguments and "ticket_type" in blocked[0].arguments


async def test_unknown_tool_writes_failure_audit(fake_repositories, empty_retriever):
    model = ScriptedChatModel(
        script=[
            tool_turn("issue_refund", {"order_id": "1001"}, call_id="call_unknown"),
            text_turn("抱歉，这个操作我做不了。"),
        ]
    )
    await run_react_agent(
        model=model,
        messages=_messages(),
        repositories=fake_repositories,
        retriever=empty_retriever,
        include_faq=True,
    )

    rows = await fake_repositories.audit.list_recent(5)
    unknown = [row for row in rows if row.tool_name == "issue_refund"]
    assert unknown, "未知工具调用没有审计"
    assert unknown[0].status == "失败"
    assert unknown[0].tool_call_id == "call_unknown"
