"""ch05 Task 11：手写 Agent 循环（祛魅热身）的离线测试。"""

from __future__ import annotations

from langchain_core.tools import tool

from scripts.bare_agent_loop import run_bare_agent
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


@tool("query_order")
def query_order_tool(order_id: str) -> str:
    """按订单号查订单状态。"""
    return f'{{"ok": true, "order_id": "{order_id}", "status": "已发货"}}'


async def test_bare_loop_executes_tool_and_feeds_result_back():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}, call_id="call_1"),
            text_turn("订单 1001 已发货。"),
        ]
    )

    result = await run_bare_agent(
        model, "订单 1001 怎么样", [query_order_tool], verbose=False
    )

    assert result.steps == 1
    assert result.tool_calls == ["query_order"]
    assert result.answer == "订单 1001 已发货。"
    assert [message.type for message in result.messages] == ["human", "ai", "tool", "ai"]
    assert "已发货" in result.messages[2].content


async def test_bare_loop_converges_without_tools():
    model = ScriptedChatModel(script=[text_turn("你好呀")])
    result = await run_bare_agent(model, "你好", [query_order_tool], verbose=False)

    assert result.steps == 0
    assert result.answer == "你好呀"
    assert [message.type for message in result.messages] == ["human", "ai"]


async def test_bare_loop_stops_at_max_steps():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}, call_id="call_1"),
            tool_turn("query_order", {"order_id": "1002"}, call_id="call_2"),
        ]
    )
    result = await run_bare_agent(
        model, "查订单", [query_order_tool], max_steps=1, verbose=False
    )

    assert result.steps == 1
    assert result.tool_calls == ["query_order"]
    assert "最大步数" in result.answer
    assert len(model.state.calls) == 1
