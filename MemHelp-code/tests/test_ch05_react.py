"""ch05 Task 7：ReAct 循环的流式、单工具步、多步、停止条件与建议动作。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agent.prompts import AGENT_EMPTY_TEXT, AGENT_HANDOFF_TEXT
from app.agent.react import AgentOutcome, run_react_agent
from app.db.models import Ticket
from app.tools.executor import ToolResult
from tests.fakes_agent import (
    ScriptedChatModel,
    empty_turn,
    multi_tool_turn,
    text_turn,
    tool_turn,
)


def _executor(payloads: dict[str, str] | None = None):
    payloads = payloads or {}

    async def run(name, args, **kwargs):
        data = payloads.get(name, json.dumps({"ok": True, "echo": name}, ensure_ascii=False))
        return ToolResult(
            tool=name,
            args=dict(args),
            ok=True,
            data=data,
            error=None,
            attempts=1,
            duration_ms=3,
        )

    return run


def _messages(text: str = "订单 1001 的物流到哪了"):
    return [SystemMessage(content="你是客服"), HumanMessage(content=text)]


async def _run(model, *, include_faq=False, max_steps=6, token_budget=6000, executor=None):
    events: list[dict] = []
    outcome = await run_react_agent(
        model=model,
        messages=_messages(),
        repositories=None,  # 工具只构造不调用；executor 已接管执行
        retriever=object(),
        include_faq=include_faq,
        evidence=(),
        max_steps=max_steps,
        token_budget=token_budget,
        emit=events.append,
        executor=executor or _executor(),
    )
    return outcome, events


async def test_one_tool_call_then_final():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_logistics", {"order_id": "1001"}),
            text_turn("已发货，正在派送。"),
        ]
    )
    outcome, events = await _run(model)

    assert outcome.agent_steps == 1
    assert outcome.llm_calls == 2
    assert outcome.stop_reason == "final"
    assert outcome.answer == "已发货，正在派送。"
    assert [m.type for m in outcome.pending_messages] == ["ai", "tool", "ai"]
    assert outcome.pending_messages[0].tool_calls[0]["name"] == "query_logistics"
    assert isinstance(outcome.pending_messages[1], ToolMessage)
    assert outcome.tool_traces[0]["name"] == "query_logistics"
    assert outcome.tool_records[0]["tool_call_id"] == "call_1"


async def test_complex_question_takes_two_tool_steps():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}, call_id="call_1"),
            tool_turn("query_logistics", {"order_id": "1001"}, call_id="call_2"),
            text_turn("订单已付款，物流运输中。"),
        ]
    )
    outcome, _ = await _run(model)

    assert outcome.agent_steps == 2
    assert [record["name"] for record in outcome.tool_records] == [
        "query_order",
        "query_logistics",
    ]
    assert [m.type for m in outcome.pending_messages] == ["ai", "tool", "ai", "tool", "ai"]


async def test_multiple_calls_in_one_step_only_first_runs():
    model = ScriptedChatModel(
        script=[
            multi_tool_turn(
                [
                    ("query_order", {"order_id": "1001"}, "call_1"),
                    ("query_logistics", {"order_id": "1001"}, "call_2"),
                ]
            ),
            text_turn("好了。"),
        ]
    )
    outcome, _ = await _run(model)

    assert outcome.agent_steps == 1
    assert len([m for m in outcome.pending_messages if m.type == "tool"]) == 1
    first_ai = outcome.pending_messages[0]
    assert len(first_ai.tool_calls) == 1
    assert first_ai.tool_calls[0]["name"] == "query_order"


async def test_max_steps_returns_handoff_and_does_not_call_again():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}),
            tool_turn("query_logistics", {"order_id": "1001"}),
        ]
    )
    outcome, _ = await _run(model, max_steps=1)

    assert outcome.agent_steps == 1
    assert outcome.stop_reason == "max_steps"
    assert outcome.answer == AGENT_HANDOFF_TEXT
    assert outcome.suggested_actions == ["transfer_human", "create_ticket"]
    assert len(model.state.calls) == 1


async def test_token_budget_stops_before_the_next_model_call():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}, usage=(5000, 2000, 7000)),
            text_turn("不该走到这里。"),
        ]
    )
    outcome, _ = await _run(model, token_budget=6000)

    assert outcome.stop_reason == "token_budget"
    assert outcome.token_usage["total_tokens"] == 7000
    assert outcome.answer == AGENT_HANDOFF_TEXT
    assert outcome.suggested_actions == ["transfer_human", "create_ticket"]
    assert len(model.state.calls) == 1


async def test_suggest_actions_is_recorded_but_not_shown_as_tool_badge():
    model = ScriptedChatModel(
        script=[
            tool_turn(
                "suggest_actions",
                {"actions": ["transfer_human"], "reason": "用户要求人工"},
                call_id="call_suggest",
            ),
            text_turn("已经帮你记下转人工的选项。"),
        ]
    )
    outcome, _ = await _run(model)

    assert outcome.suggested_actions == ["transfer_human"]
    assert outcome.tool_traces == []
    assert outcome.tool_records[0]["name"] == "suggest_actions"
    assert outcome.tool_records[0]["tool_call_id"] == "call_suggest"
    assert outcome.tool_records[0]["traces"] == []


async def test_query_faq_evidence_becomes_citations():
    payload = json.dumps(
        {
            "ok": True,
            "count": 1,
            "items": [
                {
                    "n": 1,
                    "chunk_id": 7,
                    "section_path": "售后政策 / 退货",
                    "category": "售后政策",
                    "question": "退货政策",
                    "answer": "7 天无理由退货",
                }
            ],
        },
        ensure_ascii=False,
    )
    model = ScriptedChatModel(
        script=[
            tool_turn("query_faq", {"keyword": "退货政策"}),
            text_turn("按政策可以 7 天无理由退货 [1]。"),
        ]
    )
    outcome, _ = await _run(model, include_faq=True, executor=_executor({"query_faq": payload}))

    assert outcome.citations and outcome.citations[0]["n"] == 1
    assert outcome.citations[0]["quote"] == "7 天无理由退货"


async def test_no_tool_call_converges_immediately():
    model = ScriptedChatModel(script=[text_turn("你好呀")])
    outcome, _ = await _run(model)

    assert outcome.agent_steps == 0
    assert outcome.llm_calls == 1
    assert outcome.answer == "你好呀"
    assert outcome.stop_reason == "final"


async def test_empty_response_gets_a_fixed_fallback():
    model = ScriptedChatModel(script=[empty_turn()])
    outcome, _ = await _run(model)

    assert outcome.stop_reason == "empty_response"
    assert outcome.answer == AGENT_EMPTY_TEXT
    assert outcome.suggested_actions == ["transfer_human"]
    # 空 assistant 不应进入历史：只留一条固定兜底 AI 消息。
    assert [message.type for message in outcome.pending_messages] == ["ai"]


async def test_preamble_then_empty_final_uses_last_tool_result():
    """前一步说了「我查一下」、最后一步空回复：兜底要用工具结果，不能只留前言。"""
    payload = json.dumps({"ok": True, "status": "已发货"}, ensure_ascii=False)
    model = ScriptedChatModel(
        script=[
            tool_turn(
                "query_order",
                {"order_id": "1001"},
                preamble="我帮您查一下。",
            ),
            empty_turn(),
        ]
    )
    outcome, _ = await _run(model, executor=_executor({"query_order": payload}))

    assert outcome.stop_reason == "empty_response"
    assert "状态：已发货" in outcome.answer
    assert outcome.pending_messages[-1].type == "ai"
    assert "已发货" in outcome.pending_messages[-1].content


async def test_reasoning_content_is_preserved_on_the_tool_call_message():
    model = ScriptedChatModel(
        script=[
            tool_turn(
                "query_order",
                {"order_id": "1001"},
                extra_kwargs={"reasoning_content": "需要先查订单"},
            ),
            text_turn("查到了。"),
        ]
    )
    outcome, _ = await _run(model)

    assert outcome.pending_messages[0].additional_kwargs["reasoning_content"] == "需要先查订单"


async def test_status_is_emitted_before_the_tool_runs():
    model = ScriptedChatModel(
        script=[
            tool_turn("query_order", {"order_id": "1001"}),
            text_turn("查到了。"),
        ]
    )
    _, events = await _run(model)

    kinds = [event["type"] for event in events]
    assert kinds.index("status") < kinds.index("token")
    assert events[kinds.index("status")]["tool"] == "query_order"


async def test_unknown_tool_is_rejected_and_never_executed(
    fake_repositories, empty_retriever, sqlite_session
):
    """模型编出 create_ticket 时，绝不能被 executor 按全局注册表执行。"""
    called: list[str] = []

    async def executor(name, args, **kwargs):
        called.append(name)
        return ToolResult(
            tool=name, args=dict(args), ok=True, data="{}", error=None,
            attempts=1, duration_ms=1,
        )

    model = ScriptedChatModel(
        script=[
            tool_turn("create_ticket", {"description": "偷偷建单", "ticket_type": "投诉",
                                        "conversation_id": "s-evil"}, call_id="call_evil"),
            text_turn("没有执行任何写操作。"),
        ]
    )
    outcome = await run_react_agent(
        model=model,
        messages=_messages(),
        repositories=fake_repositories,
        retriever=empty_retriever,
        include_faq=False,
        evidence=(),
        emit=lambda payload: None,
        executor=executor,
    )

    assert called == [], "不在本轮工具集的工具名不得交给 executor"
    assert outcome.tool_traces[0]["ok"] is False
    assert "不在本轮可用工具内" in outcome.tool_records[0]["content"]
    assert sqlite_session.query(Ticket).count() == 0
