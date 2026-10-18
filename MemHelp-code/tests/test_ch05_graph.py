"""ch05 Task 8：图节点、分流、闸短路、actions 落库与 checkpointer 跨轮。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.agent.prompts import CHITCHAT_REPLY, COMPLAINT_SOOTHE
from app.knowledge.citations import build_evidence
from app.knowledge.hybrid import RetrievedChunk
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from app.prompts import REFUSAL_TEXT
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


class _Retriever:
    def __init__(self, *, evidence=(), refused=False, degraded=False) -> None:
        self.evidence = tuple(evidence)
        self.refused = refused
        self.degraded = degraded
        self.calls: list[str] = []

    async def retrieve_only(self, query: str, *, category=None):
        self.calls.append(query)
        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=self.evidence,
            refused=self.refused,
            refusal_source="retrieval_low_conf" if self.refused else None,
            refusal_reason="测试里判定证据不足" if self.refused else None,
            rerank_degraded=self.degraded,
        )


def _evidence(score: float = 0.9):
    return build_evidence(
        [
            RetrievedChunk(
                chunk_id=7,
                question="退货政策",
                answer="7 天无理由退货",
                category="售后政策",
                section_path="售后政策 / 退货",
                content_type="manual",
                score=score,
                source="rerank",
            )
        ]
    )


def _intent_turn(intent: str, confidence: float = 0.9):
    return text_turn(json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False))


async def _run(graph, *, message, model, repositories, retriever, settings, session: str):
    config = {
        "configurable": {
            "thread_id": session,
            "model": model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
        }
    }
    events: list[dict] = []
    async for chunk in graph.astream(
        {"session_id": session, "turn_id": uuid.uuid4().hex, "user_message": message},
        config,
        stream_mode="custom",
    ):
        if isinstance(chunk, dict):
            events.append(chunk)
    snapshot = await graph.aget_state(config)
    return snapshot.values, events, config


def _graph():
    return build_agent_graph(checkpointer=memory_checkpointer())


async def test_policy_question_goes_through_retrieval_then_gate(
    test_settings, fake_repositories
):
    graph = _graph()
    model = ScriptedChatModel(
        script=[_intent_turn("refund_return"), text_turn("按政策可以 7 天无理由退货 [1]。")]
    )
    retriever = _Retriever(evidence=_evidence())

    values, _, _ = await _run(
        graph,
        message="退货政策是什么",
        model=model,
        repositories=fake_repositories,
        retriever=retriever,
        settings=test_settings,
        session="s-policy",
    )

    assert values["route"] == "knowledge"
    assert "knowledge_retrieve" in values["trace"]
    assert values["retrieval_attempted"] is True
    assert values["gate_passed"] is True
    assert values["agent_steps"] == 0
    assert "7 天无理由" in values["answer"]
    assert values["citations"][0]["n"] == 1


async def test_gate_failure_returns_fallback_without_agent(
    test_settings, fake_repositories
):
    graph = _graph()
    # 只给意图响应；如果 Agent 被调用，脚本会耗尽并抛错。
    model = ScriptedChatModel(script=[_intent_turn("product_consult")])
    retriever = _Retriever(evidence=(), refused=True)

    values, _, _ = await _run(
        graph,
        message="你们支持货到付款吗",
        model=model,
        repositories=fake_repositories,
        retriever=retriever,
        settings=test_settings,
        session="s-gate-fail",
    )

    assert values["answer"] == REFUSAL_TEXT
    assert values["gate_passed"] is False
    assert values["agent_steps"] == 0
    assert len(model.state.calls) == 1  # 只有意图分类
    rows = await fake_repositories.low_confidence.list_recent(5)
    assert rows and rows[0].source == "retrieval_low_conf"


async def test_complaint_gives_two_actions_and_never_calls_agent(
    test_settings, fake_repositories
):
    graph = _graph()
    model = ScriptedChatModel(script=[_intent_turn("complaint")])
    values, _, _ = await _run(
        graph,
        message="我要投诉",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-complaint",
    )

    assert values["answer"] == COMPLAINT_SOOTHE
    assert values["suggested_actions"] == ["transfer_human", "create_ticket"]
    assert values["ticket_type"] == "投诉"
    assert values["agent_steps"] == 0
    assert len(model.state.calls) == 1

    cid = await fake_repositories.messages.find_conversation_id("s-complaint")
    rows = await fake_repositories.messages.list_for_conversation(cid)
    assert rows[-1].actions == [
        {"action": "transfer_human", "ticket_type": "投诉"},
        {"action": "create_ticket", "ticket_type": "投诉"},
    ]


async def test_trivial_chitchat_has_no_model_call(test_settings, fake_repositories):
    graph = _graph()
    model = ScriptedChatModel(script=[])
    values, _, _ = await _run(
        graph,
        message="你好",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-chitchat",
    )

    assert values["answer"] == CHITCHAT_REPLY
    assert values["intent_source"] == "rule"
    assert model.state.calls == []


async def test_greeting_plus_real_question_is_not_swallowed(
    test_settings, fake_repositories
):
    graph = _graph()
    model = ScriptedChatModel(
        script=[_intent_turn("refund_return"), text_turn("退货政策如下 [1]。")]
    )
    values, _, _ = await _run(
        graph,
        message="在吗，退货政策是什么",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(evidence=_evidence()),
        settings=test_settings,
        session="s-greeting-policy",
    )

    assert values["route"] == "knowledge"
    assert values["retrieval_attempted"] is True
    assert values["answer"] == "退货政策如下 [1]。"


async def test_business_question_agent_calls_tool(test_settings, fake_repositories):
    graph = _graph()
    model = ScriptedChatModel(
        script=[
            _intent_turn("logistics"),
            tool_turn("query_logistics", {"order_id": "1001"}),
            text_turn("已发货，正在派送。"),
        ]
    )
    values, _, _ = await _run(
        graph,
        message="订单 1001 的物流到哪了",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-logistics",
    )

    assert values["route"] == "business"
    assert [tool["name"] for tool in values["tool_traces"]] == ["query_logistics"]
    assert values["answer"] == "已发货，正在派送。"


async def test_complex_question_takes_two_react_steps(
    test_settings, fake_repositories
):
    graph = _graph()
    model = ScriptedChatModel(
        script=[
            _intent_turn("order"),
            tool_turn("query_order", {"order_id": "1001"}, call_id="call_1"),
            tool_turn("query_logistics", {"order_id": "1001"}, call_id="call_2"),
            text_turn("订单已付款，物流运输中。"),
        ]
    )
    values, _, _ = await _run(
        graph,
        message="订单 1001 是什么状态？如果发货了，物流到哪了",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-complex",
    )

    assert values["agent_steps"] == 2
    assert [tool["name"] for tool in values["tool_traces"]] == [
        "query_order",
        "query_logistics",
    ]


async def test_checkpointer_keeps_history_across_turns(
    test_settings, fake_repositories
):
    graph = _graph()
    first_model = ScriptedChatModel(script=[])
    await _run(
        graph,
        message="你好",
        model=first_model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-multi",
    )

    second_model = ScriptedChatModel(
        script=[_intent_turn("logistics"), text_turn("查到了。")]
    )
    await _run(
        graph,
        message="订单 1001 的物流",
        model=second_model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
        session="s-multi",
    )

    # 第二轮的第一次调用是意图分类（只有当前问题），Agent 那次才带历史；
    # 所以看「所有调用里是否出现了上一轮的用户消息」。
    assert any(
        isinstance(message, HumanMessage) and message.content == "你好"
        for call in second_model.state.calls
        for message in call
    )


async def test_turn_log_written_with_retrieval_trace(fake_repositories):
    from app.config import Settings

    settings = Settings(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="fake-model",
        turn_log_path="tests/_ch05_turns.jsonl",
    )
    path = Path("tests/_ch05_turns.jsonl")
    if path.exists():
        path.unlink()

    graph = _graph()
    model = ScriptedChatModel(
        script=[_intent_turn("refund_return"), text_turn("退货政策 [1]。")]
    )
    try:
        await _run(
            graph,
            message="退货政策是什么",
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(evidence=_evidence()),
            settings=settings,
            session="s-log",
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines
        record = json.loads(lines[-1])
        assert "knowledge_retrieve" in record["trace"]
        assert record["retrieval_attempted"] is True
    finally:
        if path.exists():
            path.unlink()
