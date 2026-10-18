"""ch06 Task 8：退款/售后确定性子流程与图接线。"""

from __future__ import annotations

import json

from langgraph.types import Command

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from tests.fakes_agent import ScriptedChatModel, text_turn
from tests.test_ch05_graph import _evidence, _run


def _coref(resolved: str, changed: bool):
    return text_turn(
        json.dumps({"resolved": resolved, "changed": changed}, ensure_ascii=False)
    )


def _intent(intent: str, confidence: float = 0.9):
    return text_turn(
        json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False)
    )


def _expand(*queries: str):
    return text_turn(json.dumps({"queries": list(queries)}, ensure_ascii=False))


class _RefundRetriever:
    def __init__(self, *, evidence=_evidence(), refused=False) -> None:
        self.evidence = evidence
        self.refused = refused
        self.many_calls: list[list[str]] = []

    async def retrieve_only(self, query, *, category=None):
        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=self.evidence,
            refused=self.refused,
        )

    async def retrieve_many(self, queries, *, category=None):
        self.many_calls.append(list(queries))
        return RetrievalOutcome(
            query=UnderstoodQuery(
                original=queries[0], standard=queries[0], bm25_text=queries[0]
            ),
            queries=tuple(
                UnderstoodQuery(original=q, standard=q, bm25_text=q) for q in queries
            ),
            evidence=self.evidence,
            top_score=0.9,
            refused=self.refused,
        )


async def test_refund_with_order_goes_through_deterministic_subflow(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[
            _intent("logistics"),
            text_turn("运输中。"),
            _coref("订单 20240115001 能退吗", True),
            _intent("refund_return"),
            _expand("订单 20240115001 的退货条件", "退货运费谁承担"),
            text_turn("可以退，按政策…… [1]"),
        ]
    )
    retriever = _RefundRetriever()

    await _run(
        graph,
        message="订单 20240115001 的物流到哪了",
        model=model,
        repositories=fake_repositories,
        retriever=retriever,
        settings=test_settings,
        session="s-refund",
    )
    values, _, _ = await _run(
        graph,
        message="那它能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=retriever,
        settings=test_settings,
        session="s-refund",
    )

    assert values["route"] == "refund"
    assert values["intent"] == "refund_return"
    assert values["coref_changed"] is True
    assert values["refund_order"]["order_id"] == "20240115001"
    for node in (
        "refund_prepare_order",
        "refund_await_order",
        "refund_expand_queries",
        "refund_retrieve_policies",
        "refund_agent",
    ):
        assert node in values["trace"]
    assert retriever.many_calls and len(retriever.many_calls[0]) == 3
    assert values["refund_form"]["reasons"][:2] == [
        "商品质量问题",
        "尺寸/颜色/型号不合适",
    ]


async def test_missing_order_emits_picker_and_interrupts(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    values, events, config = await _run(
        graph,
        message="这个能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=_RefundRetriever(),
        settings=test_settings,
        session="s-picker",
        turn_id="t-picker",
    )
    assert [event["type"] for event in events].count("order_picker") == 1
    assert len(values["refund_order_candidates"]) == 3
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("refund_await_order",)
    assert snapshot.tasks[0].interrupts[0].value["orders"] == values[
        "refund_order_candidates"
    ]


async def test_resume_with_selected_order_finishes(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    values, _, config = await _run(
        graph,
        message="这个能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=_RefundRetriever(),
        settings=test_settings,
        session="s-resume",
        turn_id="t-resume",
    )
    picked = values["refund_order_candidates"][0]["order_id"]
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])

    events = []
    async for event in graph.astream(
        Command(resume={"order_id": picked}), config, stream_mode="custom"
    ):
        events.append(event)

    snapshot = await graph.aget_state(config)
    final = snapshot.values
    assert final["answer"] == "可以退。"
    assert final["refund_order"]["order_id"] == picked
    assert "refund_agent" in final["trace"]
    assert not any(event.get("type") == "order_picker" for event in events)


async def test_invalid_selection_reinterrupts(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    _, _, config = await _run(
        graph,
        message="这个能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=_RefundRetriever(),
        settings=test_settings,
        session="s-bad-pick",
        turn_id="t-bad",
    )
    async for _ in graph.astream(
        Command(resume={"order_id": "not-in-list"}), config, stream_mode="custom"
    ):
        pass
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("refund_await_order",)


async def test_policy_retrieval_empty_still_calls_agent(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[
            _intent("refund_return"),
            _expand("退货条件", "退货运费"),
            text_turn("这项政策我暂时无法确认。"),
        ]
    )
    retriever = _RefundRetriever(evidence=(), refused=True)
    values, _, _ = await _run(
        graph,
        message="订单 20240115001 能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=retriever,
        settings=test_settings,
        session="s-empty-policy",
    )
    assert values["answer"] == "这项政策我暂时无法确认。"
    assert "refund_agent" in values["trace"]
    assert values["stop_reason"] != "gate_failed"


async def test_other_intent_goes_business_and_asks_clarification(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_intent("other"), text_turn("你想咨询哪方面？")]
    )
    values, _, _ = await _run(
        graph,
        message="你们公司股票能买吗",
        model=model,
        repositories=fake_repositories,
        retriever=_RefundRetriever(),
        settings=test_settings,
        session="s-other",
    )
    assert values["route"] == "business"
    assert values["intent"] == "other"
    system = model.state.calls[-1][0].content
    assert "意图未确定" in system
