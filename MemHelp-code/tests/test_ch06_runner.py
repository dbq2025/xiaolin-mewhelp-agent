"""ch06 Task 9：runner 的暂停/续跑与 awaiting_input 契约。"""

from __future__ import annotations

import json

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.agent.runner import astream_resume, astream_turn, complete_turn
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from tests.fakes_agent import ScriptedChatModel, text_turn
from tests.test_ch05_graph import _evidence


def _coref(resolved: str, changed: bool):
    return text_turn(
        json.dumps({"resolved": resolved, "changed": changed}, ensure_ascii=False)
    )


def _intent(intent: str):
    return text_turn(
        json.dumps({"intent": intent, "confidence": 0.9}, ensure_ascii=False)
    )


def _expand(*queries: str):
    return text_turn(json.dumps({"queries": list(queries)}, ensure_ascii=False))


class _Retriever:
    async def retrieve_only(self, query, *, category=None):
        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=_evidence(),
            top_score=0.9,
        )

    async def retrieve_many(self, queries, *, category=None):
        return RetrievalOutcome(
            query=UnderstoodQuery(
                original=queries[0], standard=queries[0], bm25_text=queries[0]
            ),
            queries=tuple(
                UnderstoodQuery(original=q, standard=q, bm25_text=q) for q in queries
            ),
            evidence=_evidence(),
            top_score=0.9,
        )


async def _collect(agen):
    return [event async for event in agen]


async def test_astream_turn_pauses_without_done(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    events = await _collect(
        astream_turn(
            graph,
            session_id="r1",
            message="这个能退吗",
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(),
            settings=test_settings,
        )
    )
    assert any(event["type"] == "order_picker" for event in events)
    assert any(event["type"] == "awaiting_input" for event in events)
    assert not any(event["type"] == "done" for event in events)


async def test_astream_resume_finishes(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    events = await _collect(
        astream_turn(
            graph,
            session_id="r2",
            message="这个能退吗",
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(),
            settings=test_settings,
        )
    )
    first_order = next(
        event for event in events if event["type"] == "awaiting_input"
    )["payload"]["orders"][0]["order_id"]
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])

    resumed = await _collect(
        astream_resume(
            graph,
            session_id="r2",
            order_id=first_order,
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(),
            settings=test_settings,
        )
    )
    assert any(event["type"] == "token" for event in resumed)
    done = [event for event in resumed if event["type"] == "done"][-1]["payload"]
    assert done["refund_form"]["order_id"] == first_order
    assert "refund_agent" in done["trace"]


async def test_astream_turn_reprompts_when_pending(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    await _collect(
        astream_turn(
            graph,
            session_id="r3",
            message="这个能退吗",
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(),
            settings=test_settings,
        )
    )
    calls_before = model.state.index
    events = await _collect(
        astream_turn(
            graph,
            session_id="r3",
            message="那先查物流",
            model=model,
            repositories=fake_repositories,
            retriever=_Retriever(),
            settings=test_settings,
        )
    )
    assert [event["type"] for event in events] == ["awaiting_input"]
    assert model.state.index == calls_before


async def test_complete_turn_returns_awaiting_payload(
    test_settings, fake_repositories
):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    text, done = await complete_turn(
        graph,
        session_id="r4",
        message="这个能退吗",
        model=model,
        repositories=fake_repositories,
        retriever=_Retriever(),
        settings=test_settings,
    )
    assert text == "请选择要办理的订单。"
    assert done["awaiting_input"]["orders"]
