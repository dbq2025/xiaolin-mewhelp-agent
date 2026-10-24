"""ch09 Task 6：👎 后端落池与检索快照回捞。"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app import main as app_main
from app.flywheel.snapshot import snapshot_for_turn
from app.config import Settings
from tests.fakes import FakeChatModel


def _client(settings: Settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def _events(text: str) -> list[tuple[str, dict[str, Any] | None]]:
    events: list[tuple[str, dict[str, Any] | None]] = []
    for block in text.split("\n\n"):
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                data = line[6:].strip()
        if name:
            events.append((name, json.loads(data) if data else None))
    return events


async def _seed_turn(
    repositories: Any,
    *,
    session_id: str = "s1",
    question: str = "退货运费谁出",
    citations: list[dict[str, Any]] | None = None,
) -> int:
    conversation_id = await repositories.conversations.upsert_by_session(
        session_id, "anonymous"
    )
    ids = await repositories.messages.append_many(
        conversation_id,
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "这个问题我答不上来。",
                "citations": citations or [],
            },
        ],
    )
    return ids[-1]


async def test_append_many_returns_ids(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s", "anonymous")
    ids = await fake_repositories.messages.append_many(
        cid, [{"role": "user", "content": "hi"}]
    )
    assert len(ids) == 1 and ids[0] > 0


async def test_feedback_down_pools_with_snapshot(test_settings, fake_repositories):
    message_id = await _seed_turn(
        fake_repositories,
        citations=[
            {
                "n": 1,
                "chunk_id": "c1",
                "section_path": "售后/退货",
                "category": "售后",
                "quote": "运费由买家承担。",
            }
        ],
    )
    client = _client(test_settings)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "message_id": message_id, "rating": "down"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    row = (await fake_repositories.low_confidence.list_recent())[0]
    assert row.source == "user_feedback"
    assert row.raw_question == "退货运费谁出"
    assert row.retrieved_chunks is not None
    assert row.retrieved_chunks[0]["chunk_id"] == "c1"
    # 消息 citations 里没有分数，回捞时必须显式补 None 而不是 0。
    assert row.retrieved_chunks[0]["score"] is None


async def test_feedback_idempotent_by_message(test_settings, fake_repositories):
    message_id = await _seed_turn(fake_repositories)
    client = _client(test_settings)
    payload = {"session_id": "s1", "message_id": message_id, "rating": "down"}
    assert client.post("/api/feedback", json=payload).status_code == 200
    assert client.post("/api/feedback", json=payload).status_code == 200
    assert len(await fake_repositories.low_confidence.list_recent()) == 1


async def test_feedback_up_is_ignored(test_settings, fake_repositories):
    message_id = await _seed_turn(fake_repositories)
    client = _client(test_settings)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "message_id": message_id, "rating": "up"},
    )
    assert resp.status_code == 200
    assert resp.json()["ignored"] is True
    assert await fake_repositories.low_confidence.list_recent() == []


async def test_feedback_unknown_message_404(test_settings):
    client = _client(test_settings)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "message_id": 999, "rating": "down"},
    )
    assert resp.status_code == 404


class _State(TypedDict, total=False):
    user_message: str
    retrieval_snapshot: list[dict[str, Any]]


async def test_snapshot_for_turn_prefers_checkpoint():
    builder = StateGraph(_State)
    builder.add_node(
        "n",
        lambda _s: {
            "retrieval_snapshot": [
                {"n": 1, "chunk_id": "c1", "quote": "q", "score": 0.12}
            ]
        },
    )
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=MemorySaver())
    await graph.ainvoke(
        {"user_message": "怎么退"}, {"configurable": {"thread_id": "t1"}}
    )
    snapshot = await snapshot_for_turn(
        graph, session_id="t1", user_message="怎么退"
    )
    assert snapshot is not None and snapshot[0]["score"] == 0.12


async def test_snapshot_for_turn_missing_returns_none():
    builder = StateGraph(_State)
    builder.add_node("n", lambda _s: {})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=MemorySaver())
    await graph.ainvoke({"user_message": "闲聊"}, {"configurable": {"thread_id": "t2"}})
    assert await snapshot_for_turn(graph, session_id="t2", user_message="别的") is None


async def test_done_payload_carries_message_id(test_settings, fake_repositories):
    with _client(test_settings) as client:
        with client.stream(
            "POST",
            "/api/chat/completions",
            json={"session_id": "s1", "message": "在吗"},
        ) as resp:
            text = "".join(resp.iter_text())
        done = [data for name, data in _events(text) if name == "done"][0]
        assert isinstance(done["message_id"], int) and done["message_id"] > 0
        payload = client.post("/api/chat/sessions/s1/messages").json()
        assert "messages" in payload, payload
        msgs = payload["messages"]
    assert msgs[-1]["id"] == done["message_id"]
