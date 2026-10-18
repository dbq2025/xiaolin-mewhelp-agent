"""ch05 Task 9：图 runner 接 HTTP 后的事件契约、actions 与非流式路径。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import main
from tests.fakes import FailingChatModel
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


def _intent(intent: str):
    return text_turn(json.dumps({"intent": intent, "confidence": 0.9}, ensure_ascii=False))


def _client(test_settings, model):
    return TestClient(main.create_app(model=model, settings=test_settings))


def _events(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name, json.loads(data) if data else None))
    return events


def test_sse_done_has_actions_route_and_gate(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[_intent("complaint")]))
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-complaint-api", "message": "我要投诉"},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    events = _events(text)
    done = next(data for name, data in events if name == "done") or {}
    assert done["route"] == "complaint"
    assert done["intent"] == "complaint"
    assert [item["action"] for item in done["actions"]] == ["transfer_human", "create_ticket"]
    assert done["agent_steps"] == 0
    assert done["gate"]["passed"] is False


def test_sse_emits_tools_and_citations_for_knowledge_answer(test_settings):
    from tests.fakes import SearchAsAnswerContext
    from app.knowledge.retrieval import KnowledgeHit
    from tests.fakes import retrieval_outcome

    class _Retriever(SearchAsAnswerContext):
        async def search(self, query, top_k=None):
            return [KnowledgeHit("退货政策", "7 天无理由退货", "售后政策", 0.9)]

        async def retrieve_only(self, query, *, category=None):
            hits = await self.search(query)
            return retrieval_outcome(query, hits)

    # 知识出口：意图 + Agent 最终文本（带 [1]）
    client = TestClient(
        main.create_app(
            model=ScriptedChatModel(
                script=[_intent("product_consult"), text_turn("续航约 12 小时 [1]。")]
            ),
            settings=test_settings,
            retriever=_Retriever(),
        )
    )
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-know-api", "message": "XH-3000A 续航多久"},
    ) as resp:
        text = "".join(resp.iter_text())

    events = _events(text)
    names = [name for name, _ in events]
    assert "citations" in names
    assert "done" in names
    citations = next(data for name, data in events if name == "citations") or {}
    assert citations["citations"][0]["n"] == 1


def test_non_stream_response_carries_actions(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[_intent("complaint")]))
    body = client.post(
        "/api/chat/completions",
        json={"session_id": "s-nonstream", "message": "我要投诉", "stream": False},
    ).json()

    assert [item["action"] for item in body["actions"]] == ["transfer_human", "create_ticket"]
    assert body["route"] == "complaint"
    assert body["stop_reason"] == "fixed"


def test_messages_endpoint_returns_actions_after_refresh(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[_intent("complaint")]))
    client.post(
        "/api/chat/completions",
        json={"session_id": "s-actions-history", "message": "我要投诉", "stream": False},
    )

    messages = client.post("/api/chat/sessions/s-actions-history/messages").json()["messages"]
    assistant = [message for message in messages if message["role"] == "ai"][-1]
    assert assistant["actions"] == [
        {"action": "transfer_human", "ticket_type": "投诉"},
        {"action": "create_ticket", "ticket_type": "投诉"},
    ]


def test_tool_turn_restores_as_one_bot_bubble_with_badges(test_settings):
    """工具轮的前言不能单独落成一条 bot 气泡；否则刷新后序号错位、按钮锁串位。"""
    model = ScriptedChatModel(
        script=[
            _intent("logistics"),
            tool_turn(
                "query_logistics",
                {"order_id": "1001"},
                preamble="我帮您查一下。",
            ),
            text_turn("已发货，正在派送。"),
            _intent("complaint"),
        ]
    )
    client = _client(test_settings, model)
    client.post(
        "/api/chat/completions",
        json={"session_id": "s-restore", "message": "订单 1001 的物流", "stream": False},
    )
    client.post(
        "/api/chat/completions",
        json={"session_id": "s-restore", "message": "我要投诉", "stream": False},
    )

    messages = client.post("/api/chat/sessions/s-restore/messages").json()["messages"]
    # 与 web/app.js::restoreHistory 同一条过滤：空内容的工具申请单行不渲染。
    visible = [
        message
        for message in messages
        if not (
            message["role"] == "ai"
            and not (message.get("content") or "").strip()
            and "tools" in message
        )
    ]
    assistant_rows = [message for message in visible if message["role"] == "ai"]
    assert len(assistant_rows) == 2, visible
    assert assistant_rows[0]["content"] == "已发货，正在派送。"
    assert [tool["name"] for tool in assistant_rows[0]["tools"]] == ["query_logistics"]
    assert assistant_rows[1]["actions"]


def test_upstream_failure_persists_only_the_user_turn(test_settings):
    client = _client(test_settings, FailingChatModel(error=RuntimeError("boom")))
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-fail", "message": "订单 1001 的物流到哪了"},
    ) as resp:
        text = "".join(resp.iter_text())
    assert "event: error" in text

    messages = client.post("/api/chat/sessions/s-fail/messages").json()["messages"]
    assert [message["content"] for message in messages] == ["订单 1001 的物流到哪了"]
