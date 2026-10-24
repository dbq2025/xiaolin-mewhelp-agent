"""ch08 Task 11：工单预览卡片的 HTTP 接口。

前端只做两件事：把 interrupt 里的预览画成卡片；用户点按钮时 POST
``/api/chat/ticket/confirm``。确认/取消都由图内节点处理，接口本身不碰
tickets 表，也不自己补审计。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import main
from app.db.models import Ticket
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


def _intent(intent: str = "logistics"):
    return text_turn(
        json.dumps({"intent": intent, "confidence": 0.9}, ensure_ascii=False)
    )


def _events(text: str) -> list[tuple[str | None, dict | None]]:
    events: list[tuple[str | None, dict | None]] = []
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


def _client(test_settings, model) -> TestClient:
    return TestClient(main.create_app(model=model, settings=test_settings))


def _args(session: str) -> dict:
    return {
        "description": "耳机左声道没有声音",
        "ticket_type": "售后",
        "conversation_id": session,
    }


def _start_preview(client, model, session: str, args: dict) -> list[tuple]:
    model.script.extend(
        [
            _intent("logistics"),
            tool_turn("create_ticket", args, call_id="call_ticket"),
        ]
    )
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": session, "message": "帮我建个工单"},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200
    return _events(text)


def test_chat_pauses_with_ticket_preview(test_settings):
    session = "s-ticket-preview"
    model = ScriptedChatModel()
    client = _client(test_settings, model)
    events = _start_preview(client, model, session, _args(session))

    names = [name for name, _ in events]
    assert "awaiting_input" in names
    assert "done" not in names
    awaiting = next(data for name, data in events if name == "awaiting_input")
    assert awaiting["kind"] == "ticket_preview"
    assert awaiting["call_id"] == "call_ticket"
    assert awaiting["preview"] == {
        "ticket_type": "售后",
        "description": "耳机左声道没有声音",
    }
    assert awaiting["message"] == "请确认是否提交这张工单。"


def test_confirm_creates_ticket_and_returns_number(
    test_settings, fake_repositories, sqlite_session
):
    session = "s-ticket-confirm"
    model = ScriptedChatModel()
    client = _client(test_settings, model)
    _start_preview(client, model, session, _args(session))

    with client.stream(
        "POST",
        "/api/chat/ticket/confirm",
        json={"session_id": session, "confirmed": True},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    tickets = sqlite_session.scalars(select(Ticket)).all()
    assert len(tickets) == 1
    token_text = "".join(
        str(data.get("delta") or "")
        for name, data in _events(text)
        if name == "token" and isinstance(data, dict)
    )
    assert tickets[0].ticket_no in token_text

    import anyio

    rows = anyio.run(fake_repositories.audit.list_recent, 5)
    created = [row for row in rows if row.tool_name == "create_ticket"]
    assert created and created[0].status == "成功"


def test_cancel_creates_nothing_and_audits_denial(
    test_settings, fake_repositories, sqlite_session
):
    import anyio

    session = "s-ticket-cancel"
    model = ScriptedChatModel()
    client = _client(test_settings, model)
    _start_preview(client, model, session, _args(session))

    with client.stream(
        "POST",
        "/api/chat/ticket/confirm",
        json={"session_id": session, "confirmed": False},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    assert sqlite_session.scalars(select(Ticket)).all() == []
    token_text = "".join(
        str(data.get("delta") or "")
        for name, data in _events(text)
        if name == "token" and isinstance(data, dict)
    )
    assert "取消" in token_text
    rows = anyio.run(fake_repositories.audit.list_recent, 5)
    denied = [
        row
        for row in rows
        if row.tool_name == "create_ticket" and row.status == "权限拒绝"
    ]
    assert denied and "取消" in (denied[0].error_message or "")


def test_confirm_without_pending_ticket_returns_409(test_settings):
    client = _client(test_settings, ScriptedChatModel())
    resp = client.post(
        "/api/chat/ticket/confirm",
        json={"session_id": "no-such-session", "confirmed": True},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "no_pending_ticket"


def test_confirm_twice_after_done_returns_409(test_settings):
    session = "s-ticket-twice"
    model = ScriptedChatModel()
    client = _client(test_settings, model)
    _start_preview(client, model, session, _args(session))

    with client.stream(
        "POST",
        "/api/chat/ticket/confirm",
        json={"session_id": session, "confirmed": True},
    ) as resp:
        "".join(resp.iter_text())

    again = client.post(
        "/api/chat/ticket/confirm",
        json={"session_id": session, "confirmed": True},
    )
    assert again.status_code == 409
    assert again.json()["error"]["type"] == "no_pending_ticket"
