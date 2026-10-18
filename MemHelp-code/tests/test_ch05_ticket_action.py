"""ch05 Task 10：建工单端点（唯一会写 tickets 表的用户动作）。"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app import main
from app.db.models import Ticket
from app.tools.executor import ToolResult
from tests.fakes import FakeChatModel


def _client(test_settings, fake_repositories):
    return TestClient(
        main.create_app(
            model=FakeChatModel(chunks=["ok"]),
            settings=test_settings,
            repositories=fake_repositories,
        )
    )


def test_create_ticket_writes_tickets_table(test_settings, fake_repositories, sqlite_session):
    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-ticket", "description": "杯子碎了，我要投诉", "ticket_type": "投诉"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T")
    assert body["ticket_type"] == "投诉"

    sqlite_session.expire_all()
    row = sqlite_session.get(Ticket, body["ticket_no"])
    assert row is not None
    assert row.description == "杯子碎了，我要投诉"
    assert row.ticket_type == "投诉"


async def test_description_falls_back_to_last_user_message(
    test_settings, fake_repositories, sqlite_session
):
    conversation_id = await fake_repositories.conversations.upsert_by_session(
        "s-fallback", "anonymous"
    )
    await fake_repositories.messages.append(
        conversation_id, role="user", content="订单 1001 一直没送到"
    )

    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-fallback", "ticket_type": "售后"},
    )

    assert resp.status_code == 200
    sqlite_session.expire_all()
    row = sqlite_session.get(Ticket, resp.json()["ticket_no"])
    assert row is not None
    assert row.description == "订单 1001 一直没送到"


def test_invalid_ticket_type_is_422(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-bad-type", "description": "x", "ticket_type": "删除订单"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_empty_description_is_422(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-empty", "ticket_type": "售后"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_database_failure_reading_last_user_message_is_503(
    test_settings, fake_repositories, monkeypatch
):
    async def boom(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("db down"))

    monkeypatch.setattr(fake_repositories.messages, "last_user_content", boom)
    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-db-down", "ticket_type": "售后"},
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "database_error"


def test_database_failure_from_tool_is_503(test_settings, fake_repositories, monkeypatch):
    async def fake_execute(*args, **kwargs):
        return ToolResult(
            tool="create_ticket",
            args={},
            ok=False,
            data=None,
            error="db down",
            attempts=1,
            fault="database",
        )

    monkeypatch.setattr("app.api.execute_tool", fake_execute)
    client = _client(test_settings, fake_repositories)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s-tool-db", "description": "x", "ticket_type": "售后"},
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "database_error"


def test_same_idempotency_key_returns_the_same_ticket(
    test_settings, fake_repositories, sqlite_session
):
    client = _client(test_settings, fake_repositories)
    payload = {
        "session_id": "s-idem",
        "description": "我要投诉",
        "ticket_type": "投诉",
        "idempotency_key": "s-idem:0:create_ticket",
    }
    first = client.post("/api/actions/create-ticket", json=payload)
    second = client.post("/api/actions/create-ticket", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["ticket_no"] == second.json()["ticket_no"]
    assert sqlite_session.query(Ticket).count() == 1

    third = client.post(
        "/api/actions/create-ticket",
        json={**payload, "idempotency_key": "s-idem:1:create_ticket"},
    )
    assert third.json()["ticket_no"] != first.json()["ticket_no"]
    assert sqlite_session.query(Ticket).count() == 2
