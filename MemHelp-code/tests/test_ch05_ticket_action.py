"""ch05 Task 10：建工单端点（唯一会写 tickets 表的用户动作）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main
from app.db.models import Ticket
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
