"""ch06 Task 10：订单续跑、退款单与 SSE 新事件。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import main
from tests.fakes_agent import ScriptedChatModel, text_turn


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


def _client(test_settings, model):
    return TestClient(main.create_app(model=model, settings=test_settings))


def test_chat_stream_emits_order_picker_and_awaiting(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-picker-api", "message": "这个能退吗"},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    events = _events(text)
    names = [name for name, _ in events]
    assert "order_picker" in names
    assert "awaiting_input" in names
    assert "done" not in names


def test_refund_select_resumes_and_done(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-select-api", "message": "这个能退吗"},
    ) as resp:
        first = _events("".join(resp.iter_text()))
    picker = next(data for name, data in first if name == "order_picker")
    order_id = picker["orders"][0]["order_id"]
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])

    with client.stream(
        "POST",
        "/api/chat/refund/select",
        json={"session_id": "s-select-api", "order_id": order_id},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    events = _events(text)
    done = next(data for name, data in events if name == "done")
    assert done["refund_form"]["order_id"] == order_id
    assert "refund_agent" in done["trace"]


def test_refund_select_without_pending_returns_409(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[]))
    resp = client.post(
        "/api/chat/refund/select",
        json={"session_id": "no-such-session", "order_id": "1001"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "no_pending_refund"


def test_create_refund_validates_reason(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[]))
    resp = client.post(
        "/api/actions/create-refund",
        json={
            "session_id": "s1",
            "order_id": "1001",
            "reason": "瞎编的原因",
        },
    )
    assert resp.status_code == 422


def test_create_refund_idempotent(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[]))
    body = {
        "session_id": "s1",
        "order_id": "1001",
        "reason": "商品质量问题",
        "idempotency_key": "refund-1",
    }
    first = client.post("/api/actions/create-refund", json=body)
    second = client.post("/api/actions/create-refund", json=body)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["refund_no"] == second.json()["refund_no"]
