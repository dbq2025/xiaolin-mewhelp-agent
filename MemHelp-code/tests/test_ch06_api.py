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


def _start_picker(client, model, session: str) -> str:
    """跑到订单选择器，返回第一张卡的订单号。"""
    if not model.script:
        model.script.extend([_coref("这个商品能退吗", True), _intent("refund_return")])
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": session, "message": "这个能退吗"},
    ) as resp:
        events = _events("".join(resp.iter_text()))
    picker = next(data for name, data in events if name == "order_picker")
    return picker["orders"][0]["order_id"]


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


def test_refund_select_twice_after_done_returns_409(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    order_id = _start_picker(client, model, "s-select-twice")
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])
    with client.stream(
        "POST",
        "/api/chat/refund/select",
        json={"session_id": "s-select-twice", "order_id": order_id},
    ) as resp:
        "".join(resp.iter_text())
    again = client.post(
        "/api/chat/refund/select",
        json={"session_id": "s-select-twice", "order_id": order_id},
    )
    assert again.status_code == 409
    assert again.json()["error"]["type"] == "no_pending_refund"


def test_resume_failure_recovers_and_persists_user_turn(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    order_id = _start_picker(client, model, "s-resume-fail")
    with client.stream(
        "POST",
        "/api/chat/refund/select",
        json={"session_id": "s-resume-fail", "order_id": order_id},
    ) as resp:
        text = "".join(resp.iter_text())
    assert "event: error" in text

    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-resume-fail", "message": "你好"},
    ) as resp:
        after = "".join(resp.iter_text())
    assert "event: done" in after

    history = client.post("/api/chat/sessions/s-resume-fail/messages").json()
    contents = [item["content"] for item in history["messages"]]
    assert "这个能退吗" in contents


def test_refund_select_rejects_concurrent_resume(test_settings):
    class _Interrupt:
        value = {"kind": "order_picker", "orders": [{"order_id": "1001"}]}

    class _Task:
        interrupts = (_Interrupt(),)

    class _Snapshot:
        next = ("refund_await_order",)
        tasks = (_Task(),)
        values = {"turn_id": "t"}

    class _FakeGraph:
        async def aget_state(self, config):
            return _Snapshot()

    from app.api import _RESUME_INFLIGHT

    client = TestClient(
        main.create_app(
            model=ScriptedChatModel(script=[]),
            settings=test_settings,
            agent_graph=_FakeGraph(),
        )
    )
    _RESUME_INFLIGHT.add("s-inflight")
    try:
        resp = client.post(
            "/api/chat/refund/select",
            json={"session_id": "s-inflight", "order_id": "1001"},
        )
    finally:
        _RESUME_INFLIGHT.discard("s-inflight")
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "refund_resume_in_progress"


def test_refund_select_requires_an_actual_interrupt(test_settings):
    class _Snapshot:
        next = ("refund_await_order",)
        tasks = ()
        values = {}

    class _FakeGraph:
        async def aget_state(self, config):
            return _Snapshot()

    client = TestClient(
        main.create_app(
            model=ScriptedChatModel(script=[]),
            settings=test_settings,
            agent_graph=_FakeGraph(),
        )
    )
    resp = client.post(
        "/api/chat/refund/select",
        json={"session_id": "s-fake", "order_id": "1001"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "no_pending_refund"


def test_non_stream_pause_returns_awaiting_input(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s-nonstream", "message": "这个能退吗", "stream": False},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["content"] == "请选择要办理的订单。"
    assert payload["awaiting_input"]["orders"]
    assert payload["stop_reason"] == "awaiting_input"


def test_create_refund_validates_reason(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[]))
    resp = client.post(
        "/api/actions/create-refund",
        json={
            "session_id": "s1",
            "order_id": "1001",
            "reason": "瞎编的原因",
            "idempotency_key": "refund-1",
        },
    )
    assert resp.status_code == 422


def test_create_refund_rejects_order_outside_flow(test_settings):
    client = _client(test_settings, ScriptedChatModel(script=[]))
    resp = client.post(
        "/api/actions/create-refund",
        json={
            "session_id": "s-no-flow",
            "order_id": "9999",
            "reason": "商品质量问题",
            "idempotency_key": "refund-1",
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "refund_order_mismatch"


def test_create_refund_idempotent(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    order_id = _start_picker(client, model, "s-refund-idem")
    body = {
        "session_id": "s-refund-idem",
        "order_id": order_id,
        "reason": "商品质量问题",
        "idempotency_key": "refund-1",
    }
    first = client.post("/api/actions/create-refund", json=body)
    second = client.post("/api/actions/create-refund", json=body)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["refund_no"] == second.json()["refund_no"]
    assert first.json()["order_id"] == order_id
    assert first.json()["reason"] == "商品质量问题"


def test_create_refund_same_key_different_reason_conflicts(test_settings):
    model = ScriptedChatModel(
        script=[_coref("这个商品能退吗", True), _intent("refund_return")]
    )
    client = _client(test_settings, model)
    order_id = _start_picker(client, model, "s-refund-conflict")
    first = client.post(
        "/api/actions/create-refund",
        json={
            "session_id": "s-refund-conflict",
            "order_id": order_id,
            "reason": "商品质量问题",
            "idempotency_key": "refund-1",
        },
    )
    assert first.status_code == 200
    second = client.post(
        "/api/actions/create-refund",
        json={
            "session_id": "s-refund-conflict",
            "order_id": order_id,
            "reason": "与描述不符",
            "idempotency_key": "refund-1",
        },
    )
    assert second.status_code == 409
    assert second.json()["error"]["type"] == "idempotency_conflict"
