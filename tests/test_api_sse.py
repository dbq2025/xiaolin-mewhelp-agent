"""SSE 流式接口测试。

钉住三件事：
1. 事件名与顺序：``meta`` → N 个 ``token`` → ``done``；
2. 每个事件的 ``data`` 都是合法 JSON（前端直接 JSON.parse，不能有裸文本）；
3. **上游流到一半挂掉时补 ``event: error`` 再收尾**，而不是让连接裸断
   （Review Focus 5）。
"""

import json
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeChatModel


def _client(test_settings, model=None) -> TestClient:
    return TestClient(
        create_app(
            model=model or FakeChatModel(chunks=["好"]),
            settings=test_settings,
        )
    )


def _stream(client: TestClient, payload: dict[str, Any]) -> tuple[str, dict[str, str]]:
    with client.stream("POST", "/api/chat/completions", json=payload) as resp:
        text = "".join(resp.iter_text())
        headers = dict(resp.headers)
        assert resp.status_code == 200
    return text, headers


def _events(text: str) -> list[tuple[str, dict | None]]:
    """把 SSE 文本解析成 (event, data) 列表。"""
    events: list[tuple[str, dict | None]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name or "", json.loads(data) if data else None))
    return events


def test_content_type_is_event_stream(test_settings):
    _, headers = _stream(_client(test_settings), {"session_id": "s1", "message": "在吗"})
    assert headers["content-type"].startswith("text/event-stream")


def test_sse_event_sequence_and_payloads(test_settings):
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["您", "好"])),
        {"session_id": "s1", "message": "在吗"},
    )
    events = _events(text)
    names = [n for n, _ in events]

    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names.count("token") == 2

    meta = events[0][1] or {}
    assert meta["session_id"] == "s1"
    assert meta["model"]

    assert [d for n, d in events if n == "token"] == [{"delta": "您"}, {"delta": "好"}]

    done = events[-1][1] or {}
    assert done["finish_reason"] == "stop"
    assert done["trimmed"] == 0
    assert done["kept"] == 0


def test_sse_streams_content_incrementally_not_one_blob(test_settings):
    """真流式的证据：token 事件是分开的多个，而不是一个装全部内容的块。"""
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["一", "二", "三"])),
        {"message": "x"},
    )
    token_events = [d for n, d in _events(text) if n == "token"]
    assert len(token_events) == 3
    assert "".join(d["delta"] for d in token_events) == "一二三"


def test_sse_works_without_session_id_and_returns_generated_one(test_settings):
    text, _ = _stream(_client(test_settings, FakeChatModel(chunks=["好"])), {"message": "x"})
    meta = _events(text)[0][1] or {}
    assert len(meta["session_id"]) == 12


def test_sse_does_not_emit_empty_delta_events(test_settings):
    """上游会吐空串尾块，不能变成空的 token 事件。"""
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["好", "", "的"])),
        {"message": "x"},
    )
    deltas = [d["delta"] for n, d in _events(text) if n == "token"]
    assert deltas == ["好", "的"]


def test_sse_sends_error_event_when_upstream_fails_midstream(test_settings):
    """Review Focus 5：流中途上游挂了，必须补 error 事件而不是裸断。

    注意必须让失败发生在**流中途**：`fail_after=1` 表示吐完第 1 块之后抛错，
    所以 chunks 至少要有 2 块，否则循环一次就正常结束，压根不会失败。
    """
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["好", "再也发不出来"], fail_after=1)),
        {"session_id": "s1", "message": "在吗"},
    )
    events = _events(text)
    names = [n for n, _ in events]

    assert names[0] == "meta"
    assert "token" in names
    assert names[-1] == "error"

    err = events[-1][1] or {}
    assert err["type"] == "upstream_error"
    assert err["message"]


def test_sse_error_stream_still_persists_history(test_settings):
    """上游挂掉也不能让这一轮从历史里消失。"""
    client = _client(
        test_settings, FakeChatModel(chunks=["好", "再也发不出来"], fail_after=1)
    )
    _stream(client, {"session_id": "s1", "message": "在吗"})
    msgs = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [m["content"] for m in msgs] == ["在吗", "好"]


def test_every_sse_data_field_is_valid_json(test_settings):
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["a", "b", "c"])),
        {"message": "x"},
    )
    for name, data in _events(text):
        assert isinstance(data, dict), f"{name} 的 data 不是 JSON 对象"
