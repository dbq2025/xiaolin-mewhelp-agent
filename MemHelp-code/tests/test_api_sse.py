"""SSE 流式接口测试。

钉住三件事：
1. 事件名与顺序：``meta`` → ``token``* → ``done``；
2. 每个事件的 ``data`` 都是合法 JSON（前端直接 JSON.parse，不能有裸文本）；
3. **上游流到一半挂掉时补 ``event: error`` 再收尾**，而不是让连接裸断
   （Review Focus 5）。

## ch02 起「无工具路径的 token 不再是多块」

ch02 的编排为了拿到**确定的 ``tool_calls``**，第 1 次模型调用改成非流式
（见 ``app/chat_tools.py`` 的模块文档）。于是「模型不调工具」这条路上，
文本在第 1 次调用里就整段拿到了，只能**一次性**作为一个 token 帧发出；
真正逐块的流式发生在「调了工具、回灌之后」那一段。

所以这里的断言从「token 帧的个数」改成「各 token 帧拼起来的内容」：
个数是传输细节，内容才是不变量。逐块流式的证据改由
``test_streaming_after_a_tool_call_is_still_incremental`` 与
``tests/test_chat_tools.py`` 的工具路径用例提供。

## ``fail_after`` 的两种语义（ch02 起）

- ``fail_after=0``：第 1 次（非流式）调用就抛 —— 「一上来就挂」，只剩 ``error`` 帧。
- ``fail_after=N>=1``：第 1 次成功、流式那一段吐完 N 块后抛 —— 「流到一半挂」。

以前只有后者能表达，因为第 1 次调用是流式的。见 ``tests/fakes.py`` 的说明。
"""

import json
from typing import Any

from fastapi.testclient import TestClient

from app import main as app_main
from tests.fakes import FakeChatModel

#: 带工具调用的假模型要用到的 tool_call（第 1 次调用返回它）。
_TOOL_CALL = {
    "name": "query_logistics",
    "args": {"order_id": "1001"},
    "id": "call_1",
    "type": "tool_call",
}


def _client(test_settings, model=None) -> TestClient:
    return TestClient(
        app_main.create_app(
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


def _deltas(events) -> list[str]:
    return [d["delta"] for n, d in events if n == "token"]


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
    assert "token" in names

    meta = events[0][1] or {}
    assert meta["session_id"] == "s1"
    assert meta["model"]

    # 无工具路径：第 1 次非流式调用已拿到全文，一次性发出
    assert "".join(_deltas(events)) == "您好"

    done = events[-1][1] or {}
    assert done["finish_reason"] == "stop"
    assert done["trimmed"] == 0
    assert done["kept"] == 0
    assert done["tools"] == []


def test_sse_streams_the_whole_answer_not_a_prefix(test_settings):
    """ch02 改为断言内容而不是块数（见模块文档）。

    内容是契约（用户看到什么），块数是传输细节。
    """
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["一", "二", "三"])),
        {"message": "x"},
    )
    assert "".join(_deltas(_events(text))) == "一二三"


def test_streaming_after_a_tool_call_is_still_incremental(test_settings):
    """逐块流式的**证据转移到了工具路径**：第 2 次调用仍然是真流式。

    这条要保住「流式没有整体退化」这个断言 —— 只是它现在发生在回灌之后那一段。
    """
    text, _ = _stream(
        _client(
            test_settings,
            FakeChatModel(chunks=["已", "发", "货"], tool_calls_then_stream=[_TOOL_CALL]),
        ),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    deltas = _deltas(_events(text))

    assert deltas == ["已", "发", "货"], "工具回灌之后必须逐块流式"


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
    assert "" not in _deltas(_events(text))
    assert "".join(_deltas(_events(text))) == "好的"


def test_sse_sends_error_event_when_upstream_fails_midstream(test_settings):
    """Review Focus 5：流中途上游挂了，必须补 error 事件而不是裸断。

    用工具路径来表达「流中途挂」：第 1 次选择成功，第 2 次流式吐第 1 块之后抛
    （``fail_after=1`` 只作用于流式路径）。
    """
    text, _ = _stream(
        _client(
            test_settings,
            FakeChatModel(
                chunks=["好", "再也发不出来"],
                fail_after=1,
                tool_calls_then_stream=[_TOOL_CALL],
            ),
        ),
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


def test_sse_sends_error_event_when_the_first_call_fails(test_settings):
    """上游在第 1 次调用就挂（``fail_after=0``）：只剩 error 帧。"""
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["好"], fail_after=0)),
        {"session_id": "s1", "message": "在吗"},
    )
    names = [n for n, _ in _events(text)]

    assert names[0] == "meta"
    assert "token" not in names
    assert names[-1] == "error"


def test_sse_error_stream_persists_only_the_question(test_settings):
    """上游挂掉：只保留用户那一轮，不落库半截回答（半截会污染下一轮上下文）。"""
    client = _client(
        test_settings,
        FakeChatModel(
            chunks=["好", "再也发不出来"],
            fail_after=1,
            tool_calls_then_stream=[_TOOL_CALL],
        ),
    )
    _stream(client, {"session_id": "s1", "message": "在吗"})
    msgs = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [m["content"] for m in msgs] == ["在吗"]


def test_sse_emits_error_event_when_upstream_sends_only_empty_chunks(test_settings):
    """上游只吐空块：客户端要收到 error 事件，而不是「成功但回答是空的」。

    真实上游会这样：推理模型只给 reasoning 分块，或 content 是 content-block
    列表而文本在别处。以前这种响应会静默变成 finish_reason=stop 的空回答。
    """
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["", ""])),
        {"session_id": "s1", "message": "在吗"},
    )
    events = _events(text)
    names = [n for n, _ in events]

    assert names[0] == "meta"
    assert "done" not in names, "没有内容就不该发 done"
    assert names[-1] == "error"
    assert (events[-1][1] or {})["type"] == "upstream_empty_response"


def test_non_streaming_empty_response_is_an_error_not_empty_content(test_settings):
    """非流式路径同样不能把空内容当成功返回。"""
    client = _client(test_settings, FakeChatModel(reply=""))
    resp = client.post(
        "/api/chat/completions", json={"session_id": "s1", "message": "在吗", "stream": False}
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_empty_response"


def test_every_sse_data_field_is_valid_json(test_settings):
    text, _ = _stream(
        _client(test_settings, FakeChatModel(chunks=["a", "b", "c"])),
        {"message": "x"},
    )
    for name, data in _events(text):
        assert isinstance(data, dict), f"{name} 的 data 不是 JSON 对象"
