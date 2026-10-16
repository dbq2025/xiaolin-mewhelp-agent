"""ch02 的 HTTP 层测试：SSE 新契约、``/messages`` 读库、``/healthz`` 探库。

全部离线：注入假模型 + 接到 SQLite 内存库上的真仓库，不发网络请求、不连 MySQL。

## 这一层要钉住的三件事

1. **SSE 契约**：``status`` 帧在 ``token`` 之前到达；``done.tools`` 带完整成败信息。
   前端徽章完全依赖它 —— 契约漂移的表现是「徽章不显示」，而页面看起来正常。
2. **``/messages`` 读库且带 ``tools``**：徽章**刷新后仍在**的唯一保证。
3. **数据库故障的降级**：``/healthz`` 必须 503 且带可判读的字段，不能裸 500。
"""

import json
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeChatModel


def _tool_call(name="query_logistics", args=None, call_id="call_1"):
    return {
        "name": name,
        "args": args if args is not None else {"order_id": "1001"},
        "id": call_id,
        "type": "tool_call",
    }


def _client(test_settings, fake_repositories, model=None, db_ping=None) -> TestClient:
    return TestClient(
        create_app(
            model=model or FakeChatModel(chunks=["好"]),
            settings=test_settings,
            repositories=fake_repositories,
            db_ping=db_ping,
        )
    )


def _stream(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, dict | None]]:
    with client.stream("POST", "/api/chat/completions", json=payload) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200
    return _events(text)


def _events(text: str) -> list[tuple[str, dict | None]]:
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


def _tool_model(**overrides) -> FakeChatModel:
    kwargs: dict[str, Any] = {
        "chunks": ["已", "发货"],
        "tool_calls_then_stream": [_tool_call()],
        "reply": json.dumps(
            {"status": "运输中", "carrier": "顺丰速运"}, ensure_ascii=False
        ),
    }
    kwargs.update(overrides)
    return FakeChatModel(**kwargs)


# ---------------------------------------------------------------------------
# SSE 契约：status 帧
# ---------------------------------------------------------------------------


def test_status_frame_arrives_before_the_first_token(test_settings, fake_repositories):
    """用户在工具执行那几秒里要先看到「正在查物流…」，而不是一片空白。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    names = [n for n, _ in events]

    assert "status" in names
    assert names.index("status") < names.index("token")
    assert names[0] == "meta"
    assert names[-1] == "done"


def test_status_frame_payload_is_complete(test_settings, fake_repositories):
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    status = next(d for n, d in events if n == "status") or {}

    assert status["tool"] == "query_logistics"
    assert status["args"] == {"order_id": "1001"}
    assert status["label"].endswith("…")


def test_no_status_frame_when_the_model_does_not_call_a_tool(test_settings, fake_repositories):
    """不调工具是正常路径，不该有 status 帧（否则界面会闪一下假的「正在查询」）。"""
    events = _stream(
        _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"])),
        {"session_id": "s1", "message": "你好"},
    )
    assert "status" not in [n for n, _ in events]


# ---------------------------------------------------------------------------
# SSE 契约：done.tools
# ---------------------------------------------------------------------------


def test_done_frame_carries_the_tools_array(test_settings, fake_repositories):
    """``done.tools`` 是徽章的权威来源。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    done = next(d for n, d in events if n == "done") or {}

    assert isinstance(done["tools"], list)
    assert done["tools"][0]["name"] == "query_logistics"
    assert done["tools"][0]["ok"] is True
    assert "duration_ms" in done["tools"][0]


def test_done_frame_has_an_empty_tools_array_without_a_tool_call(test_settings, fake_repositories):
    events = _stream(
        _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"])),
        {"session_id": "s1", "message": "你好"},
    )
    done = next(d for n, d in events if n == "done") or {}
    assert done["tools"] == []


def test_done_frame_still_reports_context_stats(test_settings, fake_repositories):
    """ch01 的 ``trimmed`` / ``kept`` 不能被 ch02 挤掉 —— 前端统计开关还在用。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    done = next(d for n, d in events if n == "done") or {}

    assert done["finish_reason"] == "stop"
    assert done["trimmed"] == 0
    assert done["kept"] == 0
    assert done["session_id"] == "s1"


def test_every_ch02_sse_data_field_is_valid_json(test_settings, fake_repositories):
    for name, data in _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "查物流"},
    ):
        assert isinstance(data, dict), f"{name} 的 data 不是 JSON 对象"


# ---------------------------------------------------------------------------
# 落库与 /messages 读库
# ---------------------------------------------------------------------------


def test_tool_turn_is_persisted_and_readable_from_messages(test_settings, fake_repositories):
    """四行落库 + ``/messages`` 能读回来（含工具轨迹）。"""
    client = _client(test_settings, fake_repositories, _tool_model())
    _stream(client, {"session_id": "s1", "message": "订单 1001 的物流到哪了"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]

    assert [r["role"] for r in rows] == ["human", "ai", "tool", "ai"]
    assert rows[0]["content"] == "订单 1001 的物流到哪了"
    assert rows[3]["content"] == "已发货"


def test_messages_exposes_tools_for_each_assistant_row(test_settings, fake_repositories):
    """徽章**刷新后仍在**的唯一保证：``/messages`` 要回传每轮调了什么工具。"""
    client = _client(test_settings, fake_repositories, _tool_model())
    _stream(client, {"session_id": "s1", "message": "订单 1001 的物流到哪了"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    tool_turn = next(r for r in rows if r.get("tools"))

    assert tool_turn["tools"][0]["name"] == "query_logistics"
    assert tool_turn["tools"][0]["ok"] is True


def test_messages_for_a_plain_turn_has_no_tools(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    _stream(client, {"session_id": "s1", "message": "你好"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert all(not r.get("tools") for r in rows)


def test_messages_is_empty_for_an_unknown_session(test_settings, fake_repositories):
    """没聊过的会话返回空列表，而不是 404 —— 聊天页启动时会探一次。"""
    client = _client(test_settings, fake_repositories)
    resp = client.post("/api/chat/sessions/never-seen/messages")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_multi_turn_history_is_visible_on_the_second_turn(test_settings, fake_repositories):
    """第二轮必须能看到第一轮 —— 历史来自库而不是进程内存。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    _stream(client, {"session_id": "s1", "message": "第一句"})
    _stream(client, {"session_id": "s1", "message": "第二句"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == ["第一句", "你好", "第二句", "你好"]


# ---------------------------------------------------------------------------
# 非流式路径
# ---------------------------------------------------------------------------


def test_non_streaming_chat_still_works(test_settings, fake_repositories):
    """ch01 的 ``stream=false`` 契约不能破。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "你好"
    assert body["context"]["budget_tokens"] == 2000


def test_non_streaming_empty_response_is_still_an_error(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=[], reply=""))
    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_empty_response"


def test_non_streaming_turn_is_also_persisted(test_settings, fake_repositories):
    """非流式也必须落库。

    **为什么这条必须有**：ch02 的两条路（流式 / 非流式）如果历史来源不一致
    （一个读库、一个读内存），同一个 ``session_id`` 在两条路上会看到不同的历史 ——
    表现为「偶发答非所问」，而且极难复现。所以两条路共用同一个编排。
    """
    client = _client(test_settings, fake_repositories, FakeChatModel(reply="好的"))
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == ["在吗", "好的"]


def test_streaming_and_non_streaming_share_the_same_history(test_settings, fake_repositories):
    """两条路混用同一个会话时，历史必须是同一条。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(reply="好的"))
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "流式那句"},
    )
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "非流式那句", "stream": False},
    )

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == [
        "流式那句",
        "好的",
        "非流式那句",
        "好的",
    ]


# ---------------------------------------------------------------------------
# /healthz 探数据库（Review Focus 4）
# ---------------------------------------------------------------------------


def test_healthz_reports_database_ok_when_ping_succeeds(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, db_ping=lambda: True)
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_healthz_returns_503_when_the_database_is_down(test_settings, fake_repositories):
    """Review Focus 4：库挂了必须 503，且字段可判读 —— 不能裸 500。

    200 会让编排器继续把流量打进来，而每个请求都会失败。
    """
    client = _client(test_settings, fake_repositories, db_ping=lambda: False)
    resp = client.get("/healthz")

    assert resp.status_code == 503
    body = resp.json()
    assert body["database"] == "unavailable"
    assert body["status"] != "ok"


def test_healthz_database_failure_does_not_leak_the_dsn(test_settings, fake_repositories):
    """响应里不能出现连接串 —— 它含明文密码。"""
    client = _client(test_settings, fake_repositories, db_ping=lambda: False)
    raw = client.get("/healthz").text

    assert "mysql+pymysql" not in raw
    assert "mewhelp:mewhelp" not in raw


def test_healthz_without_injected_ping_still_answers(test_settings, fake_repositories):
    """没注入 ``db_ping`` 时走真实探测。

    这里用真仓库（SQLite）—— 探测会打到 SQLite 上，所以本用例不该依赖 MySQL。
    但 ``create_app`` 在没注入 repositories 时会去连 MySQL，所以这里仍注入仓库，
    只让 ``db_ping`` 缺省，验证「缺省路径不会崩」。
    """
    client = _client(test_settings, fake_repositories)
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    assert "database" in resp.json()
