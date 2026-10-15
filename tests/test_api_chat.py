"""非流式接口测试。

验收标准 2（第二轮接住上下文）在 HTTP 层的证据：两轮之后用
`POST /api/chat/sessions/{id}/messages` 能看到完整历史。
验收标准 3（拿到结构化 JSON）在 HTTP 层的证据：`POST /api/extract`。

`create_app(model=..., store=..., settings=...)` 支持注入假模型、独立会话存储和
写死的配置，所以这些测试既不联网、不读 .env，也互不干扰。
"""

from fastapi.testclient import TestClient

from app.main import create_app
from app.memory import InMemorySessionStore
from tests.fakes import FakeChatModel


def _client(test_settings, model=None, store=None) -> TestClient:
    return TestClient(
        create_app(
            model=model or FakeChatModel(reply="好的"),
            store=store,
            settings=test_settings,
        )
    )


def test_healthz_does_not_leak_api_key(test_settings):
    """密钥不得出现在任何响应里。"""
    body = _client(test_settings).get("/healthz").json()
    assert body["status"] == "ok"
    assert body["base_url_configured"] is True
    assert "api_key" not in str(body).lower()
    assert "sk-" not in str(body)


def test_healthz_reports_model_and_budget(test_settings):
    body = _client(test_settings).get("/healthz").json()
    assert body["model"]
    assert body["context_token_budget"] > 0


def test_non_streaming_chat_returns_full_json(test_settings):
    resp = _client(test_settings, FakeChatModel(reply="您好，我帮您查一下")).post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["content"] == "您好，我帮您查一下"
    assert body["context"]["budget_tokens"] > 0
    assert body["context"]["kept_messages"] == 0
    assert body["context"]["trimmed_messages"] == 0


def test_non_streaming_chat_generates_session_id_when_absent(test_settings):
    body = _client(test_settings).post(
        "/api/chat/completions", json={"message": "在吗", "stream": False}
    ).json()
    assert len(body["session_id"]) == 12


def test_second_turn_history_is_visible_via_sessions_endpoint(test_settings):
    """验收标准 2 的 HTTP 层证据。"""
    client = _client(test_settings, FakeChatModel(reply="答"))
    for turn in ("第一轮", "第二轮"):
        client.post(
            "/api/chat/completions",
            json={"session_id": "s1", "message": turn, "stream": False},
        )
    msgs = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [m["content"] for m in msgs] == ["第一轮", "答", "第二轮", "答"]
    assert [m["role"] for m in msgs] == ["human", "ai", "human", "ai"]


def test_unknown_session_returns_empty_history(test_settings):
    body = _client(test_settings).post("/api/chat/sessions/none/messages").json()
    assert body == {"session_id": "none", "messages": []}


def test_context_stats_reflect_trimming(test_settings):
    """历史塞满小预算后，响应里的统计要如实反映裁剪。

    用一个很小的预算（而不是默认 2000）：`count_tokens_approximately` 对重复
    中文估得偏低，用默认预算根本撑不满，测不出裁剪。
    """
    store = InMemorySessionStore()
    small_budget = test_settings.model_copy(update={"context_token_budget": 40})
    client = _client(small_budget, FakeChatModel(reply="答" * 20), store=store)
    for i in range(5):
        client.post(
            "/api/chat/completions",
            json={"session_id": "s1", "message": f"第{i}轮问题", "stream": False},
        )
    body = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "继续", "stream": False},
    ).json()
    assert body["context"]["trimmed_messages"] > 0
    assert body["context"]["budget_tokens"] == 40


def test_extract_endpoint_returns_structured_json(test_settings):
    """验收标准 3 的 HTTP 层证据。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": "20240115001",
            "request_type": "logistics",
            "expected_solution": "refund",
            "detail": "订单未按时送达",
        }
    )
    resp = _client(test_settings, model).post(
        "/api/extract", json={"text": "订单 20240115001 没按时到，我要退款"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["order_no"] == "20240115001"
    assert body["request_type"] == "logistics"
    assert body["expected_solution"] == "refund"
    assert body["detail"] == "订单未按时送达"


def test_extract_endpoint_returns_null_order_no_when_absent(test_settings):
    """Review Focus 4 在 HTTP 层的证据：不编造订单号。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "logistics",
            "expected_solution": "unknown",
            "detail": "快递没动",
        }
    )
    body = _client(test_settings, model).post(
        "/api/extract", json={"text": "我的快递一直没动"}
    ).json()
    assert body["order_no"] is None


def test_extract_endpoint_422_when_model_output_unparsable(test_settings):
    model = FakeChatModel(reply="我不太明白")
    resp = _client(test_settings, model).post("/api/extract", json={"text": "x"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "extraction_failed"


def test_empty_message_is_rejected(test_settings):
    resp = _client(test_settings).post(
        "/api/chat/completions", json={"message": "", "stream": False}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_unknown_session_id_is_rejected(test_settings):
    resp = _client(test_settings).post(
        "/api/chat/completions",
        json={"session_id": "with spaces!", "message": "hi", "stream": False},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_sessions_are_isolated_over_http(test_settings):
    """Review Focus 2 在 HTTP 层：甲的历史不能出现在乙的会话里。"""
    store = InMemorySessionStore()
    client = _client(test_settings, FakeChatModel(reply="答"), store=store)
    client.post(
        "/api/chat/completions",
        json={"session_id": "a", "message": "甲的问题", "stream": False},
    )
    client.post(
        "/api/chat/completions",
        json={"session_id": "b", "message": "乙的问题", "stream": False},
    )
    a = [m["content"] for m in client.post("/api/chat/sessions/a/messages").json()["messages"]]
    b = [m["content"] for m in client.post("/api/chat/sessions/b/messages").json()["messages"]]
    assert a == ["甲的问题", "答"]
    assert b == ["乙的问题", "答"]
