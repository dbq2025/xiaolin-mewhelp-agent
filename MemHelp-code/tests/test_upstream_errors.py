"""上游故障在 HTTP 层的映射测试。

这类错误在离线环境里最容易漏测：真实上游连不上时才发现「同一个连接错误，
两个接口给出了不同的错误类型」。所以用假模型抛**与真实上游同类型同消息**的
异常来离线复现（``openai.APIConnectionError`` 的 str 就是 ``"Connection error."``）。

要求：
- 基础设施错误（连不上 / 认证失败）无论走哪个接口，都应该映射成
  ``upstream_*`` + 502/503，消费者要能据此判断「该重试」还是「该查密钥」。
- 只有「模型返回了东西但不符合 schema」才算 ``extraction_failed`` + 422。
"""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeChatModel, FailingChatModel


def _connection_error() -> Exception:
    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")
    )


def _auth_error() -> Exception:
    return openai.AuthenticationError(
        "Incorrect API key provided",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")
        ),
        body=None,
    )


def _bad_model_error() -> Exception:
    """真实遇到过的形态：DeepSeek 对不存在的模型名回 400。"""
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'The supported API model names are "
        "deepseek-flash, deepseek-v4-pro, but you passed deepseek-v4.1-flash.', "
        "'type': 'invalid_request_error'}}",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        ),
        body=None,
    )


@pytest.mark.parametrize(
    ("error", "expected_type", "expected_status"),
    [
        (_connection_error(), "upstream_unavailable", 502),
        (_auth_error(), "upstream_auth", 502),
        (_bad_model_error(), "upstream_invalid_request", 502),
    ],
    ids=["connection", "auth", "bad-model-name"],
)
def test_extract_maps_infrastructure_errors_to_upstream_types(
    test_settings, error, expected_type, expected_status
):
    """缺陷复现（两轮）：连接错误与「模型名写错」都曾被误报成 422 extraction_failed。

    那样调用方会以为「这段文本抽不出来」，实际是上游不可达或 .env 里的模型名不对
    —— 排查方向完全错。评估脚本也会因此建议「去改 Prompt」，把功夫花在错地方。
    """
    client = TestClient(create_app(model=FailingChatModel(error=error), settings=test_settings))
    resp = client.post("/api/extract", json={"text": "订单 123 要退款"})
    assert resp.status_code == expected_status
    assert resp.json()["error"]["type"] == expected_type


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (_connection_error(), "upstream_unavailable"),
        (_auth_error(), "upstream_auth"),
        (_bad_model_error(), "upstream_invalid_request"),
    ],
    ids=["connection", "auth", "bad-model-name"],
)
def test_chat_maps_infrastructure_errors_to_upstream_types(
    test_settings, error, expected_type
):
    client = TestClient(create_app(model=FailingChatModel(error=error), settings=test_settings))
    resp = client.post(
        "/api/chat/completions", json={"message": "在吗", "stream": False}
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == expected_type


def test_streaming_reports_infrastructure_error_as_sse_error_event(test_settings):
    """流式路径：上游连不上时，客户端要收到 error 事件（而不是裸断的流）。"""
    client = TestClient(
        create_app(model=FailingChatModel(error=_connection_error()), settings=test_settings)
    )
    with client.stream(
        "POST", "/api/chat/completions", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    assert "event: meta" in text
    assert "event: error" in text
    assert "upstream_unavailable" in text


def test_extraction_failed_is_reserved_for_schema_mismatch(test_settings):
    """反面对照：模型「有输出但不符合 schema」才该是 422 extraction_failed。"""
    client = TestClient(
        create_app(model=FakeChatModel(reply="我不太明白"), settings=test_settings)
    )
    resp = client.post("/api/extract", json={"text": "随便说说"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "extraction_failed"


# ---------------------------------------------------------------------------
# 错误响应外壳在**所有**路径上必须一致：{"error": {"type", "message"}}
# 评审指出：早期实现只把非流式对话的错误外壳改对了，extract 与 SSE 两条路径
# 没被断言，容易回退。
# ---------------------------------------------------------------------------


def test_extract_error_uses_the_error_envelope(test_settings):
    client = TestClient(
        create_app(model=FailingChatModel(error=_connection_error()), settings=test_settings)
    )
    body = client.post("/api/extract", json={"text": "订单 123"}).json()
    assert set(body) == {"error"}, "错误响应体只能是 error 外壳"
    assert set(body["error"]) >= {"type", "message"}


def test_chat_validation_error_uses_the_error_envelope(test_settings):
    """请求体非法时也要走 {"error": {...}} 外壳。

    **必须同时注入 model**：FastAPI 先解析依赖、后校验请求体，只注入 settings
    的话模型依赖会先因「配置缺失」失败，拿到的就是 configuration_error 而不是
    validation_error（这个差异是实测出来的，不是猜的）。
    """
    client = TestClient(
        create_app(model=FakeChatModel(reply="ok"), settings=test_settings)
    )
    body = client.post("/api/chat/completions", json={"message": ""}).json()
    assert set(body) == {"error"}
    assert body["error"]["type"] == "validation_error"


def test_sse_error_event_uses_the_same_shape_as_json_errors(test_settings):
    """SSE 的 error 事件与 JSON 错误体带同样的字段，客户端可以共用解析逻辑。"""
    import json

    client = TestClient(
        create_app(model=FailingChatModel(error=_connection_error()), settings=test_settings)
    )
    with client.stream(
        "POST", "/api/chat/completions", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        text = "".join(resp.iter_text())

    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    error_block = next(b for b in blocks if "event: error" in b)
    payload = json.loads(error_block.split("data: ", 1)[1].strip())
    assert set(payload) >= {"type", "message", "session_id"}


def test_secrets_are_redacted_in_error_messages(test_settings):
    """上游异常原文里若带密钥，回给客户端时必须是脱敏的。"""
    leaky = RuntimeError("boom with key sk-abcdef1234567890 in it")
    client = TestClient(create_app(model=FailingChatModel(error=leaky), settings=test_settings))
    body = client.post("/api/extract", json={"text": "x"}).text
    assert "sk-abcdef1234567890" not in body
    assert "sk-***" in body


def test_secrets_are_redacted_in_sse_error_events(test_settings):
    leaky = RuntimeError("boom with key sk-abcdef1234567890 in it")
    client = TestClient(create_app(model=FailingChatModel(error=leaky), settings=test_settings))
    with client.stream(
        "POST", "/api/chat/completions", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        text = "".join(resp.iter_text())
    assert "sk-abcdef1234567890" not in text
    assert "sk-***" in text


def test_configuration_error_message_is_redacted(monkeypatch):
    """配置缺失时 pydantic 会带出 input_value（含密钥前几位），必须脱敏。"""
    from app.config import Settings, get_settings
    from app.main import create_app

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-partial")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        "app.config.Settings.model_config",
        {**Settings.model_config, "env_file": "/nonexistent/.env"},
    )
    get_settings.cache_clear()

    resp = TestClient(create_app()).post(
        "/api/chat/completions", json={"message": "在吗", "stream": False}
    )
    assert resp.status_code == 503
    assert "sk-super-secret" not in resp.text


# ---------------------------------------------------------------------------
# 配置完全缺失（既没有 .env 也没有环境变量）
# ---------------------------------------------------------------------------


def test_missing_config_yields_503_not_an_unhandled_exception():
    """配置全缺时必须给出规范的 503 JSON，不能把未处理异常抛到 ASGI 层。

    这里**刻意不注入 settings**，走 ``get_app_settings`` 的真实分支。
    此前的实现让 ``pydantic.ValidationError`` 穿透了异常处理器，客户端拿到的是
    "Exception in ASGI application" 而不是可解析的错误体 —— 这个缺陷由
    「从 GitHub 全新克隆后跑测试」暴露出来（开发机上有 .env 所以一直没触发）。
    """
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    client = TestClient(create_app(), raise_server_exceptions=False)

    for stream in (False, True):
        resp = client.post(
            "/api/chat/completions", json={"message": "在吗", "stream": stream}
        )
        assert resp.status_code == 503, f"stream={stream} 没有得到 503"
        assert "Exception in ASGI application" not in resp.text
        if not stream:
            assert resp.json()["error"]["type"] == "configuration_error"
