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


@pytest.mark.parametrize(
    ("error", "expected_type", "expected_status"),
    [
        (_connection_error(), "upstream_unavailable", 502),
        (_auth_error(), "upstream_auth", 502),
    ],
    ids=["connection", "auth"],
)
def test_extract_maps_infrastructure_errors_to_upstream_types(
    test_settings, error, expected_type, expected_status
):
    """缺陷复现：连接错误曾被误报成 422 extraction_failed。

    那样调用方会以为是「这段文本抽不出来」，实际是上游不可达 —— 排查方向完全错。
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
    ],
    ids=["connection", "auth"],
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
