"""重排客户端：请求形状、响应容错、降级。

## 为什么客户端不自己决定降级

``rerank()`` 拿不到结果就抛 ``RerankError``，由调用方（``HybridRetriever``）决定
「回落到 RRF 序」。客户端擅自吞掉故障，会让「重排一直没生效」变得无人发现 ——
而它恰恰是本章主要的提质手段。

## 为什么同分要按输入序稳定排序

重排服务对无关文档会给出**完全相同的分**（实测常见）。打平时若按服务端返回序或
随机序，两次跑评估会得到不同的 Recall/MRR —— 报告就不可复现了。
"""

from __future__ import annotations

import httpx
import pytest

from app.knowledge.rerank import (
    NoopReranker,
    RerankError,
    RerankHit,
    SiliconFlowReranker,
    get_reranker,
)

MODEL = "BAAI/bge-reranker-v2-m3"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _reranker(handler, **overrides) -> SiliconFlowReranker:
    options = {
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "model": MODEL,
        "timeout": 5.0,
        "client": _client(handler),
    }
    options.update(overrides)
    return SiliconFlowReranker(**options)


# ---------------------------------------------------------------------------
# 请求形状
# ---------------------------------------------------------------------------


async def test_request_body_matches_the_vendor_contract():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "results": [
                    {"index": 1, "relevance_score": 0.91, "document": None},
                    {"index": 0, "relevance_score": 0.12, "document": None},
                ],
                "meta": {},
            },
        )

    hits = await _reranker(handler).rerank("包邮", ["a", "b"], top_n=2)

    assert captured["url"] == "https://api.example.com/v1/rerank"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"] == {
        "model": MODEL,
        "query": "包邮",
        "documents": ["a", "b"],
        "top_n": 2,
        "return_documents": False,
    }
    assert hits == [RerankHit(index=1, score=0.91), RerankHit(index=0, score=0.12)]


async def test_base_url_trailing_slash_is_stripped():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"results": []})

    await _reranker(handler, base_url="https://api.example.com/v1/").rerank("q", ["a"], 1)
    assert captured["url"] == "https://api.example.com/v1/rerank"


# ---------------------------------------------------------------------------
# 响应容错
# ---------------------------------------------------------------------------


async def test_fewer_results_than_requested_are_returned_as_is():
    """服务端可能只回一条（它自己截断），不要补、不要抛。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": 0.7}]})

    hits = await _reranker(handler).rerank("q", ["a", "b", "c"], top_n=3)
    assert hits == [RerankHit(index=2, score=0.7)]


async def test_empty_results_are_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert await _reranker(handler).rerank("q", ["a"], top_n=1) == []


async def test_equal_scores_keep_input_order():
    """打平按 index 升序 —— 两次跑评估必须得到同一个顺序。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
        )

    hits = await _reranker(handler).rerank("q", ["a", "b", "c"], top_n=3)
    assert [hit.index for hit in hits] == [0, 1, 2]


async def test_results_are_truncated_to_top_n():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": 1 - i / 10} for i in range(5)]},
        )

    hits = await _reranker(handler).rerank("q", ["a", "b", "c", "d", "e"], top_n=2)
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# 失败语义
# ---------------------------------------------------------------------------


async def test_timeout_raises_rerank_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(RerankError):
        await _reranker(handler).rerank("q", ["a"], top_n=1)


async def test_server_error_raises_rerank_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    with pytest.raises(RerankError):
        await _reranker(handler).rerank("q", ["a"], top_n=1)


async def test_malformed_json_raises_rerank_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(RerankError):
        await _reranker(handler).rerank("q", ["a"], top_n=1)


# ---------------------------------------------------------------------------
# 无 Key 时的空实现
# ---------------------------------------------------------------------------


async def test_noop_reranker_keeps_input_order():
    hits = await NoopReranker().rerank("q", ["a", "b", "c"], top_n=2)
    assert [hit.index for hit in hits] == [0, 1]
    assert all(hit.score == 0.0 for hit in hits)


async def test_get_reranker_returns_noop_without_any_key():
    from app.config import Settings

    settings = Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        rerank_api_key="",
        embedding_api_key="",
    )
    assert isinstance(get_reranker(settings), NoopReranker)


async def test_get_reranker_uses_embedding_key_as_fallback():
    from app.config import Settings

    settings = Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        rerank_api_key="",
        embedding_api_key="sk-embed",
    )
    reranker = get_reranker(settings)
    assert isinstance(reranker, SiliconFlowReranker)
    assert reranker.api_key == "sk-embed"
