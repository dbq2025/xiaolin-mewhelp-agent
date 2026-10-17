"""嵌入器 —— 硅基流动云端 BGE-M3（OpenAI 兼容 HTTP）。

**默认不发任何网络请求**：用 ``httpx.MockTransport`` 当假服务端，把「请求长什么样、
响应怎么解析」钉死。真实 API 的用例在文件末尾，由 ``MEMHELP_TEST_EMBED=1`` 开启。

这一层要钉住的四件事（全部是云端特有的失败模式）：

1. **请求形状**：`{base}/embeddings`、`Authorization: Bearer <key>`、
   body 含 `model` / `input` / `encoding_format=float`；
2. **按 `index` 排序**：服务端不保证返回顺序等于输入顺序，错位不会报错，
   只会让检索答非所问；
3. **重试边界**：429/5xx 退避重试，4xx 立刻失败；
4. **失败要早**：没密钥在发请求前就抛可读错误。
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from app.knowledge.embedder import SiliconFlowEmbedder


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _embedder(handler, *, dim=4, batch_size=32, max_retries=2, key="sk-test"):
    return SiliconFlowEmbedder(
        base_url="https://api.siliconflow.cn/v1",
        api_key=key,
        model="BAAI/bge-m3",
        dim=dim,
        batch_size=batch_size,
        max_retries=max_retries,
        client=_client(handler),
    )


# ---------------------------------------------------------------------------
# 请求形状与解析
# ---------------------------------------------------------------------------


async def test_embed_posts_openai_compatible_body():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1] * 4},
                    {"index": 1, "embedding": [0.2] * 4},
                ]
            },
        )

    out = await _embedder(handler).embed(["甲", "乙"])

    assert seen["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {
        "model": "BAAI/bge-m3",
        "input": ["甲", "乙"],
        "encoding_format": "float",
    }
    assert [v[0] for v in out] == [0.1, 0.2]
    assert all(isinstance(x, float) for x in out[0])


async def test_embed_sorts_by_index_not_by_response_order():
    """服务端返回顺序不保证等于输入顺序 —— 必须按 index 对齐。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2] * 4},
                    {"index": 0, "embedding": [0.1] * 4},
                ]
            },
        )

    assert [v[0] for v in await _embedder(handler).embed(["a", "b"])] == [0.1, 0.2]


async def test_embed_splits_into_batches_and_keeps_order():
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        sizes.append(len(inputs))
        # 每批都按 index 倒序返回，验证跨批拼接后仍然是输入顺序
        data = [
            {"index": i, "embedding": [float(i)] * 4} for i in reversed(range(len(inputs)))
        ]
        return httpx.Response(200, json={"data": data})

    emb = _embedder(handler, batch_size=2)
    out = await emb.embed(["a", "b", "c", "d", "e"])

    assert sizes == [2, 2, 1]
    assert [v[0] for v in out] == [0.0, 1.0, 0.0, 1.0, 0.0]


async def test_empty_input_does_not_call_the_api():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("空输入不该发请求")

    assert await _embedder(handler).embed([]) == []


# ---------------------------------------------------------------------------
# 失败与重试
# ---------------------------------------------------------------------------


async def test_rate_limit_is_retried_then_succeeds():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"message": "rate limited"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

    out = await _embedder(handler).embed(["a"])

    assert out == [[0.1] * 4]
    assert len(calls) == 2


async def test_client_error_is_not_retried():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"message": "invalid api key"})

    with pytest.raises(RuntimeError, match="401"):
        await _embedder(handler).embed(["a"])
    assert len(calls) == 1


async def test_retries_are_bounded():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    with pytest.raises(RuntimeError, match="503"):
        await _embedder(handler, max_retries=1).embed(["a"])
    assert len(calls) == 2  # 首次 + 1 次重试


async def test_missing_key_fails_before_any_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("没有密钥不该发请求")

    emb = _embedder(handler, key="")
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        await emb.embed(["a"])


async def test_dimension_mismatch_is_reported_early():
    """趁早报错：等 Milvus 报维度不符时，往往已经跑完几百条嵌入。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 3}]})

    with pytest.raises(RuntimeError, match="维度不符"):
        await _embedder(handler, dim=4).embed(["a"])


async def test_missing_index_field_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 4}]})

    with pytest.raises(RuntimeError, match="index"):
        await _embedder(handler).embed(["a"])


async def test_short_response_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

    with pytest.raises(RuntimeError, match="条目数不符"):
        await _embedder(handler).embed(["a", "b"])


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------


def test_get_embedder_is_a_process_singleton():
    from app.config import Settings
    from app.knowledge.embedder import get_embedder, reset_embedders

    reset_embedders()
    settings = Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        embedding_api_key="sk-test",
    )
    assert get_embedder(settings) is get_embedder(settings)
    reset_embedders()


# ---------------------------------------------------------------------------
# 真 API（默认 skip，需要 .env 里的 EMBEDDING_API_KEY）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("MEMHELP_TEST_EMBED") != "1", reason="需要真实硅基流动 API"
)
async def test_real_api_is_1024_dim_and_semantic():
    """真 API 验收：1024 维、可区分同义与无关。

    这是**集成**用例，它本来就依赖宿主机配置，所以显式读项目 `.env`
    （``_env_file=`` 覆盖掉套件为隔离而置空的那个设置）。
    """
    import numpy as np

    from app.config import ENV_FILE, Settings
    from app.knowledge.embedder import get_embedder, reset_embedders

    reset_embedders()
    settings = Settings(_env_file=ENV_FILE)
    settings.require_embedding_key()  # 没密钥时给一条明确信息，而不是 401
    embedder = get_embedder(settings)

    try:
        vectors = await embedder.embed(["邮费是多少", "运费怎么收", "发票怎么开"])
    finally:
        await embedder.aclose()
        reset_embedders()

    assert all(len(v) == 1024 for v in vectors)
    a, b, c = (np.array(v) for v in vectors)
    assert float(a @ b) > float(a @ c) + 0.1  # 同义句明显比无关句更近
