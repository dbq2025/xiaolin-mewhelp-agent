"""嵌入器 —— 硅基流动云端 BGE-M3（OpenAI 兼容 ``/embeddings``）。

## 为什么是 HTTP 而不是本地模型

上游聊天模型（DeepSeek）没有 ``/embeddings`` 端点，所以嵌入必须走另一家。
用户明确要求用**硅基流动的云端 API**、不要在本机部署模型（本地
``sentence-transformers`` + 2.3GB 权重 + torch 的方案已作废）。
于是本章的依赖里没有 torch，嵌入就是一次普通的 HTTP POST。

## 云端特有的四个坑（本地版没有）

1. **返回顺序**：响应里的 ``data`` 不保证与输入同序，必须按 ``index`` 排序后再对齐。
   错位不会报错，只会让检索「答非所问」，而且不可复现。
2. **限流**：429 与 503/504 要退避重试；400/401/403 重试多少次都是同样结果，不重试。
3. **密钥**：没配 ``EMBEDDING_API_KEY`` 时**发请求前**就抛可读错误，
   不要等拿到 401 再报一个看不懂的驱动错误。
4. **批量长度**：一条请求带太多文本会被拒/变慢，所以按 ``batch_size`` 切片，
   每批独立重试。

## 维度

``BAAI/bge-m3`` 固定 1024 维（硅基流动的 ``dimensions`` 参数只对 Qwen3 系列生效），
所以维度由配置给出、并在返回时校验，不一致立刻报错 —— 让 Milvus 报
「维度不匹配」时你已经跑完几百条嵌入，是最浪费的一种失败。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["Embedder", "SiliconFlowEmbedder", "get_embedder", "reset_embedders"]

#: 仅这些状态码值得重试：限流与上游暂时不可用。
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class Embedder(Protocol):
    """上层对嵌入器的最小依赖。测试注入假实现即可完全离线。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class SiliconFlowEmbedder:
    """把一批文本编码成向量，调硅基流动的 ``/embeddings``。

    :param base_url: OpenAI 兼容根地址，例如 ``https://api.siliconflow.cn/v1``。
    :param api_key: 硅基流动账户 API Key。
    :param model: 模型名，例如 ``BAAI/bge-m3``。
    :param dim: 期望维度；响应维度不符会抛错。
    :param client: 仅供测试注入（``httpx.AsyncClient`` 带 ``MockTransport``）。
        不注入时每次调用临时建一个，用完即关 —— 脚本进程里没必要常驻连接池。
    """

    base_url: str
    api_key: str
    model: str
    dim: int = 1024
    batch_size: int = 32
    timeout: float = 30.0
    max_retries: int = 2
    client: httpx.AsyncClient | None = None
    _owned_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    # -- 客户端 -------------------------------------------------------

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    def _get_client(self) -> httpx.AsyncClient:
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            self._owned_client = httpx.AsyncClient(timeout=self.timeout)
        return self._owned_client

    # -- 单批请求 -----------------------------------------------------

    async def _request(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            # 提前失败：没有密钥时连请求都不发，错误消息指向 .env。
            raise ValueError(
                "缺少 EMBEDDING_API_KEY（硅基流动账户 API Key）。"
                "请在 MemHelp-code/.env 里设置，例如 EMBEDDING_API_KEY=sk-xxx。"
            )

        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._get_client().post(
                    self._endpoint(), json=payload, headers=headers
                )
            except httpx.TransportError as exc:  # 网络层故障，值得重试
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"嵌入请求网络故障（已重试 {attempt} 次）：{exc}") from exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                logger.warning(
                    "嵌入请求返回 %s，第 %d 次重试", response.status_code, attempt + 1
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code >= 400:
                # 4xx（尤其是 401/403）重试多少次都一样，直接抛。
                raise RuntimeError(
                    f"嵌入请求失败（HTTP {response.status_code}）：{response.text[:300]}"
                )

            return self._parse(response.json(), expected=len(texts))

        # 循环正常结束只可能在「最后一次仍可重试」时发生
        raise RuntimeError(f"嵌入请求失败：{last_error or '未知错误'}")

    def _parse(self, body: dict[str, Any], *, expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError(
                f"嵌入响应条目数不符：期望 {expected}，实际 "
                f"{len(data) if isinstance(data, list) else '非列表'}"
            )

        # **按 index 排序**：服务端不保证返回顺序等于输入顺序。
        try:
            ordered = sorted(data, key=lambda item: int(item["index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"嵌入响应缺少可用的 index 字段：{body}") from exc

        vectors: list[list[float]] = []
        for item in ordered:
            vector = [float(x) for x in item["embedding"]]
            if len(vector) != self.dim:
                raise RuntimeError(
                    f"嵌入维度不符：期望 {self.dim}，实际 {len(vector)}。"
                    "检查 EMBEDDING_MODEL 与 EMBEDDING_DIM 是否配对。"
                )
            vectors.append(vector)
        return vectors

    # -- 对外接口 -----------------------------------------------------

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本编码成向量，顺序与输入一一对应。

        空输入直接返回 ``[]`` 且**不发请求**（建库空跑时不该付一次网络往返）。
        """
        if not texts:
            return []

        items = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            vectors.extend(await self._request(batch))
        return vectors

    async def warmup(self) -> None:
        """发一次真实嵌入请求，把密钥/模型名/网络问题提前到启动阶段。"""
        await self.embed(["预热"])

    async def aclose(self) -> None:
        """关掉自己建的客户端（脚本与 lifespan 退出时用）。注入的客户端不关。"""
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None


_EMBEDDERS: dict[tuple[str, str, str, int], SiliconFlowEmbedder] = {}


def get_embedder(settings: Settings) -> SiliconFlowEmbedder:
    """进程级单例（按 base_url/model/dim 区分：换模型要拿到新实例）。

    为什么不用 ``lru_cache(settings)``：pydantic 的 ``Settings`` 不可哈希，
    直接当缓存键会抛 ``TypeError: unhashable type`` —— 而且只有真跑到这条路径
    才会发现。
    """
    key = (
        settings.embedding_base_url,
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_dim,
    )
    cached = _EMBEDDERS.get(key)
    if cached is None:
        cached = SiliconFlowEmbedder(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
        _EMBEDDERS[key] = cached
    return cached


def reset_embedders() -> None:
    """丢掉进程内缓存的嵌入器（测试与「热换配置」用）。生产代码不要调。"""
    _EMBEDDERS.clear()
