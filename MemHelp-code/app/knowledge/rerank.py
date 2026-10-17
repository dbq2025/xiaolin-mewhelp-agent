"""重排 —— ``BAAI/bge-reranker-v2-m3``（硅基流动 ``/rerank``）。

## 为什么客户端不自己决定降级

``rerank()`` 拿不到结果就抛 :class:`RerankError`，由调用方（``HybridRetriever``）
决定「回落到 RRF 序」。客户端擅自吞故障，会让「重排一直没生效」无人发现 ——
而它恰恰是本章主要的提质手段。

## 为什么同分按输入序稳定排序

重排服务对无关文档常给出**完全相同的分**。打平时若按服务端返回序或随机序，
两次跑评估会得到不同的 Recall/MRR，报告就不可复现。这里统一按
``(-score, index)`` 排序：分数优先，打平看原始下标。

## 与嵌入器的关系

同一个账户、同一种客户端注入方式（``httpx.AsyncClient`` + ``MockTransport`` 可离线测）。
没配 Key 时用 :class:`NoopReranker`：按输入序返回前 N 条、分数全 0 ——
调用方据此把 ``rerank_degraded`` 标进报告，而不是让检索失败。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "NoopReranker",
    "RerankError",
    "RerankHit",
    "Reranker",
    "SiliconFlowReranker",
    "get_reranker",
    "reset_rerankers",
]


class RerankError(RuntimeError):
    """重排服务不可用（超时、非 2xx、响应无法解析）。

    刻意**不**在这里降级：调用方才知道「没有重排结果时该退回什么序」。
    """


@dataclass(frozen=True)
class RerankHit:
    """一条重排结果：候选列表里的下标 + 相关性分（0~1，越大越相关）。"""

    index: int
    score: float


class Reranker(Protocol):
    """上层对重排器的最小依赖。"""

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankHit]: ...


@dataclass
class SiliconFlowReranker:
    """硅基流动 ``POST {base}/rerank`` 的薄封装。

    :param client: 仅供测试注入（``httpx.AsyncClient`` 带 ``MockTransport``）。
        不注入时每次调用临时建一个、用完即关 —— 与嵌入器同一套做法。
    """

    base_url: str
    api_key: str
    model: str
    timeout: float = 10.0
    client: httpx.AsyncClient | None = None
    _owned_client: httpx.AsyncClient | None = field(
        default=None, init=False, repr=False
    )

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/rerank"

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankHit]:
        """把 ``documents`` 按与 ``query`` 的相关性重排，返回前 ``top_n`` 条。

        :raises RerankError: 超时 / 非 2xx / 响应不是合法 JSON 或缺少 ``results``。
        """
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            # 只要下标与分数；把原文再传回来是纯浪费带宽。
            "return_documents": False,
        }
        client, owned = self._resolve_client()
        try:
            response = await client.post(
                self._endpoint(),
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"重排请求失败：{exc}") from exc
        finally:
            if owned is not None:
                await owned.aclose()

        if response.status_code != 200:
            raise RerankError(
                f"重排服务返回 HTTP {response.status_code}：{response.text[:200]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RerankError("重排响应不是合法 JSON") from exc
        if not isinstance(data, dict) or "results" not in data:
            raise RerankError(f"重排响应缺少 results 字段：{str(data)[:200]}")

        hits: list[RerankHit] = []
        for item in data.get("results") or []:
            try:
                hits.append(
                    RerankHit(index=int(item["index"]), score=float(item["relevance_score"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankError(f"重排结果条目格式不对：{item!r}") from exc

        # 稳定排序：分数降序，打平按原始下标升序（见模块 docstring）。
        hits.sort(key=lambda hit: (-hit.score, hit.index))
        return hits[:top_n]

    def _resolve_client(self) -> tuple[httpx.AsyncClient, httpx.AsyncClient | None]:
        """返回 ``(要用的客户端, 需要关闭的客户端)``。"""
        if self.client is not None:
            return self.client, None
        if self._owned_client is None:
            self._owned_client = httpx.AsyncClient(timeout=self.timeout)
        return self._owned_client, None


class NoopReranker:
    """没配 Key 时的空实现：保持输入序、分数全 0。

    调用方看到全 0 分就知道「这一轮没有真正的重排」，据此标记
    ``rerank_degraded``；而不是把「没有重排」伪装成「重排后这些最相关」。
    """

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankHit]:
        return [RerankHit(index=index, score=0.0) for index in range(min(top_n, len(documents)))]


#: 进程内单例。键含 base_url / model / key —— 换配置要拿到新的实例。
_RERANKERS: dict[tuple[str, str, str], Reranker] = {}


def get_reranker(settings: Settings) -> Reranker:
    """按配置取重排器（进程内单例）。

    没配 Key（``RERANK_API_KEY`` 与 ``EMBEDDING_API_KEY`` 都空）时返回
    :class:`NoopReranker` 并告警 —— 重排可降级，不该让整个服务起不来。
    """
    try:
        api_key = settings.require_rerank_key()
    except ValueError:
        logger.warning("未配置重排 API Key，重排降级为 RRF 序（rerank_degraded=true）")
        return NoopReranker()

    key = (settings.rerank_base_url, settings.rerank_model, api_key[:8])
    reranker = _RERANKERS.get(key)
    if reranker is None:
        reranker = SiliconFlowReranker(
            base_url=settings.rerank_base_url,
            api_key=api_key,
            model=settings.rerank_model,
            timeout=settings.rerank_timeout,
        )
        _RERANKERS[key] = reranker
    return reranker


def reset_rerankers() -> None:
    """清空单例（测试用）。"""
    _RERANKERS.clear()
