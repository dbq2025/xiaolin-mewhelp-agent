"""在线检索 —— ``query_faq`` 的新内核。

## 一条查询的完整路径

```
用户原话 → BGE-M3 编码 → Milvus Top-K（COSINE，Strong）→ 拿 id 回 MySQL 取原文
        → 按相似度序返回 KnowledgeHit
```

## 为什么必须按 Milvus 的分数序返回

``get_many`` 是 ``WHERE id IN (...)``，MySQL 不保证顺序（通常按主键）。如果直接把它的
结果返回，**最高分那条可能排在最后**，而工具只取前几条 —— 表现为「明明库里有正确
答案，模型却拿到一堆不相关的」。

## 为什么孤儿 id 要跳过而不是报错

``--rebuild`` 会先 drop 集合再清表，人工删行也会留下「Milvus 有、MySQL 没有」的 id。
这类 id 属于**可预期的脏数据**，跳过 + 告警即可；抛异常会打断整条聊天连接，
而用户看到的只是「客服突然不说话」。

## 不设相似度阈值

Top-K 少召回是正常业务结果（``count`` 可以小于 K）。加阈值会把「库里确实没有」
与「有一点像但不敢给」混成同一个空结果，而两者的正确回复完全不同。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.db.repositories import KnowledgeChunkRepository
from app.knowledge.embedder import Embedder
from app.knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeHit", "KnowledgeRetriever", "RetrieverLike", "warmup"]

#: 默认取数。与 ``Settings.rag_top_k`` 的默认值一致；工具层会显式传值。
DEFAULT_TOP_K = 5


async def warmup(embedder: Embedder, vector_store: VectorStore) -> None:
    """启动预热：发一次真实嵌入请求 + 确保集合存在。

    单独一个函数（而不是只挂在 retriever 上），因为 lifespan 里没有「当请求的
    chunk 仓库」可给 —— 预热只碰嵌入器与向量库。
    """
    await embedder.warmup()
    await vector_store.ensure_collection()


@dataclass(frozen=True)
class KnowledgeHit:
    """一条检索命中。字段名与 ``query_faq`` 出参一一对应。"""

    question: str
    answer: str
    category: str
    #: 余弦相似度。只用于日志/调试，**不进工具出参**（契约里没有这个字段）。
    score: float


class RetrieverLike(Protocol):
    """工具层对检索器的最小依赖。"""

    async def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeHit]: ...


class KnowledgeRetriever:
    """把嵌入器、向量库、原文仓库拼成一次检索。

    :param chunk_repo: 必须是**当请求**的仓库（它绑着该请求的 DB Session）。
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_repo: KnowledgeChunkRepository,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_repo = chunk_repo

    async def warmup(self) -> None:
        """预热：发一次真实嵌入请求 + 确认集合存在。

        lifespan 调它。云端没有「模型加载」这一步，但第一次调用要付 TLS + 鉴权，
        而且密钥/模型名配错时能在启动阶段就报出来，而不是等用户问第一句。
        """
        await warmup(self._embedder, self._vector_store)

    async def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeHit]:
        """检索与 ``query`` 语义最接近的 ``top_k`` 块。"""
        vectors = await self._embedder.embed([query])
        if not vectors:
            return []

        hits = await self._vector_store.search(vectors[0], top_k)
        if not hits:
            return []

        parsed: list[tuple[float, int]] = []
        for hit in hits:
            try:
                parsed.append((hit.score, int(hit.id)))
            except (TypeError, ValueError):
                logger.warning("向量 id %r 不是合法的 chunk 主键，已跳过", hit.id)

        rows = {row.id: row for row in await self._chunk_repo.get_many([i for _, i in parsed])}

        results: list[KnowledgeHit] = []
        for score, chunk_id in parsed:
            row = rows.get(chunk_id)
            if row is None:
                logger.warning("向量 id %s 在 MySQL 中不存在（原文已删？），已跳过", chunk_id)
                continue
            results.append(
                KnowledgeHit(
                    question=row.questions,
                    answer=row.answer,
                    category=row.category,
                    score=score,
                )
            )
        return results
