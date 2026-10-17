"""四策略检索 —— 评估与线上共用同一份实现。

## 四种策略

| 策略 | 做什么 | 用在哪 |
|---|---|---|
| ``vector`` | 只用稠密向量召回 | 评估基线（= ch03 的行为） |
| ``bm25`` | 只用 Milvus 原生 BM25 召回 | 评估基线（关键词那一路） |
| ``hybrid`` | 两路各召 N 条，**由 Milvus 做 RRF 融合** | 评估基线 |
| ``hybrid_rerank`` | 融合后取候选，``bge-reranker-v2-m3`` 精排 | **线上实际走的** |

## 为什么融合必须交给 Milvus

自己实现一遍 RRF 再和服务端比较，两边口径迟早分叉；``hybrid_search`` 已经做了融合，
我们只负责传两路输入与过滤表达式。自己写一遍还多一处要维护的排名算法。

## 为什么过滤挂在 request 上

实测：``hybrid_search`` 顶层 ``filter`` 被静默忽略、顶层 ``expr`` 直接崩。
所以 :func:`build_expr` 拼出来的表达式要交给两个 ``AnnSearchRequest.expr``
（见 ``vector_store.hybrid_search``）。

## 重排失败就退回 RRF 序

重排是提质手段，不是检索的硬依赖。超时/限流时按 RRF 序返回前 N 条，
并把 ``rerank_degraded`` 带回报告 —— 一次网络抖动不该表现为「客服什么都不知道」。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal, Sequence

from app.db.repositories import ChunkRow, KnowledgeChunkRepository
from app.knowledge.embedder import Embedder
from app.knowledge.query import UnderstoodQuery
from app.knowledge.rerank import NoopReranker, RerankError, Reranker
from app.knowledge.vector_store import VectorHit, VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "HybridRetriever",
    "RetrievedChunk",
    "Strategy",
    "StrategyResult",
    "build_expr",
]

#: 四种检索策略。评估脚本的 ``--strategies`` 就是这四个值。
Strategy = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]

#: 单一策略的默认取数（线上最终要 10 条证据，评估时也用同一个数）。
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class RetrievedChunk:
    """一条命中的知识块（已回 MySQL 取原文）。

    :param score: 含义随策略变化 —— 余弦 / BM25 / RRF / 重排分，**不可跨策略比大小**。
    :param source: ``vector`` / ``bm25`` / ``hybrid`` / ``rerank``，进报告与日志。
    """

    chunk_id: int
    question: str
    answer: str
    category: str
    section_path: str | None
    content_type: str | None
    score: float
    source: str


@dataclass(frozen=True)
class StrategyResult:
    """一次检索的结果，连同「这一轮到底发生了什么」。"""

    chunks: tuple[RetrievedChunk, ...]
    strategy: Strategy
    #: 重排没生效（没配 Key / 出错回落）时为 ``True``。报告里必须能看见。
    rerank_degraded: bool = False


def build_expr(category: str | None, content_type: str | None) -> str | None:
    """拼 Milvus 的过滤表达式；两个条件都用 ``&&`` 连接。

    值用 ``json.dumps`` 转义：模型可能把带引号的品类传进来，
    手拼字符串会直接破坏表达式语法（表现为 Milvus 报语法错，而不是过滤生效）。

    :return: 表达式；两个值都为空/空白时返回 ``None``（**不过滤**，不是"过滤空串"）。
    """
    conditions: list[str] = []
    if category and category.strip():
        conditions.append(f"category == {json.dumps(category.strip(), ensure_ascii=False)}")
    if content_type and content_type.strip():
        conditions.append(
            f"content_type == {json.dumps(content_type.strip(), ensure_ascii=False)}"
        )
    return " && ".join(conditions) if conditions else None


def _has_searchable_characters(text: str) -> bool:
    """有没有可检索的字符（字母/数字/汉字）。

    纯标点问法（``？？？``）发到 BM25 只会得到空结果，白白多一次远端调用；
    在本地直接判空更便宜，也不会因为分析器行为差异而偶发报错。
    """
    return any(character.isalnum() for character in text)


class HybridRetriever:
    """把嵌入器、向量库、重排器、原文仓库拼成一次检索。

    :param chunk_repo: 必须是**当请求**的仓库（绑着该请求的 DB Session）。
    :param hybrid_top_k: 每一路（dense / BM25）各召回多少条。
    :param candidate_k: RRF 融合后送进重排的候选数。
    :param rerank_top_n: 最终返回多少条（= 证据条数）。
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        chunk_repo: KnowledgeChunkRepository,
        hybrid_top_k: int = 50,
        candidate_k: int = 20,
        rerank_top_n: int = DEFAULT_TOP_K,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._chunk_repo = chunk_repo
        self.hybrid_top_k = hybrid_top_k
        self.candidate_k = candidate_k
        self.rerank_top_n = rerank_top_n

    async def warmup(self) -> None:
        """启动预热：发一次真实嵌入请求 + 确认集合结构。

        ``RagPipeline.warmup`` 转调这里。嵌入器没有 ``warmup`` 时跳过
        （假实现不需要预热，不该因此报错）。
        """
        warmup = getattr(self._embedder, "warmup", None)
        if warmup is not None:
            await warmup()
        await self._vector_store.ensure_collection()

    async def retrieve(
        self,
        strategy: Strategy,
        query: UnderstoodQuery,
        *,
        category: str | None = None,
        content_type: str | None = None,
    ) -> StrategyResult:
        """按指定策略检索。空结果不是错误。"""
        expr = build_expr(category, content_type)

        if strategy == "bm25":
            hits = await self._bm25_hits(query, expr)
            return StrategyResult(
                chunks=await self._hydrate(hits, source="bm25"), strategy=strategy
            )

        vector = await self._embed(query.standard)
        if vector is None:
            return StrategyResult(chunks=(), strategy=strategy)

        if strategy == "vector":
            hits = await self._vector_store.search_dense(
                vector, self.hybrid_top_k, expr=expr
            )
            return StrategyResult(
                chunks=await self._hydrate(hits, source="vector"), strategy=strategy
            )

        hits = await self._hybrid_hits(vector, query, expr)
        if strategy == "hybrid":
            return StrategyResult(
                chunks=await self._hydrate(hits, source="hybrid"), strategy=strategy
            )

        return await self._materialise_reranked(hits, query)

    # -- 各路召回 -----------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        vectors = await self._embedder.embed([text])
        if not vectors:
            logger.warning("嵌入返回空结果，本轮检索按「无命中」处理（query=%r）", text)
            return None
        return list(vectors[0])

    async def _bm25_hits(self, query: UnderstoodQuery, expr: str | None) -> list[VectorHit]:
        if not _has_searchable_characters(query.bm25_text):
            return []
        return await self._vector_store.search_bm25(
            query.bm25_text, self.hybrid_top_k, expr=expr
        )

    async def _hybrid_hits(
        self, vector: Sequence[float], query: UnderstoodQuery, expr: str | None
    ) -> list[VectorHit]:
        return await self._vector_store.hybrid_search(
            vector,
            query.bm25_text,
            self.candidate_k,
            expr=expr,
            per_path_limit=self.hybrid_top_k,
        )

    # -- 回原文 -------------------------------------------------------

    async def _hydrate(
        self, hits: Sequence[VectorHit], *, source: str
    ) -> tuple[RetrievedChunk, ...]:
        parsed, dropped = self._parse_ids(hits)
        if not parsed:
            return ()

        rows = {row.id: row for row in await self._chunk_repo.get_many([i for _, i in parsed])}
        chunks: list[RetrievedChunk] = []
        for score, chunk_id in parsed:
            row = rows.get(chunk_id)
            if row is None:
                dropped += 1
                continue
            chunks.append(self._to_chunk(row, score, source))

        self._warn_orphans(dropped, len(hits), len(chunks))
        return tuple(chunks)

    async def _materialise_reranked(
        self, hits: Sequence[VectorHit], query: UnderstoodQuery
    ) -> StrategyResult:
        """融合候选 → 重排 → 前 N 条；重排不可用时退回 RRF 序。"""
        parsed, dropped = self._parse_ids(hits)
        if not parsed:
            return StrategyResult(chunks=(), strategy="hybrid_rerank")

        rows = {row.id: row for row in await self._chunk_repo.get_many([i for _, i in parsed])}
        # 保序：候选顺序就是 RRF 序，重排的下标要能映射回它。
        candidates = [(score, chunk_id) for score, chunk_id in parsed if chunk_id in rows]
        dropped += len(parsed) - len(candidates)
        if not candidates:
            self._warn_orphans(dropped, len(hits), 0)
            return StrategyResult(chunks=(), strategy="hybrid_rerank")

        documents = [self._document(rows[chunk_id]) for _, chunk_id in candidates]
        degraded = isinstance(self._reranker, NoopReranker)
        try:
            rerank_hits = await self._reranker.rerank(
                query.standard, documents, self.rerank_top_n
            )
        except RerankError as exc:
            logger.warning("重排失败，回落到 RRF 序（%s）", exc)
            degraded = True
            rerank_hits = []

        if not rerank_hits:
            chunks = [
                self._to_chunk(rows[chunk_id], score, "hybrid")
                for score, chunk_id in candidates[: self.rerank_top_n]
            ]
            self._warn_orphans(dropped, len(hits), len(chunks))
            return StrategyResult(
                chunks=tuple(chunks), strategy="hybrid_rerank", rerank_degraded=True
            )

        chunks = []
        for hit in rerank_hits:
            if hit.index < 0 or hit.index >= len(candidates):
                logger.warning("重排返回的下标 %d 越界，已忽略", hit.index)
                continue
            _, chunk_id = candidates[hit.index]
            chunks.append(self._to_chunk(rows[chunk_id], hit.score, "rerank"))

        self._warn_orphans(dropped, len(hits), len(chunks))
        return StrategyResult(
            chunks=tuple(chunks), strategy="hybrid_rerank", rerank_degraded=degraded
        )

    # -- 小工具 -------------------------------------------------------

    @staticmethod
    def _parse_ids(hits: Sequence[VectorHit]) -> tuple[list[tuple[float, int]], int]:
        parsed: list[tuple[float, int]] = []
        dropped = 0
        for hit in hits:
            try:
                parsed.append((hit.score, int(hit.id)))
            except (TypeError, ValueError):
                dropped += 1
                logger.warning("向量 id %r 不是合法的 chunk 主键，已跳过", hit.id)
        return parsed, dropped

    @staticmethod
    def _document(row: ChunkRow) -> str:
        """重排用的文本 = BM25 索引的同一份文本（问法 + 答案）。"""
        from app.knowledge.chunker import render_bm25_text

        return render_bm25_text(row)

    @staticmethod
    def _to_chunk(row: ChunkRow, score: float, source: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row.id,
            question=row.questions,
            answer=row.answer,
            category=row.category,
            section_path=row.section_path,
            content_type=row.content_type,
            score=float(score),
            source=source,
        )

    @staticmethod
    def _warn_orphans(dropped: int, requested: int, kept: int) -> None:
        if not dropped:
            return
        # 汇总一行：单个孤儿 id 的 warning 容易淹没在日志里，这条给出比例。
        logger.warning(
            "本次检索丢弃 %d 个无效向量 id（请求 %d，返回 %d）——"
            "若持续出现，说明 Milvus 与 knowledge_chunks 不一致，考虑 --rebuild",
            dropped,
            requested,
            kept,
        )
