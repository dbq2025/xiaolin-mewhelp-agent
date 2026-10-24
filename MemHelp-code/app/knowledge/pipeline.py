"""``RagPipeline`` —— 把「理解 → 检索 → 两道闸 → 编号证据」串成一次调用。

## 为什么单独一层

``query_faq`` 只需要一个结果对象，不该知道 RRF、重排、阈值这些细节；
评估脚本又需要「把策略换掉再跑一次」。所以把策略选择留给 ``HybridRetriever``，
把闸与证据留给这一层，工具层只做 JSON 映射。

## 两道闸

| 闸 | 触发 | 判据 | ``source`` |
|---|---|---|---|
| 检索层 | 0 命中 | 确定性 | ``retrieval_low_conf`` |
| 检索层 | 最高分 < ``min_score`` | 确定性 | ``retrieval_low_conf`` |
| 生成层 | 模型自评「证据不够」 | 模型 | ``self_check`` |

## 阈值只在**有重排分**时生效

重排降级（没配 Key 或服务挂了）时 ``chunks[0].score`` 是 RRF 融合分，量级 0.03 左右 ——
拿它去比 0.3 的阈值会把**每一轮**都判成低置信度。所以降级时只保留「0 命中」这一道
确定性检查，把判断交给自评。这条是实测逼出来的，不是理论选择。

## 拒答是短路

``refused=True`` 时由编排层直接吐固定话术，**不再发起第二次模型调用** ——
验收标准不能依赖「模型今天愿不愿意拒答」。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.knowledge.citations import Evidence, build_evidence
from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.query import UnderstoodQuery
from app.knowledge.sufficiency import SufficiencyVerdict

logger = logging.getLogger(__name__)

__all__ = ["RagPipeline", "RetrievalOutcome"]

#: 线上唯一策略。其它三种只给评估对照用。
ONLINE_STRATEGY = "hybrid_rerank"


class _UnderstanderLike(Protocol):
    async def understand(self, query: str) -> UnderstoodQuery: ...


class _RetrieverLike(Protocol):
    async def retrieve(
        self,
        strategy: str,
        query: UnderstoodQuery,
        *,
        category: str | None = None,
    ) -> StrategyResult: ...


class _SufficiencyLike(Protocol):
    async def judge(
        self, question: str, evidence: tuple[Evidence, ...]
    ) -> SufficiencyVerdict: ...


@dataclass(frozen=True)
class RetrievalOutcome:
    """一次在线检索的完整结果。

    :param refused: 是否拒答。``True`` 时编排层不再调用生成模型。
    :param refusal_source: ``retrieval_low_conf`` / ``self_check``，直接对应
        ``low_confidence_questions.source`` 的 ENUM 取值。
    :param evidence: 拒答时也可能非空（自评不足那种），保留给日志与复盘。
    """

    query: UnderstoodQuery
    evidence: tuple[Evidence, ...]
    #: ch06：多查询检索时保留每一次理解结果，供日志/评估看扩写是否真的生效。
    queries: tuple[UnderstoodQuery, ...] = ()
    #: 最高检索分。拒答时 evidence 可能为空，但分数仍要留给日志/低置信度池，
    #: 否则「0 命中」与「差一点命中」在运维视角里长得一模一样。
    top_score: float = 0.0
    refused: bool = False
    refusal_source: str | None = None
    refusal_reason: str | None = None
    rerank_degraded: bool = False
    self_check_degraded: bool = False


class RagPipeline:
    """在线检索编排。

    :param min_score: 检索层低置信阈值（重排分 0~1）。用评估集正负样本标定。
    :param top_n: 最终证据条数（= ``RAG_RERANK_TOP_N``）。
    """

    def __init__(
        self,
        *,
        understander: _UnderstanderLike,
        retriever: _RetrieverLike,
        sufficiency: _SufficiencyLike,
        min_score: float = 0.3,
        top_n: int = 10,
        strategy: str = ONLINE_STRATEGY,
    ) -> None:
        self._understander = understander
        self._retriever = retriever
        self._sufficiency = sufficiency
        self._min_score = min_score
        self._top_n = top_n
        # 线上永远是 hybrid_rerank；评估脚本要换策略才能做四策略对照，
        # 所以这里留一个注入口，而不是让脚本自己绕开这一层。
        self._strategy = strategy

    async def warmup(self) -> None:
        """启动预热：交给检索器（嵌入器 + 集合自检）。"""
        warmup = getattr(self._retriever, "warmup", None)
        if warmup is not None:
            await warmup()

    async def answer_context(
        self, query: str, *, category: str | None = None
    ) -> RetrievalOutcome:
        """理解 → 检索 → 两道闸 → 证据。不抛业务异常（空结果是正常结果）。"""
        understood, result, chunks = await self._retrieve(query, category=category)
        top_score = float(chunks[0].score) if chunks else 0.0

        if not chunks:
            return self._refuse(
                understood,
                source="retrieval_low_conf",
                reason="检索没有命中任何知识块",
                rerank_degraded=result.rerank_degraded,
                top_score=0.0,
            )

        if self._strategy == ONLINE_STRATEGY and not result.rerank_degraded:
            if top_score < self._min_score:
                logger.info(
                    "检索层低置信：最高分 %.3f < %.3f（query=%r）",
                    top_score,
                    self._min_score,
                    understood.original,
                )
                return self._refuse(
                    understood,
                    source="retrieval_low_conf",
                    reason=(
                        f"最高重排分 {top_score:.3f} 低于阈值 {self._min_score}，"
                        "证据相关性不足"
                    ),
                    rerank_degraded=result.rerank_degraded,
                    top_score=top_score,
                )

        evidence = build_evidence(chunks)
        verdict = await self._sufficiency.judge(understood.original, evidence)

        if not verdict.sufficient:
            reason = verdict.reason or verdict.missing or "证据不足以回答这个问题"
            return RetrievalOutcome(
                query=understood,
                evidence=evidence,
                top_score=top_score,
                refused=True,
                refusal_source="self_check",
                refusal_reason=reason,
                rerank_degraded=result.rerank_degraded,
                self_check_degraded=verdict.degraded,
            )

        return RetrievalOutcome(
            query=understood,
            evidence=evidence,
            top_score=top_score,
            refused=False,
            rerank_degraded=result.rerank_degraded,
            self_check_degraded=verdict.degraded,
        )

    async def retrieve_only(
        self, query: str, *, category: str | None = None
    ) -> RetrievalOutcome:
        """只做「理解 → 检索 → 0 命中/分数阈值」，**不调用自评模型**。

        ch05 的置信度闸卡在 Agent 之前：它要判断的是「检索证据够不够把问题交给
        Agent」，不是「模型觉得够不够」。如果这里复用 `answer_context`，每轮知识类
        问题都会多一次自评调用，而且 ch04 的正式自评会盖掉本章的最简阈值版。
        """
        understood, result, chunks = await self._retrieve(query, category=category)
        top_score = float(chunks[0].score) if chunks else 0.0

        if not chunks:
            return self._refuse(
                understood,
                source="retrieval_low_conf",
                reason="检索没有命中任何知识块",
                rerank_degraded=result.rerank_degraded,
                top_score=0.0,
            )

        if self._strategy == ONLINE_STRATEGY and not result.rerank_degraded:
            if top_score < self._min_score:
                return self._refuse(
                    understood,
                    source="retrieval_low_conf",
                    reason=(
                        f"最高重排分 {top_score:.3f} 低于阈值 {self._min_score}，"
                        "证据相关性不足"
                    ),
                    rerank_degraded=result.rerank_degraded,
                    top_score=top_score,
                )

        return RetrievalOutcome(
            query=understood,
            evidence=build_evidence(chunks),
            top_score=top_score,
            refused=False,
            rerank_degraded=result.rerank_degraded,
        )

    async def retrieve_for_gate(
        self, query: str, *, category: str | None = None
    ) -> RetrievalOutcome:
        """给正式置信度闸用的检索：**只拒 0 命中，不按分数提前拒答**。

        ``retrieve_only`` 在最高分低于阈值时直接返回空证据，闸就看不到
        Top1/Top2 分差与有效证据数了。ch09 把「分数低」的判断挪到闸里，
        所以这里必须把低分证据原样带出去。
        """
        understood, result, chunks = await self._retrieve(query, category=category)
        top_score = float(chunks[0].score) if chunks else 0.0

        if not chunks:
            return self._refuse(
                understood,
                source="retrieval_low_conf",
                reason="检索没有命中任何知识块",
                rerank_degraded=result.rerank_degraded,
                top_score=0.0,
            )

        return RetrievalOutcome(
            query=understood,
            evidence=build_evidence(chunks),
            top_score=top_score,
            refused=False,
            rerank_degraded=result.rerank_degraded,
        )

    async def retrieve_many(
        self, queries: Sequence[str], *, category: str | None = None
    ) -> RetrievalOutcome:
        """逐条检索，按 `chunk_id` 去重合并；单条失败不拖垮整轮。

        退款/售后子流程用：扩写出的几条查询各走一次「理解→混合召回→重排」，
        同一块知识可能被多条查询命中，只保留最高分的一条。单条查询网络失败只
        记 warning，其他查询继续；全部失败时返回空证据 + refused，由 Agent 明说
        无法确认，不把聊天变成 500。
        """
        understood_queries: list[UnderstoodQuery] = []
        merged: dict[int, RetrievedChunk] = {}
        rerank_degraded = False
        failures = 0

        for query in queries:
            text = (query or "").strip()
            if not text:
                continue
            try:
                understood, result, chunks = await self._retrieve(
                    text, category=category
                )
            except Exception:  # noqa: BLE001 - 单条检索失败不拖垮整轮
                failures += 1
                logger.warning("多查询检索的单条查询失败（query=%r）", text, exc_info=True)
                continue

            understood_queries.append(understood)
            rerank_degraded = rerank_degraded or bool(result.rerank_degraded)
            for chunk in chunks:
                current = merged.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    merged[chunk.chunk_id] = chunk

        ordered = tuple(
            sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[
                : self._top_n
            ]
        )
        first_query = (
            understood_queries[0]
            if understood_queries
            else UnderstoodQuery(original="", standard="", bm25_text="")
        )

        if not ordered:
            return RetrievalOutcome(
                query=first_query,
                evidence=(),
                queries=tuple(understood_queries),
                top_score=0.0,
                refused=True,
                refusal_source="retrieval_low_conf",
                refusal_reason=(
                    "多查询检索全部失败"
                    if failures
                    else "检索没有命中任何知识块"
                ),
                rerank_degraded=rerank_degraded,
            )

        top_score = float(ordered[0].score)
        refused = False
        reason: str | None = None
        if (
            self._strategy == ONLINE_STRATEGY
            and not rerank_degraded
            and top_score < self._min_score
        ):
            refused = True
            reason = (
                f"最高重排分 {top_score:.3f} 低于阈值 {self._min_score}，"
                "证据相关性不足"
            )

        return RetrievalOutcome(
            query=first_query,
            evidence=build_evidence(ordered),
            queries=tuple(understood_queries),
            top_score=top_score,
            refused=refused,
            refusal_source="retrieval_low_conf" if refused else None,
            refusal_reason=reason,
            rerank_degraded=rerank_degraded,
        )

    async def _retrieve(
        self, query: str, *, category: str | None = None
    ) -> tuple[UnderstoodQuery, StrategyResult, tuple[RetrievedChunk, ...]]:
        """理解 + 检索 + 截断到 top_n。两个公开入口共用，避免闸的口径漂移。"""
        understood = await self._understander.understand(query)
        result = await self._retriever.retrieve(
            self._strategy, understood, category=category
        )
        return understood, result, tuple(result.chunks[: self._top_n])

    @staticmethod
    def _refuse(
        understood: UnderstoodQuery,
        *,
        source: str,
        reason: str,
        rerank_degraded: bool,
        top_score: float = 0.0,
    ) -> RetrievalOutcome:
        return RetrievalOutcome(
            query=understood,
            evidence=(),
            top_score=top_score,
            refused=True,
            refusal_source=source,
            refusal_reason=reason,
            rerank_degraded=rerank_degraded,
        )
