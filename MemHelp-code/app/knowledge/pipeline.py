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
        understood = await self._understander.understand(query)
        result = await self._retriever.retrieve(
            self._strategy, understood, category=category
        )
        chunks: tuple[RetrievedChunk, ...] = tuple(result.chunks[: self._top_n])

        if not chunks:
            return self._refuse(
                understood,
                source="retrieval_low_conf",
                reason="检索没有命中任何知识块",
                rerank_degraded=result.rerank_degraded,
            )

        if self._strategy == ONLINE_STRATEGY and not result.rerank_degraded:
            top_score = float(chunks[0].score)
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
                )

        evidence = build_evidence(chunks)
        verdict = await self._sufficiency.judge(understood.original, evidence)

        if not verdict.sufficient:
            reason = verdict.reason or verdict.missing or "证据不足以回答这个问题"
            return RetrievalOutcome(
                query=understood,
                evidence=evidence,
                refused=True,
                refusal_source="self_check",
                refusal_reason=reason,
                rerank_degraded=result.rerank_degraded,
                self_check_degraded=verdict.degraded,
            )

        return RetrievalOutcome(
            query=understood,
            evidence=evidence,
            refused=False,
            rerank_degraded=result.rerank_degraded,
            self_check_degraded=verdict.degraded,
        )

    @staticmethod
    def _refuse(
        understood: UnderstoodQuery,
        *,
        source: str,
        reason: str,
        rerank_degraded: bool,
    ) -> RetrievalOutcome:
        return RetrievalOutcome(
            query=understood,
            evidence=(),
            refused=True,
            refusal_source=source,
            refusal_reason=reason,
            rerank_degraded=rerank_degraded,
        )
