"""审核队列的业务动作：详情组装、通过落库、驳回。

这一层只做「业务」，不碰 HTTP：API 负责状态码，这里负责把一次审核变成
「知识库新增一块 + 队列状态推进」的原子动作。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.db.repositories import LowConfidenceRow, Repositories, ReviewRow
from app.knowledge.ingest import vectorize_ids
from app.knowledge.mining import MINED_CATEGORY, MINED_CONTENT_TYPE

logger = logging.getLogger(__name__)

__all__ = [
    "ReviewAlreadyDecided",
    "ReviewDetail",
    "ReviewNotFound",
    "approve_review",
    "reject_review",
    "review_detail",
]


class ReviewNotFound(LookupError):
    """待审行不存在。"""


class ReviewAlreadyDecided(RuntimeError):
    """已经通过或驳回，不能再次处置。"""


@dataclass(frozen=True)
class ReviewDetail:
    review: ReviewRow
    questions: list[LowConfidenceRow]
    snapshot: list[dict[str, Any]]


class _ReviewChunk:
    """给 ``upsert_chunk`` 用的行形状（审核通过写入知识库的那一块）。"""

    __slots__ = (
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "ordinal",
    )

    def __init__(
        self, *, question: str, answer: str, section_path: str, ordinal: int
    ) -> None:
        self.category = MINED_CATEGORY
        self.questions = question
        self.answer = answer
        self.section_path = section_path
        self.content_type = MINED_CONTENT_TYPE
        self.is_key_clause = False
        self.ordinal = ordinal


def _merge_snapshots(questions: Sequence[LowConfidenceRow]) -> list[dict[str, Any]]:
    """把多条原话的召回快照按 chunk_id（缺省用 quote）去重合并。"""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in questions:
        for item in question.retrieved_chunks or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("chunk_id") or item.get("quote") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


async def review_detail(
    repositories: Repositories, review_id: int
) -> ReviewDetail | None:
    review = await repositories.reviews.get(review_id)
    if review is None:
        return None
    questions = await repositories.low_confidence.list_for_review(review_id)
    return ReviewDetail(
        review=review, questions=questions, snapshot=_merge_snapshots(questions)
    )


async def approve_review(
    repositories: Repositories,
    review_id: int,
    approved_answer: str,
    *,
    vector_store: Any | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """通过审核：写知识库、标记队列、尽力向量化。

    向量化失败**不回滚**审核状态：原文已经进 MySQL（权威源），块处于 pending，
    下一轮 ``vectorize_pending`` 会补上；返回 ``vectorize_status="pending"``
    让调用方（和审核页）知道「现在还不能检索到，但不用重新审」。
    """
    review = await repositories.reviews.get(review_id)
    if review is None:
        raise ReviewNotFound(f"待审问题 {review_id} 不存在")
    if review.review_status != "待审":
        raise ReviewAlreadyDecided(
            f"待审问题 {review_id} 已经是「{review.review_status}」，不能重复处置"
        )

    answer = approved_answer.strip()
    if not answer:
        raise ValueError("核准答案不能为空")

    chunk_id, outcome = await repositories.knowledge.upsert_chunk(
        _ReviewChunk(
            question=review.normalized_question,
            answer=answer,
            section_path=f"人工审核 / {review_id}",
            ordinal=int(review_id),
        )
    )
    await repositories.reviews.set_status(review_id, "通过", answer)

    vectorize_status = "pending"
    if vector_store is not None and embedder is not None:
        try:
            written = await vectorize_ids(
                [chunk_id],
                chunk_repo=repositories.knowledge,
                vector_store=vector_store,
                embedder=embedder,
            )
            if written:
                vectorize_status = "ok"
        except Exception:  # noqa: BLE001 - 检索索引可后补，审核状态不能丢
            logger.exception("审核通过后向量化失败（chunk_id=%s），留待续跑", chunk_id)

    return {
        "ok": True,
        "review_id": review_id,
        "chunk_id": chunk_id,
        "chunk_outcome": outcome,
        "vectorize_status": vectorize_status,
    }


async def reject_review(repositories: Repositories, review_id: int) -> bool:
    review = await repositories.reviews.get(review_id)
    if review is None:
        raise ReviewNotFound(f"待审问题 {review_id} 不存在")
    if review.review_status != "待审":
        raise ReviewAlreadyDecided(
            f"待审问题 {review_id} 已经是「{review.review_status}」，不能重复处置"
        )
    return await repositories.reviews.set_status(review_id, "驳回")
