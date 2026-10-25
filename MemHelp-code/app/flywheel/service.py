"""飞轮编排：入池问题 → 标准化 → 查重 → 待审队列 / 累加出现次数。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.flywheel.deduper import find_duplicate
from app.flywheel.normalizer import normalize_question

logger = logging.getLogger(__name__)

__all__ = ["FlywheelResult", "FlywheelService"]


@dataclass(frozen=True)
class FlywheelResult:
    low_confidence_id: int
    review_id: int | None
    matched: bool
    degraded: bool
    reason: str


class FlywheelService:
    """同步处理一条已入池的问题。

    同步而不是后台任务：审核页刷新后应该立刻能看到新缺口。代价是每轮
    「答不上来」多 1–3 秒模型调用，这是 ch09 明确接受的取舍。
    """

    def __init__(self, repositories: Any, model: Any, settings: Any) -> None:
        self._repositories = repositories
        self._model = model
        self._settings = settings

    async def ingest(self, low_confidence_id: int) -> FlywheelResult:
        try:
            row = await self._repositories.low_confidence.get(low_confidence_id)
        except Exception:  # noqa: BLE001 - 飞轮失败不该影响用户侧回答
            logger.exception("飞轮读取问题池失败（id=%s）", low_confidence_id)
            return FlywheelResult(
                low_confidence_id, None, False, True, "read_failed"
            )

        if row is None:
            return FlywheelResult(
                low_confidence_id, None, False, True, "not_found"
            )
        if row.matched_review_id is not None:
            return FlywheelResult(
                low_confidence_id, row.matched_review_id, True, False, "already_matched"
            )

        try:
            normalized = await normalize_question(
                self._model,
                raw_question=row.raw_question,
                evidence=row.retrieved_chunks or [],
                timeout=float(getattr(self._settings, "flywheel_normalize_timeout", 20.0)),
            )
            candidates = await self._repositories.reviews.pending_questions(50)
            dedupe = await find_duplicate(
                self._model,
                question=normalized.normalized_question,
                candidates=candidates,
                timeout=float(getattr(self._settings, "flywheel_dedupe_timeout", 15.0)),
            )

            if dedupe.matched_id is not None:
                if not await self._repositories.reviews.increment(dedupe.matched_id):
                    # 竞态：查重后队列行被删了；退化为新建，不丢这条缺口。
                    logger.warning("查重命中的待审行不存在，改为新建（id=%s）", dedupe.matched_id)
                    dedupe = type(dedupe)(matched_id=None, reason="目标行已不存在")
                else:
                    review_id = dedupe.matched_id
                    await self._repositories.low_confidence.attach_review(
                        low_confidence_id, review_id
                    )
                    return FlywheelResult(
                        low_confidence_id,
                        review_id,
                        True,
                        normalized.degraded or dedupe.degraded,
                        dedupe.reason,
                    )

            review_id = await self._repositories.reviews.add(
                normalized.normalized_question,
                normalized.suggested_answer or None,
            )
            await self._repositories.low_confidence.attach_review(
                low_confidence_id, review_id
            )
            return FlywheelResult(
                low_confidence_id,
                review_id,
                False,
                normalized.degraded or dedupe.degraded,
                normalized.degraded and "normalize_degraded" or dedupe.reason,
            )
        except Exception:  # noqa: BLE001 - 飞轮失败不该影响用户侧回答
            logger.exception("飞轮处理失败（id=%s）", low_confidence_id)
            return FlywheelResult(
                low_confidence_id, None, False, True, "ingest_failed"
            )
