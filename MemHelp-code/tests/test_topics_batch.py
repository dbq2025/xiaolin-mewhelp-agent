"""ch10 Task 14：批跑编排的编排逻辑（不碰 HTTP、不碰 torch）。"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, TopicClassification
from app.db.repositories import Repositories
from app.topics.batch import TopicServiceUnavailable, run_batch
from app.topics.infer import TopicPrediction
from app.topics.taxonomy import CATEGORY_NAMES

NOW = datetime(2026, 10, 25, 12, 0, 0)


def _pred(labels, fallback=False):
    return TopicPrediction(labels=labels, scores={}, fallback=fallback)


async def _seed(session, n: int = 3) -> None:
    session.add_all(
        [
            LowConfidenceQuestion(raw_question=f"问题{i}", source="retrieval_low_conf")
            for i in range(n)
        ]
    )
    session.commit()


async def test_batch_classifies_pending_and_reports(sqlite_session):
    await _seed(sqlite_session, 2)
    seen: list[str] = []

    async def classify(texts):
        seen.extend(texts)
        return [_pred(["物流"]), _pred(["其他"], fallback=True)]

    out = await run_batch(
        repositories=Repositories.from_session(sqlite_session),
        classify=classify,
        limit=10,
        now=NOW,
    )
    assert out.processed == 2 and out.fallback_other == 1
    assert out.pending_before == 2 and out.pending_after == 0
    assert sorted(seen) == ["问题0", "问题1"]
    rows = list(sqlite_session.scalars(select(TopicClassification)).all())
    assert {tuple(r.labels) for r in rows} == {("物流",), ("其他",)}
    assert all(r.classified_at == NOW for r in rows)


async def test_second_run_is_noop(sqlite_session):
    await _seed(sqlite_session, 2)

    async def classify(texts):
        return [_pred(["物流"]) for _ in texts]

    repo = Repositories.from_session(sqlite_session)
    await run_batch(repositories=repo, classify=classify, limit=10, now=NOW)
    out = await run_batch(repositories=repo, classify=classify, limit=10, now=NOW)
    assert out.processed == 0 and out.pending_after == 0


async def test_limit_is_respected(sqlite_session):
    await _seed(sqlite_session, 5)

    async def classify(texts):
        return [_pred(["物流"]) for _ in texts]

    out = await run_batch(
        repositories=Repositories.from_session(sqlite_session),
        classify=classify,
        limit=2,
        now=NOW,
    )
    assert out.processed == 2 and out.pending_after == 3


async def test_service_error_becomes_domain_error(sqlite_session):
    await _seed(sqlite_session, 1)

    async def classify(texts):
        raise TopicServiceUnavailable("连不上 sidecar")

    with pytest.raises(TopicServiceUnavailable):
        await run_batch(
            repositories=Repositories.from_session(sqlite_session),
            classify=classify,
            limit=10,
            now=NOW,
        )


async def test_count_mismatch_does_not_write_anything(sqlite_session):
    """条数对不上＝不能落库：错位一格等于整批标签全串。"""
    await _seed(sqlite_session, 3)

    async def classify(texts):
        return [_pred(["物流"]) for _ in texts][:-1]

    with pytest.raises(TopicServiceUnavailable):
        await run_batch(
            repositories=Repositories.from_session(sqlite_session),
            classify=classify,
            limit=10,
            now=NOW,
        )
    assert list(sqlite_session.scalars(select(TopicClassification)).all()) == []


async def test_empty_pending_is_not_an_error(sqlite_session):
    async def classify(texts):
        return []

    out = await run_batch(
        repositories=Repositories.from_session(sqlite_session),
        classify=classify,
        limit=10,
        now=NOW,
    )
    assert out.processed == 0 and out.pending_before == 0


async def test_bad_labels_are_rejected_before_write(sqlite_session):
    """「其他」不许与别的类共现 —— 落库前必须炸，不能等前端画图才发现。"""
    await _seed(sqlite_session, 1)

    async def classify(texts):
        return [_pred(["其他", "物流"])]

    with pytest.raises(Exception) as excinfo:
        await run_batch(
            repositories=Repositories.from_session(sqlite_session),
            classify=classify,
            limit=10,
            now=NOW,
        )
    assert "其他" in str(excinfo.value)
    assert list(sqlite_session.scalars(select(TopicClassification)).all()) == []
    assert CATEGORY_NAMES[-1] == "其他"
