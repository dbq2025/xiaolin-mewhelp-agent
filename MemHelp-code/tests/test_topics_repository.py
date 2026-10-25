"""``topic_classifications`` 的仓储行为。

只测仓储层承诺的四件事：pending 的定义（**没有归类结果的才算**）、
upsert 的语义（同一道题重跑是覆盖不是追加）、计数的口径（多标签题的
每个标签各算一次）、以及「同步 DB 调用必须进工作线程」这条约定。
"""

from datetime import datetime

from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, TopicClassification
from app.db.repositories import Repositories

NOW = datetime(2026, 10, 25, 12, 0, 0)


async def _seed(session, n=3):
    session.add_all(
        [LowConfidenceQuestion(raw_question=f"问题{i}", source="retrieval_low_conf") for i in range(n)]
    )
    session.commit()
    return list(session.scalars(select(LowConfidenceQuestion.id)).all())


async def test_pending_lists_unclassified_only(sqlite_session):
    ids = await _seed(sqlite_session)
    repo = Repositories.from_session(sqlite_session).topics
    assert [p.id for p in await repo.list_pending(10)] == ids
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=NOW)
    assert [p.id for p in await repo.list_pending(10)] == ids[1:]
    assert await repo.count_pending() == 2


async def test_upsert_twice_overwrites_not_duplicates(sqlite_session):
    ids = await _seed(sqlite_session, 1)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=NOW)
    await repo.upsert_many([(ids[0], ["退换货", "运费"])], classified_at=NOW)
    rows = list(sqlite_session.scalars(select(TopicClassification)).all())
    assert len(rows) == 1 and rows[0].labels == ["退换货", "运费"]


async def test_label_counts_counts_each_label(sqlite_session):
    ids = await _seed(sqlite_session, 3)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many(
        [(ids[0], ["退换货", "运费"]), (ids[1], ["退换货"]), (ids[2], ["其他"])],
        classified_at=NOW,
    )
    counts = {c.label: c.count for c in await repo.label_counts()}
    assert counts == {"退换货": 2, "运费": 1, "其他": 1}
    assert await repo.total_classified() == 3
    assert await repo.multi_label_count() == 1


async def test_last_classified_at_is_max(sqlite_session):
    ids = await _seed(sqlite_session, 2)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=datetime(2026, 1, 1))
    await repo.upsert_many([(ids[1], ["物流"])], classified_at=datetime(2026, 5, 1))
    assert await repo.last_classified_at() == datetime(2026, 5, 1)


async def test_empty_table_returns_none_and_zero(sqlite_session):
    repo = Repositories.from_session(sqlite_session).topics
    assert await repo.last_classified_at() is None
    assert await repo.total_classified() == 0
    assert await repo.count_pending() == 0


async def test_upsert_many_wraps_in_thread(sqlite_session, monkeypatch):
    # 仓库层约定：同步 DB 调用必须经 anyio.to_thread，否则会阻塞事件循环
    from app.db import repositories as mod

    calls = []
    real = mod.anyio.to_thread.run_sync

    async def spy(fn, *a, **kw):
        calls.append(fn)
        return await real(fn, *a, **kw)

    monkeypatch.setattr(mod.anyio.to_thread, "run_sync", spy)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.count_pending()
    assert calls
