"""ch09 数据层：审核队列、评估记录、问题池快照列。

这三张表/列支撑飞轮闭环与评估趋势；字段与 ``sql/ch09-ddl.sql`` 逐列对齐。
仓库 API 是 async（同步 DB 调用在工作线程跑），所以用例都是 async。
"""

from __future__ import annotations


async def test_review_queue_defaults_and_increment(fake_repositories):
    repo = fake_repositories.reviews
    review_id = await repo.add("退款多久到账？", "1-3 个工作日")
    row = await repo.get(review_id)
    assert row is not None
    assert row.review_status == "待审" and row.occurrence_count == 1
    assert await repo.increment(review_id) is True
    assert (await repo.get(review_id)).occurrence_count == 2
    assert await repo.pending_questions() == [(review_id, "退款多久到账？")]


async def test_low_confidence_snapshot_and_match(fake_repositories):
    lc = fake_repositories.low_confidence
    snapshot = [{"n": 1, "chunk_id": 7, "quote": "原文", "score": 0.42}]
    qid = await lc.add(
        None, "怎么退", "retrieval_low_conf", "分低", retrieved_chunks=snapshot
    )
    rid = await fake_repositories.reviews.add("如何退款？", "在订单页申请")
    await lc.attach_review(qid, rid)
    row = await lc.get(qid)
    assert row is not None
    assert row.retrieved_chunks == snapshot and row.matched_review_id == rid


async def test_eval_runs_roundtrip(fake_repositories):
    run_id = await fake_repositories.eval_runs.add(
        "手动", 48, {"retrieval": {"recall@5": 0.9}}
    )
    rows = await fake_repositories.eval_runs.list_recent()
    assert rows[0].id == run_id
    assert rows[0].metrics["retrieval"]["recall@5"] == 0.9
    assert rows[0].triggered_by == "手动" and rows[0].dataset_size == 48


async def test_review_status_filter_hides_non_pending(fake_repositories):
    repo = fake_repositories.reviews
    rid = await repo.add("怎么改地址？", "在订单页修改")
    assert await repo.set_status(rid, "通过", approved_answer="订单页可改") is True
    row = await repo.get(rid)
    assert row.review_status == "通过"
    assert row.approved_answer == "订单页可改"
    assert await repo.list_rows(status="待审") == []
    assert [r.id for r in await repo.list_rows(status="通过")] == [rid]
