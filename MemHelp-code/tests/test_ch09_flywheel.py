"""ch09 飞轮：标准化、查重、入队、回写。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from app.flywheel.deduper import DedupeResult, find_duplicate
from app.flywheel.normalizer import NormalizedQuestion, normalize_question
from app.flywheel.service import FlywheelResult, FlywheelService


class _StubModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("stub 模型没有更多响应")
        return AIMessage(content=self.responses.pop(0))


class _BrokenModel:
    async def ainvoke(self, messages):
        raise RuntimeError("上游超时")


async def test_normalize_falls_back_on_bad_json():
    out = await normalize_question(_StubModel("这不是 JSON"), raw_question="这个怎么弄", evidence=[])
    assert out.degraded is True
    assert out.normalized_question == "这个怎么弄"
    assert out.suggested_answer == ""


async def test_normalize_strips_code_fence():
    payload = json.dumps(
        {"question": "如何申请退款？", "answer": "在订单页点申请退款。"},
        ensure_ascii=False,
    )
    out = await normalize_question(
        _StubModel(f"```json\n{payload}\n```"),
        raw_question="钱怎么退回来",
        evidence=[{"quote": "7 天无理由退款"}],
    )
    assert out.degraded is False
    assert out.normalized_question == "如何申请退款？"
    assert out.suggested_answer == "在订单页点申请退款。"


async def test_normalize_model_failure_degrades():
    out = await normalize_question(_BrokenModel(), raw_question="  发票怎么开  ", evidence=[])
    assert out.degraded is True
    assert out.normalized_question == "发票怎么开"


async def test_dedupe_without_candidates_skips_model():
    model = _StubModel()
    result = await find_duplicate(model, question="如何退款？", candidates=[])
    assert result == DedupeResult(matched_id=None, reason="待审队列为空")
    assert model.calls == 0


async def test_dedupe_validates_id_against_candidates():
    candidates = [(3, "退款多久到账？"), (4, "怎么开发票？")]
    hit = await find_duplicate(
        _StubModel(json.dumps({"matched_id": 3, "reason": "同义"}, ensure_ascii=False)),
        question="钱几天回来",
        candidates=candidates,
    )
    assert hit.matched_id == 3

    unknown = await find_duplicate(
        _StubModel(json.dumps({"matched_id": 99, "reason": "幻觉 id"}, ensure_ascii=False)),
        question="钱几天回来",
        candidates=candidates,
    )
    assert unknown.matched_id is None
    assert unknown.degraded is True


async def test_dedupe_bad_json_degrades_to_no_match():
    result = await find_duplicate(
        _StubModel("不是 JSON"), question="如何退款？", candidates=[(3, "退款多久到账？")]
    )
    assert result.matched_id is None and result.degraded is True


async def test_ingest_merges_duplicate(fake_repositories, test_settings):
    rid = await fake_repositories.reviews.add("退款多久到账？", "1-3 个工作日")
    qid = await fake_repositories.low_confidence.add(
        None, "钱啥时候回来", "user_feedback"
    )
    model = _StubModel(
        json.dumps({"question": "退款多久到账？", "answer": "1-3 个工作日"}, ensure_ascii=False),
        json.dumps({"matched_id": rid, "reason": "问的是同一件事"}, ensure_ascii=False),
    )
    service = FlywheelService(fake_repositories, model, test_settings)
    result = await service.ingest(qid)

    assert result.matched is True and result.review_id == rid
    assert result.degraded is False
    review = await fake_repositories.reviews.get(rid)
    assert review is not None and review.occurrence_count == 2
    row = await fake_repositories.low_confidence.get(qid)
    assert row is not None and row.matched_review_id == rid


async def test_ingest_creates_new_review(fake_repositories, test_settings):
    qid = await fake_repositories.low_confidence.add(
        None, "赠品坏了能补发吗", "retrieval_low_conf", retrieved_chunks=[]
    )
    model = _StubModel(
        json.dumps(
            {"question": "赠品损坏如何补发？", "answer": "联系客服补发。"},
            ensure_ascii=False,
        )
    )
    service = FlywheelService(fake_repositories, model, test_settings)
    result = await service.ingest(qid)

    assert result.matched is False and result.review_id is not None
    review = await fake_repositories.reviews.get(result.review_id)
    assert review is not None
    assert review.normalized_question == "赠品损坏如何补发？"
    assert review.ai_suggested_answer == "联系客服补发。"


async def test_ingest_is_idempotent(fake_repositories, test_settings):
    rid = await fake_repositories.reviews.add("退款多久到账？", "1-3 个工作日")
    qid = await fake_repositories.low_confidence.add(None, "钱啥时候回来", "self_check")
    await fake_repositories.low_confidence.attach_review(qid, rid)
    model = _StubModel()
    service = FlywheelService(fake_repositories, model, test_settings)
    result = await service.ingest(qid)

    assert result.matched is True and result.review_id == rid
    assert model.calls == 0


async def test_ingest_missing_question_degrades(fake_repositories, test_settings):
    service = FlywheelService(fake_repositories, _StubModel(), test_settings)
    result = await service.ingest(999_999)
    assert result.degraded is True
    assert result.reason == "not_found"


async def test_ingest_survives_model_failure(fake_repositories, test_settings):
    qid = await fake_repositories.low_confidence.add(None, "这个怎么弄", "self_check")
    service = FlywheelService(fake_repositories, _BrokenModel(), test_settings)
    result = await service.ingest(qid)
    assert result.degraded is True
    assert result.review_id is not None  # 标准化降级也要把原话送进待审
    review = await fake_repositories.reviews.get(result.review_id)
    assert review is not None and review.normalized_question == "这个怎么弄"
