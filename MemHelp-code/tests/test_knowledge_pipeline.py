"""``RagPipeline``：理解 → 检索 → 两道闸 → 编号证据。

## 两道闸，谁判

| 闸 | 触发 | 判据 | ``source`` |
|---|---|---|---|
| 检索层 | 0 命中 | 确定性 | ``retrieval_low_conf`` |
| 检索层 | 最高分 < 阈值 | 确定性 | ``retrieval_low_conf`` |
| 生成层 | 模型自评「证据不够」 | 模型 | ``self_check`` |

## 阈值只在**有重排分**时生效

实测：重排降级（没配 Key 或服务挂了）时 ``chunks[0].score`` 是 RRF 融合分，
量级 0.03 左右 —— 拿它去比 0.3 的阈值会把**每一轮**都判成低置信度。
所以降级时只做「0 命中」这一道确定性检查，把判断交给自评。

## 拒答是短路，不是请求模型合作

``refused=True`` 时由编排层直接吐固定话术，**不再发起第二次模型调用** ——
验收 4 不能依赖「模型今天愿不愿意拒答」。
"""

from __future__ import annotations

import pytest

from app.knowledge.citations import Evidence
from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.pipeline import RagPipeline, RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from app.knowledge.sufficiency import SufficiencyVerdict


class _FakeUnderstander:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def understand(self, query: str) -> UnderstoodQuery:
        self.queries.append(query)
        return UnderstoodQuery(original=query, standard=query, bm25_text=query)


class _FakeRetriever:
    def __init__(self) -> None:
        self.chunks: tuple[RetrievedChunk, ...] = ()
        self.degraded = False
        self.calls: list[dict] = []

    async def retrieve(self, strategy, understood, *, category=None, content_type=None):
        self.calls.append({"strategy": strategy, "category": category})
        return StrategyResult(
            chunks=self.chunks, strategy="hybrid_rerank", rerank_degraded=self.degraded
        )


class _FakeSufficiency:
    def __init__(self, verdict: SufficiencyVerdict | None = None) -> None:
        self.verdict = verdict or SufficiencyVerdict(sufficient=True)
        self.calls: list[tuple[str, tuple[Evidence, ...]]] = []

    async def judge(self, question, evidence):
        self.calls.append((question, tuple(evidence)))
        return self.verdict


def chunk(chunk_id: int, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        question=f"问法{chunk_id}",
        answer=f"答案{chunk_id}",
        category="售后政策",
        section_path=f"路径{chunk_id}",
        content_type="manual",
        score=score,
        source="rerank",
    )


def pipeline(sufficiency=None, *, min_score=0.3, top_n=10):
    retriever = _FakeRetriever()
    understander = _FakeUnderstander()
    judge = sufficiency or _FakeSufficiency()
    return (
        RagPipeline(
            understander=understander,
            retriever=retriever,
            sufficiency=judge,
            min_score=min_score,
            top_n=top_n,
        ),
        retriever,
        understander,
        judge,
    )


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


async def test_sufficient_returns_numbered_evidence():
    pipe, retriever, _, judge = pipeline()
    retriever.chunks = (chunk(1, 0.9), chunk(2, 0.8))

    outcome = await pipe.answer_context("运费怎么算")

    assert outcome.refused is False
    assert [item.n for item in outcome.evidence] == [1, 2]
    assert outcome.refusal_source is None
    assert judge.calls[0][0] == "运费怎么算"


async def test_category_is_forwarded_to_the_retriever():
    pipe, retriever, _, _ = pipeline()
    retriever.chunks = (chunk(1),)
    await pipe.answer_context("q", category="售后政策")
    assert retriever.calls[0]["category"] == "售后政策"


async def test_always_uses_hybrid_rerank():
    """线上只有一条路：混合 + 重排。其它三种策略只给评估用。"""
    pipe, retriever, _, _ = pipeline()
    retriever.chunks = (chunk(1),)
    await pipe.answer_context("q")
    assert retriever.calls[0]["strategy"] == "hybrid_rerank"


async def test_top_n_limits_the_evidence():
    pipe, retriever, _, _ = pipeline(top_n=2)
    retriever.chunks = (chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7))
    outcome = await pipe.answer_context("q")
    assert [item.n for item in outcome.evidence] == [1, 2]


# ---------------------------------------------------------------------------
# 闸 1：检索层
# ---------------------------------------------------------------------------


async def test_zero_hits_refuse_with_retrieval_source():
    pipe, _, _, judge = pipeline()
    outcome = await pipe.answer_context("你们支持货到付款吗")

    assert isinstance(outcome, RetrievalOutcome)
    assert outcome.refused is True
    assert outcome.refusal_source == "retrieval_low_conf"
    assert outcome.evidence == ()
    assert judge.calls == []          # 没有证据就不用自评


async def test_low_score_refuses():
    pipe, retriever, _, judge = pipeline(min_score=0.3)
    retriever.chunks = (chunk(1, 0.12),)

    outcome = await pipe.answer_context("q")

    assert outcome.refused is True
    assert outcome.refusal_source == "retrieval_low_conf"
    assert "0.12" in (outcome.refusal_reason or "")
    assert judge.calls == []


async def test_score_gate_is_skipped_when_rerank_degraded():
    """降级时 score 是 RRF 分（0.0x 量级），拿它比 0.3 会把每一轮都拒掉。"""
    pipe, retriever, _, judge = pipeline(min_score=0.3)
    retriever.chunks = (chunk(1, 0.032),)
    retriever.degraded = True

    outcome = await pipe.answer_context("q")

    assert outcome.refused is False
    assert outcome.rerank_degraded is True
    assert judge.calls  # 交给自评判


# ---------------------------------------------------------------------------
# 闸 2：自评
# ---------------------------------------------------------------------------


async def test_self_check_insufficient_refuses_with_self_check_source():
    judge = _FakeSufficiency(SufficiencyVerdict(False, "缺赔付上限", "证据只讲流程"))
    pipe, retriever, _, _ = pipeline(judge)
    retriever.chunks = (chunk(1, 0.9),)

    outcome = await pipe.answer_context("赔付上限多少")

    assert outcome.refused is True
    assert outcome.refusal_source == "self_check"
    assert outcome.refusal_reason == "证据只讲流程"


async def test_self_check_degraded_is_surfaced():
    judge = _FakeSufficiency(SufficiencyVerdict(True, degraded=True))
    pipe, retriever, _, _ = pipeline(judge)
    retriever.chunks = (chunk(1, 0.9),)

    outcome = await pipe.answer_context("q")

    assert outcome.refused is False
    assert outcome.self_check_degraded is True


async def test_refusal_source_is_always_one_of_the_ddl_values():
    """``low_confidence_questions.source`` 是 ENUM —— 写别的值会在 flush 时炸。"""
    pipe, _, _, judge = pipeline(
        _FakeSufficiency(SufficiencyVerdict(False, "", "不够"))
    )
    outcome = await pipe.answer_context("q")
    assert outcome.refused is True
    assert outcome.refusal_source in {"retrieval_low_conf", "self_check", "user_feedback"}
