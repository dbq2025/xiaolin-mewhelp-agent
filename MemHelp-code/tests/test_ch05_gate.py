"""ch05 Task 5：只检索入口 + 置信度闸（最简阈值版）。

ch04 的 `answer_context` 带模型自评；本章的闸在 Agent 之前，不能为了判「证据够不够」
再花一次自评模型调用。`retrieve_only` 只做理解 → 检索 → 0 命中/分数阈值。
"""

from __future__ import annotations

import pytest

from app.agent.gate import GateDecision, evaluate_gate
from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.pipeline import RagPipeline
from app.knowledge.query import UnderstoodQuery


class _Understander:
    async def understand(self, query: str) -> UnderstoodQuery:
        return UnderstoodQuery(original=query, standard=query, bm25_text=query)


class _Retriever:
    def __init__(self, chunks=(), *, degraded=False) -> None:
        self.chunks = tuple(chunks)
        self.degraded = degraded
        self.calls = 0

    async def retrieve(self, strategy, understood, *, category=None, content_type=None):
        self.calls += 1
        return StrategyResult(
            chunks=self.chunks, strategy="hybrid_rerank", rerank_degraded=self.degraded
        )


class _ExplodingJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, question, evidence):
        self.calls += 1
        raise AssertionError("retrieve_only 不该调自评模型")


class _NeverJudge:
    async def judge(self, question, evidence):
        raise AssertionError("retrieve_only 不该调自评模型")


def _chunk(chunk_id: int, score: float) -> RetrievedChunk:
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


def _pipeline(chunks=(), *, degraded=False, min_score=0.3):
    retriever = _Retriever(chunks, degraded=degraded)
    judge = _ExplodingJudge()
    pipe = RagPipeline(
        understander=_Understander(),
        retriever=retriever,
        sufficiency=judge,
        min_score=min_score,
        top_n=10,
    )
    return pipe, retriever, judge


async def test_retrieve_only_does_not_call_sufficiency():
    pipe, _, judge = _pipeline([_chunk(1, 0.9)])
    outcome = await pipe.retrieve_only("退货政策")

    assert outcome.refused is False
    assert [item.n for item in outcome.evidence] == [1]
    assert judge.calls == 0


async def test_retrieve_only_refuses_on_zero_hits():
    pipe, _, judge = _pipeline([])
    outcome = await pipe.retrieve_only("知识库没有的问题")

    assert outcome.refused is True
    assert outcome.refusal_source == "retrieval_low_conf"
    assert "没有命中" in (outcome.refusal_reason or "")
    assert judge.calls == 0


async def test_retrieve_only_refuses_low_score():
    pipe, _, _ = _pipeline([_chunk(1, 0.12)])
    outcome = await pipe.retrieve_only("差一点命中的问题")

    assert outcome.refused is True
    assert outcome.top_score == pytest.approx(0.12)
    assert "低于阈值" in (outcome.refusal_reason or "")


async def test_retrieve_only_does_not_apply_score_gate_when_rerank_degraded():
    """重排降级时 score 是 RRF 分（0.03 量级），不能拿去比 0.3。"""
    pipe, _, _ = _pipeline([_chunk(1, 0.03)], degraded=True)
    outcome = await pipe.retrieve_only("降级时的问题")

    assert outcome.refused is False
    assert outcome.rerank_degraded is True


def test_gate_rejects_low_score():
    decision = evaluate_gate(
        refused=False,
        evidence_count=3,
        top_score=0.12,
        rerank_degraded=False,
        min_score=0.3,
    )
    assert decision.passed is False
    assert "低于阈值" in decision.reason


def test_gate_passes_when_rerank_degraded_but_evidence_exists():
    decision = evaluate_gate(
        refused=False,
        evidence_count=2,
        top_score=0.0,
        rerank_degraded=True,
        min_score=0.3,
    )
    assert decision.passed is True


def test_gate_rejects_empty_evidence_even_if_not_marked_refused():
    decision = evaluate_gate(
        refused=False,
        evidence_count=0,
        top_score=0.99,
        rerank_degraded=False,
        min_score=0.3,
    )
    assert decision.passed is False
    assert "0 命中" in decision.reason


def test_gate_decision_is_frozen():
    decision = GateDecision(passed=True, reason="ok", score=0.9, evidence_count=1)
    with pytest.raises(Exception):
        decision.passed = False  # type: ignore[misc]
