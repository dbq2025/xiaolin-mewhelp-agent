"""ch06 Task 6：多查询检索去重合并与降级。"""

from __future__ import annotations

from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.pipeline import RagPipeline
from app.knowledge.query import UnderstoodQuery


class _Understander:
    async def understand(self, query: str) -> UnderstoodQuery:
        return UnderstoodQuery(original=query, standard=query, bm25_text=query)


class _Retriever:
    def __init__(self, mapping, *, fail=()):
        self.mapping = mapping
        self.fail = set(fail)
        self.calls: list[str] = []

    async def retrieve(self, strategy, query, *, category=None):
        self.calls.append(query.original)
        if query.original in self.fail:
            raise RuntimeError("milvus down")
        return StrategyResult(chunks=self.mapping.get(query.original, ()), strategy="hybrid_rerank")


def _chunk(chunk_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        question="q",
        answer="a",
        category="c",
        section_path="s",
        content_type="manual",
        score=score,
        source="rerank",
    )


def _pipeline(mapping, *, fail=()):
    return RagPipeline(
        understander=_Understander(),
        retriever=_Retriever(mapping, fail=fail),
        sufficiency=object(),
        min_score=0.3,
        top_n=3,
    )


async def test_merge_dedupes_and_keeps_max_score():
    pipeline = _pipeline(
        {
            "a": (_chunk(1, 0.4), _chunk(2, 0.9)),
            "b": (_chunk(1, 0.8), _chunk(3, 0.7)),
        }
    )
    outcome = await pipeline.retrieve_many(["a", "b"])
    assert [chunk.chunk_id for chunk in outcome.evidence] == [2, 1, 3]
    assert outcome.top_score == 0.9
    assert outcome.queries[0].original == "a"
    assert len(outcome.queries) == 2


async def test_single_query_failure_does_not_abort():
    pipeline = _pipeline({"a": (_chunk(1, 0.9),)}, fail={"b"})
    outcome = await pipeline.retrieve_many(["a", "b"])
    assert outcome.refused is False
    assert [chunk.chunk_id for chunk in outcome.evidence] == [1]


async def test_all_fail_returns_refused():
    pipeline = _pipeline({}, fail={"a", "b"})
    outcome = await pipeline.retrieve_many(["a", "b"])
    assert outcome.refused is True
    assert outcome.evidence == ()


async def test_low_score_marks_refused_but_keeps_evidence():
    pipeline = _pipeline({"a": (_chunk(1, 0.1),)})
    outcome = await pipeline.retrieve_many(["a"])
    assert outcome.refused is True
    assert [chunk.chunk_id for chunk in outcome.evidence] == [1]
