"""ch09 正式置信度闸：特征闸、自评节点、入池快照。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage

from app.agent.checkpoint import memory_checkpointer
from app.agent.evidence_confidence import (
    ConfidenceModel,
    features_from_scores,
    model_to_dict,
)
from app.agent.gate import evaluate_evidence_gate
from app.agent.graph import build_agent_graph
from app.knowledge.citations import Evidence, build_evidence
from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.pipeline import RagPipeline, RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from app.knowledge.sufficiency import SufficiencyVerdict
from app.prompts import REFUSAL_TEXT
from tests.fakes_agent import ScriptedChatModel, text_turn

FIXTURE_DIR = Path(".uv-cache") / "ch09-gate-fixtures"


def _chunk(chunk_id: int = 7, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        question="退货政策",
        answer="7 天无理由退货",
        category="售后政策",
        section_path="售后政策 / 退货",
        content_type="manual",
        score=score,
        source="rerank",
    )


def _evidence(score: float = 0.9) -> tuple[Evidence, ...]:
    return build_evidence([_chunk(score=score)])


class _PipelineUnderstander:
    async def understand(self, query: str) -> UnderstoodQuery:
        return UnderstoodQuery(original=query, standard=query, bm25_text=query)


class _PipelineRetriever:
    def __init__(self, chunks: tuple[RetrievedChunk, ...]) -> None:
        self.chunks = chunks

    async def retrieve(self, strategy, understood, *, category=None, content_type=None):
        return StrategyResult(
            chunks=self.chunks, strategy="hybrid_rerank", rerank_degraded=False
        )


class _PassJudge:
    async def judge(self, question, evidence):
        return SufficiencyVerdict(sufficient=True)


class _FailJudge:
    async def judge(self, question, evidence):
        return SufficiencyVerdict(
            sufficient=False, reason="证据只提到部分退货条件"
        )


async def test_retrieve_for_gate_keeps_low_score_evidence():
    pipeline = RagPipeline(
        understander=_PipelineUnderstander(),
        retriever=_PipelineRetriever((_chunk(score=0.12),)),
        sufficiency=_PassJudge(),
        min_score=0.3,
    )
    outcome = await pipeline.retrieve_for_gate("退货多久到账")
    assert outcome.refused is False
    assert outcome.evidence and outcome.top_score == 0.12


async def test_retrieve_for_gate_still_refuses_zero_hits():
    pipeline = RagPipeline(
        understander=_PipelineUnderstander(),
        retriever=_PipelineRetriever(()),
        sufficiency=_PassJudge(),
        min_score=0.3,
    )
    outcome = await pipeline.retrieve_for_gate("知识库里没有的问题")
    assert outcome.refused is True and outcome.evidence == ()


def _model() -> ConfidenceModel:
    return ConfidenceModel(
        weights=[2.0, 0.0, 0.0],
        intercept=0.0,
        threshold=0.5,
        mean=[0.5, 0.2, 2.0],
        std=[0.3, 0.2, 1.0],
        version="test",
    )


def test_gate_rejects_no_evidence_and_passes_degraded():
    model = _model()
    no_evidence = evaluate_evidence_gate(
        evidence_count=0,
        features=features_from_scores([]),
        confidence=0.0,
        rerank_degraded=False,
        model=model,
    )
    assert no_evidence.passed is False
    degraded = evaluate_evidence_gate(
        evidence_count=3,
        features=features_from_scores([0.03, 0.02]),
        confidence=0.0,
        rerank_degraded=True,
        model=model,
    )
    assert degraded.passed is True


def test_gate_uses_confidence_threshold():
    model = _model()
    high = evaluate_evidence_gate(
        evidence_count=1,
        features=features_from_scores([0.9]),
        confidence=0.93,
        rerank_degraded=False,
        model=model,
    )
    low = evaluate_evidence_gate(
        evidence_count=1,
        features=features_from_scores([0.1]),
        confidence=0.07,
        rerank_degraded=False,
        model=model,
    )
    assert high.passed is True
    assert low.passed is False
    assert "0.07" in low.reason and "0.50" in low.reason


class _GateRetriever:
    def __init__(self, score: float, *, refused: bool = False) -> None:
        self.score = score
        self.refused = refused
        self.calls: list[str] = []

    async def retrieve_for_gate(self, query: str, *, category=None):
        self.calls.append(query)
        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=_evidence(self.score) if not self.refused else (),
            top_score=self.score,
            refused=self.refused,
            refusal_reason="测试里判定没有命中" if self.refused else None,
        )

    async def retrieve_only(self, query: str, *, category=None):  # pragma: no cover
        raise AssertionError("闸路径必须优先用 retrieve_for_gate")


class _Flywheel:
    def __init__(self) -> None:
        self.ids: list[int] = []

    async def ingest(self, low_confidence_id: int):
        self.ids.append(low_confidence_id)


def _artifact(settings):
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    folder = FIXTURE_DIR / uuid.uuid4().hex
    folder.mkdir()
    path = folder / "model.json"
    path.write_text(json.dumps(model_to_dict(_model())), encoding="utf-8")
    return settings.model_copy(
        update={
            "evidence_confidence_model": str(path),
            "knowledge_self_check": False,
        }
    )


async def _run(
    graph,
    *,
    message: str,
    model,
    repositories,
    retriever,
    settings,
    session: str,
    sufficiency=None,
    flywheel=None,
):
    config = {
        "configurable": {
            "thread_id": session,
            "model": model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
            "sufficiency": sufficiency,
            "flywheel": flywheel,
        }
    }
    async for _ in graph.astream(
        {"session_id": session, "turn_id": uuid.uuid4().hex, "user_message": message},
        config,
        stream_mode="custom",
    ):
        pass
    return (await graph.aget_state(config)).values


def _intent_turn() -> AIMessage:
    return text_turn(json.dumps({"intent": "product_consult", "confidence": 0.9}, ensure_ascii=False))


async def test_gate_failure_pools_with_retrieval_snapshot(
    test_settings, fake_repositories
):
    settings = _artifact(test_settings)
    flywheel = _Flywheel()
    values = await _run(
        build_agent_graph(checkpointer=memory_checkpointer()),
        message="赠品可以退吗",
        model=ScriptedChatModel(script=[_intent_turn()]),
        repositories=fake_repositories,
        retriever=_GateRetriever(0.12),
        settings=settings,
        session="s-ch09-gate",
        flywheel=flywheel,
    )

    assert values["answer"] == REFUSAL_TEXT
    assert values["gate_passed"] is False
    rows = await fake_repositories.low_confidence.list_recent(5)
    assert rows and rows[0].source == "retrieval_low_conf"
    stored = await fake_repositories.low_confidence.get(rows[0].id)
    assert stored is not None
    assert stored.retrieved_chunks[0]["score"] == 0.12
    assert stored.retrieved_chunks[0]["quote"].startswith("7 天无理由退货")
    assert flywheel.ids == [rows[0].id]


async def test_self_check_failure_pools_and_returns_fallback(
    test_settings, fake_repositories
):
    settings = _artifact(test_settings).model_copy(
        update={"knowledge_self_check": True}
    )
    flywheel = _Flywheel()
    values = await _run(
        build_agent_graph(checkpointer=memory_checkpointer()),
        message="拆封后还能退吗",
        model=ScriptedChatModel(script=[_intent_turn()]),
        repositories=fake_repositories,
        retriever=_GateRetriever(0.9),
        settings=settings,
        session="s-ch09-selfcheck",
        sufficiency=_FailJudge(),
        flywheel=flywheel,
    )

    assert values["answer"] == REFUSAL_TEXT
    assert values["gate_passed"] is False
    assert values["self_check_degraded"] is False
    rows = await fake_repositories.low_confidence.list_recent(5)
    assert rows and rows[0].source == "self_check"
    stored = await fake_repositories.low_confidence.get(rows[0].id)
    assert stored is not None and stored.retrieved_chunks
    assert flywheel.ids == [rows[0].id]
