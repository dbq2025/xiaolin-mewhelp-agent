"""ch09 Task 11：eval_runs 记录、手动触发与趋势。"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.db.repositories import EvalRunRow
from app.eval_runs import (
    build_script_args,
    extract_metrics,
    record_report,
    run_evaluation,
    summarize_trend,
)


def _report(*, recall: float = 0.9, faith: float = 0.88) -> dict:
    return {
        "meta": {"cases": 4},
        "retrieval": {
            "hybrid_rerank": {
                "overall": {"n": 4, "recall@1": 0.7, "recall@5": recall, "mrr@10": 0.8}
            }
        },
        "generation": {
            "hybrid_rerank": {
                "overall": {"n": 4, "faithfulness": faith, "refusal_accuracy": 1.0}
            }
        },
    }


def _row(value: float, *, minutes: int, run_id: int = 1) -> EvalRunRow:
    return EvalRunRow(
        id=run_id,
        triggered_by="定时",
        dataset_size=48,
        metrics={
            "retrieval": {"recall@5": value},
            "generation": {"faithfulness": value},
        },
        created_at=datetime(2026, 10, 24, 12, 0) + timedelta(minutes=minutes),
    )


def test_extract_metrics_uses_hybrid_rerank_overall():
    metrics = extract_metrics(_report())

    assert metrics["retrieval"]["recall@5"] == 0.9
    assert metrics["retrieval"]["mrr@10"] == 0.8
    assert metrics["generation"]["faithfulness"] == 0.88
    assert metrics["generation"]["refusal_accuracy"] == 1.0
    assert "n" not in metrics["retrieval"]


def test_extract_metrics_tolerates_missing_sections():
    metrics = extract_metrics({"meta": {"cases": 0}})

    assert metrics["retrieval"] == {}
    assert metrics["generation"] == {}


def test_trend_marks_downward():
    rows = [_row(0.9, minutes=0, run_id=1), _row(0.8, minutes=10, run_id=2)]

    trend = summarize_trend(rows)

    assert trend["faithfulness"]["direction"] == "down"
    assert trend["faithfulness"]["delta"] < 0
    assert trend["recall@5"]["direction"] == "down"


def test_trend_marks_upward_and_flat():
    rows = [
        _row(0.5, minutes=0, run_id=1),
        _row(0.7, minutes=10, run_id=2),
        _row(0.7, minutes=20, run_id=3),
    ]

    trend = summarize_trend(rows)

    assert trend["faithfulness"]["direction"] == "flat"
    assert trend["faithfulness"]["delta"] == 0.0
    assert trend["faithfulness"]["previous"] == 0.7


def test_trend_with_single_run_has_no_direction():
    trend = summarize_trend([_row(0.9, minutes=0)])

    assert trend["faithfulness"]["latest"] == 0.9
    assert trend["faithfulness"]["direction"] is None
    assert trend["faithfulness"]["previous"] is None


async def test_record_report_writes_row(fake_repositories):
    run_id = await record_report(fake_repositories, _report(), triggered_by="手动")

    rows = await fake_repositories.eval_runs.list_recent()
    assert rows[0].id == run_id
    assert rows[0].triggered_by == "手动"
    assert rows[0].dataset_size == 4
    assert rows[0].metrics["retrieval"]["recall@5"] == 0.9


def test_build_script_args_includes_generation_limit(test_settings):
    settings = test_settings.model_copy(update={"eval_generation_limit": 8})

    args = build_script_args(settings)

    assert args[:2] == ["--generation", "hybrid_rerank"]
    assert args[-2:] == ["--limit", "8"]


def test_build_script_args_omits_zero_limit(test_settings):
    settings = test_settings.model_copy(update={"eval_generation_limit": 0})

    assert "--limit" not in build_script_args(settings)


async def test_run_evaluation_uses_injected_report(fake_repositories, monkeypatch, test_settings):
    async def fake_report(settings):
        return _report(recall=0.75, faith=0.5)

    monkeypatch.setattr("app.eval_runs._run_report", fake_report)

    run_id = await run_evaluation(
        repositories=fake_repositories, trigger="定时", settings=test_settings
    )

    rows = await fake_repositories.eval_runs.list_recent()
    assert rows[0].id == run_id
    assert rows[0].triggered_by == "定时"
    assert rows[0].metrics["generation"]["faithfulness"] == 0.5


async def test_run_evaluation_rejects_unknown_trigger(fake_repositories, test_settings):
    import pytest

    with pytest.raises(ValueError):
        await run_evaluation(
            repositories=fake_repositories, trigger="回填", settings=test_settings
        )


def _client(settings):
    from fastapi.testclient import TestClient

    from app import main as app_main
    from tests.fakes import FakeChatModel

    return TestClient(app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings))


def test_api_lists_runs_and_trend(fake_repositories, test_settings):
    import anyio

    async def seed():
        await record_report(fake_repositories, _report(recall=0.9, faith=0.88), triggered_by="定时")
        await record_report(fake_repositories, _report(recall=0.7, faith=0.5), triggered_by="手动")

    anyio.run(seed)

    with _client(test_settings) as client:
        resp = client.get("/api/eval-runs")

    assert resp.status_code == 200
    body = resp.json()
    assert [item["triggered_by"] for item in body["items"]] == ["手动", "定时"]
    assert body["trend"]["faithfulness"]["direction"] == "down"
    assert body["trend"]["faithfulness"]["latest"] == 0.5


def test_api_manual_run_records_a_row(fake_repositories, test_settings, monkeypatch):
    import time

    async def fake_run(*, repositories, trigger, settings):
        return await record_report(repositories, _report(), triggered_by=trigger)

    monkeypatch.setattr("app.api.run_evaluation", fake_run)

    with _client(test_settings) as client:
        started = client.post("/api/eval-runs/run")
        assert started.status_code == 202

        deadline = time.time() + 5
        rows = []
        while time.time() < deadline:
            rows = client.get("/api/eval-runs").json()["items"]
            if rows:
                break
            time.sleep(0.05)

    assert len(rows) == 1
    assert rows[0]["triggered_by"] == "手动"
