"""RAG 评估看板的数据适配与 HTTP 接口。

只钉住三件事：
1. 报告里没跑的策略必须是 ``None``（页面显示 ``—``），不能补 0；
2. 逐条记录要能和评估集题面按 ``case_id`` 关联上；
3. 报告缺失 / 损坏 / 重跑并发时，接口给出可判读的错误类型。

重跑任务本身（真的拉起子进程）不在单测里跑 —— 那需要 MySQL / Milvus /
嵌入与重排服务；测试只验注入的 runner 与 HTTP 语义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.rag_eval import EvalAlreadyRunning, build_dashboard

REPORT_FIXTURE = Path(__file__).with_name("_rag_eval_report.json")


def _report() -> dict:
    return {
        "meta": {
            "generated_at": "2026-01-02T03:04:05",
            "cases": 2,
            "milvus_rows": 9,
            "strategies": ["vector", "hybrid_rerank"],
            "generation_strategies": ["hybrid_rerank"],
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "min_rerank_score": 0.3,
        },
        "retrieval": {
            "vector": {
                "A_policy": {
                    "n": 1,
                    "recall@1": 0.0,
                    "recall@3": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 0.5,
                },
                "overall": {
                    "n": 1,
                    "recall@1": 0.0,
                    "recall@3": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 0.5,
                },
            },
            "hybrid_rerank": {
                "A_policy": {
                    "n": 1,
                    "recall@1": 1.0,
                    "recall@3": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 1.0,
                },
                "overall": {
                    "n": 1,
                    "recall@1": 1.0,
                    "recall@3": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 1.0,
                },
            },
        },
        "generation": {
            "hybrid_rerank": {
                "A_policy": {
                    "n": 1,
                    "faithfulness": 1.0,
                    "refusal_accuracy": 1.0,
                    "citation_validity": 1.0,
                },
                "overall": {
                    "n": 1,
                    "faithfulness": 1.0,
                    "refusal_accuracy": 1.0,
                    "citation_validity": 1.0,
                },
            }
        },
        "warnings": {"A01": {"99 元包邮": 4}},
        "notes": {"gold": "gold 是并集"},
        "cases": [
            {
                "case_id": "A01",
                "bucket": "A_policy",
                "behavior": "answer",
                "strategy": "vector",
                "hits": [11, 12],
                "gold": [12],
                "rerank_degraded": False,
            }
        ],
        "generation_cases": [
            {
                "case_id": "A01",
                "bucket": "A_policy",
                "behavior": "answer",
                "strategy": "hybrid_rerank",
                "refused": False,
                "refusal_source": None,
                "refusal_reason": None,
                "answer": "满 99 元包邮 [1]。",
                "claims": [{"text": "满 99 元包邮", "supported": True, "evidence_n": 1}],
                "faithfulness": 1.0,
                "citations": {"valid": [1], "dropped": []},
                "rerank_degraded": False,
            }
        ],
    }


@pytest.fixture
def report_path():
    REPORT_FIXTURE.write_text(json.dumps(_report(), ensure_ascii=False), encoding="utf-8")
    try:
        yield REPORT_FIXTURE
    finally:
        REPORT_FIXTURE.unlink(missing_ok=True)


@pytest.fixture
def client(report_path, monkeypatch, test_settings):
    monkeypatch.setattr("app.rag_eval.REPORT_PATH", report_path)
    monkeypatch.setattr("app.rag_eval.CASES_PATH", REPORT_FIXTURE.with_name("_missing_cases.jsonl"))
    return TestClient(main.create_app(settings=test_settings))


def test_missing_strategy_is_none_not_zero():
    dashboard = build_dashboard(_report(), index={"A01": {"query": "满多少包邮"}})

    # 只跑了一个生成策略：另一个策略不能凭空出现
    assert set(dashboard["generation"]) == {"hybrid_rerank"}
    # 报告没给的桶不能补 0
    assert dashboard["retrieval"]["vector"]["B_model"]["mrr@10"] is None
    assert dashboard["generation"]["hybrid_rerank"]["B_model"]["faithfulness"] is None


def test_cases_are_joined_with_case_queries():
    index = {"A01": {"query": "满多少包邮", "difficulty": "easy"}}

    dashboard = build_dashboard(_report(), index=index)

    retrieval_case = dashboard["cases"]["retrieval"][0]
    assert retrieval_case["query"] == "满多少包邮"
    assert retrieval_case["difficulty"] == "easy"
    assert retrieval_case["first_gold_rank"] == 2
    assert retrieval_case["top5_hit"] is True
    assert dashboard["cases"]["generation"][0]["query"] == "满多少包邮"


def test_summary_uses_final_generation_strategy():
    dashboard = build_dashboard(_report(), index={})

    summary = dashboard["summary"]
    assert summary["best_strategy"] == "hybrid_rerank"
    assert summary["best_mrr"] == 1.0
    assert summary["final_generation_strategy"] == "hybrid_rerank"
    assert summary["answer_coverage"] == 1.0
    assert summary["citation_coverage"] == 1.0
    assert summary["acceptance"]["status"] == "pass"
    assert dashboard["warnings"][0] == {"case_id": "A01", "anchor": "99 元包邮", "count": 4}


def test_api_returns_dashboard(client):
    response = client.get("/api/rag-eval")

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["cases"] == 2
    assert body["summary"]["best_strategy_label"] == "混合 + 重排"
    assert body["run"]["status"] == "idle"


def test_api_missing_report_returns_404(monkeypatch, test_settings):
    monkeypatch.setattr(
        "app.rag_eval.REPORT_PATH",
        Path(__file__).with_name("_definitely_missing_report.json"),
    )
    client = TestClient(main.create_app(settings=test_settings))

    response = client.get("/api/rag-eval")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "report_not_found"


def test_api_invalid_report_returns_500(report_path, monkeypatch, test_settings):
    report_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("app.rag_eval.REPORT_PATH", report_path)
    client = TestClient(main.create_app(settings=test_settings))

    response = client.get("/api/rag-eval")

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "report_invalid"


class _BusyRunner:
    def status(self):
        return {"status": "running", "running": True}

    async def start(self):
        raise EvalAlreadyRunning("已有一个评估任务在跑")


class _StartingRunner:
    def __init__(self) -> None:
        self.started = False

    def status(self):
        status = "running" if self.started else "idle"
        return {"status": status, "running": self.started, "log_tail": ""}

    async def start(self):
        self.started = True
        return self.status()


def test_rerun_while_running_returns_409(report_path, monkeypatch, test_settings):
    monkeypatch.setattr("app.rag_eval.REPORT_PATH", report_path)
    client = TestClient(
        main.create_app(settings=test_settings, rag_eval_runner=_BusyRunner())
    )

    response = client.post("/api/rag-eval/rerun")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "eval_running"


def test_rerun_is_accepted_and_status_can_be_polled(report_path, monkeypatch, test_settings):
    monkeypatch.setattr("app.rag_eval.REPORT_PATH", report_path)
    runner = _StartingRunner()
    client = TestClient(main.create_app(settings=test_settings, rag_eval_runner=runner))

    started = client.post("/api/rag-eval/rerun")
    status = client.get("/api/rag-eval/run")

    assert started.status_code == 202
    assert started.json()["run"]["running"] is True
    assert status.status_code == 200
    assert status.json()["run"]["running"] is True
    assert runner.started is True


def test_eval_page_is_served(test_settings):
    client = TestClient(main.create_app(settings=test_settings))

    response = client.get("/rag-eval/")

    assert response.status_code == 200
    assert "RAG 评估" in response.text
    assert "/rag-eval/app.js" in response.text
