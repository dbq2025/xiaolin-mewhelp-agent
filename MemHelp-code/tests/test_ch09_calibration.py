"""ch09 Task 7：置信度校准脚本的纯函数与采样。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "scripts" / "calibrate_confidence.py"
    spec = importlib.util.spec_from_file_location("calibrate_confidence_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibration():
    return _load_module()


def test_load_cases_skips_comments(calibration):
    cases = calibration.load_cases(ROOT / "evals" / "ch04_cases.jsonl")
    assert len(cases) == 48
    assert {case["behavior"] for case in cases} == {"answer", "refuse"}


def test_label_mapping(calibration):
    assert calibration.label_for("answer") == 1
    assert calibration.label_for("refuse") == 0
    with pytest.raises(ValueError):
        calibration.label_for("maybe")


def test_published_artifact_is_loadable(calibration):
    from app.agent.evidence_confidence import load_model

    model = load_model(calibration.MODEL_PATH)
    assert model is not None
    assert model.version == "ch09-v1"
    assert 0.0 <= model.threshold <= 1.0


class _Pipeline:
    def __init__(self, scores_by_query: dict[str, list[float]]) -> None:
        self._scores = scores_by_query
        self.calls = 0

    async def retrieve_for_gate(self, query, *, category=None):
        self.calls += 1
        return SimpleNamespace(
            evidence=[
                SimpleNamespace(score=score)
                for score in self._scores.get(query, [])
            ]
        )


async def test_collect_samples_uses_gate_retrieval_and_labels(calibration):
    cases = [
        {"id": "A1", "query": "怎么退", "behavior": "answer", "bucket": "A_policy"},
        {"id": "R1", "query": "月球", "behavior": "refuse", "bucket": "B_model"},
    ]
    pipeline = _Pipeline({"怎么退": [0.9, 0.5], "月球": []})

    samples = await calibration.collect_samples(cases, pipeline)

    assert pipeline.calls == 2
    assert [label for _features, label in samples] == [1, 0]
    assert samples[0][0].top1 == pytest.approx(0.9)
    assert samples[0][0].effective_count == 2
    assert samples[1][0].effective_count == 0


async def test_collect_samples_survives_retrieval_failure(calibration):
    class _Broken:
        async def retrieve_for_gate(self, query, *, category=None):
            raise RuntimeError("milvus down")

    samples = await calibration.collect_samples(
        [{"id": "X", "query": "q", "behavior": "answer"}], _Broken()
    )

    assert samples[0][0].top1 == 0.0
    assert calibration.failed_ids == ["X"]


def test_write_report_contains_confusion_and_bucket_table(calibration, tmp_path_factory=None):
    output = ROOT / ".uv-cache" / "ch09-calibration-report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_report(
        output,
        cases=[
            {"id": "A1", "bucket": "A_policy", "behavior": "answer", "query": "怎么退"},
            {"id": "R1", "bucket": "B_model", "behavior": "refuse", "query": "月球"},
        ],
        samples=[(SimpleNamespace(top1=0.9), 1), (SimpleNamespace(top1=0.0), 0)],
        model={"threshold": 0.35, "mean": 0.0, "std": 1.0, "weights": [1, 1, 1, 1], "intercept": 0.0},
        calibration_report={
            "balanced_accuracy": 1.0,
            "threshold": 0.35,
            "confusion": {"tp": 1, "fp": 0, "tn": 1, "fn": 0},
        },
        published=True,
        failed_ids=[],
    )
    text = output.read_text(encoding="utf-8")
    assert "balanced accuracy" in text.lower()
    assert "A_policy" in text and "B_model" in text
    output.unlink()
