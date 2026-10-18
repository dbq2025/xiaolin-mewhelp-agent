"""ch06 Task 12：多轮评估集的离线校验与指标汇总。"""

from __future__ import annotations

import pytest

from evals.run_ch06_evals import load_cases, summarise, validate_case


def test_validate_rejects_unknown_intent():
    with pytest.raises(ValueError):
        validate_case({"id": "X", "turns": [{"user": "a", "intent": "nope"}]})


def test_validate_rejects_empty_turns():
    with pytest.raises(ValueError):
        validate_case({"id": "X", "turns": []})


def test_load_cases_reads_multiturn_file():
    cases = load_cases()
    assert len(cases) >= 8
    assert {case["id"] for case in cases} >= {"M01", "M04", "M08"}


def test_metrics_counts_other_and_coref():
    rows = [
        {
            "id": "M01",
            "turn": 1,
            "expected_intent": "logistics",
            "actual_intent": "logistics",
            "intent_ok": True,
            "coref_ok": True,
            "json_ok": True,
            "expansion_checked": False,
            "expansion_ok": True,
        },
        {
            "id": "M03",
            "turn": 1,
            "expected_intent": "other",
            "actual_intent": "other",
            "intent_ok": True,
            "coref_ok": False,
            "json_ok": True,
            "expansion_checked": False,
            "expansion_ok": True,
        },
        {
            "id": "M03",
            "turn": 2,
            "expected_intent": "refund_return",
            "actual_intent": "other",
            "intent_ok": False,
            "coref_ok": True,
            "json_ok": True,
            "expansion_checked": True,
            "expansion_ok": False,
        },
    ]
    report = summarise(rows)
    assert report["total"] == 3
    assert report["intent_accuracy"] == pytest.approx(2 / 3)
    assert report["coref_accuracy"] == pytest.approx(2 / 3)
    assert report["other_accuracy"] == 1.0
    assert report["json_failures"] == 0
    assert report["expansion_failures"] == 1
    assert len(report["failures"]) == 2
