"""ch05 Task 12：意图评估脚本的判定逻辑（离线）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "evals" / "run_ch05_intent_evals.py"
    spec = importlib.util.spec_from_file_location("run_ch05_intent_evals_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_cases_reads_seven_intents():
    module = _load_module()
    cases = module.load_cases(ROOT / "evals" / "ch05_intent_cases.jsonl")

    assert len(cases) >= 56
    assert {case.expected_intent for case in cases} == {
        "logistics",
        "order",
        "product_consult",
        "refund_return",
        "after_sales",
        "complaint",
        "chitchat",
    }


def test_aggregate_computes_accuracy_and_confusion():
    module = _load_module()
    cases = [
        module.Case("1", "q1", "logistics", "business", "", ""),
        module.Case("2", "q2", "logistics", "business", "", ""),
        module.Case("3", "q3", "chitchat", "chitchat", "", ""),
    ]
    predictions = [
        module.Prediction(cases[0], "logistics", "llm", "", 1.0),
        module.Prediction(cases[1], "order", "llm", "", 1.0),
        # 把闲聊误判成物流：意图错、出口也错
        module.Prediction(cases[2], "logistics", "llm", "", 1.0),
    ]

    report = module.aggregate(predictions)

    assert report["accuracy"] == 1 / 3
    # 第二条 logistics→order 仍在 business 出口，不算路由错
    assert report["route_accuracy"] == 2 / 3
    assert report["per_intent_recall"]["logistics"] == 0.5
    assert report["per_intent_recall"]["chitchat"] == 0.0
    assert report["confusion"]["logistics"]["order"] == 1
    assert report["confusion"]["chitchat"]["logistics"] == 1
    assert len(report["failures"]) == 2


def test_exit_code_requires_overall_and_per_class_threshold():
    module = _load_module()
    assert module.exit_code(
        {"accuracy": 0.9, "route_accuracy": 0.9, "per_intent_recall": {"chitchat": 0.7}}
    ) == 0
    assert module.exit_code(
        {"accuracy": 0.9, "route_accuracy": 0.9, "per_intent_recall": {"chitchat": 0.4}}
    ) == 1
    assert module.exit_code(
        {"accuracy": 0.7, "route_accuracy": 0.9, "per_intent_recall": {"chitchat": 0.9}}
    ) == 1
