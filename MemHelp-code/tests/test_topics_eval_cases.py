"""多诉求专项统计：只数 gold 真有两个以上诉求的用例。

第一版把 4 条单诉求用例也算进「命中 ≥2 个标签」的分母里，于是报告写
``10/20`` —— 那 4 条永远不可能命中两个，分母本身就是错的。验收要求的是
「多诉求句子能同时命中多个类目」，分母就该是真正的多诉求句子。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.topics.infer import TopicPrediction
from scripts.eval_topics import _multilabel_cases

CASES = Path(__file__).resolve().parent / "_tmp_multilabel_cases.jsonl"


class _StubClassifier:
    def __init__(self, labels_by_text: dict[str, list[str]]) -> None:
        self._labels = labels_by_text

    def predict(self, texts):  # noqa: ANN001, ANN201 - 测试桩
        return [
            TopicPrediction(labels=list(self._labels[text]), scores={}, fallback=False)
            for text in texts
        ]


def _write_cases(rows: list[dict]) -> None:
    CASES.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def teardown_function() -> None:
    CASES.unlink(missing_ok=True)


def test_single_request_case_is_not_counted_as_multi():
    _write_cases(
        [
            {"text": "开发票要税号吗", "labels": ["发票"]},
            {"text": "买大了想退", "labels": ["尺码", "退换货"]},
        ]
    )
    classifier = _StubClassifier({"开发票要税号吗": ["发票"], "买大了想退": ["尺码"]})

    out = _multilabel_cases(classifier, CASES)

    assert out["total"] == 2
    assert out["multi_cases"] == 1, "单诉求用例不进多诉求分母"
    assert out["hit2"] == 0
    assert out["target"] == "1/1"
    assert [miss["text"] for miss in out["misses"]] == ["买大了想退"]


def test_two_label_case_needs_both_gold_labels():
    _write_cases([{"text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]}])
    classifier = _StubClassifier({"买大了想退，运费谁出": ["尺码", "退换货"]})

    out = _multilabel_cases(classifier, CASES)

    assert out["hit2"] == 0
    assert out["misses"][0]["missing"] == ["运费"], "漏了哪个类目要写清楚"


def test_both_gold_labels_hit_counts():
    _write_cases([{"text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]}])
    classifier = _StubClassifier({"买大了想退，运费谁出": ["退换货", "尺码", "运费"]})

    out = _multilabel_cases(classifier, CASES)

    assert out["hit2"] == 1
    assert out["misses"] == []
