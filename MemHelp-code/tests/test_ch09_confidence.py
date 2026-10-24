"""ch09 证据置信度模型：特征、训练、留一法校准、模型落盘。"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from app.agent.evidence_confidence import (
    ConfidenceModel,
    EvidenceFeatures,
    calibrate_loo,
    features_from_scores,
    fit_logistic,
    load_model,
    model_from_dict,
    model_to_dict,
)


def test_features_margin_and_effective_count():
    f = features_from_scores([0.9, 0.8, 0.2])
    assert (f.top1, f.top2, round(f.margin, 3), f.effective_count) == (0.9, 0.8, 0.1, 2)


def test_features_empty_scores_are_zero():
    f = features_from_scores([])
    assert f == EvidenceFeatures(top1=0.0, top2=0.0, margin=0.0, effective_count=0)


def test_logistic_separates_synthetic_data():
    # 正例：高分 + 有一定分差；负例：低分 + 分差也很小。plan 里的原始样例
    # （正例 [0.9, 0.85] 分差 0.05、负例 [0.2] 分差 0.2）会让模型学出
    # 「分差越大越不可信」的反向权重，那是样例本身构造矛盾，不是模型缺陷。
    samples = [(features_from_scores([0.9, 0.6]), 1), (features_from_scores([0.2, 0.19]), 0)] * 20
    model = fit_logistic(samples)
    assert model.score(features_from_scores([0.9])) > 0.5
    assert model.score(features_from_scores([0.1])) < 0.5


def test_loo_report_has_confusion_matrix():
    samples = [(features_from_scores([0.9, 0.8]), 1), (features_from_scores([0.1]), 0)] * 5
    model, report = calibrate_loo(samples)
    assert report["balanced_accuracy"] >= 0.9
    assert set(report["confusion"]) == {"tp", "fp", "tn", "fn"}
    assert 0.05 <= report["threshold"] <= 0.95


@pytest.fixture
def model_dir():
    """工作区内的临时目录：沙箱下 ``tmp_path`` 退出后会被收回权限。"""
    root = Path(".uv-cache") / "ch09-confidence-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    made = root / uuid.uuid4().hex
    made.mkdir()
    try:
        yield made
    finally:
        shutil.rmtree(made, ignore_errors=True)


def test_model_roundtrip(model_dir):
    model = ConfidenceModel(
        weights=[1.0, -0.5, 0.25],
        intercept=0.1,
        threshold=0.6,
        mean=[0.5, 0.2, 1.0],
        std=[0.1, 0.05, 0.5],
        version="test",
    )
    path = model_dir / "model.json"
    path.write_text(json.dumps(model_to_dict(model)), encoding="utf-8")
    loaded = load_model(path)
    assert loaded is not None
    assert loaded.threshold == 0.6
    assert loaded.score(features_from_scores([0.9, 0.8])) == model.score(
        features_from_scores([0.9, 0.8])
    )


def test_load_model_missing_or_corrupt_returns_none(model_dir):
    assert load_model(model_dir / "missing.json") is None
    bad = model_dir / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_model(bad) is None


def test_model_from_dict_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        model_from_dict(
            {
                "weights": [1.0, 2.0],
                "intercept": 0.0,
                "threshold": 0.5,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
                "version": "x",
            }
        )


def test_fit_requires_two_classes():
    with pytest.raises(ValueError):
        fit_logistic([(features_from_scores([0.9]), 1)] * 5)
