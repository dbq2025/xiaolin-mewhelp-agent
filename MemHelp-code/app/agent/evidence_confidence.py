"""证据置信度模型（ch09）。

正式闸不再只看「Top1 分数是否过 0.3」，而是把检索/精排结果压成三个信号：

* ``top1``：精排 Top1 的相关性分（重排降级时是 RRF 融合分，量纲不同）；
* ``margin``：Top1 与 Top2 的分差，分差小说明「有命中但说不准是哪条」；
* ``effective_count``：达到「Top1 一半分且不低于 0.1」的证据条数，衡量有效覆盖。

三个信号喂给一个手写逻辑回归（不引第三方 ML 依赖），阈值用 ch04 评估集
留一法（LOO）校准 —— 评估集只有 8 条拒答，自拟合自评会虚假漂亮，留一法
至少保证每个样本的判定来自没见过它的模型。模型以 JSON artifact 落盘，
加载失败一律退回 ``None``，由闸调用方决定退回 ch05 阈值版。

这一层是纯函数/纯数据，没有 IO（除 ``load_model``）；训练与校准放在
``scripts/calibrate_confidence.py``，生产只读 artifact。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "FEATURE_NAMES",
    "ConfidenceModel",
    "EvidenceFeatures",
    "calibrate_loo",
    "features_from_scores",
    "fit_logistic",
    "load_model",
    "model_from_dict",
    "model_to_dict",
]

FEATURE_NAMES = ("top1", "margin", "effective_count")

#: 三个特征在模型里的顺序，落盘 artifact 与训练必须一致。
_EPS = 1e-9


@dataclass(frozen=True)
class EvidenceFeatures:
    top1: float
    top2: float
    margin: float
    effective_count: int

    def vector(self) -> tuple[float, float, float]:
        return (self.top1, self.margin, float(self.effective_count))


@dataclass(frozen=True)
class ConfidenceModel:
    weights: list[float]
    intercept: float
    threshold: float
    mean: list[float]
    std: list[float]
    version: str

    def score(self, features: EvidenceFeatures) -> float:
        total = self.intercept
        for value, weight, mean, std in zip(
            features.vector(), self.weights, self.mean, self.std
        ):
            total += weight * ((value - mean) / (std if std else 1.0))
        return _sigmoid(total)

    def passes(self, features: EvidenceFeatures) -> bool:
        return self.score(features) >= self.threshold


def features_from_scores(scores: Sequence[float]) -> EvidenceFeatures:
    """把一组检索分（已按相关性降序）压成三个信号。"""
    values = [float(score) for score in scores]
    if not values:
        return EvidenceFeatures(top1=0.0, top2=0.0, margin=0.0, effective_count=0)
    top1 = values[0]
    top2 = values[1] if len(values) > 1 else 0.0
    cutoff = max(0.1, top1 * 0.5)
    effective = sum(1 for value in values if value >= cutoff)
    return EvidenceFeatures(
        top1=top1,
        top2=top2,
        margin=top1 - top2,
        effective_count=effective,
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _standardizer(rows: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    count = len(rows)
    mean = [sum(row[i] for row in rows) / count for i in range(3)]
    variance = [
        sum((row[i] - mean[i]) ** 2 for row in rows) / count for i in range(3)
    ]
    std = [math.sqrt(value) if value > _EPS else 1.0 for value in variance]
    # 标准差为 0 时置 1：该特征在训练集里没有区分度，标准化后恒为 0。
    if any(value <= _EPS for value in variance):
        std = [1.0 if value <= _EPS else math.sqrt(value) for value in variance]
    return mean, std


def _train(
    rows: list[tuple[float, float, float]],
    labels: list[int],
    *,
    epochs: int,
    lr: float,
    l2: float,
    threshold: float,
    version: str,
) -> ConfidenceModel:
    mean, std = _standardizer(rows)
    normalized = [
        tuple((row[i] - mean[i]) / std[i] for i in range(3)) for row in rows
    ]
    weights = [0.0, 0.0, 0.0]
    intercept = 0.0
    count = len(normalized)
    for _ in range(epochs):
        grad_w = [0.0, 0.0, 0.0]
        grad_b = 0.0
        for row, label in zip(normalized, labels):
            linear = intercept + sum(w * x for w, x in zip(weights, row))
            error = _sigmoid(linear) - label
            for i in range(3):
                grad_w[i] += error * row[i]
            grad_b += error
        for i in range(3):
            weights[i] -= lr * (grad_w[i] / count + l2 * weights[i])
        intercept -= lr * (grad_b / count)
    return ConfidenceModel(
        weights=weights,
        intercept=intercept,
        threshold=threshold,
        mean=mean,
        std=std,
        version=version,
    )


def fit_logistic(
    samples: Sequence[tuple[EvidenceFeatures, int]],
    *,
    epochs: int = 3000,
    lr: float = 0.1,
    l2: float = 1e-3,
    threshold: float = 0.5,
    version: str = "ch09-v1",
) -> ConfidenceModel:
    """固定轮数梯度下降；样本必须同时含正负类。"""
    if not samples:
        raise ValueError("训练样本为空")
    labels = [int(label) for _, label in samples]
    if len(set(labels)) < 2:
        raise ValueError("训练样本必须同时包含正类与负类")
    rows = [features.vector() for features, _ in samples]
    return _train(
        rows,
        labels,
        epochs=epochs,
        lr=lr,
        l2=l2,
        threshold=threshold,
        version=version,
    )


def _balanced_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    positive = tp + fn
    negative = tn + fp
    if positive == 0 or negative == 0:
        return 0.0
    return 0.5 * (tp / positive + tn / negative)


def calibrate_loo(
    samples: Sequence[tuple[EvidenceFeatures, int]],
) -> tuple[ConfidenceModel, dict[str, Any]]:
    """留一法评估 + 阈值扫描；返回全量重训的最终模型与报告。

    阈值在 0.05..0.95 之间按 0.05 扫描，取 balanced accuracy 最大者；
    并列时取更接近 0.5 的（少一点极端阈值）。
    """
    if len(samples) < 2:
        raise ValueError("留一法至少需要 2 个样本")
    if len({int(label) for _, label in samples}) < 2:
        raise ValueError("留一法样本必须同时包含正类与负类")

    probabilities: list[float] = []
    for index, (features, _) in enumerate(samples):
        rest = list(samples[:index]) + list(samples[index + 1 :])
        model = fit_logistic(rest)
        probabilities.append(model.score(features))

    labels = [int(label) for _, label in samples]
    best_threshold = 0.5
    best_score = -1.0
    best_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for step in range(1, 20):
        threshold = step * 0.05
        tp = fp = tn = fn = 0
        for probability, label in zip(probabilities, labels):
            predicted = 1 if probability >= threshold else 0
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        score = _balanced_accuracy(tp, fp, tn, fn)
        if score > best_score + _EPS or (
            abs(score - best_score) <= _EPS
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = threshold
            best_confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    final = fit_logistic(samples, threshold=best_threshold)
    report = {
        "n": len(samples),
        "positives": sum(labels),
        "threshold": round(best_threshold, 2),
        "balanced_accuracy": round(best_score, 4),
        "confusion": best_confusion,
        "probabilities": [round(value, 4) for value in probabilities],
        "labels": labels,
    }
    return final, report


def model_to_dict(model: ConfidenceModel) -> dict[str, Any]:
    return {
        "version": model.version,
        "feature_names": list(FEATURE_NAMES),
        "weights": list(model.weights),
        "intercept": model.intercept,
        "threshold": model.threshold,
        "mean": list(model.mean),
        "std": list(model.std),
    }


def model_from_dict(data: dict[str, Any]) -> ConfidenceModel:
    weights = [float(value) for value in data["weights"]]
    mean = [float(value) for value in data["mean"]]
    std = [float(value) for value in data["std"]]
    if not (len(weights) == len(mean) == len(std) == len(FEATURE_NAMES)):
        raise ValueError("模型维度与特征数量不一致")
    return ConfidenceModel(
        weights=weights,
        intercept=float(data["intercept"]),
        threshold=float(data["threshold"]),
        mean=mean,
        std=std,
        version=str(data.get("version") or "unknown"),
    )


def load_model(path: Path) -> ConfidenceModel | None:
    """缺失/坏 JSON/维度不符一律返回 None，由调用方决定降级路径。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return model_from_dict(data)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
