"""ch10 分层抽样：按**完整标签组合**做迭代分层，划 80/10/10。

为什么不是逐标签分层：多标签样本（「尺码+退换货」）在逐标签分层里会被
当成两件独立的事各切一刀，结果是训练集和测试集的多诉求比例对不上，
评测出来的 macro-F1 会系统性偏好单标签样本。

为什么小组合整组进 train：一个只出现 2 次的组合，按 10% 切会切出 0.2 条 ——
四舍五入成 1 条就是把 val 的一半样本赌在这一条上。宁可在报告里标
「该组合只在 train」，也不要制造分布外的 val/test 样本。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.topics.taxonomy import CATEGORY_NAMES

#: 小于这个数的组合不切 val/test，整组留 train。
_MIN_GROUP_FOR_SPLIT = 3


@dataclass(frozen=True, slots=True)
class SplitResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    report: dict[str, Any] = field(default_factory=dict)


def _stratum_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(set(row["labels"])))


def iterative_stratify(
    rows: Sequence[dict[str, Any]],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    real_rows: Sequence[dict[str, Any]] = (),
) -> SplitResult:
    """切三份。``real_rows`` 是真实问题，**强制进 test**（最像线上，放训练集等于背答案）。"""

    if len(ratios) != 3:
        raise ValueError("ratios 必须是三个数：train / val / test")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 之和必须是 1，当前 {sum(ratios)}")

    rng = random.Random(seed)

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_stratum_key(row), []).append(dict(row))

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for key in sorted(groups):
        bucket = groups[key]
        rng.shuffle(bucket)
        if len(bucket) < _MIN_GROUP_FOR_SPLIT:
            train.extend(bucket)
            continue
        n_val = max(1, round(len(bucket) * ratios[1]))
        n_test = max(1, round(len(bucket) * ratios[2]))
        # 极端小组合（3 条）三个份额加起来会超过组大小，按比例让 train 兜底。
        n_val = min(n_val, max(0, len(bucket) - n_test - 1))
        val.extend(bucket[:n_val])
        test.extend(bucket[n_val : n_val + n_test])
        train.extend(bucket[n_val + n_test :])

    test.extend(dict(row) for row in real_rows)

    per_label: dict[str, dict[str, int]] = {}
    zero_in_split: list[dict[str, str]] = []
    for label in CATEGORY_NAMES:
        counts = {
            "train": sum(1 for x in train if label in x["labels"]),
            "val": sum(1 for x in val if label in x["labels"]),
            "test": sum(1 for x in test if label in x["labels"]),
        }
        per_label[label] = counts
        for name, value in counts.items():
            if value == 0:
                zero_in_split.append({"label": label, "split": name})

    report: dict[str, Any] = {
        "total": len(train) + len(val) + len(test),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "per_label": per_label,
        "zero_in_split": zero_in_split,
        "strata_count": len(groups),
        "real_in_test": len(real_rows),
    }
    return SplitResult(train=train, val=val, test=test, report=report)
