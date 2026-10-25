"""ch10 评测：所有数字都要能手算复核。

这份测试里的期望值都是**按混淆矩阵手算**出来的，不是从实现里抄的 —— 报告要
拿给运营看「哪类总漏」，算错了没人能发现。特别地：

- ``GOLD=[[退换货],[物流],[尺码,退换货]]``、``PRED=[[退换货],[物流],[尺码]]``
- gold 正例 2+1+1=4，pred 正例 1+1+1=3，tp=3 → micro-P=1.0、micro-R=0.75、micro-F1=6/7
- 第 3 行 gold 有「退换货」而 pred 没有 → 这是漏标（FN），**不是**把 A 判成 B，
  所以「类对混淆」应为空
"""

from __future__ import annotations

import pytest

from app.topics.evaluate import (
    baseline_regression,
    label_pair_confusions,
    overall_metrics,
    per_class_metrics,
    render_confusion_html,
    render_error_review_csv,
    render_report_md,
)

NAMES = ["退换货", "物流", "尺码"]
GOLD = [["退换货"], ["物流"], ["尺码", "退换货"]]
PRED = [["退换货"], ["物流"], ["尺码"]]


def test_overall_micro_and_macro():
    m = overall_metrics(GOLD, PRED, NAMES)
    assert m["micro_precision"] == pytest.approx(1.0)
    assert m["micro_recall"] == pytest.approx(0.75)
    assert m["micro_f1"] == pytest.approx(6 / 7)
    assert m["exact_match"] == pytest.approx(2 / 3)
    assert m["hamming_loss"] == pytest.approx(1 / 9)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert "weighted_f1" in m


def test_per_class_has_all_fields():
    rows = per_class_metrics(GOLD, PRED, NAMES)
    assert [r["label"] for r in rows] == NAMES
    assert rows[0]["tp"] == 1 and rows[0]["fn"] == 1 and rows[0]["support"] == 2
    assert rows[0]["precision"] == pytest.approx(1.0)
    assert rows[0]["recall"] == pytest.approx(0.5)
    # 尺码：只有第 3 行是正例且判对；前两行 gold 与 pred 都是 0 → tn=2
    assert rows[2]["tp"] == 1 and rows[2]["tn"] == 2


def test_pair_confusions_exclude_correct():
    assert label_pair_confusions(GOLD, PRED, NAMES) == []


def test_pair_confusions_counts_wrong_label():
    gold = [["退换货"]]
    pred = [["尺码"]]
    assert label_pair_confusions(gold, pred, NAMES) == [
        {"gold_label": "退换货", "pred_label": "尺码", "count": 1}
    ]


def test_report_flags_small_support():
    md = render_report_md(
        overall_metrics(GOLD, PRED, NAMES),
        per_class_metrics(GOLD, PRED, NAMES),
        [],
        run_id="r1",
    )
    assert "样本不足" in md and "r1" in md


def test_confusion_html_is_self_contained():
    html = render_confusion_html(GOLD, PRED, NAMES)
    assert "<table" in html and "退换货" in html
    assert "http" not in html.split("<style>")[0]  # 不许引外部脚本/样式


def test_error_review_csv_lists_only_errors():
    csv_text = render_error_review_csv(GOLD, PRED, ["a", "b", "c"], NAMES)
    lines = [line for line in csv_text.splitlines() if line]
    assert lines[0].startswith("text,")
    # 只有第 3 条判错（漏了「退换货」），对的两条不该进复核清单
    assert len(lines) == 2
    assert lines[1].startswith("c,") and "退换货" in lines[1]


def test_baseline_regression_threshold():
    assert baseline_regression(0.80, 0.82) == "ok"
    assert baseline_regression(0.78, 0.82) == "regressed"
