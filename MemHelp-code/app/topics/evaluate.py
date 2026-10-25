"""ch10 评测：整体/每类指标、2×2 混淆矩阵、类对混淆、可复核的报告。

全部纯 Python（不引 numpy / matplotlib）。理由不是「不想装」，而是报告要能直接
用浏览器打开、随仓库提交、在 review 里 diff —— 一张 PNG 做不到这三件事，而
17 个 2×2 色块用 ``rgba()`` 就是十几行 CSS。

口径：

- 多标签下 micro 按「所有类目的 tp/fp/fn 汇总」算，macro 是每类 P/R/F1 的算术
  平均（无正例且无预测的类目记 0，不是 NaN），weighted 按 support 加权。
- ``support < 10`` 的类目在报告里必须标「样本不足」：150 条测试集摊到 17 类，
  末尾几类本来就只有个位数样本，不标注会让读者把 0.5 当真实水平。
- 类对混淆只统计「gold 有 A、pred 有 B」这种**判错**；漏标（gold 有、pred 没有）
  单独由 recall 反映，混在一起会让「最易混的两类」失真。
"""

from __future__ import annotations

import csv
import io
from typing import Any, Sequence

#: support 低于这个数的类目，报告里标「样本不足」。
SMALL_SUPPORT = 10


def _sets(labels: Sequence[Sequence[str]], label_names: Sequence[str]) -> list[set[str]]:
    known = set(label_names)
    out: list[set[str]] = []
    for row in labels:
        values = set(row)
        unknown = values - known
        if unknown:
            raise ValueError(f"未知类目：{sorted(unknown)}")
        out.append(values)
    return out


def _counts(
    gold: Sequence[set[str]], pred: Sequence[set[str]], name: str
) -> tuple[int, int, int, int]:
    tp = sum(1 for g, p in zip(gold, pred) if name in g and name in p)
    fp = sum(1 for g, p in zip(gold, pred) if name not in g and name in p)
    fn = sum(1 for g, p in zip(gold, pred) if name in g and name not in p)
    tn = sum(1 for g, p in zip(gold, pred) if name not in g and name not in p)
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def per_class_metrics(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], label_names: Sequence[str]
) -> list[dict[str, Any]]:
    gold_sets = _sets(gold, label_names)
    pred_sets = _sets(pred, label_names)
    rows: list[dict[str, Any]] = []
    for name in label_names:
        tp, fp, fn, tn = _counts(gold_sets, pred_sets, name)
        precision, recall, f1 = _prf(tp, fp, fn)
        rows.append(
            {
                "label": name,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def overall_metrics(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], label_names: Sequence[str]
) -> dict[str, float]:
    gold_sets = _sets(gold, label_names)
    pred_sets = _sets(pred, label_names)
    per_class = per_class_metrics(gold, pred, label_names)

    tp = sum(row["tp"] for row in per_class)
    fp = sum(row["fp"] for row in per_class)
    fn = sum(row["fn"] for row in per_class)
    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)

    macro_p = sum(row["precision"] for row in per_class) / len(per_class)
    macro_r = sum(row["recall"] for row in per_class) / len(per_class)
    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class)

    total_support = sum(row["support"] for row in per_class)
    if total_support:
        weighted_f1 = sum(row["f1"] * row["support"] for row in per_class) / total_support
    else:
        weighted_f1 = 0.0

    cells = len(gold_sets) * len(label_names)
    mismatches = sum(
        len(g.symmetric_difference(p)) for g, p in zip(gold_sets, pred_sets)
    )
    exact = sum(1 for g, p in zip(gold_sets, pred_sets) if g == p)
    return {
        "micro_precision": round(micro_p, 6),
        "micro_recall": round(micro_r, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "hamming_loss": round(mismatches / cells if cells else 0.0, 6),
        "exact_match": round(exact / len(gold_sets) if gold_sets else 0.0, 6),
        "samples": float(len(gold_sets)),
    }


def label_pair_confusions(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], label_names: Sequence[str]
) -> list[dict[str, Any]]:
    gold_sets = _sets(gold, label_names)
    pred_sets = _sets(pred, label_names)
    counter: dict[tuple[str, str], int] = {}
    for g, p in zip(gold_sets, pred_sets):
        for gold_label in g - p:
            for pred_label in p - g:
                key = (gold_label, pred_label)
                counter[key] = counter.get(key, 0) + 1
    rows = [
        {"gold_label": gold_label, "pred_label": pred_label, "count": count}
        for (gold_label, pred_label), count in counter.items()
    ]
    rows.sort(key=lambda row: (-row["count"], row["gold_label"], row["pred_label"]))
    return rows


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_report_md(
    metrics: dict[str, float],
    per_class: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    run_id: str,
    real_subset: dict[str, Any] | None = None,
    attr: dict[str, int] | None = None,
    multilabel_cases: dict[str, Any] | None = None,
) -> str:
    parts = [
        f"# ch10 主题分类评测报告（run `{run_id}`）",
        "",
        "## 整体指标",
        "",
        _md_table(
            ["指标", "值"],
            [
                ["micro-P", metrics["micro_precision"]],
                ["micro-R", metrics["micro_recall"]],
                ["micro-F1", metrics["micro_f1"]],
                ["macro-P", metrics["macro_precision"]],
                ["macro-R", metrics["macro_recall"]],
                ["macro-F1", metrics["macro_f1"]],
                ["weighted-F1", metrics["weighted_f1"]],
                ["Hamming loss", metrics["hamming_loss"]],
                ["完全一致率", metrics["exact_match"]],
                ["样本数", int(metrics.get("samples", 0))],
            ],
        ),
        "",
        "## 每类指标",
        "",
    ]

    rows = []
    for row in per_class:
        note = "（样本不足）" if row["support"] < SMALL_SUPPORT else ""
        rows.append(
            [
                row["label"],
                row["precision"],
                row["recall"],
                f"{row['f1']}{note}",
                row["support"],
                row["tp"],
                row["fp"],
                row["fn"],
            ]
        )
    parts.append(_md_table(["类目", "P", "R", "F1", "support", "TP", "FP", "FN"], rows))
    parts.append("")
    parts.append(
        f"> support < {SMALL_SUPPORT} 的类目已标「样本不足」：测试集 150 条的样本量"
        "摊到 17 类，尾部类目的分数只能当趋势看。"
    )

    parts += ["", "## 最易归错的类对（top 10）", ""]
    if pairs:
        parts.append(
            _md_table(
                ["gold 类目", "被预测成", "次数"],
                [[p["gold_label"], p["pred_label"], p["count"]] for p in pairs[:10]],
            )
        )
    else:
        parts.append("没有「判成别的类」的样本（漏标不算类对混淆，见 recall）。")

    if multilabel_cases:
        multi_total = multilabel_cases.get("multi_cases", multilabel_cases.get("total", 0))
        parts += [
            "",
            "## 多诉求专项（验收标准 3）",
            "",
            f"- 用例数：{multilabel_cases.get('total', 0)}",
            f"- 其中多诉求句：{multi_total}（单诉求句不进这个分母）",
            f"- 多诉求句全部类目命中：{multilabel_cases.get('hit2', 0)} / {multi_total}",
        ]
        misses = multilabel_cases.get("misses") or []
        if misses:
            parts.append("- 未达标的句子：")
            for item in misses:
                missing = item.get("missing") or []
                parts.append(
                    f"  - `{item['text']}` gold={item['labels']} pred={item['pred']}"
                    f" 漏={missing}"
                )

    if real_subset:
        parts += [
            "",
            "## 真实问题子集（只在 test 里）",
            "",
            f"- 条数：{real_subset.get('samples', 0)}",
            f"- macro-F1：{real_subset.get('macro_f1', 0.0)}",
            f"- 完全一致率：{real_subset.get('exact_match', 0.0)}",
            "",
            "读这个数之前先看两条：真实问题只有十几条、且集中在少数几类，"
            "macro-F1 是按 17 类平均的 —— 没出现的类目按 0 计入，"
            "少数类各占一个点，所以这个数会被结构性地压低，"
            "不代表模型在真实问题上「全错」。判断分布差异请看完全一致率和混淆矩阵；"
            "要拿到可信的真实语料 F1，得先攒够每类几十条真实标注。",
        ]

    if attr:
        parts += [
            "",
            "## 人工复核（抽判错样本）",
            "",
            _md_table(["归因", "条数"], [[k, v] for k, v in attr.items()]),
        ]

    parts.append("")
    return "\n".join(parts)


def render_confusion_html(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], label_names: Sequence[str]
) -> str:
    """17 个 2×2 混淆矩阵，纯 HTML+CSS，双击就能开。"""

    rows = per_class_metrics(gold, pred, label_names)
    pairs = label_pair_confusions(gold, pred, label_names)
    max_cell = max([max(r["tp"], r["fp"], r["fn"], r["tn"]) for r in rows] or [1]) or 1

    def cell(count: int, kind: str, caption: str) -> str:
        alpha = 0.08 + 0.72 * (count / max_cell)
        color = "220,38,38" if kind in ("fn", "fp") else "22,163,74"
        return (
            f'<td class="cell" style="background:rgba({color},{alpha:.3f})">'
            f'<span class="n">{count}</span><span class="cap">{caption}</span></td>'
        )

    blocks = []
    for row in rows:
        blocks.append(
            "<section class='card'>"
            f"<h3>{row['label']}<span class='sup'>support {row['support']}</span></h3>"
            "<table class='cm'>"
            "<tr><td class='corner'>gold \\ pred</td><td class='head'>预测有</td>"
            "<td class='head'>预测无</td></tr>"
            f"<tr><td class='head'>实际有</td>{cell(row['tp'], 'tp', 'TP')}"
            f"{cell(row['fn'], 'fn', 'FN')}</tr>"
            f"<tr><td class='head'>实际无</td>{cell(row['fp'], 'fp', 'FP')}"
            f"{cell(row['tn'], 'tn', 'TN')}</tr>"
            "</table></section>"
        )

    pair_rows = "".join(
        f"<tr><td>{p['gold_label']}</td><td>{p['pred_label']}</td><td>{p['count']}</td></tr>"
        for p in pairs[:15]
    ) or "<tr><td colspan='3'>没有判成别的类的样本</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ch10 混淆矩阵</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 12px}}
.card h3{{font-size:14px;margin:2px 0 8px;display:flex;justify-content:space-between}}
.sup{{color:#64748b;font-weight:400}}
table.cm{{border-collapse:collapse;width:100%}} .cell{{padding:8px;text-align:center;border:1px solid #fff}}
.cell .n{{display:block;font-size:18px;font-weight:600}} .cell .cap{{font-size:11px;color:#475569}}
.head{{font-size:11px;color:#475569;padding:4px}} .corner{{font-size:10px;color:#94a3b8}}
table.pairs{{border-collapse:collapse;margin-top:8px;background:#fff}}
table.pairs td,table.pairs th{{border:1px solid #e2e8f0;padding:5px 10px;font-size:13px}}
</style></head><body>
<h1>ch10 主题分类 · 每类 2×2 混淆矩阵</h1>
<p>颜色越深计数越大；红=错误格（FN/FP），绿=正确格（TP/TN）。</p>
<div class="grid">{''.join(blocks)}</div>
<h2>最易归错的类对（top 15）</h2>
<table class="pairs"><tr><th>gold 类目</th><th>被预测成</th><th>次数</th></tr>{pair_rows}</table>
</body></html>
"""


def render_error_review_csv(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    texts: Sequence[str],
    label_names: Sequence[str],
) -> str:
    """给人工复核用的错例清单：一条样本一行，漏了/多了哪些标签一目了然。"""

    gold_sets = _sets(gold, label_names)
    pred_sets = _sets(pred, label_names)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["text", "gold", "pred", "missing", "extra", "verdict", "note"])
    for text, g, p in zip(texts, gold_sets, pred_sets):
        if g == p:
            continue
        writer.writerow(
            [
                text,
                "|".join(sorted(g, key=label_names.index)),
                "|".join(sorted(p, key=label_names.index)),
                "|".join(sorted(g - p, key=label_names.index)),
                "|".join(sorted(p - g, key=label_names.index)),
                "",
                "",
            ]
        )
    return buffer.getvalue()


def baseline_regression(
    current_macro_f1: float, baseline: float, *, tolerance: float = 0.03
) -> str:
    """掉超过 ``tolerance`` 才算回归 —— 小数据集的正常抖动不该每次标红。"""

    return "regressed" if current_macro_f1 < baseline - tolerance else "ok"
