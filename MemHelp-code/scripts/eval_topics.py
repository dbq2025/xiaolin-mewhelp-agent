"""ch10 评测入口：在留出测试集上出 F1/混淆矩阵，并跑多诉求专项。

    python scripts/eval_topics.py --run-id latest

产物落在 ``reports/topics/<run_id>/``：

- ``metrics.json``：整体 + 每类 + 真实子集 + 多诉求专项
- ``report.md``：给人看的报告（类目表、类对混淆、人工复核段）
- ``confusion.html``：17 个 2×2 色块，双击就开
- ``error_review.csv``：错例清单，人工抽判用
- ``predictions.jsonl``：逐条 gold/pred，便于复查
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from app.topics.evaluate import (
    baseline_regression,
    label_pair_confusions,
    overall_metrics,
    per_class_metrics,
    render_confusion_html,
    render_error_review_csv,
    render_report_md,
)
from app.topics.infer import TopicClassifier
from app.topics.taxonomy import CATEGORY_NAMES

ROOT = Path(__file__).resolve().parents[1]
#: 多诉求专项的目标：每句至少命中 2 个标签（验收标准 3）。
MIN_LABELS_FOR_MULTI = 2


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"找不到文件：{path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _multilabel_cases(classifier: TopicClassifier, cases_path: Path) -> dict[str, Any]:
    """多诉求专项：**分母只算 gold 真有两个以上诉求的句子**。

    单诉求用例永远不可能命中两个标签，把它们算进分母，得出的比例既不能
    验收也不能回归（第一版就报成了 10/20，其中 4 条注定失败）。
    """

    cases = _read_jsonl(cases_path)
    predictions = classifier.predict([case["text"] for case in cases])
    multi_cases = 0
    misses: list[dict[str, Any]] = []
    for case, prediction in zip(cases, predictions):
        gold = set(case["labels"])
        if len(gold) < MIN_LABELS_FOR_MULTI:
            continue
        multi_cases += 1
        missing = [label for label in case["labels"] if label not in prediction.labels]
        if not missing:
            continue
        misses.append(
            {
                "text": case["text"],
                "labels": case["labels"],
                "pred": prediction.labels,
                "missing": missing,
            }
        )
    return {
        "total": len(cases),
        "multi_cases": multi_cases,
        "hit2": multi_cases - len(misses),
        "target": f"{multi_cases}/{multi_cases}",
        "misses": misses,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch10 主题分类评测")
    parser.add_argument("--model", default="models/topics/latest")
    parser.add_argument("--test", default="data/topics/splits/test.jsonl")
    parser.add_argument("--cases", default="evals/ch10_multilabel_cases.jsonl")
    parser.add_argument("--reports", default="reports/topics")
    parser.add_argument("--baseline", default="evals/ch10_baseline.json")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args(argv)

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"找不到模型目录：{model_dir}（先跑 scripts/train_topics.py）")

    test_rows = _read_jsonl(Path(args.test))
    if not test_rows:
        raise SystemExit("测试集为空")

    classifier = TopicClassifier.load(model_dir)
    started = time.perf_counter()
    predictions = classifier.predict([row["text"] for row in test_rows])
    pred_labels = [item.labels for item in predictions]
    gold_labels = [row["labels"] for row in test_rows]

    metrics = overall_metrics(gold_labels, pred_labels, CATEGORY_NAMES)
    per_class = per_class_metrics(gold_labels, pred_labels, CATEGORY_NAMES)
    pairs = label_pair_confusions(gold_labels, pred_labels, CATEGORY_NAMES)

    real_rows = [row for row in test_rows if row.get("source") == "real"]
    real_subset = None
    if real_rows:
        real_index = {id(row) for row in real_rows}
        gold_real = [row["labels"] for row in real_rows]
        pred_real = [
            pred.labels for row, pred in zip(test_rows, predictions) if id(row) in real_index
        ]
        real_metrics = overall_metrics(gold_real, pred_real, CATEGORY_NAMES)
        real_subset = {
            "samples": len(real_rows),
            "macro_f1": real_metrics["macro_f1"],
            "exact_match": real_metrics["exact_match"],
        }

    cases_summary = _multilabel_cases(classifier, Path(args.cases)) if Path(args.cases).is_file() else None

    out_dir = Path(args.reports) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "model_dir": str(model_dir),
        "overall": metrics,
        "per_class": per_class,
        "pair_confusions": pairs,
        "real_subset": real_subset,
        "multilabel_cases": cases_summary,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    _write(out_dir / "metrics.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write(
        out_dir / "report.md",
        render_report_md(
            metrics,
            per_class,
            pairs,
            run_id=args.run_id,
            real_subset=real_subset,
            multilabel_cases=cases_summary,
        ),
    )
    _write(
        out_dir / "confusion.html",
        render_confusion_html(gold_labels, pred_labels, CATEGORY_NAMES),
    )
    _write(
        out_dir / "error_review.csv",
        render_error_review_csv(
            gold_labels, pred_labels, [row["text"] for row in test_rows], CATEGORY_NAMES
        ),
    )
    _write(
        out_dir / "predictions.jsonl",
        "\n".join(
            json.dumps(
                {
                    "id": row.get("id"),
                    "text": row["text"],
                    "gold": row["labels"],
                    "pred": prediction.labels,
                    "fallback": prediction.fallback,
                    "source": row.get("source"),
                },
                ensure_ascii=False,
            )
            for row, prediction in zip(test_rows, predictions)
        )
        + "\n",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "exact_match": metrics["exact_match"],
                "multi_label_hit2": f"{cases_summary['hit2']}/{cases_summary['multi_cases']}"
                if cases_summary
                else "n/a",
                "real_subset": real_subset,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    baseline_path = Path(args.baseline)
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        verdict = baseline_regression(metrics["macro_f1"], float(baseline.get("macro_f1", 0.0)))
        print(
            f"[baseline] {verdict}: {metrics['macro_f1']} vs {baseline.get('macro_f1')}"
            f"（容差 0.03，标红不阻断）",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
