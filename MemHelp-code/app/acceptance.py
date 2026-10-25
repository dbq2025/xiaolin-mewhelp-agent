"""ch10 验收页的数据源：训练/验证/测试集体量、评测指标、错例清单、类目表。

这些数据全在仓库文件里（``data/topics`` / ``reports/topics`` / ``evals``），不进库：
评测报告是「某一版模型的证据」，跟着 commit 走比放进数据库更可信 —— 库里的数会变，
报告快照不会。

页面只读不写；文件缺失返回 404 而不是 500，前端据此显示「还没跑评测」。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "topics"
REPORTS_DIR = PROJECT_ROOT / "reports" / "topics"
BASELINE_FILE = PROJECT_ROOT / "evals" / "ch10_baseline.json"

SMALL_SUPPORT = 10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _latest_run_id() -> str | None:
    if not BASELINE_FILE.is_file():
        return None
    try:
        return str(_read_json(BASELINE_FILE).get("run_id") or "") or None
    except (OSError, json.JSONDecodeError):
        return None


def _metrics_path() -> Path | None:
    run_id = _latest_run_id()
    if not run_id:
        return None
    path = REPORTS_DIR / run_id / "metrics.json"
    return path if path.is_file() else None


def _split_sizes() -> dict[str, int]:
    splits = DATA_DIR / "splits"
    return {
        "train": _count_lines(splits / "train_aug.jsonl"),
        "val": _count_lines(splits / "val.jsonl"),
        "test": _count_lines(splits / "test.jsonl"),
        "test_gold": _count_lines(splits / "test_gold.jsonl"),
        "hard_cases": _count_lines(DATA_DIR / "extra" / "hard_cases.jsonl"),
    }


def _multi_counts() -> dict[str, int]:
    """每份语料里 gold 标签 ≥2 的句子数 —— 多标签能力要靠这个喂养。"""

    out: dict[str, int] = {}
    for name, file_name in (("train", "train_aug.jsonl"), ("val", "val.jsonl"), ("test", "test.jsonl")):
        path = DATA_DIR / "splits" / file_name
        multi = 0
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    labels = json.loads(line).get("labels") or []
                except json.JSONDecodeError:
                    continue
                if len(set(labels)) >= 2:
                    multi += 1
        out[name] = multi
    return out


def _error_rows() -> list[dict[str, str]]:
    run_id = _latest_run_id()
    if not run_id:
        return []
    path = REPORTS_DIR / run_id / "error_review.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def register_acceptance_routes(app: FastAPI) -> None:
    """挂验收页的三个只读接口（数据页、评测页、错例页各一个）。"""

    @app.get("/api/acceptance/summary")
    async def acceptance_summary() -> JSONResponse:
        metrics_path = _metrics_path()
        if metrics_path is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"type": "no_report", "message": "还没有评测报告，先跑 scripts/eval_topics.py"}},
            )
        metrics = _read_json(metrics_path)
        overall = metrics.get("overall") or {}
        per_class = metrics.get("per_class") or []
        worst = sorted(per_class, key=lambda row: row.get("f1", 0))[:5]
        baseline = _read_json(BASELINE_FILE) if BASELINE_FILE.is_file() else {}
        errors = _error_rows()
        return JSONResponse(
            status_code=200,
            content={
                "run_id": metrics.get("run_id"),
                "generated_at": datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "dataset": _split_sizes(),
                "multi": _multi_counts(),
                "overall": overall,
                "multilabel_cases": metrics.get("multilabel_cases"),
                "real_subset": metrics.get("real_subset"),
                "worst_classes": [
                    {
                        "label": row.get("label"),
                        "f1": row.get("f1"),
                        "precision": row.get("precision"),
                        "recall": row.get("recall"),
                        "support": row.get("support"),
                        "small_support": (row.get("support") or 0) < SMALL_SUPPORT,
                    }
                    for row in worst
                ],
                "error_count": len(errors),
                "baseline": baseline,
                "skipped_unlabeled": metrics.get("skipped_unlabeled", 0),
            },
        )

    @app.get("/api/acceptance/eval")
    async def acceptance_eval() -> JSONResponse:
        metrics_path = _metrics_path()
        if metrics_path is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"type": "no_report", "message": "还没有评测报告"}},
            )
        metrics = _read_json(metrics_path)
        per_class = metrics.get("per_class") or []
        return JSONResponse(
            status_code=200,
            content={
                "run_id": metrics.get("run_id"),
                "overall": metrics.get("overall"),
                "multilabel_cases": metrics.get("multilabel_cases"),
                "real_subset": metrics.get("real_subset"),
                "per_class": [
                    {
                        "label": row.get("label"),
                        "precision": row.get("precision"),
                        "recall": row.get("recall"),
                        "f1": row.get("f1"),
                        "support": row.get("support"),
                        "tp": row.get("tp"),
                        "fp": row.get("fp"),
                        "fn": row.get("fn"),
                        "tn": row.get("tn"),
                        "small_support": (row.get("support") or 0) < SMALL_SUPPORT,
                    }
                    for row in per_class
                ],
                "pair_confusions": (metrics.get("pair_confusions") or [])[:8],
                "confusion_url": f"/acceptance/reports/{metrics.get('run_id')}/confusion.html",
            },
        )

    @app.get("/api/acceptance/data")
    async def acceptance_data() -> JSONResponse:
        taxonomy_path = DATA_DIR / "taxonomy.json"
        taxonomy = _read_json(taxonomy_path) if taxonomy_path.is_file() else {"categories": []}
        files = []
        for path in (
            DATA_DIR / "taxonomy.json",
            DATA_DIR / "taxonomy.md",
            DATA_DIR / "splits" / "train_aug.jsonl",
            DATA_DIR / "splits" / "val.jsonl",
            DATA_DIR / "splits" / "test.jsonl",
            DATA_DIR / "splits" / "test_gold.jsonl",
            DATA_DIR / "extra" / "hard_cases.jsonl",
            DATA_DIR / "review" / "test_gold_review.csv",
        ):
            files.append(
                {
                    "name": path.name,
                    "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "exists": path.is_file(),
                    "size": path.stat().st_size if path.is_file() else 0,
                }
            )
        return JSONResponse(
            status_code=200,
            content={
                "taxonomy": taxonomy.get("categories", []),
                "taxonomy_hash": taxonomy.get("hash"),
                "splits": _split_sizes(),
                "multi": _multi_counts(),
                "files": files,
            },
        )

    @app.get("/api/acceptance/errors")
    async def acceptance_errors() -> JSONResponse:
        rows = _error_rows()
        return JSONResponse(
            status_code=200,
            content={"run_id": _latest_run_id(), "count": len(rows), "rows": rows},
        )
