"""评估流水线：跑 ch04 评估集、写 ``eval_runs``、算趋势。

指标口径跟 ch04 评估脚本一致，只取线上策略 ``hybrid_rerank`` 的 overall：
检索段 recall@K / MRR，生成段 faithfulness / refusal_accuracy / citation_*。
一行一轮，只追加；趋势由最近两轮相减得到。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from app.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

TRIGGERS = ("定时", "手动")
REPORT_PATH = PROJECT_ROOT / "evals" / "reports" / "ch09-latest.json"
MARKDOWN_PATH = PROJECT_ROOT / "evals" / "reports" / "ch09-latest.md"
LOG_PATH = PROJECT_ROOT / "evals" / "reports" / "ch09-run.log"
RETRIEVAL_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")
GENERATION_KEYS = (
    "faithfulness",
    "refusal_accuracy",
    "citation_validity",
    "citation_coverage",
)
TREND_THRESHOLD = 1e-9
TIMEOUT_SECONDS = 1800.0


def _overall(report: dict, section: str) -> dict:
    block = report.get(section) or {}
    strategy = block.get("hybrid_rerank") or {}
    overall = strategy.get("overall") or {}
    return overall if isinstance(overall, dict) else {}


def extract_metrics(report: dict) -> dict[str, dict[str, float]]:
    """从 ch04 报告里抽线上策略的 overall 指标（缺失记空，不补 0）。"""
    retrieval_raw = _overall(report, "retrieval")
    generation_raw = _overall(report, "generation")
    return {
        "retrieval": {
            key: float(retrieval_raw[key])
            for key in RETRIEVAL_KEYS
            if isinstance(retrieval_raw.get(key), (int, float))
        },
        "generation": {
            key: float(generation_raw[key])
            for key in GENERATION_KEYS
            if isinstance(generation_raw.get(key), (int, float))
        },
    }


def _dataset_size(report: dict) -> int:
    meta = report.get("meta") or {}
    if isinstance(meta.get("cases"), int):
        return int(meta["cases"])
    cases = report.get("cases")
    return len(cases) if isinstance(cases, list) else 0


async def record_report(repositories, report: dict, *, triggered_by: str) -> int:
    """把一轮报告写进 ``eval_runs``，返回行 id。"""
    if triggered_by not in TRIGGERS:
        raise ValueError(f"triggered_by 必须是 {TRIGGERS} 之一")
    return await repositories.eval_runs.add(
        triggered_by, _dataset_size(report), extract_metrics(report)
    )


def build_script_args(settings: Settings) -> list[str]:
    """拼评估脚本参数。

    ``eval_generation_limit > 0`` 时只跑每桶这么多个生成样本（检索仍跑全量），
    用于在生成模型偶发卡住时把一轮控制在可接受的时间里；报告里会写明。
    """
    args = ["--generation", "hybrid_rerank"]
    if settings.eval_generation_limit > 0:
        args.extend(["--limit", str(settings.eval_generation_limit)])
    return args


async def _run_report(settings: Settings) -> dict:
    """跑评估脚本（子进程），返回它写出的 JSON 报告。"""
    script = PROJECT_ROOT / settings.eval_script_path
    if not script.is_file():
        raise RuntimeError(f"评估脚本不存在：{script}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 不接 PIPE：Windows 沙箱里给子进程建管道会 WinError 5；写日志文件即可，
    # 失败时再从文件尾部取上下文。生产环境同样受益（长跑的输出不会撑爆内存）。
    with LOG_PATH.open("ab") as log:
        log.write("\n=== ch09 eval start ===\n".encode("utf-8"))
        log.flush()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            *build_script_args(settings),
            "--json",
            str(REPORT_PATH),
            "--markdown",
            str(MARKDOWN_PATH),
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError(f"评估超时（>{TIMEOUT_SECONDS:.0f}s）") from exc

    if process.returncode != 0:
        tail = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-800:].strip()
        raise RuntimeError(f"评估脚本退出码 {process.returncode}：{tail}")
    if not REPORT_PATH.is_file():
        raise RuntimeError(f"评估脚本没有写出报告：{REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


async def run_evaluation(*, repositories, trigger: str, settings: Settings) -> int:
    """跑一轮评估并落库，返回 ``eval_runs`` 行 id。"""
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger 必须是 {TRIGGERS} 之一")
    logger.info("ch09 评估开始（trigger=%s）", trigger)
    report = await _run_report(settings)
    run_id = await record_report(repositories, report, triggered_by=trigger)
    logger.info("ch09 评估完成（run_id=%s）", run_id)
    return run_id


def _flatten(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section in ("retrieval", "generation"):
        block = metrics.get(section) or {}
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, (int, float)):
                flat[str(key)] = float(value)
    return flat


def summarize_trend(rows: Sequence) -> dict[str, dict[str, Any]]:
    """最近两轮对比：每个指标给 latest / previous / delta / direction。

    入参顺序不限（仓库返回新→旧），内部按时间升序；只有一轮时
    ``direction`` / ``previous`` 为 ``None``，前端据此不画箭头。
    """
    ordered = sorted(rows, key=lambda row: (row.created_at, row.id))
    if not ordered:
        return {}

    latest = _flatten(ordered[-1].metrics)
    previous = _flatten(ordered[-2].metrics) if len(ordered) > 1 else {}

    trend: dict[str, dict[str, Any]] = {}
    for key, value in latest.items():
        prior = previous.get(key)
        if prior is None:
            trend[key] = {
                "latest": value,
                "previous": None,
                "delta": None,
                "direction": None,
            }
            continue
        delta = value - prior
        if delta > TREND_THRESHOLD:
            direction = "up"
        elif delta < -TREND_THRESHOLD:
            direction = "down"
        else:
            direction = "flat"
        trend[key] = {
            "latest": value,
            "previous": prior,
            "delta": delta,
            "direction": direction,
        }
    return trend
