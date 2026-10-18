"""ch04 评估看板：报告适配 + 后台重跑。

为什么页面不直接 ``fetch('evals/reports/ch04.json')``：

- 报告是脚本产物，字段会随评估口径演进；把「取哪些字段、怎么补空值」集中在
  这里，前端只消费一份稳定的 dashboard 契约。
- 报告里的逐条记录只有 ``case_id``，没有题目原文；这里用
  ``evals/ch04_cases.jsonl`` 做一次关联，页面才能直接展示「哪道题挂在哪」。
- 重跑是一个长任务（分钟级），不能让 HTTP 请求一直挂着；这里用后台任务 +
  状态轮询，页面只负责展示进度。

报告口径的两条硬规则（沿用 ``evals/run_ch04_evals.py``）：

1. 没跑的策略是 ``None`` 而不是 0 —— 页面上必须显示 ``—``；
2. 检索指标只算 ``behavior == "answer"`` 的题，拒答准确率单独统计。
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT

#: 评估脚本的默认产物路径。页面与接口都只认这一份最新报告。
REPORT_PATH = PROJECT_ROOT / "evals" / "reports" / "ch04.json"
CASES_PATH = PROJECT_ROOT / "evals" / "ch04_cases.jsonl"
MARKDOWN_PATH = PROJECT_ROOT / "dev-notes" / "ch04-eval-report.md"
LOG_PATH = PROJECT_ROOT / "evals" / "reports" / "ch04-run.log"
SCRIPT_PATH = PROJECT_ROOT / "evals" / "run_ch04_evals.py"

STRATEGY_ORDER: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")
GENERATION_STRATEGY_ORDER: tuple[str, ...] = ("hybrid", "hybrid_rerank")
BUCKETS: tuple[str, ...] = ("A_policy", "B_model", "C_colloquial", "E_multi")
BUCKET_KEYS: tuple[str, ...] = (*BUCKETS, "overall")

STRATEGY_LABELS = {
    "vector": "纯向量",
    "bm25": "纯 BM25",
    "hybrid": "混合",
    "hybrid_rerank": "混合 + 重排",
}

BUCKET_LABELS = {
    "A_policy": "售后政策",
    "B_model": "型号参数",
    "C_colloquial": "口语问法",
    "E_multi": "跨节多跳",
    "overall": "整体",
}

NOTE_LABELS = {
    "threshold": "阈值口径",
    "generation_sampling": "生成段抽样",
    "gold": "金标口径",
}

RETRIEVAL_METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")
GENERATION_METRIC_KEYS = ("faithfulness", "refusal_accuracy", "citation_validity", "citation_coverage")


class ReportNotFound(FileNotFoundError):
    """最新报告还没生成。页面应展示「先跑一次评估」的空状态，而不是报错。"""


class ReportInvalid(ValueError):
    """报告存在但不是本看板认识的形状（JSON 坏了 / 缺 retrieval 段）。"""


class EvalAlreadyRunning(RuntimeError):
    """同一进程内只允许一个评估任务 —— 两个任务会同时写同一份报告。"""


class EvalStartError(RuntimeError):
    """评估进程没起来（解释器或脚本路径不对）。"""


# ---------------------------------------------------------------------------
# 报告读取与归一
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """只接受真正的数字。``None`` / 字符串 / 布尔值一律当归「没测」。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ordered(names: Any, preferred: tuple[str, ...]) -> list[str]:
    if not isinstance(names, dict):
        return []
    present = [name for name in preferred if name in names]
    return present + [name for name in names if name not in preferred]


def _retrieval_bucket(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    bucket: dict[str, Any] = {"n": _integer(data.get("n"))}
    for key in RETRIEVAL_METRIC_KEYS:
        bucket[key] = _number(data.get(key))
    return bucket


def _generation_bucket(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    bucket: dict[str, Any] = {"n": _integer(data.get("n"))}
    for key in GENERATION_METRIC_KEYS:
        bucket[key] = _number(data.get(key))
    return bucket


def _citation_coverage(rows: list[dict], strategy: str, bucket: str) -> tuple[float | None, int]:
    """被回答的题里，有多少条真的带出了有效引用。

    这是「引用覆盖」：答案没有引用 = 证据没覆盖到答案。拒答题不计入分母
    （它们本来就不该有引用）。
    """
    answered = [
        row
        for row in rows
        if row.get("strategy") == strategy
        and (bucket == "overall" or row.get("bucket") == bucket)
        and row.get("behavior") == "answer"
        and not row.get("refused")
    ]
    if not answered:
        return None, 0
    with_citation = 0
    for row in answered:
        citations = row.get("citations")
        valid = citations.get("valid") if isinstance(citations, dict) else None
        if valid:
            with_citation += 1
    return with_citation / len(answered), len(answered)


def _answer_coverage(rows: list[dict], strategy: str, bucket: str) -> tuple[float | None, int]:
    """可答题里真的被作答的比例。

    与 ``citation_coverage`` 的分母不同：这里把「本该作答却被拒答」也算作
    没覆盖，所以它能直接反映端到端答案覆盖率。
    """
    answerable = [
        row
        for row in rows
        if row.get("strategy") == strategy
        and (bucket == "overall" or row.get("bucket") == bucket)
        and row.get("behavior") == "answer"
    ]
    if not answerable:
        return None, 0
    answered = [row for row in answerable if not row.get("refused")]
    return len(answered) / len(answerable), len(answerable)


def _acceptance(report: dict[str, Any], strategies: list[str]) -> dict[str, Any]:
    """把脚本的退出码判定复刻成页面可展示的结论。

    与 ``run_ch04_evals.py::_exit_code`` 保持一致；这里不 import 脚本，
    避免 Web 进程因为导入评估脚本而连带导入 Milvus / 模型客户端。
    """
    retrieval = report.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}

    def recall(strategy: str, bucket: str) -> float | None:
        raw = retrieval.get(strategy)
        bucket_raw = raw.get(bucket) if isinstance(raw, dict) else None
        return _number(bucket_raw.get("recall@5")) if isinstance(bucket_raw, dict) else None

    model_bm25 = recall("bm25", "B_model")
    if model_bm25 is not None and model_bm25 < 0.8:
        return {
            "status": "fail",
            "message": f"型号桶 BM25 Recall@5 {model_bm25:.3f} < 0.800",
        }

    if "hybrid_rerank" not in strategies or "vector" not in strategies:
        return {"status": "unknown", "message": "本轮未同时运行 vector 与 hybrid_rerank"}

    overall_hybrid = recall("hybrid_rerank", "overall")
    overall_vector = recall("vector", "overall")
    if overall_hybrid is None or overall_vector is None:
        return {"status": "unknown", "message": "报告缺少整体 Recall@5"}
    if overall_hybrid < overall_vector:
        return {
            "status": "fail",
            "message": (
                f"整体 Recall@5：hybrid_rerank {overall_hybrid:.3f} < "
                f"vector {overall_vector:.3f}"
            ),
        }

    if "bm25" in strategies:
        model_hybrid = recall("hybrid_rerank", "B_model")
        model_bm25 = recall("bm25", "B_model")
        if model_hybrid is not None and model_bm25 is not None and model_hybrid < model_bm25:
            return {
                "status": "fail",
                "message": (
                    f"型号桶 Recall@5：hybrid_rerank {model_hybrid:.3f} < "
                    f"bm25 {model_bm25:.3f}"
                ),
            }

    return {"status": "pass", "message": "四策略验收条件均满足"}


def _best_strategy(
    metrics: dict[str, dict[str, dict[str, Any]]],
    key: str,
    preferred: tuple[str, ...],
) -> tuple[str | None, float | None]:
    """按 ``key`` 找整体值最高的策略；同分时偏向更简单的策略。

    评估顺序是 vector → bm25 → hybrid → hybrid_rerank，后一个是在前一个基础上
    加能力。同分时用更简单的策略更划算，所以同分应偏向更靠前的。实现上按
    preferred 顺序遍历，只有严格大于才替换。
    """
    best_name: str | None = None
    best_value: float | None = None
    candidates = list(preferred) + [name for name in metrics if name not in preferred]
    for name in candidates:
        bucket = metrics.get(name, {}).get("overall", {})
        value = bucket.get(key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name, best_value = name, value
    return best_name, best_value


def _case_index(path: Path) -> dict[str, dict[str, Any]]:
    """读评估集的题面信息（问题 / 难度）。文件缺失时降级为空索引。"""
    if not path.is_file():
        return {}
    index: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id"):
            index[str(row["id"])] = row
    return index


def _case_common(row: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    source = index.get(case_id, {})
    bucket = str(row.get("bucket") or source.get("bucket") or "")
    return {
        "case_id": case_id,
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "behavior": row.get("behavior") or source.get("behavior"),
        "query": source.get("query") or "",
        "difficulty": source.get("difficulty") or "",
    }


def _retrieval_cases(
    rows: Any, index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return cases
    for row in rows:
        if not isinstance(row, dict):
            continue
        common = _case_common(row, index)
        hits = row.get("hits") if isinstance(row.get("hits"), list) else []
        gold = row.get("gold") if isinstance(row.get("gold"), list) else []
        gold_set = set(gold)
        first_rank = next(
            (position + 1 for position, chunk in enumerate(hits) if chunk in gold_set),
            None,
        )
        common.update(
            {
                "strategy": row.get("strategy"),
                "strategy_label": STRATEGY_LABELS.get(str(row.get("strategy")), row.get("strategy")),
                "first_gold_rank": first_rank,
                "top5_hit": bool(gold_set.intersection(hits[:5])),
                "hits": hits[:10],
                "hit_count": len(hits),
                "gold": gold,
                "gold_count": len(gold),
                "rerank_degraded": bool(row.get("rerank_degraded")),
            }
        )
        cases.append(common)
    return cases


def _generation_cases(
    rows: Any, index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return cases
    for row in rows:
        if not isinstance(row, dict):
            continue
        common = _case_common(row, index)
        claims = row.get("claims") if isinstance(row.get("claims"), list) else []
        citations = row.get("citations") if isinstance(row.get("citations"), dict) else {}
        valid = citations.get("valid") if isinstance(citations.get("valid"), list) else []
        dropped = citations.get("dropped") if isinstance(citations.get("dropped"), list) else []
        common.update(
            {
                "strategy": row.get("strategy"),
                "strategy_label": STRATEGY_LABELS.get(str(row.get("strategy")), row.get("strategy")),
                "refused": bool(row.get("refused")),
                "refusal_source": row.get("refusal_source"),
                "refusal_reason": row.get("refusal_reason"),
                "answer": row.get("answer") or "",
                "faithfulness": _number(row.get("faithfulness")),
                "claims": [
                    {
                        "text": claim.get("text") or "",
                        "supported": bool(claim.get("supported")),
                        "evidence_n": claim.get("evidence_n"),
                    }
                    for claim in claims
                    if isinstance(claim, dict)
                ],
                "claims_total": len(claims),
                "claims_supported": sum(
                    1 for claim in claims if isinstance(claim, dict) and claim.get("supported")
                ),
                "citations": {"valid": valid, "dropped": dropped},
                "rerank_degraded": bool(row.get("rerank_degraded")),
            }
        )
        cases.append(common)
    return cases


def build_dashboard(
    report: dict[str, Any],
    index: dict[str, dict[str, Any]] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """把脚本报告归一成页面契约。

    页面只读这里产出的字段；报告新增字段不会影响页面，报告缺字段时页面拿到
    的是 ``None`` / 空数组，由前端统一显示 ``—`` 与空状态。
    """
    if not isinstance(report, dict):
        raise ReportInvalid("报告根节点不是 JSON 对象")
    retrieval_raw = report.get("retrieval")
    if not isinstance(retrieval_raw, dict) or not retrieval_raw:
        raise ReportInvalid("报告缺少 retrieval 段，无法生成评估看板")

    case_index = index if index is not None else _case_index(CASES_PATH)
    meta_raw = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    strategies = _ordered(retrieval_raw, STRATEGY_ORDER)
    generation_raw = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    generation_strategies = _ordered(generation_raw, GENERATION_STRATEGY_ORDER)

    retrieval: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy in strategies:
        buckets = retrieval_raw.get(strategy)
        retrieval[strategy] = {
            bucket: _retrieval_bucket(buckets.get(bucket) if isinstance(buckets, dict) else None)
            for bucket in BUCKET_KEYS
        }

    generation_cases_raw = report.get("generation_cases")
    generation_cases = _generation_cases(generation_cases_raw, case_index)

    generation: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy in generation_strategies:
        buckets = generation_raw.get(strategy)
        generation[strategy] = {}
        for bucket in BUCKET_KEYS:
            normalized = _generation_bucket(buckets.get(bucket) if isinstance(buckets, dict) else None)
            coverage, answered = _citation_coverage(
                generation_cases if isinstance(generation_cases_raw, list) else [],
                strategy,
                bucket,
            )
            normalized["citation_coverage"] = coverage
            normalized["answered"] = answered
            answer_coverage, answerable = _answer_coverage(
                generation_cases if isinstance(generation_cases_raw, list) else [],
                strategy,
                bucket,
            )
            normalized["answer_coverage"] = answer_coverage
            normalized["answerable"] = answerable
            generation[strategy][bucket] = normalized

    best_strategy, best_mrr = _best_strategy(retrieval, "mrr@10", STRATEGY_ORDER)
    best_generation, best_faithfulness = _best_strategy(
        generation, "faithfulness", GENERATION_STRATEGY_ORDER
    )
    # 生产链路固定用 hybrid_rerank；KPI 不跟着「哪轮忠实度略高」漂移，
    # 否则同一份报告里两个指标可能来自不同策略，读者会误读成同一条链路。
    final_generation = (
        "hybrid_rerank" if "hybrid_rerank" in generation else best_generation
    )

    colloquial_gain: float | None = None
    hybrid_c = retrieval.get("hybrid_rerank", {}).get("C_colloquial", {}).get("mrr@10")
    bm25_c = retrieval.get("bm25", {}).get("C_colloquial", {}).get("mrr@10")
    if hybrid_c is not None and bm25_c is not None:
        colloquial_gain = hybrid_c - bm25_c

    best_generation_overall = (
        generation.get(final_generation, {}).get("overall", {}) if final_generation else {}
    )
    refusal_rows = [
        row
        for row in generation_cases
        if row.get("strategy") == final_generation and row.get("behavior") == "refuse"
    ]
    refusal_correct = sum(1 for row in refusal_rows if row.get("refused"))
    answered_rows = [
        row
        for row in generation_cases
        if row.get("strategy") == final_generation
        and row.get("behavior") == "answer"
        and not row.get("refused")
    ]

    warnings: list[dict[str, Any]] = []
    warnings_raw = report.get("warnings")
    if isinstance(warnings_raw, dict):
        for case_id, anchors in warnings_raw.items():
            if not isinstance(anchors, dict):
                continue
            for anchor, count in anchors.items():
                warnings.append(
                    {"case_id": case_id, "anchor": anchor, "count": _integer(count) or 0}
                )
    warnings.sort(key=lambda item: (-item["count"], item["case_id"]))

    notes = report.get("notes")
    note_rows = []
    if isinstance(notes, dict):
        for key, value in notes.items():
            note_rows.append(
                {
                    "key": key,
                    "label": NOTE_LABELS.get(key, key),
                    "text": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                }
            )

    relative_report = "evals/reports/ch04.json"
    if report_path is not None:
        try:
            relative_report = str(report_path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            relative_report = report_path.name

    return {
        "meta": {
            "generated_at": meta_raw.get("generated_at") or "",
            "cases": _integer(meta_raw.get("cases")),
            "milvus_rows": _integer(meta_raw.get("milvus_rows")),
            "strategies": strategies,
            "generation_strategies": generation_strategies,
            "rerank_model": meta_raw.get("rerank_model") or "",
            "min_rerank_score": _number(meta_raw.get("min_rerank_score")),
            "report_path": relative_report,
        },
        "labels": {
            "strategies": STRATEGY_LABELS,
            "buckets": BUCKET_LABELS,
            "buckets_order": list(BUCKET_KEYS),
        },
        "summary": {
            "best_strategy": best_strategy,
            "best_strategy_label": STRATEGY_LABELS.get(best_strategy or "", best_strategy),
            "best_mrr": best_mrr,
            "colloquial_gain": colloquial_gain,
            "best_generation_strategy": best_generation,
            "best_faithfulness": best_faithfulness,
            "final_generation_strategy": final_generation,
            "final_generation_label": STRATEGY_LABELS.get(
                final_generation or "", final_generation
            ),
            "citation_validity": best_generation_overall.get("citation_validity"),
            "citation_coverage": best_generation_overall.get("citation_coverage"),
            "answer_coverage": best_generation_overall.get("answer_coverage"),
            "answered_total": len(answered_rows),
            "answerable_total": best_generation_overall.get("answerable") or 0,
            "refusal_accuracy": best_generation_overall.get("refusal_accuracy"),
            "refusal_correct": refusal_correct,
            "refusal_total": len(refusal_rows),
            "acceptance": _acceptance(report, strategies),
        },
        "retrieval": retrieval,
        "generation": generation,
        "warnings": warnings,
        "notes": note_rows,
        "cases": {
            "retrieval": _retrieval_cases(report.get("cases"), case_index),
            "generation": generation_cases,
        },
        "rerank_degraded": report.get("rerank_degraded")
        if isinstance(report.get("rerank_degraded"), dict)
        else {},
    }


def _display_path(path: Path) -> str:
    """给客户端看的路径：工作区内用相对路径，避免把本机绝对路径带出去。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def load_report(path: Path | None = None) -> dict[str, Any]:
    target = path or REPORT_PATH
    if not target.is_file():
        raise ReportNotFound(f"还没有评估报告：{_display_path(target)}")
    try:
        report = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReportInvalid(f"报告读取失败：{exc}") from exc
    except ValueError as exc:
        raise ReportInvalid(f"报告不是合法 JSON：{exc}") from exc
    if not isinstance(report, dict):
        raise ReportInvalid("报告根节点不是 JSON 对象")
    return report


def load_dashboard() -> dict[str, Any]:
    return build_dashboard(load_report(), _case_index(CASES_PATH), REPORT_PATH)


# ---------------------------------------------------------------------------
# 后台重跑
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _report_generated_at(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = report.get("meta", {}).get("generated_at") if isinstance(report, dict) else None
    return value if isinstance(value, str) else None


class EvalRunner:
    """同一进程内最多跑一个评估任务。

    任务用**子进程**而不是 ``asyncio`` 直接调脚本：评估脚本会创建自己的事件循环
    （``asyncio.run``），在 Web 进程里再跑一遍会撞 loop；子进程还能让崩溃
    不拖垮 API 进程。
    """

    def __init__(
        self,
        *,
        script_path: Path | None = None,
        report_path: Path | None = None,
        markdown_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._script = script_path or SCRIPT_PATH
        self._report = report_path or REPORT_PATH
        self._markdown = markdown_path or MARKDOWN_PATH
        self._log = log_path or LOG_PATH
        self._task: asyncio.Task[None] | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "message": "尚未在页面上重跑过",
        }
        self._lock = asyncio.Lock()

    def status(self) -> dict[str, Any]:
        state = dict(self._state)
        state["report_generated_at"] = _report_generated_at(self._report)
        state["running"] = self._task is not None and not self._task.done()
        if state["running"]:
            state["status"] = "running"
        state["log_tail"] = self._read_log_tail()
        return state

    async def aclose(self) -> None:
        """应用退出时收掉后台任务，别让评估子进程变成孤儿。"""
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _read_log_tail(self, limit: int = 4000) -> str:
        if not self._log.is_file():
            return ""
        try:
            return self._log.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise EvalAlreadyRunning("已有一个评估任务在跑，等它结束再点。")
            if not self._script.is_file():
                raise EvalStartError(f"找不到评估脚本：{_display_path(self._script)}")
            self._state = {
                "status": "running",
                "started_at": _now(),
                "finished_at": None,
                "exit_code": None,
                "message": "评估已启动，检索与生成两段跑完通常需要几分钟",
            }
            try:
                self._task = asyncio.create_task(self._run())
            except Exception as exc:  # noqa: BLE001 - 启动失败要回到可重试状态
                self._state.update(
                    status="failed", finished_at=_now(), message=f"评估启动失败：{exc}"
                )
                raise EvalStartError(str(exc)) from exc
            return self.status()

    async def _run(self) -> None:
        command = [
            sys.executable,
            "-u",
            str(self._script),
            "--strategies",
            "vector,bm25,hybrid,hybrid_rerank",
            "--generation",
            "hybrid,hybrid_rerank",
            "--limit",
            "4",
            "--json",
            str(self._report),
            "--markdown",
            str(self._markdown),
        ]
        self._log.parent.mkdir(parents=True, exist_ok=True)
        process: asyncio.subprocess.Process | None = None
        try:
            with self._log.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n===== {_now()} 重跑评估 =====\n"
                    f"$ {' '.join(command)}\n"
                )
                log_file.flush()
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
                return_code = await process.wait()
            if return_code == 0:
                message = "评估完成，四策略验收通过"
                status = "succeeded"
            else:
                message = f"评估完成，但验收未通过（退出码 {return_code}）；报告已更新，可查看明细"
                status = "failed"
            self._state.update(
                status=status,
                finished_at=_now(),
                exit_code=return_code,
                message=message,
            )
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
            self._state.update(
                status="cancelled",
                finished_at=_now(),
                message="评估任务被中止",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务不能把异常抛给事件循环
            self._state.update(
                status="failed",
                finished_at=_now(),
                message=f"评估进程启动失败：{exc}",
            )
            try:
                with self._log.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n评估进程启动失败：{exc}\n")
            except OSError:
                pass
