"""意图分类评估 —— 纯 Prompt 任务的「TDD 替代」。

意图识别不是可单测的业务逻辑（同一句话，模型可能给出不同措辞），所以按用户要求
用**标注样例**验证：先写标签，再跑真实模型；不达标先改 Prompt/示例，不改标签。

## 指标

- 整体 accuracy：七类逐条对；
- route accuracy：由预测意图按 `INTENT_TO_ROUTE` 映射后，是否落到期望出口；
- 每类 recall：某一类里答对的比例；
- 混淆矩阵：谁被误判成谁；
- source 计数：`llm` / `heuristic` / `rule`，用来看模型是不是大面积降级。

退出码：0 达标；1 未达标；2 配置/上游故障（或 >50% 走启发式，说明模型没真正跑起来）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.intent import IntentClassifier  # noqa: E402
from app.agent.state import ALL_INTENTS, route_for  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "ch05_intent_cases.jsonl"
OVERALL_THRESHOLD = 0.85
ROUTE_THRESHOLD = 0.85
PER_CLASS_THRESHOLD = 0.60


@dataclass(frozen=True)
class Case:
    id: str
    message: str
    expected_intent: str
    expected_route: str
    bucket: str = ""
    note: str = ""


@dataclass(frozen=True)
class Prediction:
    case: Case
    predicted_intent: str
    source: str
    raw: str
    latency_ms: float


def load_cases(path: str | Path = DEFAULT_CASES) -> list[Case]:
    cases: list[Case] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
            try:
                case = Case(
                    id=str(payload["id"]),
                    message=str(payload["message"]),
                    expected_intent=str(payload["expected_intent"]),
                    expected_route=str(payload["expected_route"]),
                    bucket=str(payload.get("bucket") or ""),
                    note=str(payload.get("note") or ""),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} 缺少字段 {exc}") from exc
            if case.expected_intent not in ALL_INTENTS:
                raise ValueError(f"{path}:{line_no} 非法 intent {case.expected_intent!r}")
            if case.expected_route != route_for(case.expected_intent):
                raise ValueError(
                    f"{path}:{line_no} expected_route 与写死的分流表不一致："
                    f"{case.expected_route!r} != {route_for(case.expected_intent)!r}"
                )
            cases.append(case)
    if not cases:
        raise ValueError(f"{path} 里没有用例")
    return cases


def aggregate(predictions: Sequence[Prediction]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for item in predictions if item.predicted_intent == item.case.expected_intent)
    route_correct = sum(
        1 for item in predictions if route_for(item.predicted_intent) == item.case.expected_route
    )

    per_intent_recall: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for intent in ALL_INTENTS:
        expected = [item for item in predictions if item.case.expected_intent == intent]
        if expected:
            hit = sum(1 for item in expected if item.predicted_intent == intent)
            per_intent_recall[intent] = hit / len(expected)
        else:
            per_intent_recall[intent] = 1.0

    for item in predictions:
        row = confusion.setdefault(item.case.expected_intent, {})
        row[item.predicted_intent] = row.get(item.predicted_intent, 0) + 1
        if item.predicted_intent != item.case.expected_intent:
            failures.append(
                {
                    "id": item.case.id,
                    "message": item.case.message,
                    "expected": item.case.expected_intent,
                    "predicted": item.predicted_intent,
                    "source": item.source,
                    "raw": item.raw,
                }
            )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "route_accuracy": route_correct / total if total else 0.0,
        "per_intent_recall": per_intent_recall,
        "confusion": confusion,
        "sources": dict(Counter(item.source for item in predictions)),
        "avg_latency_ms": (
            sum(item.latency_ms for item in predictions) / total if total else 0.0
        ),
        "failures": failures,
    }


def exit_code(report: dict[str, Any]) -> int:
    if report.get("accuracy", 0.0) < OVERALL_THRESHOLD:
        return 1
    if report.get("route_accuracy", 0.0) < ROUTE_THRESHOLD:
        return 1
    for recall in (report.get("per_intent_recall") or {}).values():
        if recall is not None and recall < PER_CLASS_THRESHOLD:
            return 1
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ch05 意图分类评估",
        "",
        f"- 题数：**{report['total']}**",
        f"- 整体 accuracy：**{report['accuracy']:.3f}**（阈值 {OVERALL_THRESHOLD}）",
        f"- 出口 route accuracy：**{report['route_accuracy']:.3f}**（阈值 {ROUTE_THRESHOLD}）",
        f"- 平均延迟：{report['avg_latency_ms']:.0f} ms",
        f"- 来源计数：`{report['sources']}`",
        "",
        "> 口径说明：`INTENT_PROMPT` 的边界示例与部分标注用例（L01/O01/C01/C03/R01/H01）"
        "高度重叠，56/56 里含一定的提示记忆；正式版应补 held-out 集。",
        "",
        "## 每类 recall",
        "",
        "| 意图 | recall |",
        "|---|---|",
    ]
    for intent, recall in report["per_intent_recall"].items():
        lines.append(f"| `{intent}` | {recall:.3f} |")

    lines += ["", "## 混淆矩阵", "", "| 期望 \\ 预测 | " + " | ".join(ALL_INTENTS) + " |"]
    lines.append("|---" * (len(ALL_INTENTS) + 1) + "|")
    for expected in ALL_INTENTS:
        row = report["confusion"].get(expected, {})
        lines.append(
            f"| `{expected}` | "
            + " | ".join(str(row.get(predicted, 0)) for predicted in ALL_INTENTS)
            + " |"
        )

    lines += ["", "## 失败明细", ""]
    if not report["failures"]:
        lines.append("无。")
    else:
        for failure in report["failures"]:
            lines.append(
                f"- `{failure['id']}` {failure['message']}：期望 `{failure['expected']}`，"
                f"预测 `{failure['predicted']}`（{failure['source']}）"
            )
    return "\n".join(lines) + "\n"


async def _run_real(cases: Sequence[Case]) -> tuple[list[Prediction], int]:
    from app.config import get_settings
    from app.llm import get_chat_model

    settings = get_settings()
    model = get_chat_model(settings)
    classifier = IntentClassifier(model, timeout=10.0)

    predictions: list[Prediction] = []
    for case in cases:
        started = asyncio.get_running_loop().time()
        result = await classifier.classify(case.message)
        latency_ms = (asyncio.get_running_loop().time() - started) * 1000
        predictions.append(
            Prediction(
                case=case,
                predicted_intent=result.intent,
                source=result.source,
                raw=result.raw,
                latency_ms=latency_ms,
            )
        )
    return predictions, 0


def _self_test_predictions(cases: Sequence[Case]) -> list[Prediction]:
    return [
        Prediction(
            case=case,
            predicted_intent=case.expected_intent,
            source="rule" if case.expected_intent == "chitchat" else "llm",
            raw="self-test",
            latency_ms=0.0,
        )
        for case in cases
    ]


def _write_outputs(report: dict[str, Any], *, json_path: str | None, markdown_path: str | None) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")


async def _main(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print(f"评估集不可用：{exc}", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]

    if args.self_test:
        predictions = _self_test_predictions(cases)
    else:
        try:
            predictions, _ = await _run_real(cases)
        except Exception as exc:  # noqa: BLE001 - 配置/上游故障用退出码 2 表达
            print(f"真实模型不可用：{exc}", file=sys.stderr)
            return 2

    report = aggregate(predictions)
    heuristic_ratio = report["sources"].get("heuristic", 0) / max(1, report["total"])
    if not args.self_test and heuristic_ratio > 0.5:
        print(
            f"超过一半（{heuristic_ratio:.0%}）的样本走了启发式兜底，"
            "说明意图模型没有真正跑起来；先修配置/上游，不要改 Prompt。",
            file=sys.stderr,
        )
        return 2

    print(render_markdown(report))
    _write_outputs(report, json_path=args.json, markdown_path=args.markdown)
    return exit_code(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="ch05 意图分类评估")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--self-test", action="store_true", help="不联网，只验证聚合与退出码")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
