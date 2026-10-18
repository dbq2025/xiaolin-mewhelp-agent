"""ch06 多轮评估：指代消解、八类意图、退款/售后 Query 扩写。

不连 MySQL/Milvus；只调真实聊天模型。输出：
- ``dev-notes/ch06-evals.json``：逐轮明细与指标；
- ``dev-notes/ch06-eval-report.md``：人读报告。

退出码：达标 0，不达标 1。阈值故意写死在本文件，改阈值必须同时改报告口径。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.coref import CorefResolver
from app.agent.intent import IntentClassifier
from app.agent.query_expansion import QueryExpander
from app.config import PROJECT_ROOT, get_settings
from app.llm import get_chat_model

CASES_FILE = PROJECT_ROOT / "evals" / "ch06_multiturn_cases.jsonl"
REPORT_JSON = PROJECT_ROOT / "dev-notes" / "ch06-evals.json"
REPORT_MD = PROJECT_ROOT / "dev-notes" / "ch06-eval-report.md"
REFUND_INTENTS = {"refund_return", "after_sales"}

INTENT_THRESHOLD = 0.85
COREF_THRESHOLD = 0.85
OTHER_THRESHOLD = 0.80
EXPANSION_THRESHOLD = 0.85


def validate_case(case: dict[str, Any]) -> None:
    """一条多轮用例的结构校验；坏数据在加载时炸，不要带病跑评估。"""
    if not isinstance(case, dict) or not case.get("id"):
        raise ValueError("用例必须有 id")
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{case.get('id')}: turns 不能为空")
    for turn in turns:
        if not isinstance(turn, dict) or not turn.get("user"):
            raise ValueError(f"{case.get('id')}: 每轮必须有 user")
        intent = turn.get("intent")
        if intent not in {
            "logistics", "order", "product_consult", "refund_return",
            "after_sales", "complaint", "chitchat", "other",
        }:
            raise ValueError(f"{case.get('id')}: 非法 intent {intent!r}")
        for key in ("coref_contains", "coref_forbidden"):
            value = turn.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{case.get('id')}: {key} 必须是数组")


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or CASES_FILE
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_no} 不是合法 JSON") from exc
        validate_case(case)
        cases.append(case)
    return cases


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    intent_correct = sum(1 for row in rows if row.get("intent_ok"))
    coref_correct = sum(1 for row in rows if row.get("coref_ok"))
    json_failures = sum(1 for row in rows if not row.get("json_ok", True))
    other_rows = [row for row in rows if row.get("expected_intent") == "other"]
    other_correct = sum(1 for row in other_rows if row.get("intent_ok"))
    expansion_rows = [row for row in rows if row.get("expansion_checked")]
    expansion_failures = sum(1 for row in expansion_rows if not row.get("expansion_ok"))
    failures = [
        row
        for row in rows
        if not row.get("intent_ok")
        or not row.get("coref_ok")
        or not row.get("json_ok", True)
        or (row.get("expansion_checked") and not row.get("expansion_ok"))
    ]
    return {
        "total": total,
        "intent_correct": intent_correct,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "coref_accuracy": coref_correct / total if total else 0.0,
        "other_total": len(other_rows),
        "other_correct": other_correct,
        "other_accuracy": other_correct / len(other_rows) if other_rows else 0.0,
        "json_failures": json_failures,
        "expansion_checked": len(expansion_rows),
        "expansion_failures": expansion_failures,
        "expansion_accuracy": (
            (len(expansion_rows) - expansion_failures) / len(expansion_rows)
            if expansion_rows
            else 0.0
        ),
        "failures": failures,
    }


def _coref_ok(
    resolved: str, contains: Sequence[str], forbidden: Sequence[str]
) -> bool:
    if any(token and token not in resolved for token in contains):
        return False
    if any(token and token in resolved for token in forbidden):
        return False
    return True


def _expansion_ok(original: str, queries: Sequence[str], degraded: bool) -> bool:
    if degraded:
        return False
    if not 2 <= len(queries) <= 4:
        return False
    return any(original in query or query in original for query in queries)


async def run(cases: Sequence[dict[str, Any]], model: Any, settings: Any) -> list[dict[str, Any]]:
    resolver = CorefResolver(
        model,
        timeout=settings.coref_timeout,
        history_limit=settings.coref_history_messages,
        enabled=settings.coref_enabled,
    )
    classifier = IntentClassifier(
        model,
        timeout=settings.intent_timeout,
        confidence_threshold=settings.intent_confidence_threshold,
        cost_mode=False,
    )
    expander = QueryExpander(
        model,
        timeout=settings.query_expansion_timeout,
        max_queries=settings.query_expansion_max_queries,
        enabled=settings.query_expansion_enabled,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        history: list[Any] = []
        for index, turn in enumerate(case["turns"], 1):
            user = str(turn["user"])
            coref = await resolver.resolve(user, history)
            intent = await classifier.classify(coref.resolved)
            contains = turn.get("coref_contains") or []
            forbidden = turn.get("coref_forbidden") or []
            row: dict[str, Any] = {
                "id": case["id"],
                "turn": index,
                "user": user,
                "resolved": coref.resolved,
                "coref_source": coref.source,
                "coref_ok": _coref_ok(coref.resolved, contains, forbidden),
                "expected_intent": turn["intent"],
                "actual_intent": intent.intent,
                "intent_confidence": intent.confidence,
                "intent_source": intent.source,
                "intent_ok": intent.intent == turn["intent"],
                "json_ok": intent.source != "heuristic",
                "expansion_checked": False,
                "expansion_ok": True,
            }
            if intent.intent in REFUND_INTENTS:
                expansion = await expander.expand(
                    coref.resolved, order={"order_id": "20240115001"}
                )
                row["expansion_checked"] = True
                row["queries"] = list(expansion.queries)
                row["expansion_degraded"] = expansion.degraded
                row["expansion_ok"] = _expansion_ok(
                    coref.resolved, expansion.queries, expansion.degraded
                )
                row["json_ok"] = row["json_ok"] and not expansion.degraded
            rows.append(row)
            history.extend(
                [HumanMessage(content=user), AIMessage(content="（上一轮客服回答）")]
            )
    return rows


def write_report(report: dict[str, Any], rows: Sequence[dict[str, Any]], model_name: str) -> None:
    REPORT_JSON.write_text(
        json.dumps({"model": model_name, "report": report, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ch06 多轮指代/意图/扩写评估",
        "",
        f"- 模型：`{model_name}`",
        f"- 轮数：**{report['total']}**",
        f"- 意图 accuracy：**{report['intent_accuracy']:.3f}**（阈值 {INTENT_THRESHOLD}）",
        f"- 指代通过率：**{report['coref_accuracy']:.3f}**（阈值 {COREF_THRESHOLD}）",
        f"- other 落点：**{report['other_correct']}/{report['other_total']} = "
        f"{report['other_accuracy']:.3f}**（阈值 {OTHER_THRESHOLD}）",
        f"- JSON 解析失败：**{report['json_failures']}**（要求 0）",
        f"- 扩写通过：**{report['expansion_checked'] - report['expansion_failures']}"
        f"/{report['expansion_checked']}**（阈值 {EXPANSION_THRESHOLD}）",
        "",
        "## 逐轮明细",
        "",
        "| 用例 | 轮 | 用户 | 补全后 | 期望 | 实际 | 置信度 | 指代 | 意图 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['turn']} | {row['user']} | {row['resolved']} | "
            f"{row['expected_intent']} | {row['actual_intent']} | "
            f"{row['intent_confidence']:.2f} | "
            f"{'PASS' if row['coref_ok'] else 'FAIL'} | "
            f"{'PASS' if row['intent_ok'] else 'FAIL'} |"
        )

    lines.extend(["", "## 失败明细", ""])
    if report["failures"]:
        for row in report["failures"]:
            lines.append(
                f"- {row['id']} 第 {row['turn']} 轮：`{row['user']}` → "
                f"`{row['resolved']}` / 期望 {row['expected_intent']}，"
                f"实际 {row['actual_intent']}（{row['intent_confidence']:.2f}）；"
                f"coref={'PASS' if row['coref_ok'] else 'FAIL'}，"
                f"expansion={'PASS' if row['expansion_ok'] else 'FAIL'}"
            )
    else:
        lines.append("无。")

    lines.extend([
        "",
        "> 口径：本评估只测 coref/意图/扩写，不连 Milvus；订单上下文固定使用 "
        "`20240115001`。少数用例与 Prompt few-shot 语义相近，不能当作完全 held-out。",
        "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def passed(report: dict[str, Any]) -> bool:
    return (
        report["intent_accuracy"] >= INTENT_THRESHOLD
        and report["coref_accuracy"] >= COREF_THRESHOLD
        and report["other_accuracy"] >= OTHER_THRESHOLD
        and report["json_failures"] == 0
        and report["expansion_accuracy"] >= EXPANSION_THRESHOLD
    )


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    model = get_chat_model(settings)
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    rows = await run(cases, model, settings)
    report = summarise(rows)
    write_report(report, rows, settings.intent_model or settings.openai_model)

    print(json.dumps({key: report[key] for key in (
        "total", "intent_accuracy", "coref_accuracy", "other_accuracy",
        "json_failures", "expansion_accuracy", "expansion_failures",
    )}, ensure_ascii=False, indent=2))
    if report["failures"]:
        print("failures:")
        for row in report["failures"]:
            print(
                f"- {row['id']}#{row['turn']} {row['user']!r} -> "
                f"{row['resolved']!r} | expected={row['expected_intent']} "
                f"actual={row['actual_intent']} conf={row['intent_confidence']:.2f} "
                f"coref={row['coref_ok']} expansion={row['expansion_ok']}"
            )
    return 0 if passed(report) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
