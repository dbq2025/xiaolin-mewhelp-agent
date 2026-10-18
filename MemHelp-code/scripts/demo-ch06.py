"""ch06 功能演示：多轮指代/意图、other 落点、退款子流程、订单选择器。

用法（先起 uvicorn 8000；refund/picker 需要真实 MySQL/Milvus）：

    uv run python scripts/demo-ch06.py --case all
    uv run python scripts/demo-ch06.py --case multiturn
    uv run python scripts/demo-ch06.py --case other
    uv run python scripts/demo-ch06.py --case refund --base-url http://127.0.0.1:8000
    uv run python scripts/demo-ch06.py --case picker --base-url http://127.0.0.1:8000

退出码 0 = 当前 case 全部 PASS；1 = 有 FAIL 或依赖不可用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.coref import CorefResolver
from app.agent.intent import IntentClassifier
from app.config import PROJECT_ROOT, get_settings
from app.llm import get_chat_model

CASES = {"all", "multiturn", "other", "refund", "picker"}


def parse_case(value: str) -> str:
    if value not in CASES:
        raise ValueError(f"未知 case {value!r}；可选 {sorted(CASES)}")
    return value


def _parse_sse(text: str) -> list[tuple[str | None, dict[str, Any] | None]]:
    events: list[tuple[str | None, dict[str, Any] | None]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name, json.loads(data) if data else None))
    return events


async def _post_sse(
    client: httpx.AsyncClient, path: str, body: dict[str, Any]
) -> list[tuple[str | None, dict[str, Any] | None]]:
    async with client.stream("POST", path, json=body) as response:
        text = "".join([chunk async for chunk in response.aiter_text()])
        response.raise_for_status()
    return _parse_sse(text)


def _say(lines: list[str], text: str) -> None:
    lines.append(text)
    print(text)


async def _multiturn(lines: list[str]) -> bool:
    settings = get_settings()
    model = get_chat_model(settings)
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
    )
    turns = [
        ("订单 20240115001 的物流到哪了", "logistics"),
        ("那它能退吗", "refund_return"),
        ("那现在到哪了", "logistics"),
    ]
    history: list[Any] = []
    ok = True
    _say(lines, "[multiturn] 物流 → 退款 → 物流")
    for user, expected in turns:
        coref = await resolver.resolve(user, history)
        intent = await classifier.classify(coref.resolved)
        passed = intent.intent == expected and (
            "20240115001" in coref.resolved or user == "订单 20240115001 的物流到哪了"
        )
        ok = ok and passed
        _say(
            lines,
            f"  {'PASS' if passed else 'FAIL'} | {user} -> {coref.resolved} | "
            f"intent={intent.intent} ({intent.confidence:.2f}) expected={expected}",
        )
        history.extend(
            [HumanMessage(content=user), AIMessage(content="（上一轮客服回答）")]
        )
    return ok


async def _other(lines: list[str]) -> bool:
    settings = get_settings()
    model = get_chat_model(settings)
    classifier = IntentClassifier(model, timeout=settings.intent_timeout)
    cases = [
        ("你们公司股票能买吗", "other"),
        ("帮我看看那个", "other"),
        ("这个怎么弄", "other"),
        ("今天天气怎么样", "chitchat"),
    ]
    ok = True
    _say(lines, "[other] 拿不准/怪问题落点")
    for user, expected in cases:
        intent = await classifier.classify(user)
        passed = intent.intent == expected and intent.source != "heuristic"
        ok = ok and passed
        _say(
            lines,
            f"  {'PASS' if passed else 'FAIL'} | {user} -> {intent.intent} "
            f"({intent.confidence:.2f}, {intent.source}) expected={expected}",
        )
    return ok


async def _refund(client: httpx.AsyncClient, lines: list[str]) -> bool:
    session = "demo-ch06-refund"
    _say(lines, "[refund] 带订单号问「这个能退吗」")
    first = await _post_sse(
        client,
        "/api/chat/completions",
        {"session_id": session, "message": "订单 20240115001 的物流到哪了"},
    )
    second = await _post_sse(
        client,
        "/api/chat/completions",
        {"session_id": session, "message": "这个能退吗"},
    )
    names = [name for name, _ in second]
    coref = next((data for name, data in second if name == "coref"), {}) or {}
    done = next((data for name, data in second if name == "done"), {}) or {}
    trace = done.get("trace") or []
    ok = (
        "done" in names
        and bool(coref.get("changed"))
        and "20240115001" in str(coref.get("resolved") or "")
        and "refund_agent" in trace
        and bool(done.get("refund_form"))
    )
    _say(lines, f"  first_done={'done' in [n for n, _ in first]}")
    _say(lines, f"  coref={coref.get('resolved')} changed={coref.get('changed')}")
    _say(lines, f"  trace={' -> '.join(trace)}")
    _say(
        lines,
        f"  refund_form_order={(done.get('refund_form') or {}).get('order_id')}",
    )
    _say(lines, f"  {'PASS' if ok else 'FAIL'} | 退款子流程完成")
    return ok


async def _picker(client: httpx.AsyncClient, lines: list[str]) -> bool:
    session = "demo-ch06-picker"
    _say(lines, "[picker] 不带订单号问「这个能退吗」")
    first = await _post_sse(
        client,
        "/api/chat/completions",
        {"session_id": session, "message": "这个能退吗"},
    )
    picker = next((data for name, data in first if name == "order_picker"), None)
    awaiting = next((data for name, data in first if name == "awaiting_input"), None)
    if not picker or not awaiting:
        _say(lines, "  FAIL | 没有收到 order_picker / awaiting_input")
        return False
    orders = picker.get("orders") or []
    _say(lines, f"  order_picker={len(orders)} 张，awaiting_input=yes")
    chosen = orders[0]["order_id"]
    resumed = await _post_sse(
        client,
        "/api/chat/refund/select",
        {"session_id": session, "order_id": chosen},
    )
    done = next((data for name, data in resumed if name == "done"), {}) or {}
    form = done.get("refund_form") or {}
    ok = form.get("order_id") == chosen and "refund_agent" in (done.get("trace") or [])
    _say(
        lines,
        f"  selected={chosen} -> refund_form_order={form.get('order_id')} "
        f"trace_tail={' -> '.join((done.get('trace') or [])[-3:])}",
    )
    _say(lines, f"  {'PASS' if ok else 'FAIL'} | 选单后续跑完成")
    return ok


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=sorted(CASES))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    parse_case(args.case)

    lines: list[str] = []
    results: dict[str, bool] = {}
    if args.case in ("all", "multiturn"):
        results["multiturn"] = await _multiturn(lines)
    if args.case in ("all", "other"):
        results["other"] = await _other(lines)
    if args.case in ("all", "refund", "picker"):
        async with httpx.AsyncClient(base_url=args.base_url, timeout=180.0) as client:
            if args.case in ("all", "refund"):
                results["refund"] = await _refund(client, lines)
            if args.case in ("all", "picker"):
                results["picker"] = await _picker(client, lines)

    _say(lines, "")
    _say(lines, "SUMMARY " + " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written: {path}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
