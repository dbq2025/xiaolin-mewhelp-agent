"""ch09 飞轮端到端演示：从「答不上」到「审核通过后答得上」。

前置：MySQL 已应用 ``sql/ch09-ddl.sql``；服务已启动（默认 127.0.0.1:8000）；
``.env`` 里配好模型与（可选）Langfuse。

    uv run python scripts/demo_ch09.py --base-url http://127.0.0.1:8000

脚本按验收顺序推进，任一步不符合预期就以非 0 退出。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Iterator

import httpx

from app.prompts import REFUSAL_TEXT

UNKNOWN_QUESTION = "收纳箱 F98 能承受多少公斤？"
APPROVED_ANSWER = (
    "收纳箱 F98 的静态承重为 25 公斤，建议单箱装载不超过 20 公斤，"
    "堆叠时最多叠 3 层，重物放下层。"
)


class DemoError(RuntimeError):
    """演示不满足验收预期。"""


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _frames(text: str) -> Iterator[tuple[str | None, dict]]:
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        yield name, (json.loads(data) if data else {})


def ask(client: httpx.Client, session_id: str, message: str) -> dict[str, Any]:
    """发一条聊天请求，返回 ``done`` 帧，并把流式 token 拼成 ``answer``。"""
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": session_id, "message": message},
    ) as response:
        body = "".join(response.iter_text())
        if response.status_code != 200:
            raise DemoError(f"聊天接口 HTTP {response.status_code}：{body[:300]}")

    done = None
    error = None
    tokens: list[str] = []
    for name, data in _frames(body):
        if name == "done":
            done = data
        elif name == "error":
            error = data
        elif name == "token":
            tokens.append(str(data.get("delta") or ""))
    if error:
        raise DemoError(f"聊天返回错误帧：{error}")
    if done is None:
        raise DemoError("聊天流里没有 done 帧（可能进了退款/工单的点选暂停）")
    done["answer"] = "".join(tokens)
    return done


def _find_review(client: httpx.Client, raw_question: str) -> dict[str, Any] | None:
    listed = client.get("/api/review-queue", params={"status": "待审", "limit": 50})
    listed.raise_for_status()
    for item in listed.json()["items"]:
        detail = client.get(f"/api/review-queue/{item['id']}").json()
        for question in detail["questions"]:
            if raw_question in question["raw_question"]:
                return detail
    return None


def _wait_review(
    client: httpx.Client, raw_question: str, *, timeout: float
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = _find_review(client, raw_question)
        if detail is not None:
            return detail
        time.sleep(1.0)
    raise DemoError(f"{timeout:.0f}s 内待审队列里没有出现：{raw_question}")


def step_review_then_answer(
    client: httpx.Client, *, timeout: float, question: str, approved_answer: str
) -> None:
    _hr("验收 2 / 3：答不上 → 落池 → 审核通过 → 再问答得上")
    first = ask(client, f"demo-ch09-gap-{int(time.time())}", question)
    answer = str(first.get("answer") or "")
    print("第一次回答：", answer[:120])

    if REFUSAL_TEXT not in answer:
        print("！这个知识库已经有答案了，跳过「落池 → 审核」两步；")
        print("  用 --question/--answer 换一个真正没有的问题再跑，才能验完整飞轮。")
        return

    detail = _wait_review(client, question, timeout=timeout)
    review = detail["review"]
    print(f"待审行：#{review['id']} {review['normalized_question']}")
    print(f"出现次数：{review['occurrence_count']}")
    for row in detail["questions"]:
        print(f"  - 用户原话（{row['source']}）：{row['raw_question']}")
    snapshot = detail.get("snapshot") or []
    print(f"召回片段快照：{len(snapshot)} 条")
    for chunk in snapshot[:3]:
        score = chunk.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  · {chunk.get('chunk_id')} score={score_text} quote={str(chunk.get('quote'))[:50]}")
    if not snapshot:
        raise DemoError("落池行没有召回片段快照，需求 3 未满足")
    if not detail["questions"]:
        raise DemoError("待审详情里看不到用户原话")

    approved = client.post(
        f"/api/review-queue/{review['id']}/approve",
        json={"approved_answer": approved_answer},
    )
    if approved.status_code != 200:
        raise DemoError(f"审核通过失败：{approved.status_code} {approved.text[:200]}")
    print("审核通过：", approved.json())

    second = ask(client, f"demo-ch09-answered-{int(time.time())}", question)
    answer = str(second.get("answer") or "")
    print("第二次回答：", answer[:160])
    if REFUSAL_TEXT in answer:
        raise DemoError("审核通过后仍返回兜底话术，飞轮没有转完")
    print("✓ 通过后能回答了")


def step_feedback(client: httpx.Client, *, timeout: float, question: str) -> None:
    _hr("验收 4：👎 落池 → 标准化 → 进待审队列")
    done = ask(client, f"demo-ch09-feedback-{int(time.time())}", question)
    message_id = done.get("message_id")
    if not message_id:
        raise DemoError("done 帧没有 message_id，前端无法定位这一轮")
    print("回答 message_id：", message_id)

    response = client.post(
        "/api/feedback",
        json={
            "session_id": done.get("session_id"),
            "message_id": message_id,
            "rating": "down",
            "reason": "demo：答非所问",
        },
    )
    if response.status_code != 200:
        raise DemoError(f"👎 接口失败：{response.status_code} {response.text[:200]}")
    payload = response.json()
    print("落池结果：", payload)
    if not payload.get("question_id"):
        raise DemoError("👎 没有返回 question_id")

    detail = _wait_review(client, question, timeout=timeout)
    review = detail["review"]
    print(f"待审行：#{review['id']} {review['normalized_question']}")
    snapshot = detail.get("snapshot") or []
    print(f"召回片段快照：{len(snapshot)} 条（TOP1 score={snapshot[0].get('score') if snapshot else 'n/a'}）")
    if not any(q["source"] == "user_feedback" for q in detail["questions"]):
        raise DemoError("待审详情里没有 user_feedback 来源的原话")
    print("✓ 事后落池进了待审队列")


def step_intent_cost(client: httpx.Client) -> None:
    _hr("验收 5：按意图汇总 token 花销")
    response = client.get("/api/intent-cost", params={"days": 7})
    if response.status_code != 200:
        raise DemoError(f"意图成本接口失败：{response.status_code} {response.text[:200]}")
    payload = response.json()
    if not payload.get("configured"):
        print("！Langfuse 未配置，接口返回 configured=false（这是允许的降级）")
        return
    print(f"窗口：最近 {payload['days']} 天，总 token：{payload['total_tokens']}")
    print(f"{'意图':<12}{'观测组':>6}{'tokens':>10}{'成本':>10}")
    for row in payload["items"]:
        print(f"{row['intent']:<12}{row['observations']:>6}{row['tokens']:>10}{row['cost']:>10.4f}")
    if not payload["items"]:
        print("（窗口内还没有带 intent 标签的 trace）")


def step_eval_trend(client: httpx.Client) -> None:
    _hr("验收 6：评估流水线至少两轮 + 趋势")
    payload = client.get("/api/eval-runs", params={"limit": 20}).json()
    items = payload.get("items") or []
    if len(items) < 2:
        raise DemoError(
            f"eval_runs 只有 {len(items)} 轮；先跑 "
            "`uv run python scripts/run_eval_round.py --trigger 手动` 两轮"
        )
    print(f"{'时间':<22}{'触发':<6}{'条数':>4}{'recall@5':>10}{'mrr@10':>9}{'faith':>9}{'refusal':>9}")
    for item in reversed(items):
        metrics = item["metrics"]
        retrieval = metrics.get("retrieval", {})
        generation = metrics.get("generation", {})

        def fmt(value: Any) -> str:
            return f"{value:.3f}" if isinstance(value, (int, float)) else "—"

        print(
            f"{item['created_at'][:19]:<22}{item['triggered_by']:<6}{item['dataset_size']:>4}"
            f"{fmt(retrieval.get('recall@5')):>10}{fmt(retrieval.get('mrr@10')):>9}"
            f"{fmt(generation.get('faithfulness')):>9}{fmt(generation.get('refusal_accuracy')):>9}"
        )
    print("\n趋势（最新 vs 上一轮）：")
    for key, value in payload["trend"].items():
        arrow = {"up": "↑", "down": "↓", "flat": "→", None: "·"}[value["direction"]]
        print(f"  {key:<20}{value['latest']:>8.3f}  {arrow}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch09 数据飞轮端到端演示")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0, help="等待落池/标准化的秒数")
    parser.add_argument("--skip-feedback", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--question", default=UNKNOWN_QUESTION, help="一个知识库里没有的问题")
    parser.add_argument("--answer", default=APPROVED_ANSWER, help="审核时填入的核准答案")
    args = parser.parse_args(argv)

    with httpx.Client(base_url=args.base_url, timeout=180.0) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        print("healthz：", health.json().get("status", health.text[:80]))

        step_review_then_answer(
            client, timeout=args.timeout, question=args.question, approved_answer=args.answer
        )
        if not args.skip_feedback:
            step_feedback(client, timeout=args.timeout, question=args.question)
        step_intent_cost(client)
        if not args.skip_eval:
            step_eval_trend(client)

    print("\n全部步骤通过。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DemoError as exc:
        print(f"\n演示失败：{exc}", file=sys.stderr)
        sys.exit(1)
