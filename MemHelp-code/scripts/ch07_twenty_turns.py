"""ch07 二十轮压测：默认窗口「装得下就不压」，演示窗口「超了就压」。

为什么单独一个脚本而不是 pytest 用例：验收第 3 条要的是**二十轮的真实轨迹**
（每轮层 1/层 2 各占多少、有没有降级），跑在测试里既慢又难读。这里把同一套
``plan_layers`` 拿纯数据喂二十轮，逐轮打印；测试只钉结论那几行。

两种模式：

* ``--fake``（默认）纯规划模拟：不连数据库、不发网络，只让 ``plan_layers``
  在逐轮增长的消息上跑，看预算怎么分配、锚点怎么走；
* ``--real`` 跑真图（真模型、真落库），慢且依赖 ``.env``，用于人工复核。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.context.budget import BudgetReport, compute_budget  # noqa: E402
from app.context.layers import LayerPlan, plan_layers  # noqa: E402
from app.context.tokens import counter_from_settings  # noqa: E402
from app.db.repositories import MessageRow  # noqa: E402

#: 二十轮的用户问句。都短，模拟真实客服会话里「一句话问一件事」。
QUESTIONS: tuple[str, ...] = (
    "订单 20240115001 到哪了",
    "我上周买的耳机有降噪吗",
    "发票能补开吗",
    "收货地址要改一下",
    "这个能便宜点吗",
    "退货怎么弄",
    "退款多久到账",
    "你们周末上班吗",
    "我要转人工",
    "订单 20240115001 支持七天无理由吗",
    "耳机盒子还没拆",
    "运费谁出",
    "能换个颜色吗",
    "保修多久",
    "订单 20240115001 能加急吗",
    "我要投诉快递",
    "发票抬头写公司",
    "有没有优惠券",
    "上次那个客服说可以退",
    "那算了先不退",
)

#: 每轮客服答复的长度。二十轮 × 这个长度决定总占用，故意留出余量：
#: 默认窗口层 1 预算 5599，二十轮合计要稳稳落在它下面才叫「装得下」。
ANSWER_CHARS = 200

_DEFAULT = Settings(
    openai_base_url="https://offline.invalid/v1",
    openai_api_key="sk-offline",
    openai_model="offline-model",
    turn_log_path="",
    app_log_path="",
)

_DEMO_WINDOW: dict[str, Any] = {
    "model_context_window": 18000,
    "max_output_tokens": 2000,
    "max_user_input_tokens": 2000,
    "agent_max_steps": 3,
    "tool_result_max_tokens": 1200,
    "rag_rerank_top_n": 5,
}


def _settings(*, demo_window: bool = False, **overrides: Any) -> Settings:
    values: dict[str, Any] = {}
    if demo_window:
        values.update(_DEMO_WINDOW)
    values.update(overrides)
    return _DEFAULT.model_copy(update=values)


def _answer(index: int) -> str:
    """一句**长度精确为 ``ANSWER_CHARS``** 的客服答复。

    长度是这段压测唯一的自变量，所以必须钉死：先用一句话开头，再拿填充句补齐。
    早先写成 ``"还请您稍等。" * 4``，看着不长不短，实际每轮才六十来字，
    二十轮下来连默认窗口的层 1 预算都撑不满 —— 「演示窗口下必定压缩」那条直接
    假绿。
    """
    head = f"关于您第 {index} 个问题，我这边查到的情况是：已为您核对订单信息，"
    filler = "还请您稍等一下，有结果我第一时间告诉您。"
    body = head
    while len(body) < ANSWER_CHARS:
        body += filler
    return body[:ANSWER_CHARS]


def _rows(turns: int) -> list[MessageRow]:
    """造 ``turns`` 轮 user/assistant 交替的消息行，id 从 1 递增。"""
    base = datetime(2026, 10, 20, 10, 0, 0)
    rows: list[MessageRow] = []
    for index in range(turns):
        rows.append(
            MessageRow(
                id=len(rows) + 1,
                role="user",
                content=QUESTIONS[index % len(QUESTIONS)],
                tool_calls=None,
                tool_call_id=None,
                created_at=base + timedelta(seconds=len(rows)),
            )
        )
        rows.append(
            MessageRow(
                id=len(rows) + 1,
                role="assistant",
                content=_answer(index + 1),
                tool_calls=None,
                tool_call_id=None,
                created_at=base + timedelta(seconds=len(rows)),
            )
        )
    return rows


@dataclass(frozen=True)
class TurnStat:
    """一轮规划完的观测值。"""

    turn: int
    layer1_messages: int
    layer2_messages: int
    layer1_tokens: int
    layer2_tokens: int
    degraded: bool
    summary_due: bool
    layer1_from: int | None
    summary_upto: int | None


def _simulate(
    *,
    turns: int,
    budget: BudgetReport,
    settings: Settings,
) -> tuple[list[TurnStat], list[LayerPlan], bool, bool]:
    """逐轮规划，锚点按「真发生了什么」前移，模拟图里的行为。

    这一步不能只用一次 ``plan_layers`` 跑完全部：第一次跑时
    ``layer1_from is None``，所有消息都算层 1，超预算只会被裁剪；
    层 2 是**下一轮**才出现的（锚点已经前移，落在锚点之间的那段才是层 2）。
    """
    counter = counter_from_settings(settings)
    rows = _rows(turns)
    stats: list[TurnStat] = []
    plans: list[LayerPlan] = []
    summary_upto: int | None = None
    layer1_from: int | None = None
    any_degraded = False
    any_summary = False

    for turn in range(1, turns + 1):
        visible = rows[: turn * 2]
        plan = plan_layers(
            visible,
            summary_upto=summary_upto,
            layer1_from=layer1_from,
            layer1_budget=budget.layer1_budget,
            layer2_budget=budget.layer2_budget,
            counter=counter,
        )
        plans.append(plan)
        stats.append(
            TurnStat(
                turn=turn,
                layer1_messages=len(plan.layer1),
                layer2_messages=len(plan.layer2),
                layer1_tokens=plan.layer1_tokens,
                layer2_tokens=plan.layer2_tokens,
                degraded=plan.degraded,
                summary_due=plan.summary_due,
                layer1_from=plan.layer1_from_msg_id,
                summary_upto=summary_upto,
            )
        )
        any_degraded = any_degraded or plan.degraded
        if plan.degraded and plan.layer1_from_msg_id is not None:
            layer1_from = plan.layer1_from_msg_id
        if plan.summary_due and plan.summary_rows:
            any_summary = True
            summary_upto = plan.summary_rows[-1].id
            layer1_from = summary_upto + 1
    return stats, plans, any_degraded, any_summary


def run_twenty_turns_fake(*, turns: int = 20, demo_window: bool = False) -> dict[str, Any]:
    """二十轮纯规划压测（不发网络、不连库）。

    返回的 ``degraded`` / ``summary_due`` 是**全程有没有发生过一次**，
    不是最后一轮的状态 —— 验收第 3 条问的是「二十轮聊下来压没压过」。
    """
    settings = _settings(demo_window=demo_window)
    budget = compute_budget(settings)
    stats, _plans, any_degraded, any_summary = _simulate(
        turns=turns, budget=budget, settings=settings
    )
    return {
        "turns": stats,
        "degraded": any_degraded,
        "summary_due": any_summary,
        "max_layer1_tokens": max((s.layer1_tokens for s in stats), default=0),
        "max_layer2_tokens": max((s.layer2_tokens for s in stats), default=0),
        "layer1_budget": budget.layer1_budget,
        "layer2_budget": budget.layer2_budget,
        "budget": budget,
        "ok": budget.ok,
    }


def format_turns(summary: dict[str, Any]) -> str:
    budget = summary["budget"]
    lines = [
        f"  预算：历史 {budget.history_budget}"
        f"（层 1 {budget.layer1_budget} / 层 2 {budget.layer2_budget}）",
        f"  {'轮':>3}{'层1条':>6}{'层1tok':>8}{'层2条':>6}{'层2tok':>8}"
        f"{'降级':>6}{'待摘要':>7}{'层1起点':>9}",
    ]
    for stat in summary["turns"]:
        lines.append(
            f"  {stat.turn:>3}{stat.layer1_messages:>6}{stat.layer1_tokens:>8}"
            f"{stat.layer2_messages:>6}{stat.layer2_tokens:>8}"
            f"{'是' if stat.degraded else '否':>6}{'是' if stat.summary_due else '否':>7}"
            f"{(stat.layer1_from if stat.layer1_from is not None else '-'):>9}"
        )
    lines.append(
        f"  峰值：层 1 {summary['max_layer1_tokens']} / {budget.layer1_budget}，"
        f"层 2 {summary['max_layer2_tokens']} / {budget.layer2_budget}"
    )
    return "\n".join(lines)


def _real_retriever(settings: Settings, model: Any, repositories: Any) -> Any:
    """按**生产路径**造检索管线（``RagPipeline``）。

    为什么不能像首版那样直接传 ``None``：``--real`` 跑的是真图，问题里只要有
    「耳机有降噪吗」这种知识类问句，就会走到 ``knowledge_retrieve`` 节点并
    ``retriever.retrieve_only(...)``，``None`` 当场 ``AttributeError`` —— 那是
    脚本没接好，不是业务缺陷，但会让「二十轮压测」只能跑全是 business 路由的
    会话，等于没验到真实分布。

    这里刻意复用 ``app.api`` 的两个装配辅助（``_vector_store_for`` /
    ``_understander_for``），而不是就地再拼一套：那两处带着进程内复用缓存
    （Milvus 客户端、Query 理解器），另拼一套会绕开缓存、也让线上与脚本的检索
    参数各走各的。
    """
    from app.api import _understander_for, _vector_store_for
    from app.knowledge.embedder import get_embedder
    from app.knowledge.hybrid import HybridRetriever
    from app.knowledge.pipeline import RagPipeline
    from app.knowledge.rerank import get_reranker
    from app.knowledge.sufficiency import SufficiencyJudge

    store = _vector_store_for(
        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
    )
    return RagPipeline(
        understander=_understander_for(settings, model),
        retriever=HybridRetriever(
            embedder=get_embedder(settings),
            vector_store=store,
            reranker=get_reranker(settings),
            chunk_repo=repositories.knowledge,
            hybrid_top_k=settings.rag_hybrid_top_k,
            candidate_k=settings.rag_candidate_k,
            rerank_top_n=settings.rag_rerank_top_n,
        ),
        sufficiency=SufficiencyJudge(model, timeout=settings.rag_rewrite_timeout),
        min_score=settings.rag_min_rerank_score,
        top_n=settings.rag_rerank_top_n,
    )


async def _run_real(turns: int, *, demo_window: bool) -> dict[str, Any]:
    """跑真图：真模型、真落库。慢，用于人工复核，不进 CI。"""
    from app.agent.checkpoint import memory_checkpointer
    from app.agent.graph import build_agent_graph
    from app.agent.summarizer import ContextSummarizer
    from app.config import get_settings
    from app.db.repositories import Repositories
    from app.db.session import create_engine_from, session_factory
    from app.llm import get_chat_model
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base

    settings = get_settings()
    if demo_window:
        settings = settings.model_copy(update=_DEMO_WINDOW)
    budget = compute_budget(settings)
    counter = counter_from_settings(settings)

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    repositories = Repositories.from_session(session)
    model = get_chat_model(settings)
    summarizer = ContextSummarizer(
        model=model,
        repositories_factory=lambda: Repositories.from_session(session),
        settings=settings,
    )
    retriever = _real_retriever(settings, model, repositories)
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    config = {
        "configurable": {
            "thread_id": "s-20",
            "model": model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
            "summarizer": summarizer,
        }
    }

    stats: list[TurnStat] = []
    layers: list[tuple[int, int, bool, bool, int, int]] = []
    try:
        for turn in range(1, turns + 1):
            payload = {
                "session_id": "s-20",
                "turn_id": f"t{turn}",
                "user_message": QUESTIONS[(turn - 1) % len(QUESTIONS)],
            }
            async for _event in graph.astream(payload, config, stream_mode="custom"):
                pass
            snapshot = await graph.aget_state(config)
            ctx = (snapshot.values or {}).get("context") or {}
            layers.append(
                (
                    int(ctx.get("layer1_messages") or 0),
                    int(ctx.get("layer2_messages") or 0),
                    bool(ctx.get("degraded")),
                    bool(ctx.get("summary_due")),
                    int(ctx.get("layer1_tokens") or 0),
                    int(ctx.get("layer2_tokens") or 0),
                )
            )
            await asyncio.sleep(0.1)
            conversation_id = await repositories.conversations.find_id("s-20")
            anchors = (
                await repositories.conversations.get_anchors(conversation_id)
                if conversation_id
                else None
            )
            stats.append(
                TurnStat(
                    turn=turn,
                    layer1_messages=layers[-1][0],
                    layer2_messages=layers[-1][1],
                    layer1_tokens=layers[-1][4],
                    layer2_tokens=layers[-1][5],
                    degraded=layers[-1][2],
                    summary_due=layers[-1][3],
                    layer1_from=anchors.layer1_from_msg_id if anchors else None,
                    summary_upto=anchors.summary_upto_msg_id if anchors else None,
                )
            )
    finally:
        await summarizer.aclose()
        session.close()

    degraded_any = any(item[2] for item in layers)
    summary_any = any(item[3] for item in layers)
    return {
        "turns": stats,
        "degraded": degraded_any,
        "summary_due": summary_any,
        "max_layer1_tokens": max((item[4] for item in layers), default=0),
        "max_layer2_tokens": max((item[5] for item in layers), default=0),
        "layer1_budget": budget.layer1_budget,
        "layer2_budget": budget.layer2_budget,
        "budget": budget,
        "ok": budget.ok,
        "counter": counter,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch07 二十轮上下文压测")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument(
        "--demo-window",
        action="store_true",
        help="用演示窗口（18000）跑，期望看到压缩发生",
    )
    parser.add_argument("--real", action="store_true", help="跑真图真模型（慢）")
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.window is not None:
        overrides["model_context_window"] = args.window

    mode = "真图（真模型）" if args.real else "纯规划（离线）"
    window_label = "演示窗口 18000" if args.demo_window else f"窗口 {args.window or 32000}"
    print(f"== ch07 二十轮压测：{mode}，{window_label}，{args.turns} 轮 ==")

    if args.real:
        summary = asyncio.run(_run_real(args.turns, demo_window=args.demo_window))
    else:
        if overrides:
            settings = _settings(demo_window=args.demo_window, **overrides)
            budget = compute_budget(settings)
            stats, _plans, degraded, summary_due = _simulate(
                turns=args.turns, budget=budget, settings=settings
            )
            summary = {
                "turns": stats,
                "degraded": degraded,
                "summary_due": summary_due,
                "max_layer1_tokens": max((s.layer1_tokens for s in stats), default=0),
                "max_layer2_tokens": max((s.layer2_tokens for s in stats), default=0),
                "layer1_budget": budget.layer1_budget,
                "layer2_budget": budget.layer2_budget,
                "budget": budget,
                "ok": budget.ok,
            }
        else:
            summary = run_twenty_turns_fake(
                turns=args.turns, demo_window=args.demo_window
            )

    print(format_turns(summary))

    budget = summary["budget"]
    print(f"\n  budget.ok = {summary['ok']}（{budget.reason}）")
    if not summary["ok"]:
        print("  ✗ 预算本身就不足，先调窗口，别调别的")
        return 1

    if args.real:
        # ``--real`` 验的是**验收 1**（连着聊二十轮以上，token 不爆、系统不崩），
        # 不是验收 3。真模型的答复长度不受控（实测单条 300–600 字，比离线脚本
        # 钉死的 200 字长一倍多），默认窗口下二十多轮本来就该压缩 —— 拿
        # 「不许压缩」去卡它会得出「实现坏了」的错误结论。
        # 这里卡的是不变量：每轮层 1 不超自己的预算，且超了就一定有降级标记。
        over = [
            stat
            for stat in summary["turns"]
            if stat.layer1_tokens > summary["layer1_budget"]
            and not stat.degraded
        ]
        if over:
            for stat in over:
                print(
                    f"  ✗ 第 {stat.turn} 轮层 1 用了 {stat.layer1_tokens} token，"
                    f"预算 {summary['layer1_budget']}，却没有降级标记"
                )
            return 1
        print(
            f"  ✓ 跑了 {len(summary['turns'])} 轮没崩；"
            f"层 1 峰值 {summary['max_layer1_tokens']} / {summary['layer1_budget']}，"
            f"层 2 峰值 {summary['max_layer2_tokens']} / {summary['layer2_budget']}"
        )
        compressed = summary["degraded"] or summary["summary_due"]
        print(f"  发生过压缩：{compressed}（真模型答复更长，默认窗口下压了也正常）")
        return 0

    if args.demo_window:
        compressed = summary["degraded"] or summary["summary_due"]
        print(f"  压缩发生过：{compressed}（演示窗口下期望 True）")
        if not compressed:
            print("  ✗ 演示窗口下二十轮都没压，说明对话太短或预算没生效")
            return 1
        print("  ✓ 演示窗口下如期压缩")
        return 0

    print(f"  降级：{summary['degraded']}  摘要：{summary['summary_due']}（默认窗口下期望都是 False）")
    if summary["degraded"] or summary["summary_due"]:
        print("  ✗ 默认窗口装得下，不该压 —— 压缩是成本不是美德")
        return 1
    print("  ✓ 二十轮全程未压缩：装得下就不压")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
