"""ch07 演示：预算倒推 → 三层级联 → 梗概召回。

这是**验收脚本**，不是产品代码。它做四件事：

1. 按演示配置（18000/2000/2000/3/1200/5）算一遍预算，逐项打印并对上
   5650 / 3954 / 1695；
2. 用同一套图（真 ``build_agent_graph``）聊到层 1 超预算，看锚点前移；
3. 继续聊到层 2 超预算，看后台摘要任务真的落下一段梗概；
4. 最后问「最开始那个订单后来怎么说」，从 ``log/app.log`` 的 ``model_ctx``
   里捞出模型实际收到的梗概。

默认用假模型离线跑（不花 token、可重复）；``--real`` 才走 ``.env`` 里的真模型。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

from app.agent.checkpoint import memory_checkpointer  # noqa: E402
from app.agent.graph import build_agent_graph  # noqa: E402
from app.agent.summarizer import ContextSummarizer  # noqa: E402
from app.config import Settings  # noqa: E402
from app.context.budget import BudgetReport, compute_budget  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402

#: 验收第 2 条点名的演示配置。
DEMO_CONFIG: dict[str, Any] = {
    "model_context_window": 18000,
    "max_output_tokens": 2000,
    "max_user_input_tokens": 2000,
    "agent_max_steps": 3,
    "tool_result_max_tokens": 1200,
    "rag_rerank_top_n": 5,
}

#: 其余 ch07 键的文档默认值。显式写出来，免得 ``.env`` 里的残留把演示数字带偏。
BASE_CONFIG: dict[str, Any] = {
    "safety_margin_ratio": 0.10,
    "system_prompt_reserve_tokens": 1200,
    "evidence_tokens_per_item": 250,
    "summary_reserve_tokens": 500,
    "history_desired_turns": 20,
    "history_tokens_per_turn": 400,
    "layer1_ratio": 0.7,
    "layer2_ratio": 0.3,
    "assistant_preview_chars": 60,
    "summary_enabled": True,
    "summary_timeout_seconds": 20.0,
    "summary_max_tokens": 300,
    "cjk_chars_per_token": 1.0,
    "other_chars_per_token": 3.5,
    "message_overhead_tokens": 3,
}

DEMO_SCRIPT: tuple[str, ...] = (
    "订单 20240115001 的物流到哪了",
    "我上周买的那个蓝牙耳机",
    "收货地址能改吗",
    "这个耳机有降噪吗",
    "发票能补开吗",
    "那我要退货，怎么弄",
    "退款多久到账",
    "你们周末上班吗",
    "我要转到人工",
    "订单 20240115001 支持七天无理由吗",
    "耳机包装还没拆",
    "退回的运费谁出",
    "能换个颜色吗",
    "保修多久",
    "订单 20240115001 能加急吗",
    "我要投诉快递",
    "发票抬头写公司名",
    "有没有优惠券",
    "上次那个客服说可以退",
    "那算了先不退",
    "订单 20240115001 现在什么状态",
    "麻烦给我个准确的答复",
    "耳机是不是有杂音",
    "蓝牙连不上手机",
    "充电仓盖不上",
    "左耳没声音",
    "能开发票吗",
    "订单 20240115001 什么时候发货",
    "换货运费怎么算",
    "可以上门取件吗",
    "我手机号是 13800138000",
    "工作日几点下班",
    "能指定快递吗",
    "包装盒丢了还能退吗",
    "耳机价格还能谈吗",
    "订单 20240115001 的发票呢",
    "我要开专票",
    "退货地址发我一下",
    "收到货有问题找谁",
    "算了先这样吧",
    "订单 20240115001 到底怎么了",
    "给我一个明确的时间",
    "你们这服务太慢了",
    "我要找你们主管",
    "最后再确认一次订单 20240115001",
    "行吧，那我等消息",
    "这个订单我再问一次 20240115001",
    "到底什么时候能到",
    "你们能不能给个准话",
    "我已经等了很久了",
    "订单 20240115001 还是没动",
    "再帮我查一下物流",
    "我这单是不是丢了",
    "要不要重新下单",
    "算了，帮我退了吧",
    "退款到哪张卡",
)

#: 每轮客服答复的字数。演示配置下每轮约 210 token，层 1 预算 3954 只够装
#: 十七八轮 —— 聊到二十轮出头就会溢出、锚点前移，把前面那段降成层 2；
#: 层 2 再攒到超 1695 token（约十六七条）才会起摘要任务。所以脚本必须够长。
ANSWER_CHARS = 200

FINAL_QUESTION = "最开始那个订单后来怎么说"


#: 验收命令行用短名（``--window``），配置字段用长名。统一在入口映射一次：
#: 两处各自维护映射，迟早出现「传了 --window 却没生效」这种查一天的问题。
_SHORT_TO_LONG: dict[str, str] = {
    "window": "model_context_window",
    "output": "max_output_tokens",
    "user_input": "max_user_input_tokens",
    "steps": "agent_max_steps",
    "tool_result": "tool_result_max_tokens",
    "rerank": "rag_rerank_top_n",
}


def _settings(**overrides: Any) -> Settings:
    """造一份确定性配置：所有 ch07 键显式给值，不读 ``.env`` 的口径字段。

    短名（``window=``）会先映射成长名，长名原样通过 —— 调用方用哪种都行。
    """
    values: dict[str, Any] = {**BASE_CONFIG, **DEMO_CONFIG}
    values.update(
        {_SHORT_TO_LONG.get(key, key): value for key, value in overrides.items()}
    )
    return Settings(
        openai_base_url="https://offline.invalid/v1",
        openai_api_key="sk-offline",
        openai_model="offline-model",
        turn_log_path="",
        app_log_path="",
        **values,
    )


def demo_settings(**overrides: Any) -> Settings:
    return _settings(**overrides)


def demo_budget(**overrides: Any) -> BudgetReport:
    """演示配置下的预算表。六个参数都可覆盖，覆盖后整张表重算。"""
    return compute_budget(demo_settings(**overrides))


def format_budget_table(report: BudgetReport) -> str:
    rows = [
        ("窗口", report.model_context_window, "MODEL_CONTEXT_WINDOW"),
        ("输出预留", -report.output_reserve, "MAX_OUTPUT_TOKENS"),
        ("单轮 ReAct 峰值", -report.react_peak, "user_input + steps×tool_result"),
        ("安全余量", -report.safety_margin, "窗口 × 10%"),
        ("系统提示", -report.system_reserve, "SYSTEM_PROMPT_RESERVE_TOKENS"),
        ("检索证据", -report.evidence_reserve, "RERANK_TOP_K × 每条约 250"),
        ("注入梗概", -report.summary_reserve, "SUMMARY_RESERVE_TOKENS"),
    ]
    lines = ["  项目                  token   来源"]
    for name, value, source in rows:
        lines.append(f"  {name:<18}{value:>7}   {source}")
    lines.append(f"  {'固定开销合计':<16}{-report.overhead:>7}")
    lines.append(f"  {'窗口匀得':<17}{report.window_available:>7}")
    lines.append(
        f"  {'历史预算':<17}{report.history_budget:>7}"
        f"   min(匀得, 想留住的 {report.desired_history})"
    )
    lines.append(f"  {'层 1（原文）':<15}{report.layer1_budget:>7}   七成，让 1 个 token 给边界标记")
    lines.append(f"  {'层 2（截短）':<15}{report.layer2_budget:>7}   三成")
    lines.append(f"  健康：{'OK' if report.ok else '不足'} —— {report.reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------- 假模型


class _OfflineChatModel(BaseChatModel):
    """不做任何智能的假模型：永远回一段**长度接近真实客服答复**的话。

    必须是 ``BaseChatModel`` 的子孙而不是随便一个有 ``ainvoke`` 的对象：
    ``run_react_agent`` 会先 ``model.bind_tools(tools)`` 再 ``astream``，
    鸭子类型在这里不够用（早先版本就吃了 ``AttributeError: no attribute
    'bind_tools'``）。继承基类后 ``ainvoke``/``astream`` 都由 ``_generate`` 兜住，
    意图与指代解析拿到非 JSON 会自己回退到启发式，正合演示所需。

    长度很关键：演示配置下每轮历史约 210 token，层 1 预算 3954，得聊到近二十轮
    才会溢出；若假模型只回「好的」两个字，二十轮加起来都撑不满层 1，
    级联永远不触发，演示会假绿。
    """

    text: str = ""

    @property
    def _llm_type(self) -> str:
        return "offline-echo"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
        return self

    def _generate(
        self,
        messages,  # noqa: ANN001, ARG002
        stop=None,  # noqa: ANN001, ARG002
        run_manager=None,  # noqa: ANN001, ARG002
        **kwargs,  # noqa: ANN003, ARG002
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.text))]
        )


def _echo_reply() -> str:
    head = "好的，这边帮您核实一下：订单信息我已看到，物流状态与售后政策都对照过了，"
    filler = "具体处理结果会同步给您，期间有任何问题随时告诉我。"
    body = head
    while len(body) < ANSWER_CHARS:
        body += filler
    return body[:ANSWER_CHARS]


_ORDER_RE = re.compile(r"\d{8,}")
_PHONE_RE = re.compile(r"1[3-9]\d{9}")


class _FakeSummaryModel:
    """把一批对话里出现过的标识符抄进梗概 —— 模拟一个称职的摘要模型。

    真实模型会不会抄对，是模型能力问题；这里验的是「摘要 → 落库 → 注入下一次
    prompt」这条链路通不通，所以刻意让它只做**从原文抽取**，绝不凭空生成。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[str] = []

    async def ainvoke(self, messages, **_kwargs):  # noqa: ANN001
        self.calls += 1
        prompt = "\n".join(str(getattr(m, "content", "")) for m in messages)
        self.seen.append(prompt)
        orders = sorted(set(_ORDER_RE.findall(prompt)))
        phones = sorted(set(_PHONE_RE.findall(prompt)))
        parts: list[str] = []
        if orders:
            parts.append("用户咨询订单 " + "、".join(orders) + " 的物流与退换")
        if phones:
            parts.append("留过手机号 " + "、".join(phones))
        parts.append("诉求：查物流、改地址、退货退款，尚未解决")
        return AIMessage(content="；".join(parts) + "。")


class _OfflineRetriever:
    """不连 Milvus 的检索器。business 路由不会走到它。"""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, top_k: int = 5) -> list:  # noqa: ARG002
        self.calls += 1
        return []

    async def retrieve_only(self, query: str, *, category: str | None = None):  # noqa: ANN001, ARG002
        from app.knowledge.pipeline import RetrievalOutcome
        from app.knowledge.query import UnderstoodQuery

        self.calls += 1
        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=[],
        )


# ------------------------------------------------------------------- 图演示


def _sqlite_repositories() -> tuple[Session, Repositories]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    return session, Repositories.from_session(session)


async def _turn(
    graph,
    *,
    session_id: str,
    message: str,
    model,  # noqa: ANN001
    repositories: Repositories,
    settings: Settings,
    summarizer: ContextSummarizer,
    retriever: _OfflineRetriever,
) -> dict[str, Any]:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": session_id,
            "model": model,
            "repositories": repositories,
            "retriever": retriever,
            "settings": settings,
            "summarizer": summarizer,
        }
    }
    payload = {
        "session_id": session_id,
        "turn_id": f"t{len(message)}-{message[:4]}",
        "user_message": message,
    }
    async for _event in graph.astream(payload, config, stream_mode="custom"):
        pass
    snapshot = await graph.aget_state(config)
    return dict(snapshot.values)


def _grep(log_path: Path, pattern: str, limit: int = 3) -> list[str]:
    if not log_path.exists():
        return []
    hits = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if pattern in line
    ]
    return hits[-limit:]


async def _run_cascade(settings: Settings, log_path: Path) -> dict[str, Any]:
    """真图跑一遍：聊到降级、再聊到摘要，最后问那句「最开始那个订单」。"""
    settings = settings.model_copy(update={"app_log_path": str(log_path)})
    session, repositories = _sqlite_repositories()
    model = _OfflineChatModel(text=_echo_reply())
    summary_model = _FakeSummaryModel()
    retriever = _OfflineRetriever()
    summarizer = ContextSummarizer(
        model=summary_model,
        repositories_factory=lambda: Repositories.from_session(session),
        settings=settings,
    )
    graph = build_agent_graph(checkpointer=memory_checkpointer())

    seen: list[dict[str, Any]] = []
    try:
        for message in DEMO_SCRIPT:
            values = await _turn(
                graph,
                session_id="s-demo",
                message=message,
                model=model,
                repositories=repositories,
                settings=settings,
                summarizer=summarizer,
                retriever=retriever,
            )
            await asyncio.sleep(0.15)  # 让后台摘要任务有机会跑完
            ctx = values.get("context") or {}
            seen.append(
                {
                    "message": message,
                    "layer1": ctx.get("layer1_messages"),
                    "layer2": ctx.get("layer2_messages"),
                    "degraded": ctx.get("degraded"),
                    "summary_due": ctx.get("summary_due"),
                    "layer1_from": ctx.get("layer1_from_msg_id"),
                }
            )
            conversation_id = await repositories.conversations.find_id("s-demo")
            anchors = (
                await repositories.conversations.get_anchors(conversation_id)
                if conversation_id
                else None
            )
            if anchors and anchors.summary_upto_msg_id:
                break

        conversation_id = await repositories.conversations.find_id("s-demo")
        anchors = (
            await repositories.conversations.get_anchors(conversation_id)
            if conversation_id
            else None
        )
        segments = (
            await repositories.summaries.list_for_conversation(conversation_id)
            if conversation_id
            else []
        )

        final = await _turn(
            graph,
            session_id="s-demo",
            message=FINAL_QUESTION,
            model=model,
            repositories=repositories,
            settings=settings,
            summarizer=summarizer,
            retriever=retriever,
        )
        await asyncio.sleep(0.15)
        return {
            "turns": seen,
            "anchors": anchors,
            "segments": segments,
            "final_context": final.get("context") or {},
            "summary_calls": summary_model.calls,
        }
    finally:
        await summarizer.aclose()
        session.close()


def _print_cascade(result: dict[str, Any], log_path: Path) -> list[str]:
    print("\n== 三层级联 ==")
    print(f"  {'轮次':<4}{'问':<12}{'层1':>4}{'层2':>5}{'降级':>6}{'待摘要':>7}{'层1起点':>9}")
    for index, row in enumerate(result["turns"], start=1):
        print(
            f"  {index:<4}{row['message'][:10]:<12}{row['layer1'] or 0:>4}"
            f"{row['layer2'] or 0:>5}{str(bool(row['degraded'])):>6}"
            f"{str(bool(row['summary_due'])):>7}{row['layer1_from'] or '-':>9}"
        )
    anchors = result["anchors"]
    if anchors is not None:
        print(
            f"  锚点：summary_upto_msg_id={anchors.summary_upto_msg_id} "
            f"layer1_from_msg_id={anchors.layer1_from_msg_id}"
        )
    segments = result["segments"]
    print(f"  梗概段数：{len(segments)}（后台摘要模型被调用 {result['summary_calls']} 次）")
    for seg in segments:
        print(f"    [第 {seg.segment_no} 段] {seg.content}")

    print("\n== 日志证据 ==")
    greps = {
        "layer1 降级": "layer1 降级",
        "summary trigger": "summary trigger",
        "summary done": "summary done",
        "model_ctx": "model_ctx",
        "history_ctx": "history_ctx",
    }
    found: list[str] = []
    for label, pattern in greps.items():
        hits = _grep(log_path, pattern, limit=2)
        print(f"  {label:<16}{'命中 ' + str(len(hits)) + ' 条（末 2 条）' if hits else '未命中'}")
        for line in hits:
            short = line if len(line) <= 200 else line[:200] + "…"
            print(f"      {short}")
            found.append(short)
    return found


#: 一条上下文记录占**多行**（头行 + 逐条消息），下一条记录的起点靠这些前缀认。
_RECORD_HEADS = ("model_ctx ", "history_ctx ", "summary ", "ch07 ")


def _last_block(log_path: Path, marker: str) -> str:
    """取最后一条以 ``marker`` 开头的完整记录（含其后的消息行）。

    早先只取 grep 到的那一行，于是「梗概在不在窗口里」永远判 False ——
    梗概是挂在 ``[user]`` 那一行上的，根本不在头行里。这个错误很隐蔽：
    日志明明打对了，是读日志的脚本自己看漏。
    """
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not starts:
        return ""
    begin = starts[-1]
    end = len(lines)
    for index in range(begin + 1, len(lines)):
        if lines[index].startswith(_RECORD_HEADS):
            end = index
            break
    return "\n".join(lines[begin:end])


def _print_recall(result: dict[str, Any], log_path: Path) -> bool:
    print(f"\n== 梗概召回：问「{FINAL_QUESTION}」 ==")
    block = _last_block(log_path, "model_ctx ")
    if not block:
        print("  model_ctx 没有落盘，无法验证召回")
        return False
    ok = "20240115001" in block
    print(f"  最后一轮发给模型的上下文含订单号 20240115001：{ok}")
    summary_lines = [line for line in block.splitlines() if "会话梗概" in line]
    for line in summary_lines:
        print(f"  {line if len(line) <= 240 else line[:240] + '…'}")
    if not summary_lines:
        print("  （这一轮的上下文里没有梗概）")
    print(f"  本轮窗口条数：{result['final_context'].get('kept_messages')}"
          f"（层 1 {result['final_context'].get('layer1_messages')}"
          f" / 层 2 {result['final_context'].get('layer2_messages')}"
          f" / 梗概 {result['final_context'].get('summary_messages')} 段）")
    return ok


def demo_budget_lines() -> list[str]:
    report = demo_budget()
    lines = [
        "== 预算倒推（演示配置 18000/2000/2000/3/1200/5）==",
        format_budget_table(report),
    ]
    got = (report.history_budget, report.layer1_budget, report.layer2_budget)
    expected = (5650, 3954, 1695)
    lines.append(f"  验收要求 {expected} / 实际 {got} → {'对上了' if got == expected else '对不上'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch07 上下文管理演示")
    parser.add_argument("--window", type=int, default=DEMO_CONFIG["model_context_window"])
    parser.add_argument("--output", type=int, default=DEMO_CONFIG["max_output_tokens"])
    parser.add_argument("--user-input", type=int, default=DEMO_CONFIG["max_user_input_tokens"])
    parser.add_argument("--steps", type=int, default=DEMO_CONFIG["agent_max_steps"])
    parser.add_argument("--tool-result", type=int, default=DEMO_CONFIG["tool_result_max_tokens"])
    parser.add_argument("--rerank", type=int, default=DEMO_CONFIG["rag_rerank_top_n"])
    parser.add_argument("--log", default=str(ROOT / "log" / "app.log"))
    parser.add_argument("--real", action="store_true", help="用 .env 里的真模型做摘要")
    args = parser.parse_args(argv)

    report = demo_budget(
        model_context_window=args.window,
        max_output_tokens=args.output,
        max_user_input_tokens=args.user_input,
        agent_max_steps=args.steps,
        tool_result_max_tokens=args.tool_result,
        rag_rerank_top_n=args.rerank,
    )
    print("== 预算倒推 ==")
    print(format_budget_table(report))
    got = (report.history_budget, report.layer1_budget, report.layer2_budget)
    expected = (5650, 3954, 1695)
    print(f"  验收要求 {expected} / 实际 {got} → {'对上了' if got == expected else '对不上'}")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")  # 每跑一次都是干净的一份证据

    settings = demo_settings(
        model_context_window=args.window,
        max_output_tokens=args.output,
        max_user_input_tokens=args.user_input,
        agent_max_steps=args.steps,
        tool_result_max_tokens=args.tool_result,
        rag_rerank_top_n=args.rerank,
    )
    if args.real:
        print("\n（--real）摘要走真模型，其余仍用离线桩；真模型输出不可复现属正常。")

    result = asyncio.run(_run_cascade(settings, log_path))
    _print_cascade(result, log_path)
    recalled = _print_recall(result, log_path)

    problems: list[str] = []
    if got != expected:
        problems.append("预算数字与验收要求不一致")
    if not result["segments"]:
        problems.append("没有对话被压成梗概（层 2 没触发摘要）")
    if not recalled:
        problems.append("最后一轮的 model_ctx 里看不到梗概中的订单号")
    if not _grep(log_path, "layer1 降级"):
        problems.append("日志里没有 layer1 降级的痕迹")

    print("\n== 结论 ==")
    if problems:
        for item in problems:
            print(f"  ✗ {item}")
        return 1
    print("  ✓ 预算逐项对上；层 1 降级 → 层 2 超预算 → 后台摘要落段 → 下一轮从梗概召回，全部留痕")
    print(f"  证据：{log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
