"""State 契约与「八类意图 → 五个出口」映射。

## 为什么分流是常量表

意图识别可以模型化，但**分流规则不能**。把八类归到五个出口是产品决策，
不是模型的自由；写成常量表后，任何一次路由变化都会在 diff 里出现，
也能被 `test_eight_intents_map_to_five_exits` 逐项钉住。

## 为什么 messages 用 add_messages

LangGraph 的 checkpointer 每一轮都会把 state 落盘；`add_messages` 负责按消息 id
追加/覆盖，避免节点返回 `messages` 时把整段历史覆盖掉。MySQL 仍是审计与页面
回读的权威副本，`messages` 在这里是**给模型看的运行时上下文**。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

__all__ = [
    "ALL_INTENTS",
    "INTENT_TO_ROUTE",
    "AgentState",
    "Intent",
    "Route",
    "SuggestedAction",
    "route_for",
]

Intent = Literal[
    "logistics",
    "order",
    "product_consult",
    "refund_return",
    "after_sales",
    "complaint",
    "chitchat",
    "other",
]

Route = Literal["knowledge", "business", "refund", "complaint", "chitchat"]

SuggestedAction = Literal["transfer_human", "create_ticket"]

#: 八类意图的权威顺序（也是评估集与日志里的顺序）。
ALL_INTENTS: tuple[Intent, ...] = (
    "logistics",
    "order",
    "product_consult",
    "refund_return",
    "after_sales",
    "complaint",
    "chitchat",
    "other",
)

#: 写死的分流表。改它必须同时改 spec §4.1 与测试。
#:
#: ch06：退款退货与售后统一走 `refund` 确定性子流程；`other` 走 business，
#: 由主力 Agent 做需求澄清，不硬塞业务意图。
INTENT_TO_ROUTE: dict[Intent, Route] = {
    "logistics": "business",
    "order": "business",
    "product_consult": "knowledge",
    "refund_return": "refund",
    "after_sales": "refund",
    "complaint": "complaint",
    "chitchat": "chitchat",
    "other": "business",
}

#: 未知意图的兜底出口。宁可多走一次 Agent，也不要被当成闲聊糊弄掉。
_FALLBACK_ROUTE: Route = "business"


def route_for(intent: str) -> Route:
    """把意图字符串映射到出口；未知值回落 business。"""
    return INTENT_TO_ROUTE.get(intent, _FALLBACK_ROUTE)  # type: ignore[arg-type]


class AgentState(TypedDict, total=False):
    """图里唯一的一份状态。

    字段按用途分组；`total=False` 是因为不同出口只写自己那部分（投诉出口不会
    有 `evidence`，知识出口也不一定有工具轨迹）。不要为了「类型完整」给所有
    字段塞空值 —— 那会让日志分不清「没走到」与「走到了但结果为空」。
    """

    # -- 输入 / 会话 -----------------------------------------------------
    session_id: str
    turn_id: str
    user_message: str
    messages: Annotated[list[AnyMessage], add_messages]
    last_user_turn_id: str
    last_committed_turn_id: str

    # -- 指代消解 / 意图 -------------------------------------------------
    resolved_message: str
    coref_changed: bool
    coref_raw: str
    coref_source: Literal["llm", "passthrough", "fallback"]
    coref_reason: str
    coref_llm_calls: int
    intent: Intent
    intent_confidence: float
    intent_raw: str
    intent_source: Literal["llm", "llm_cheap", "llm_escalated", "heuristic", "rule"]
    intent_llm_calls: int
    route: Route

    # -- 退款/售后确定性子流程 --------------------------------------------
    refund_order_id: str
    refund_order_candidates: list[dict[str, Any]]
    refund_order: dict[str, Any]
    refund_order_error: str
    #: 通用政策问题（不指向某一单）：跳过选择器，但仍走扩写+政策检索。
    refund_order_optional: bool
    #: 候选生成失败，连「可选订单」都没有：直接进 Agent 说明无法确认。
    refund_order_blocked: bool
    expanded_queries: list[str]
    query_expansion_raw: str
    query_expansion_degraded: bool
    query_expansion_llm_calls: int
    refund_form: dict[str, Any]

    # -- 知识类 -----------------------------------------------------------
    retrieval_attempted: bool
    retrieval_score: float
    retrieval_degraded: bool
    retrieval_refused: bool
    evidence: list[dict[str, Any]]
    #: ch09：当轮召回片段快照（含分数），入池与 👎 回捞都用它。
    retrieval_snapshot: list[dict[str, Any]]
    #: ch09：正式闸算出的证据置信度（0~1）。
    evidence_confidence: float
    gate_passed: bool
    gate_reason: str
    #: ch09：生成前自评是否降级（模型失败/坏 JSON 时 fail-open）。
    self_check_degraded: bool

    # -- Agent ------------------------------------------------------------
    answer: str
    tool_traces: list[dict[str, Any]]
    tool_records: list[dict[str, Any]]
    pending_messages: list[AnyMessage]
    citations: list[dict[str, Any]]
    suggested_actions: list[SuggestedAction]
    ticket_type: str
    agent_steps: int
    llm_calls: int
    token_usage: dict[str, int]
    stop_reason: str
    #: ch08：模型发起、等待用户确认的建单调用 ``{"args", "call_id"}``。
    pending_ticket_call: dict[str, Any] | None
    #: ch08：出示给用户的工单预览 ``{"ticket_type", "description"}``。
    ticket_preview: dict[str, Any] | None
    context: dict[str, Any]
    #: ch07：``history_ctx`` 已为哪个 turn 写过。coref 与 intent 共用同一份
    #: 历史，按 turn 去重，避免一轮写两条。
    history_ctx_logged_turn: str

    # -- 可观测 -----------------------------------------------------------
    trace: list[str]
    turn_started_at: float
    turn_log: dict[str, Any]
    error: str | None
