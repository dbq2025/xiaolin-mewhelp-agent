"""LangGraph 图：确定性骨架 + 一个 ReAct 节点 + 退款确定性子流程。

## 节点顺序

``load_history → begin_turn → resolve_coref → classify_intent → route_intent``
之后按出口分流：

- ``knowledge``：检索 → 置信度闸 → Agent
- ``business``：Agent（含 other 澄清）
- ``refund``：订单槽位 → 确定性订单快照 → Query 扩写 → 多查询政策检索 → Agent
- ``complaint`` / ``chitchat``：固定节点

所有完成出口最终汇到 ``log_turn``；退款子流程在缺订单号时由
``refund_await_order`` 的 ``interrupt()`` 暂停，续跑后同一条路径走完。

## 依赖从 config 进，不进 state

`model` / `intent_model` / `intent_cheap_model` / `repositories` / `retriever` /
`settings` 是**每请求**依赖；把它们塞进 state 会被 checkpointer 序列化，而 DB
Session 不可跨请求共享。所以它们走 ``config["configurable"]``，state 只放可序列化
的业务数据。

## 每个节点都写 trace

`trace` 是验收的直接证据（能看到退款子流程真的按顺序走过）。节点返回**完整**的
trace 列表（覆盖写），而不是用 reducer 追加 —— `begin_turn` 才能把上一轮的 trace 清掉。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt

from app.agent.checkpoint import memory_checkpointer
from app.agent.coref import CorefResolver
from app.agent.ctxlog import AppLogger, log_history_ctx, log_model_ctx
from app.agent.gate import evaluate_gate
from app.agent.intent import IntentClassifier
from app.agent.logging import TurnLogger, build_turn_log
from app.agent.persist import persist_turn
from app.agent.prompts import (
    CHITCHAT_REPLY,
    COMPLAINT_SOOTHE,
    build_agent_system_prompt,
)
from app.agent.query_expansion import QueryExpander
from app.agent.react import run_react_agent
from app.agent.refund_orders import (
    build_refund_form,
    extract_order_id,
    is_general_refund_policy,
    list_demo_orders,
)
from app.agent.state import AgentState, route_for
from app.chat import _message_text
from app.context import assemble_context, select_summary_texts
from app.context.assemble import build_legacy_context
from app.context.budget import compute_budget, format_budget_report_parts
from app.context.layers import plan_layers, row_to_message
from app.context.tokens import counter_from_settings
from app.knowledge.citations import (
    Evidence,
    evidence_from_payload,
    evidence_payload,
)
from app.prompts import REFUSAL_TEXT
from app.tools.audit import AuditEntry, write_audit
from app.tools.executor import execute_tool, to_tool_message_content
from app.tools.registry import build_registry
from app.tools.business import query_order, query_product

logger = logging.getLogger(__name__)

__all__ = ["build_agent_graph"]

#: 检索闸失败时的兜底话术。
_FALLBACK_TEXT = REFUSAL_TEXT


@dataclass(frozen=True)
class GraphDeps:
    """每请求依赖。放 config，不放 state（不可序列化）。"""

    model: Any
    intent_model: Any
    intent_cheap_model: Any
    repositories: Any
    retriever: Any
    settings: Any
    summarizer: Any = None
    """后台摘要任务（``app.agent.summarizer.ContextSummarizer``）。

    默认 ``None``：ch07 之前的图测试与 ``app/chat.py`` 的调用点都只传六个依赖，
    加一个必填字段会让它们全部 TypeError。为 ``None`` 时层 2 只记账不压缩，
    ``summary_due`` 照样会出现在 ``state["context"]`` 里 —— 可观测性不能因为
    没接摘要器就消失。
    """

    mcp_provider: Any = None
    """进程级 MCP 工具发现器（``app.tools.mcp_client.McpToolProvider``）。

    默认 ``None`` 表示本实例不接 MCP：图只登记内置工具，行为与 ch07 一致。
    """


def _deps(config: RunnableConfig) -> GraphDeps:
    configurable = (config or {}).get("configurable") or {}
    return GraphDeps(
        model=configurable.get("model"),
        intent_model=configurable.get("intent_model"),
        intent_cheap_model=configurable.get("intent_cheap_model"),
        repositories=configurable.get("repositories"),
        retriever=configurable.get("retriever"),
        settings=configurable.get("settings"),
        summarizer=configurable.get("summarizer"),
        mcp_provider=configurable.get("mcp_provider"),
    )


async def _turn_registry(deps: GraphDeps, repositories: Any, retriever: Any) -> Any:
    """构造本轮工具表并刷新 MCP 清单。

    ``mcp_provider`` 为 ``None``（未接 MCP / 测试注入）时只登记内置工具，
    ``refresh_mcp`` 是空操作。MCP 发现失败由 provider 兜住，最坏退回旧清单，
    绝不会让这一轮聊天失败。
    """
    provider = deps.mcp_provider
    registry = build_registry(
        repositories,
        retriever,
        mcp_provider=provider,
        enable_mcp=provider is not None,
    )
    await registry.refresh_mcp()
    return registry


def _trace(state: Mapping[str, Any], node: str) -> list[str]:
    return [*list(state.get("trace") or []), node]


def _emit(writer: StreamWriter | None, payload: dict[str, Any]) -> None:
    if writer is not None:
        writer(payload)


def _stream_writer() -> StreamWriter | None:
    """取当前节点的 stream writer；非流式调用（如 `ainvoke`）时返回 None。

    **为什么不用参数注入**：实测在 `StateGraph` 里给异步节点加
    `writer: StreamWriter | None = None` 参数时，`writer` 仍是 None（token 帧
    不会发出）。官方推荐是节点内 `get_stream_writer()`；它在没有流式消费者时
    会抛错，所以这里兜一层。
    """
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - 没有流式消费者时不该影响节点执行
        return None


async def _load_history(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("messages"):
        return {"trace": _trace(state, "load_history")}
    repositories = _deps(config).repositories
    history = await repositories.messages.history_as_messages(state["session_id"])
    return {"messages": history, "trace": _trace(state, "load_history")}


def _reset_turn(state: Mapping[str, Any], node: str) -> dict[str, Any]:
    """清掉上一轮的派生字段；`messages` 不在这里清（那是跨轮上下文）。"""
    return {
        "trace": [node],
        "turn_started_at": time.time(),
        "answer": "",
        "pending_messages": [],
        "tool_traces": [],
        "tool_records": [],
        "citations": [],
        "suggested_actions": [],
        "ticket_type": "售后",
        "agent_steps": 0,
        "llm_calls": 0,
        "intent_llm_calls": 0,
        "coref_raw": "",
        "coref_source": "",
        "coref_reason": "",
        "coref_llm_calls": 0,
        "refund_order_id": "",
        "refund_order_candidates": [],
        "refund_order": {},
        "refund_order_error": "",
        "refund_order_optional": False,
        "refund_order_blocked": False,
        "expanded_queries": [],
        "query_expansion_raw": "",
        "query_expansion_degraded": False,
        "query_expansion_llm_calls": 0,
        "refund_form": {},
        "token_usage": {},
        "stop_reason": "",
        "retrieval_attempted": False,
        "retrieval_score": 0.0,
        "retrieval_degraded": False,
        "retrieval_refused": False,
        "evidence": [],
        "gate_passed": False,
        "gate_reason": "",
        "context": {},
        "history_ctx_logged_turn": "",
        "turn_log": {},
        "error": None,
    }


async def _begin_turn(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    turn_id = str(state.get("turn_id") or "")
    if state.get("last_user_turn_id") == turn_id:
        # 同一 turn 的重试：用户消息已经追加过，不要再追加一次。
        return _reset_turn(state, "begin_turn")
    return {
        **_reset_turn(state, "begin_turn"),
        "messages": [HumanMessage(content=state["user_message"])],
        "last_user_turn_id": turn_id,
    }


def _history_before_current(state: Mapping[str, Any]) -> list[Any]:
    """当前用户消息之前的消息；重放时最后一条可能已经是上一轮 AI。"""
    messages = list(state.get("messages") or [])
    current = str(state.get("user_message") or "")
    if (
        messages
        and isinstance(messages[-1], HumanMessage)
        and str(messages[-1].content) == current
    ):
        return messages[:-1]
    return messages


async def _log_history_ctx(
    state: AgentState, config: RunnableConfig, history: Sequence[Any]
) -> dict[str, Any]:
    """把前置节点（指代消解 / 意图识别）看到的历史原样留痕。

    为什么单独一条而不是复用 ``model_ctx``：``model_ctx`` 只在真正进 ``agent``
    节点的轮次才有，闲聊、投诉、兜底这些固定回复轮根本不调主力模型。可指代消解
    与意图识别是**每轮都跑**的，它们看到的历史同样决定这一轮怎么走 —— 只记
    ``model_ctx`` 的话，闲聊轮「模型看到了什么」就成了黑洞。

    同一轮只写一条：两个节点读的是同一份历史，写两遍只会把日志读花。按
    ``turn_id`` 去重，而不是按节点名。
    """
    settings = _deps(config).settings
    turn_id = str(state.get("turn_id") or "")
    if turn_id and state.get("history_ctx_logged_turn") == turn_id:
        return {}
    log_history_ctx(
        logger=AppLogger(getattr(settings, "app_log_path", "log/app.log")),
        session_id=str(state.get("session_id") or ""),
        turn=turn_id,
        summary_texts=await _summary_texts(state, config),
        messages=history,
    )
    return {"history_ctx_logged_turn": turn_id}


async def _summary_texts(state: AgentState, config: RunnableConfig) -> list[str]:
    """该会话已有的全部梗概正文（按段号升序）。读不到就当没有。"""
    repositories = _deps(config).repositories
    session_id = str(state.get("session_id") or "")
    if not session_id:
        return []
    conversation_id = await repositories.conversations.find_id(session_id)
    if conversation_id is None:
        return []
    return list(await repositories.summaries.latest_texts(conversation_id))


async def _resolve_coref(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    settings = deps.settings
    resolver = CorefResolver(
        deps.model,
        timeout=float(getattr(settings, "coref_timeout", 5.0)),
        history_limit=int(getattr(settings, "coref_history_messages", 8)),
        enabled=bool(getattr(settings, "coref_enabled", True)),
    )
    history = _history_before_current(state)
    # 先把历史留痕，再调模型：即使本轮后面超时/抛错，日志里也有迹可循。
    ctx_patch = await _log_history_ctx(state, config, history)
    result = await resolver.resolve(state["user_message"], history)
    _emit(
        _stream_writer(),
        {
            "type": "coref",
            "original": state["user_message"],
            "resolved": result.resolved,
            "changed": result.changed,
        },
    )
    # 只要真的发过模型调用就计 1：fallback 也可能是超时/坏 JSON，计 0 会低估成本。
    calls = 1 if result.raw else 0
    return {
        **ctx_patch,
        "resolved_message": result.resolved,
        "coref_changed": result.changed,
        "coref_raw": result.raw,
        "coref_source": result.source,
        "coref_reason": result.reason,
        "coref_llm_calls": calls,
        "llm_calls": calls,
        "trace": _trace(state, "resolve_coref"),
    }


async def _classify_intent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    settings = deps.settings
    classifier = IntentClassifier(
        deps.intent_model or deps.model,
        cheap_model=deps.intent_cheap_model,
        timeout=float(getattr(settings, "intent_timeout", 5.0)),
        confidence_threshold=float(
            getattr(settings, "intent_confidence_threshold", 0.6)
        ),
        cost_mode=bool(
            getattr(settings, "intent_cost_mode", False)
            and deps.intent_cheap_model is not None
        ),
    )
    result = await classifier.classify(state["resolved_message"])
    _emit(
        _stream_writer(),
        {
            "type": "intent",
            "intent": result.intent,
            "confidence": result.confidence,
            "source": result.source,
        },
    )
    total_calls = int(state.get("coref_llm_calls") or 0) + result.llm_calls
    return {
        # 兜底去重：正常情况下 coref 已经写过这一轮了，这里返回空补丁。
        # 但「coref 被关掉又只跑了意图」这种配置下，这一句保证仍然有痕迹。
        **(await _log_history_ctx(state, config, _history_before_current(state))),
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_raw": result.raw,
        "intent_source": result.source,
        "intent_llm_calls": result.llm_calls,
        # 非 Agent 出口（投诉/闲聊/兜底）也要把 coref + 意图调用记进总数。
        "llm_calls": total_calls,
        "trace": _trace(state, "classify_intent"),
    }


async def _route_intent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    route = route_for(state.get("intent", ""))
    ticket_type = "投诉" if route == "complaint" else "售后"
    return {
        "route": route,
        "ticket_type": ticket_type,
        "trace": _trace(state, "route_intent"),
    }


async def _knowledge_retrieve(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    retriever = _deps(config).retriever
    outcome = await retriever.retrieve_only(state["resolved_message"])
    evidence = list(evidence_payload(outcome.evidence))
    top_score = float(outcome.top_score)
    if not top_score and outcome.evidence:
        top_score = float(outcome.evidence[0].score)
    return {
        "retrieval_attempted": True,
        "retrieval_refused": bool(outcome.refused),
        "retrieval_degraded": bool(outcome.rerank_degraded),
        "retrieval_score": top_score,
        "evidence": evidence,
        "gate_reason": outcome.refusal_reason or "",
        "trace": _trace(state, "knowledge_retrieve"),
    }


async def _confidence_gate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    settings = deps.settings
    repositories = deps.repositories
    decision = evaluate_gate(
        refused=bool(state.get("retrieval_refused")),
        evidence_count=len(state.get("evidence") or []),
        top_score=float(state.get("retrieval_score") or 0.0),
        rerank_degraded=bool(state.get("retrieval_degraded")),
        min_score=float(getattr(settings, "rag_min_rerank_score", 0.3)),
    )

    if not decision.passed:
        reason = state.get("gate_reason") or decision.reason
        try:
            conversation_id = await repositories.conversations.upsert_by_session(
                state["session_id"], "anonymous"
            )
            await repositories.low_confidence.add(
                conversation_id,
                state["user_message"],
                "retrieval_low_conf",
                reason,
            )
        except Exception:  # noqa: BLE001 - 入池失败不该吞掉兜底话术
            logger.exception("低置信度入池失败（已忽略，仍然回兜底话术）")

    return {
        "gate_passed": decision.passed,
        "gate_reason": state.get("gate_reason") or decision.reason,
        "trace": _trace(state, "confidence_gate"),
    }


# ---------------------------------------------------------------------------
# 退款 / 售后确定性子流程
# ---------------------------------------------------------------------------


def _order_snapshot(order_id: str) -> dict[str, Any]:
    """确定性查订单 + 商品；失败返回空 dict，由调用方记错误。"""
    try:
        order = json.loads(query_order(order_id))
        product = json.loads(query_product(str(order["product_id"])))
    except Exception:  # noqa: BLE001 - 演示工具本应永不抛，坏 JSON 也不该打断聊天
        logger.warning("订单快照查询失败（order_id=%r）", order_id, exc_info=True)
        return {}

    name = str(product.get("name") or "商品")
    model = str(product.get("model") or "")
    return {
        "order_id": order_id,
        "status": str(order.get("status") or ""),
        "amount_yuan": str(order.get("amount_yuan") or ""),
        "product_id": str(order.get("product_id") or ""),
        "product_name": name,
        "product_model": model,
        "created_at": str(order.get("created_at") or ""),
        "title": f"{name} {model}".strip(),
    }


async def _refund_prepare_order(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """确定性抽订单号；抽不到就发卡片，交给下一个节点 interrupt。"""
    if state.get("refund_order_id"):
        return {"trace": _trace(state, "refund_prepare_order")}

    resolved = state.get("resolved_message") or ""
    order_id = extract_order_id(resolved)
    if order_id:
        return {
            "refund_order_id": order_id,
            "trace": _trace(state, "refund_prepare_order"),
        }

    # 通用政策问题（「退货政策是什么」）不指向某一单，跳过选择器但仍走
    # 扩写 + 政策检索；「这个能退吗」/「我的订单能退吗」仍必须选单。
    if is_general_refund_policy(resolved):
        return {
            "refund_order_optional": True,
            "trace": _trace(state, "refund_prepare_order"),
        }

    settings = _deps(config).settings
    try:
        candidates = list_demo_orders(
            state["session_id"], int(getattr(settings, "refund_order_count", 3))
        )
    except Exception:  # noqa: BLE001 - 候选生成失败不该打断聊天
        logger.exception("订单选择器候选生成失败")
        return {
            "refund_order_error": "订单列表暂时不可用",
            "refund_order_blocked": True,
            "trace": _trace(state, "refund_prepare_order"),
        }

    _emit(
        _stream_writer(),
        {
            "type": "order_picker",
            "turn_id": state.get("turn_id"),
            "orders": candidates,
        },
    )
    return {
        "refund_order_candidates": candidates,
        "refund_order_error": "" if candidates else "订单列表暂时不可用",
        "refund_order_blocked": not candidates,
        "trace": _trace(state, "refund_prepare_order"),
    }


async def _refund_await_order(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """缺订单号时 interrupt；用户点选后从 checkpoint 续跑。"""
    if state.get("refund_order_optional"):
        return {"trace": _trace(state, "refund_await_order")}

    order_id = str(state.get("refund_order_id") or "")
    candidates = list(state.get("refund_order_candidates") or [])

    if not order_id:
        if not candidates:
            return {
                "refund_order_error": "订单列表暂时不可用",
                "refund_order_blocked": True,
                "trace": _trace(state, "refund_await_order"),
            }

        payload = {
            "kind": "order_picker",
            "turn_id": state.get("turn_id"),
            "orders": candidates,
        }
        valid_ids = {str(item.get("order_id")) for item in candidates}
        for _ in range(3):
            picked = interrupt(payload)
            if isinstance(picked, Mapping):
                picked_id = str(picked.get("order_id") or "")
            else:
                picked_id = str(picked or "")
            if picked_id in valid_ids:
                order_id = picked_id
                break
            payload = {**payload, "error": "请从列表里选择订单"}
        else:
            return {
                "refund_order_error": "没有选择有效订单",
                "trace": _trace(state, "refund_await_order"),
            }

    snapshot = _order_snapshot(order_id)
    if not snapshot:
        return {
            "refund_order_id": order_id,
            "refund_order_error": "订单信息查询失败",
            "trace": _trace(state, "refund_await_order"),
        }

    return {
        "refund_order_id": order_id,
        "refund_order": snapshot,
        "refund_order_error": "",
        "trace": _trace(state, "refund_await_order"),
    }


async def _refund_expand_queries(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    if state.get("expanded_queries"):
        return {"trace": _trace(state, "refund_expand_queries")}

    deps = _deps(config)
    settings = deps.settings
    resolved = state.get("resolved_message") or state.get("user_message") or ""
    if not bool(getattr(settings, "query_expansion_enabled", True)):
        return {
            "expanded_queries": [resolved],
            "query_expansion_raw": "",
            "query_expansion_degraded": True,
            "query_expansion_llm_calls": 0,
            "trace": _trace(state, "refund_expand_queries"),
        }

    expander = QueryExpander(
        deps.model,
        timeout=float(getattr(settings, "query_expansion_timeout", 5.0)),
        max_queries=int(getattr(settings, "query_expansion_max_queries", 4)),
        enabled=True,
    )
    result = await expander.expand(resolved, order=state.get("refund_order") or None)
    queries = list(result.queries) or [resolved]
    return {
        "expanded_queries": queries,
        "query_expansion_raw": result.raw,
        "query_expansion_degraded": result.degraded,
        "query_expansion_llm_calls": 1,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "trace": _trace(state, "refund_expand_queries"),
    }


def _merge_evidence(
    outcomes: list[Any], *, top_n: int
) -> tuple[tuple[Evidence, ...], float, bool, bool]:
    """兼容只有 retrieve_only 的旧 fake：按 chunk_id 去重取最高分并重新编号。"""
    merged: dict[int, Evidence] = {}
    degraded = False
    for outcome in outcomes:
        degraded = degraded or bool(getattr(outcome, "rerank_degraded", False))
        for item in getattr(outcome, "evidence", ()) or ():
            current = merged.get(item.chunk_id)
            if current is None or float(item.score) > float(current.score):
                merged[item.chunk_id] = item
    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_n]
    renumbered = tuple(replace(item, n=index) for index, item in enumerate(ordered, 1))
    top_score = float(renumbered[0].score) if renumbered else 0.0
    return renumbered, top_score, bool(not renumbered), degraded


async def _retrieve_policies_fallback(
    retriever: Any, queries: list[str], *, top_n: int
) -> tuple[tuple[Evidence, ...], float, bool, bool, str]:
    outcomes = []
    failures = 0
    for query in queries:
        try:
            outcomes.append(await retriever.retrieve_only(query))
        except Exception:  # noqa: BLE001 - 单条失败不拖垮整轮
            failures += 1
            logger.warning("政策检索单条失败（query=%r）", query, exc_info=True)
    evidence, top_score, empty, degraded = _merge_evidence(outcomes, top_n=top_n)
    if empty:
        reason = "多查询检索全部失败" if failures else "检索没有命中任何知识块"
    else:
        reason = ""
    return evidence, top_score, empty, degraded, reason


async def _refund_retrieve_policies(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    deps = _deps(config)
    settings = deps.settings
    retriever = deps.retriever
    queries = list(state.get("expanded_queries") or [state["resolved_message"]])
    top_n = int(getattr(settings, "rag_rerank_top_n", 10))
    min_score = float(getattr(settings, "rag_min_rerank_score", 0.3))

    if hasattr(retriever, "retrieve_many"):
        try:
            outcome = await retriever.retrieve_many(queries)
        except Exception:  # noqa: BLE001 - 政策检索失败也要进 Agent 明说无法确认
            logger.exception("多查询政策检索失败，按空证据继续")
            evidence = ()
            top_score = 0.0
            refused = True
            degraded = True
            reason = "多查询检索失败"
        else:
            evidence = outcome.evidence
            top_score = float(outcome.top_score)
            refused = bool(outcome.refused)
            degraded = bool(outcome.rerank_degraded)
            reason = outcome.refusal_reason or ""
    else:
        evidence, top_score, empty, degraded, reason = await _retrieve_policies_fallback(
            retriever, queries, top_n=top_n
        )
        refused = empty
        if evidence and not degraded and top_score < min_score:
            refused = True
            reason = (
                f"最高重排分 {top_score:.3f} 低于阈值 {min_score}，证据相关性不足"
            )

    return {
        "retrieval_attempted": True,
        "retrieval_refused": refused,
        "retrieval_degraded": degraded,
        "retrieval_score": top_score,
        "evidence": list(evidence_payload(evidence)),
        "gate_reason": reason,
        "trace": _trace(state, "refund_retrieve_policies"),
    }


async def _load_rows_and_anchors(
    state: AgentState, repositories: Any
) -> tuple[list[Any], int | None, int | None, int | None]:
    """取「带数据库 id 的历史行」与两个锚点。

    返回 ``(rows, conversation_id, summary_upto, layer1_from)``。

    为什么非要走数据库：三层划分的边界是两个 ``messages.id``，而
    ``state["messages"]`` 里的 ``BaseMessage`` 只有内容没有 id。拿不到 id 就
    无从判断某条属于哪一层，只能整段算层 1（那就是 ch01 的行为）。

    库里查不到会话（首轮、或 State 里有但还没落库）时返回空 rows，调用方
    退回 ``state["messages"]`` + legacy 裁剪：**宁可少压，不能把上下文弄丢**。
    """
    session_id = str(state.get("session_id") or "")
    if not session_id:
        return [], None, None, None
    conversation_id = await repositories.conversations.find_id(session_id)
    if conversation_id is None:
        return [], None, None, None
    rows = await repositories.messages.list_for_conversation(conversation_id)
    anchors = await repositories.conversations.get_anchors(conversation_id)
    if anchors is None:
        return list(rows), conversation_id, None, None
    return list(rows), conversation_id, anchors.summary_upto_msg_id, anchors.layer1_from_msg_id


async def _current_question(state: Mapping[str, Any]) -> str:
    """本轮要发给模型的问题文本（已做指代消解）。"""
    messages = list(state.get("messages") or [])
    question = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if question is None:
        return str(state.get("user_message") or "")
    return str(state.get("resolved_message") or _message_text(question))


def _state_history_messages(state: Mapping[str, Any]) -> list[Any]:
    """``state["messages"]`` 里当前问题之前的消息（无 id 的兜底历史）。"""
    messages = list(state.get("messages") or [])
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[:index]
    return messages[:-1] if messages else []


async def _run_agent_node(
    state: AgentState, config: RunnableConfig, *, refund_mode: bool
) -> dict[str, Any]:
    deps = _deps(config)
    model = deps.model
    repositories = deps.repositories
    retriever = deps.retriever
    settings = deps.settings
    summarizer = deps.summarizer
    messages = list(state.get("messages") or [])
    if not messages:
        raise RuntimeError("agent 节点没有可用的消息历史")

    resolved_question = await _current_question(state)
    model_question = HumanMessage(content=resolved_question)

    evidence = evidence_from_payload(state.get("evidence") or [])
    system = SystemMessage(
        content=build_agent_system_prompt(
            state["session_id"],
            state.get("evidence") or [],
            resolved_question=resolved_question,
            refund_order=state.get("refund_order") or None,
            refund_order_optional=bool(state.get("refund_order_optional")),
            refund_mode=refund_mode,
            retrieval_attempted=bool(state.get("retrieval_attempted")),
            other_intent=(
                state.get("route") == "business" and state.get("intent") == "other"
            ),
        )
    )

    budget = compute_budget(settings)
    counter = counter_from_settings(settings)
    rows, conversation_id, summary_upto, layer1_from = await _load_rows_and_anchors(
        state, repositories
    )
    summary_texts: list[str] = []
    if conversation_id is not None:
        summary_texts = await repositories.summaries.latest_texts(conversation_id)

    # 梗概是「一段段攒」的，而预算里的 ``SUMMARY_RESERVE_TOKENS`` 是固定值。
    # 攒够几段就会超出那份预留，历史预算却仍按 500 算 —— 总量会悄悄突破窗口。
    # 这里按预留截一遍：头尾都留、丢中间，丢了几段如实记账。
    summary_texts, summary_dropped = select_summary_texts(
        summary_texts,
        budget_tokens=budget.summary_reserve,
        counter=counter,
    )

    if rows:
        plan = plan_layers(
            rows,
            summary_upto=summary_upto,
            layer1_from=layer1_from,
            layer1_budget=budget.layer1_budget,
            layer2_budget=budget.layer2_budget,
            assistant_preview_chars=int(
                getattr(settings, "assistant_preview_chars", 60)
            ),
            counter=counter,
        )
        ctx = assemble_context(
            system_message=system,
            question=model_question,
            layer2=[row_to_message(row) for row in plan.layer2],
            layer1=[row_to_message(row) for row in plan.layer1],
            summary_text="\n".join(summary_texts),
            budget_tokens=budget.history_budget,
            counter=counter,
            summary_upto_msg_id=plan.summary_upto_msg_id,
            layer1_from_msg_id=plan.layer1_from_msg_id,
            degraded=plan.degraded,
            summary_due=plan.summary_due,
            trimmed_messages=plan.dropped_layer1_rows,
            summary_dropped_segments=summary_dropped,
        )
    else:
        # 库里没有带 id 的历史：退回 ch01 的单层裁剪。这条路径**不触发摘要**
        # （没有 id 就没有边界），但统计字段仍然如实填。
        plan = None
        legacy_history = _state_history_messages(state)
        ctx = build_legacy_context(
            history=legacy_history,
            question=model_question,
            system_message=system,
            budget_tokens=int(getattr(settings, "context_token_budget", 2000)),
        )

    # ---- 可观测：调模型前把实际发出去的上下文原样落盘 ----
    ctx_logger = AppLogger(getattr(settings, "app_log_path", "log/app.log"))
    log_model_ctx(
        logger=ctx_logger,
        session_id=str(state.get("session_id") or ""),
        turn=str(state.get("turn_id") or ""),
        messages=ctx.messages,
        layer1_from_msg_id=ctx.layer1_from_msg_id,
        summary_upto_msg_id=ctx.summary_upto_msg_id,
        budget=budget,
        layer1_tokens=plan.layer1_tokens if plan is not None else ctx.used_tokens,
        layer2_tokens=plan.layer2_tokens if plan is not None else 0,
        summary_texts=summary_texts,
    )

    # 降级了就把锚点落库：下一轮直接按新边界取历史，不用重新算一遍。
    # 只前进不后退由 repositories.set_anchors 的守卫强制。
    if plan is not None and conversation_id is not None and ctx.layer1_from_msg_id:
        if ctx.layer1_from_msg_id != layer1_from:
            await repositories.conversations.set_anchors(
                conversation_id=conversation_id,
                layer1_from_msg_id=ctx.layer1_from_msg_id,
            )
            # 这一行是验收第 2 条的抓手：`grep "layer1 降级"` 就能看到
            # 「层 1 从哪条挪到哪条、丢掉几行、预算是多少」。
            # 只打预算表的话，日志里看不出边界到底动没动。
            #
            # 同时落 `log/app.log`：验收要求「日志里能看到完整级联」，
            # 而级联的其余几环（model_ctx / summary trigger / summary done）
            # 都写在那个文件里。只走 stdlib logger 的话，grep 同一个文件
            # 只能看到摘要那两环，降级那一环永远「未命中」。
            detail = (
                f"ch07 layer1 降级 {layer1_from if layer1_from is not None else '-'}"
                f"→{ctx.layer1_from_msg_id} session={state.get('session_id') or ''}"
                f" turn={state.get('turn_id') or ''}"
                f" dropped={plan.dropped_layer1_rows}"
                f" layer1_tokens={plan.layer1_tokens}"
                f" layer1_budget={budget.layer1_budget}"
            )
            logger.info(
                "%s 预算表 %s",
                detail,
                format_budget_report_parts(
                    window=budget.model_context_window,
                    overhead=budget.overhead,
                    history=budget.history_budget,
                    layer1=budget.layer1_budget,
                    layer2=budget.layer2_budget,
                ),
            )
            ctx_logger.write(detail)

    # ---- 层 2 超预算：调度后台摘要，**不 await** ----
    # 顺序很重要：调度放在调模型之前，摘要任务才有机会与本轮回复并行跑。
    # 失败只吞掉留痕 —— 摘要晚一轮不影响这一轮的回答。
    # 先判 `summary_due` 再调：不该压缩时连任务都不该建，否则每轮都留一条
    # trigger 日志，日志里就分不出「真的超了」和「每轮例行公事」。
    if (
        plan is not None
        and plan.summary_due
        and summarizer is not None
        and conversation_id is not None
    ):
        # ``expected_layer1_from`` 必须是**本次降级落库之后**的锚点。
        #
        # 这里踩过一次：上面刚把 `ctx.layer1_from_msg_id` 写进库，这里却传了
        # 规划之前的旧值 `layer1_from`，于是后台任务复核时**永远**看到
        # 「anchor moved: expected 57 now 59」，每轮都 skip，摘要一次也不会落。
        # 乐观并发要挡的是「别的进程在我算完之后又挪了锚点」，不是我们自己
        # 刚刚那次降级。
        expected_anchor = (
            ctx.layer1_from_msg_id if ctx.layer1_from_msg_id else layer1_from
        )
        summarizer.schedule(
            session_id=str(state.get("session_id") or ""),
            conversation_id=conversation_id,
            plan=plan,
            layer2_budget=budget.layer2_budget,
            expected_layer1_from=expected_anchor,
            background="\n".join(summary_texts),
        )

    outcome = await run_react_agent(
        model=model,
        messages=ctx.messages,
        repositories=repositories,
        retriever=retriever,
        include_faq=(state.get("route") == "business"),
        include_order=not refund_mode,
        evidence=evidence,
        max_steps=int(getattr(settings, "agent_max_steps", 6)),
        token_budget=int(getattr(settings, "agent_token_budget", 6000)),
        timeout=float(getattr(settings, "tool_timeout_seconds", 3.0)),
        max_retries=int(getattr(settings, "tool_max_retries", 1)),
        emit=lambda payload: _emit(_stream_writer(), payload),
        registry=await _turn_registry(deps, repositories, retriever),
        conversation_id=conversation_id,
    )

    payload: dict[str, Any] = {
        "answer": outcome.answer,
        "pending_messages": outcome.pending_messages,
        "tool_traces": outcome.tool_traces,
        "tool_records": outcome.tool_records,
        "citations": outcome.citations,
        "suggested_actions": outcome.suggested_actions,
        "agent_steps": outcome.agent_steps,
        "llm_calls": outcome.llm_calls + int(state.get("llm_calls") or 0),
        "token_usage": outcome.token_usage,
        "stop_reason": outcome.stop_reason,
        "pending_ticket_call": outcome.pending_ticket_call,
        "context": {
            "kept_messages": ctx.kept_messages,
            "trimmed_messages": ctx.trimmed_messages,
            "budget_tokens": ctx.budget_tokens,
            "used_tokens": ctx.used_tokens,
            "system_prompt_tokens": ctx.system_prompt_tokens,
            "summary_tokens": ctx.summary_tokens,
            "layer1_messages": ctx.layer1_messages,
            "layer2_messages": ctx.layer2_messages,
            "layer1_tokens": plan.layer1_tokens if plan is not None else 0,
            "layer2_tokens": plan.layer2_tokens if plan is not None else 0,
            "history_budget": budget.history_budget,
            "layer1_budget": budget.layer1_budget,
            "layer2_budget": budget.layer2_budget,
            "summary_messages": len(summary_texts),
            "layer1_from_msg_id": ctx.layer1_from_msg_id,
            "summary_upto_msg_id": ctx.summary_upto_msg_id,
            "degraded": ctx.degraded,
            "summary_due": ctx.summary_due,
        },
        "trace": _trace(state, "refund_agent" if refund_mode else "agent"),
    }
    if refund_mode:
        order = state.get("refund_order") or {}
        payload["refund_form"] = build_refund_form(order) if order else {}
        payload["ticket_type"] = "售后"
    return payload


async def _agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    return await _run_agent_node(state, config, refund_mode=False)


async def _refund_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    return await _run_agent_node(state, config, refund_mode=True)


async def _ticket_await_confirm(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """出示工单预览，等用户点「确认提交」/「取消」。

    ``interrupt()`` 会让本节点在恢复时**从头重跑**，所以 ``interrupt`` 之前的
    代码不能有副作用：这里只读 state、拼预览。真正的写库在拿到确认值之后，
    且通过执行引擎（``confirmed=True``）走同一套校验与审计。
    """
    deps = _deps(config)
    repositories = deps.repositories
    settings = deps.settings
    pending = state.get("pending_ticket_call") or {}
    args = dict(pending.get("args") or {})
    call_id = str(pending.get("call_id") or "")
    preview = {
        "ticket_type": str(args.get("ticket_type") or state.get("ticket_type") or "售后"),
        "description": str(args.get("description") or ""),
    }

    started = time.monotonic()
    decision = interrupt(
        {
            "kind": "ticket_preview",
            "turn_id": state.get("turn_id"),
            "call_id": call_id,
            "preview": preview,
        }
    )
    # 用户在预览卡片上停留的时间：取消路径也要进审计，否则「为什么没建」
    # 只能看到一条没有耗时的拒绝记录。
    waited_ms = int((time.monotonic() - started) * 1000)
    confirmed = bool(decision.get("confirmed")) if isinstance(decision, dict) else False
    trace = _trace(state, "ticket_await_confirm")
    pending_messages = list(state.get("pending_messages") or [])

    conversation_id: int | None = None
    try:
        conversation_id = await repositories.conversations.find_id(
            str(state.get("session_id") or "")
        )
    except Exception:  # noqa: BLE001 - 审计拿不到会话 id 也要能记
        conversation_id = None

    if not confirmed:
        await write_audit(
            repositories,
            AuditEntry(
                tool_name="create_ticket",
                tool_source="builtin",
                status="权限拒绝",
                tool_call_id=call_id or None,
                conversation_id=conversation_id,
                args=args,
                error="用户取消建单，未执行",
                duration_ms=waited_ms,
            ),
        )
        answer = "已取消建单，没有创建工单。"
        pending_messages.append(
            ToolMessage(content="用户取消建单，未执行。", tool_call_id=call_id)
        )
        pending_messages.append(AIMessage(content=answer))
        _emit(_stream_writer(), {"type": "token", "delta": answer})
        return {
            "answer": answer,
            "pending_messages": pending_messages,
            "pending_ticket_call": None,
            "ticket_preview": preview,
            "stop_reason": "ticket_cancelled",
            "trace": trace,
        }

    result = await execute_tool(
        "create_ticket",
        args,
        registry=build_registry(repositories, deps.retriever),
        repositories=repositories,
        retriever=deps.retriever,
        session_id=str(state.get("session_id") or "") or None,
        conversation_id=conversation_id,
        tool_call_id=call_id or None,
        timeout=float(getattr(settings, "tool_timeout_seconds", 3.0)),
        max_retries=int(getattr(settings, "tool_max_retries", 1)),
        confirmed=True,
    )

    answer = "建工单没有成功，请稍后再试或转人工。"
    if result.ok and result.data:
        try:
            answer = str(json.loads(result.data).get("message") or answer)
        except (ValueError, TypeError, AttributeError):
            answer = answer
    pending_messages.append(
        ToolMessage(content=to_tool_message_content(result), tool_call_id=call_id)
    )
    pending_messages.append(AIMessage(content=answer))
    _emit(_stream_writer(), {"type": "token", "delta": answer})
    return {
        "answer": answer,
        "pending_messages": pending_messages,
        "pending_ticket_call": None,
        "ticket_preview": preview,
        "stop_reason": "ticket_created" if result.ok else "ticket_failed",
        "tool_traces": [*list(state.get("tool_traces") or []), result.summary()],
        "trace": trace,
    }


def _fixed_node(
    node: str,
    text: str,
    *,
    actions: list[str] | None = None,
    stop_reason: str = "fixed",
    ticket_type: str | None = None,
):
    async def run(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        _emit(_stream_writer(), {"type": "token", "delta": text})
        payload: dict[str, Any] = {
            "answer": text,
            "pending_messages": [AIMessage(content=text)],
            "suggested_actions": list(actions or []),
            "stop_reason": stop_reason,
            "trace": _trace(state, node),
        }
        if ticket_type is not None:
            payload["ticket_type"] = ticket_type
        return payload

    return run


async def _log_turn(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    repositories = deps.repositories
    settings = deps.settings
    turn_id = str(state.get("turn_id") or "")
    started = float(state.get("turn_started_at") or time.time())
    latency_ms = int((time.time() - started) * 1000)

    already_committed = state.get("last_committed_turn_id") == turn_id
    if not already_committed:
        await persist_turn(
            repositories=repositories,
            session_id=state["session_id"],
            user_message=state["user_message"],
            pending_messages=state.get("pending_messages") or [],
            citations=state.get("citations") or [],
            suggested_actions=state.get("suggested_actions") or [],
            ticket_type=str(state.get("ticket_type") or "售后"),
            tool_records=state.get("tool_records") or [],
        )

    record = build_turn_log(
        {**state, "trace": _trace(state, "log_turn")}, latency_ms
    )
    TurnLogger(getattr(settings, "turn_log_path", "")).write(record)

    payload: dict[str, Any] = {
        "turn_log": record,
        "trace": _trace(state, "log_turn"),
    }
    if not already_committed:
        payload["messages"] = list(state.get("pending_messages") or [])
        payload["last_committed_turn_id"] = turn_id
    return payload


def build_agent_graph(*, checkpointer: BaseCheckpointSaver | None = None):
    """编译图。测试传 `memory_checkpointer()`；生产传 SQLite saver。"""
    builder = StateGraph(AgentState)

    builder.add_node("load_history", _load_history)
    builder.add_node("begin_turn", _begin_turn)
    builder.add_node("resolve_coref", _resolve_coref)
    builder.add_node("classify_intent", _classify_intent)
    builder.add_node("route_intent", _route_intent)
    builder.add_node("knowledge_retrieve", _knowledge_retrieve)
    builder.add_node("confidence_gate", _confidence_gate)
    builder.add_node("refund_prepare_order", _refund_prepare_order)
    builder.add_node("refund_await_order", _refund_await_order)
    builder.add_node("refund_expand_queries", _refund_expand_queries)
    builder.add_node("refund_retrieve_policies", _refund_retrieve_policies)
    builder.add_node("refund_agent", _refund_agent)
    builder.add_node("agent", _agent)
    builder.add_node("ticket_await_confirm", _ticket_await_confirm)
    builder.add_node(
        "complaint",
        _fixed_node(
            "complaint",
            COMPLAINT_SOOTHE,
            actions=["transfer_human", "create_ticket"],
            ticket_type="投诉",
        ),
    )
    builder.add_node("chitchat", _fixed_node("chitchat", CHITCHAT_REPLY))
    builder.add_node(
        "fallback",
        _fixed_node("fallback", _FALLBACK_TEXT, stop_reason="gate_failed"),
    )
    builder.add_node("log_turn", _log_turn)

    builder.add_edge(START, "load_history")
    builder.add_edge("load_history", "begin_turn")
    builder.add_edge("begin_turn", "resolve_coref")
    builder.add_edge("resolve_coref", "classify_intent")
    builder.add_edge("classify_intent", "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        lambda state: state.get("route", "business"),
        {
            "knowledge": "knowledge_retrieve",
            "business": "agent",
            "refund": "refund_prepare_order",
            "complaint": "complaint",
            "chitchat": "chitchat",
        },
    )
    builder.add_edge("knowledge_retrieve", "confidence_gate")
    builder.add_conditional_edges(
        "confidence_gate",
        lambda state: "agent" if state.get("gate_passed") else "fallback",
        {"agent": "agent", "fallback": "fallback"},
    )
    builder.add_edge("refund_prepare_order", "refund_await_order")
    builder.add_conditional_edges(
        "refund_await_order",
        lambda state: (
            "refund_agent"
            if state.get("refund_order_blocked")
            else "refund_expand_queries"
        ),
        {
            "refund_agent": "refund_agent",
            "refund_expand_queries": "refund_expand_queries",
        },
    )
    builder.add_edge("refund_expand_queries", "refund_retrieve_policies")
    builder.add_edge("refund_retrieve_policies", "refund_agent")
    builder.add_conditional_edges(
        "refund_agent",
        lambda state: (
            "ticket_await_confirm"
            if state.get("pending_ticket_call")
            else "log_turn"
        ),
        {"ticket_await_confirm": "ticket_await_confirm", "log_turn": "log_turn"},
    )
    builder.add_conditional_edges(
        "agent",
        lambda state: (
            "ticket_await_confirm"
            if state.get("pending_ticket_call")
            else "log_turn"
        ),
        {"ticket_await_confirm": "ticket_await_confirm", "log_turn": "log_turn"},
    )
    builder.add_edge("ticket_await_confirm", "log_turn")
    builder.add_edge("complaint", "log_turn")
    builder.add_edge("chitchat", "log_turn")
    builder.add_edge("fallback", "log_turn")
    builder.add_edge("log_turn", END)

    return builder.compile(checkpointer=checkpointer or memory_checkpointer())
