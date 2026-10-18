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
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt

from app.agent.checkpoint import memory_checkpointer
from app.agent.coref import CorefResolver
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
from app.agent.refund_orders import build_refund_form, extract_order_id, list_demo_orders
from app.agent.state import AgentState, route_for
from app.chat import _message_text
from app.context import build_context
from app.knowledge.citations import (
    Evidence,
    evidence_from_payload,
    evidence_payload,
)
from app.prompts import REFUSAL_TEXT
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


def _deps(config: RunnableConfig) -> GraphDeps:
    configurable = (config or {}).get("configurable") or {}
    return GraphDeps(
        model=configurable.get("model"),
        intent_model=configurable.get("intent_model"),
        intent_cheap_model=configurable.get("intent_cheap_model"),
        repositories=configurable.get("repositories"),
        retriever=configurable.get("retriever"),
        settings=configurable.get("settings"),
    )


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


async def _resolve_coref(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    settings = deps.settings
    resolver = CorefResolver(
        deps.model,
        timeout=float(getattr(settings, "coref_timeout", 5.0)),
        history_limit=int(getattr(settings, "coref_history_messages", 8)),
        enabled=bool(getattr(settings, "coref_enabled", True)),
    )
    result = await resolver.resolve(
        state["user_message"], _history_before_current(state)
    )
    _emit(
        _stream_writer(),
        {
            "type": "coref",
            "original": state["user_message"],
            "resolved": result.resolved,
            "changed": result.changed,
        },
    )
    calls = 1 if result.source == "llm" else 0
    return {
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

    order_id = extract_order_id(state.get("resolved_message") or "")
    if order_id:
        return {
            "refund_order_id": order_id,
            "trace": _trace(state, "refund_prepare_order"),
        }

    settings = _deps(config).settings
    candidates = list_demo_orders(
        state["session_id"], int(getattr(settings, "refund_order_count", 3))
    )
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
        "trace": _trace(state, "refund_prepare_order"),
    }


async def _refund_await_order(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """缺订单号时 interrupt；用户点选后从 checkpoint 续跑。"""
    order_id = str(state.get("refund_order_id") or "")
    candidates = list(state.get("refund_order_candidates") or [])

    if not order_id:
        if not candidates:
            return {
                "refund_order_error": "订单列表暂时不可用",
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
        outcome = await retriever.retrieve_many(queries)
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


async def _run_agent_node(
    state: AgentState, config: RunnableConfig, *, refund_mode: bool
) -> dict[str, Any]:
    deps = _deps(config)
    model = deps.model
    repositories = deps.repositories
    retriever = deps.retriever
    settings = deps.settings
    messages = list(state.get("messages") or [])
    if not messages:
        raise RuntimeError("agent 节点没有可用的消息历史")

    # 取**最后一条 HumanMessage**，不要假设它在列表末尾：同一 turn_id 重放时
    # `begin_turn` 不会再追加用户消息，messages[-1] 会是上一轮的 AI 回答。
    question_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        len(messages) - 1,
    )
    question = messages[question_index]
    history = messages[:question_index]
    resolved_question = state.get("resolved_message") or _message_text(question)
    model_question = HumanMessage(content=resolved_question)

    evidence = evidence_from_payload(state.get("evidence") or [])
    system = SystemMessage(
        content=build_agent_system_prompt(
            state["session_id"],
            state.get("evidence") or [],
            resolved_question=resolved_question,
            refund_order=state.get("refund_order") or None,
            refund_mode=refund_mode,
            other_intent=(
                state.get("route") == "business" and state.get("intent") == "other"
            ),
        )
    )
    ctx = build_context(
        history=history,
        question=model_question,
        system_message=system,
        budget_tokens=int(getattr(settings, "context_token_budget", 2000)),
    )

    outcome = await run_react_agent(
        model=model,
        messages=ctx.messages,
        repositories=repositories,
        retriever=retriever,
        include_faq=(state.get("route") == "business" and not refund_mode),
        evidence=evidence,
        max_steps=int(getattr(settings, "agent_max_steps", 6)),
        token_budget=int(getattr(settings, "agent_token_budget", 6000)),
        timeout=float(getattr(settings, "tool_timeout_seconds", 3.0)),
        max_retries=int(getattr(settings, "tool_max_retries", 1)),
        emit=lambda payload: _emit(_stream_writer(), payload),
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
        "context": {
            "kept_messages": ctx.kept_messages,
            "trimmed_messages": ctx.trimmed_messages,
            "budget_tokens": ctx.budget_tokens,
            "used_tokens": ctx.used_tokens,
            "system_prompt_tokens": ctx.system_prompt_tokens,
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
            "refund_agent" if state.get("refund_order_error") else "refund_expand_queries"
        ),
        {
            "refund_agent": "refund_agent",
            "refund_expand_queries": "refund_expand_queries",
        },
    )
    builder.add_edge("refund_expand_queries", "refund_retrieve_policies")
    builder.add_edge("refund_retrieve_policies", "refund_agent")
    builder.add_edge("refund_agent", "log_turn")
    builder.add_edge("agent", "log_turn")
    builder.add_edge("complaint", "log_turn")
    builder.add_edge("chitchat", "log_turn")
    builder.add_edge("fallback", "log_turn")
    builder.add_edge("log_turn", END)

    return builder.compile(checkpointer=checkpointer or memory_checkpointer())
