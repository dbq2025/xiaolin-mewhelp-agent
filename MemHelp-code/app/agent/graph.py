"""LangGraph 图：确定性骨架 + 一个 ReAct 节点。

## 节点顺序

``load_history → begin_turn → resolve_coref → classify_intent → route_intent``
之后按出口分流；所有出口最终汇到 ``log_turn``。

## 依赖从 config 进，不进 state

`model` / `repositories` / `retriever` / `settings` 是**每请求**依赖；把它们塞进
state 会被 checkpointer 序列化，而 DB Session 不可跨请求共享。所以它们走
``config["configurable"]``，state 只放可序列化的业务数据。

## 每个节点都写 trace

`trace` 是验收 1 的直接证据（能看到 `knowledge_retrieve` 真的被走到）。
节点返回**完整**的 trace 列表（覆盖写），而不是用 reducer 追加 ——
`begin_turn` 才能把上一轮的 trace 清掉。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.agent.checkpoint import memory_checkpointer
from app.agent.coref import resolve_coref
from app.agent.gate import evaluate_gate
from app.agent.intent import IntentClassifier
from app.agent.logging import TurnLogger, build_turn_log
from app.agent.persist import persist_turn
from app.agent.prompts import (
    CHITCHAT_REPLY,
    COMPLAINT_SOOTHE,
    build_agent_system_prompt,
)
from app.agent.react import run_react_agent
from app.agent.state import AgentState, route_for
from app.context import build_context
from app.knowledge.citations import evidence_from_payload, evidence_payload
from app.prompts import REFUSAL_TEXT

logger = logging.getLogger(__name__)

__all__ = ["build_agent_graph"]

#: ch05 的错误兜底：gate 失败不调 Agent。
_FALLBACK_TEXT = REFUSAL_TEXT


def _trace(state: Mapping[str, Any], node: str) -> list[str]:
    return [*list(state.get("trace") or []), node]


def _deps(config: RunnableConfig) -> tuple[Any, Any, Any, Any]:
    configurable = (config or {}).get("configurable") or {}
    return (
        configurable.get("model"),
        configurable.get("repositories"),
        configurable.get("retriever"),
        configurable.get("settings"),
    )


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
    _, repositories, _, _ = _deps(config)
    history = await repositories.messages.history_as_messages(state["session_id"])
    return {"messages": history, "trace": _trace(state, "load_history")}


def _reset_turn(state: Mapping[str, Any], node: str) -> dict[str, Any]:
    """清掉上一轮的派生字段；`messages` 不在这里清（那是跨轮上下文）。"""
    return {
        "trace": [*list(state.get("trace") or []), node],
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


async def _resolve_coref(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    resolved, changed = resolve_coref(state["user_message"])
    return {
        "resolved_message": resolved,
        "coref_changed": changed,
        "trace": _trace(state, "resolve_coref"),
    }


async def _classify_intent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    model, _, _, settings = _deps(config)
    timeout = max(5.0, float(getattr(settings, "rag_rewrite_timeout", 3.0)))
    result = await IntentClassifier(model, timeout=timeout).classify(state["resolved_message"])
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_raw": result.raw,
        "intent_source": result.source,
        "trace": _trace(state, "classify_intent"),
    }


async def _route_intent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    route = route_for(state.get("intent", ""))
    return {
        "route": route,
        "ticket_type": "投诉" if route == "complaint" else "售后",
        "trace": _trace(state, "route_intent"),
    }


async def _knowledge_retrieve(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _, _, retriever, _ = _deps(config)
    outcome = await retriever.retrieve_only(state["resolved_message"])
    evidence = list(evidence_payload(outcome.evidence))
    return {
        "retrieval_attempted": True,
        "retrieval_refused": bool(outcome.refused),
        "retrieval_degraded": bool(outcome.rerank_degraded),
        "retrieval_score": float(outcome.evidence[0].score) if outcome.evidence else 0.0,
        "evidence": evidence,
        "gate_reason": outcome.refusal_reason or "",
        "trace": _trace(state, "knowledge_retrieve"),
    }


async def _confidence_gate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _, repositories, _, settings = _deps(config)
    decision = evaluate_gate(
        refused=bool(state.get("retrieval_refused")),
        evidence_count=len(state.get("evidence") or []),
        top_score=float(state.get("retrieval_score") or 0.0),
        rerank_degraded=bool(state.get("retrieval_degraded")),
        min_score=float(getattr(settings, "rag_min_rerank_score", 0.3)),
    )

    if not decision.passed:
        try:
            conversation_id = await repositories.conversations.upsert_by_session(
                state["session_id"], "anonymous"
            )
            await repositories.low_confidence.add(
                conversation_id,
                state["user_message"],
                "retrieval_low_conf",
                decision.reason,
            )
        except Exception:  # noqa: BLE001 - 入池失败不该吞掉兜底话术
            logger.exception("低置信度入池失败（已忽略，仍然回兜底话术）")

    return {
        "gate_passed": decision.passed,
        "gate_reason": decision.reason,
        "trace": _trace(state, "confidence_gate"),
    }


async def _agent(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    model, repositories, retriever, settings = _deps(config)
    messages = list(state.get("messages") or [])
    if not messages:
        raise RuntimeError("agent 节点没有可用的消息历史")

    evidence = evidence_from_payload(state.get("evidence") or [])
    system = SystemMessage(
        content=build_agent_system_prompt(state["session_id"], state.get("evidence") or [])
    )
    ctx = build_context(
        history=messages[:-1],
        question=messages[-1],
        system_message=system,
        budget_tokens=int(getattr(settings, "context_token_budget", 2000)),
    )

    outcome = await run_react_agent(
        model=model,
        messages=ctx.messages,
        repositories=repositories,
        retriever=retriever,
        include_faq=state.get("route") == "business",
        evidence=evidence,
        max_steps=int(getattr(settings, "agent_max_steps", 6)),
        token_budget=int(getattr(settings, "agent_token_budget", 6000)),
        timeout=float(getattr(settings, "tool_timeout_seconds", 3.0)),
        max_retries=int(getattr(settings, "tool_max_retries", 1)),
        emit=lambda payload: _emit(_stream_writer(), payload),
    )

    return {
        "answer": outcome.answer,
        "pending_messages": outcome.pending_messages,
        "tool_traces": outcome.tool_traces,
        "tool_records": outcome.tool_records,
        "citations": outcome.citations,
        "suggested_actions": outcome.suggested_actions,
        "agent_steps": outcome.agent_steps,
        "llm_calls": outcome.llm_calls,
        "token_usage": outcome.token_usage,
        "stop_reason": outcome.stop_reason,
        "context": {
            "kept_messages": ctx.kept_messages,
            "trimmed_messages": ctx.trimmed_messages,
            "budget_tokens": ctx.budget_tokens,
            "used_tokens": ctx.used_tokens,
            "system_prompt_tokens": ctx.system_prompt_tokens,
        },
        "trace": _trace(state, "agent"),
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
    _, repositories, _, settings = _deps(config)
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
    builder.add_edge("agent", "log_turn")
    builder.add_edge("complaint", "log_turn")
    builder.add_edge("chitchat", "log_turn")
    builder.add_edge("fallback", "log_turn")
    builder.add_edge("log_turn", END)

    return builder.compile(checkpointer=checkpointer or memory_checkpointer())
