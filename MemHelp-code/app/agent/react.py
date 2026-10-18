"""ReAct 循环 —— 图里唯一一个「会自己决定多走几步」的节点。

## 一次只执行一个工具

模型偶尔一步返回多个 tool_calls。这里只执行第一个，并把 assistant 消息里的
tool_calls 裁到只剩它 —— 与 ch02 的单轮策略一致。原因不是保守，而是：
1. 每个 tool_call 必须有对应的 ToolMessage，漏一个上游就拒绝整段历史；
2. 一次一个工具，`agent_steps` 才是真正的「ReAct 走了几步」，验收可观测；
3. 多个 `query_faq` 会带出多组 `[n]`，引用编号必然打架。

## 停止条件

- 模型不再调工具 → `final`；
- 工具轮数到 `max_steps` → `max_steps` + 转人工/建单建议；
- 累计 token 到 `token_budget` → `token_budget` + 同样的建议；
- 既没文本也没工具 → `empty_response` + 固定兜底。
上游异常**不吞**：交给 HTTP 层转 `error`，不能伪装成「答完了」。

## 流式与落库分离

节点边流边发 token，但本轮产生的消息先放 `pending_messages`；由 `log_turn`
在 MySQL 写成功之后才提交进 state。这样上游中途挂掉不会留下半截回答。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from app.agent.evidence import citations_for_answer, evidence_from_tool_result
from app.agent.prompts import AGENT_EMPTY_TEXT, AGENT_HANDOFF_TEXT
from app.agent.tools import agent_tools
from app.chat import _message_text
from app.chat_tools import fallback_answer
from app.knowledge.citations import Evidence
from app.tools.executor import ToolResult, execute_tool, to_tool_message_content
from app.tools.registry import TOOL_LABELS

logger = logging.getLogger(__name__)

__all__ = ["AgentOutcome", "run_react_agent"]


@dataclass(frozen=True)
class AgentOutcome:
    """一轮 ReAct 的完整产物（纯数据，可直接落库/断言）。"""

    answer: str
    pending_messages: list[BaseMessage]
    tool_traces: list[dict[str, Any]]
    tool_records: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    suggested_actions: list[str]
    agent_steps: int
    llm_calls: int
    token_usage: dict[str, int]
    stop_reason: str


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _usage_of(message: BaseMessage, sent: Sequence[BaseMessage]) -> dict[str, int]:
    """取一次模型调用的用量；网关没给 usage 时用官方近似计数器兜底。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("total_tokens"):
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
    return {
        "input_tokens": count_tokens_approximately(list(sent)),
        "output_tokens": count_tokens_approximately([message]),
        "total_tokens": count_tokens_approximately(list(sent))
        + count_tokens_approximately([message]),
    }


def _add_usage(total: dict[str, int], delta: dict[str, int]) -> None:
    for key in total:
        total[key] += int(delta.get(key) or 0)


def _to_ai_message(chunk: AIMessageChunk) -> AIMessage:
    """把合并后的 chunk 转成一条可回灌的 AIMessage。

    **保留 additional_kwargs**：思考模型的 `reasoning_content` 住在这里，
    回灌时丢掉它，上游会间歇性 400。也保留 usage_metadata 供预算统计。
    """
    return AIMessage(
        content=_message_text(chunk),
        tool_calls=list(chunk.tool_calls or []),
        additional_kwargs=dict(chunk.additional_kwargs),
        response_metadata=dict(chunk.response_metadata),
        usage_metadata=chunk.usage_metadata,
        id=chunk.id,
    )


def _first_call_only(message: AIMessage, chosen: dict[str, Any]) -> AIMessage:
    if len(message.tool_calls) <= 1:
        return message
    return message.model_copy(update={"tool_calls": [chosen]})


def _emit(emit: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if emit is not None:
        emit(payload)


def _status_payload(
    name: str, args: dict[str, Any], call_id: str = ""
) -> dict[str, Any]:
    label = (
        "正在记录建议…"
        if name == "suggest_actions"
        else TOOL_LABELS.get(name, f"正在调用 {name}…")
    )
    return {
        "type": "status",
        "tool": name,
        "args": args,
        "label": label,
        # 前端靠它是把「执行中」那条记录精确对应到结果帧，而不是按工具名猜。
        "call_id": call_id,
    }


async def run_react_agent(
    *,
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    repositories: Any,
    retriever: Any,
    include_faq: bool,
    include_order: bool = True,
    evidence: Sequence[Evidence] = (),
    max_steps: int = 6,
    token_budget: int = 6000,
    timeout: float = 3.0,
    max_retries: int = 1,
    emit: Callable[[dict[str, Any]], None] | None = None,
    executor: Callable[..., Any] | None = None,
) -> AgentOutcome:
    """跑一轮 ReAct，返回可落库的完整产物。"""
    record_actions: list[str] = []
    tools = agent_tools(
        repositories,
        retriever,
        include_faq=include_faq,
        include_order=include_order,
        record_actions=record_actions,  # type: ignore[arg-type]
    )
    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}
    run_tool = executor if executor is not None else execute_tool

    loop_messages: list[BaseMessage] = list(messages)
    pending_messages: list[BaseMessage] = []
    answer_parts: list[str] = []
    tool_traces: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []
    usage = _empty_usage()
    llm_calls = 0
    agent_steps = 0
    stop_reason = "final"
    faq_evidence: tuple[Evidence, ...] = ()
    query_faq_used = False
    last_tool_result: ToolResult | None = None

    while True:
        if llm_calls >= max_steps + 1:
            stop_reason = "max_steps"
            break
        if usage["total_tokens"] >= token_budget:
            stop_reason = "token_budget"
            break

        full: AIMessageChunk | None = None
        bound = model.bind_tools(tools)
        async for chunk in bound.astream(loop_messages):
            full = chunk if full is None else full + chunk
            delta = _message_text(chunk)
            if delta:
                answer_parts.append(delta)
                _emit(emit, {"type": "token", "delta": delta})

        if full is None:  # pragma: no cover - 上游至少会给一个 chunk
            stop_reason = "empty_response"
            break

        llm_calls += 1
        _add_usage(usage, _usage_of(full, loop_messages))
        assistant = _to_ai_message(full)
        calls = list(assistant.tool_calls or [])

        if not calls:
            # 看**这一步**的文本，而不是累计 answer_parts：前一步可能只说了
            # 「我查一下」就调了工具；如果最后一步空回复，累计值会把它误判成 final。
            current_text = _message_text(assistant).strip()
            if current_text:
                pending_messages.append(assistant)
                stop_reason = "final"
            else:
                # 不要把「空 assistant」提交进历史：回读时它会渲染成一个多余的
                # 「（客服没有返回内容）」气泡，而下面已经会补固定兜底。
                stop_reason = "empty_response"
            break

        if len(calls) > 1:
            logger.warning(
                "模型一步返回了 %d 个工具调用，只执行第一个（ReAct 单工具步）：已忽略 %s",
                len(calls),
                [call.get("name") for call in calls[1:]],
            )
        chosen = calls[0]
        assistant = _first_call_only(assistant, chosen)
        loop_messages.append(assistant)
        pending_messages.append(assistant)

        name = str(chosen.get("name", ""))
        args = dict(chosen.get("args") or {})
        call_id = str(chosen.get("id", ""))
        _emit(emit, _status_payload(name, args, call_id))

        if name == "suggest_actions":
            content, record = await _run_suggest_actions(tools_by_name, args, call_id)
            tool_records.append(record)
        elif name not in tools_by_name:
            # 模型可能编出本轮**没有绑定**的工具名（最危险的是 create_ticket）。
            # 绝不能把它交给全局 registry 的 execute_tool —— 那会绕过工具集过滤，
            # 在没有用户点按钮的情况下真的写 tickets。
            result = ToolResult(
                tool=name,
                args=args,
                ok=False,
                data=None,
                error=(
                    f"工具 {name!r} 不在本轮可用工具内。"
                    f"可用工具：{', '.join(tools_by_name)}"
                ),
                attempts=0,
                duration_ms=0,
                started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )
            tool_traces.append(result.summary())
            content = to_tool_message_content(result)
            tool_records.append(
                {
                    "name": name,
                    "args": args,
                    "tool_call_id": call_id,
                    "content": content,
                    "traces": [result.summary()],
                }
            )
        else:
            result = await run_tool(
                name,
                args,
                repositories=repositories,
                retriever=retriever,
                timeout=timeout,
                max_retries=max_retries,
            )
            last_tool_result = result
            tool_traces.append(result.summary())
            content = to_tool_message_content(result)
            tool_records.append(
                {
                    "name": name,
                    "args": args,
                    "tool_call_id": call_id,
                    "content": content,
                    "traces": [result.summary()],
                }
            )
            if name == "query_faq" and not query_faq_used:
                query_faq_used = True
                faq_evidence = await evidence_from_tool_result(result, repositories)
                tools = [tool for tool in tools if tool.name != "query_faq"]
                tools_by_name.pop("query_faq", None)

        # 工具刚执行完就播报结果：用户不必等最终回答生成完才看到「成功/失败」。
        # suggest_actions 是内部控制面工具，不属于用户可见的工具调用记录。
        if name != "suggest_actions":
            _emit(
                emit,
                {
                    "type": "tool_result",
                    "call_id": call_id,
                    "tool": result.summary(),
                },
            )

        tool_message = ToolMessage(content=content, tool_call_id=call_id)
        loop_messages.append(tool_message)
        pending_messages.append(tool_message)
        agent_steps += 1

        if agent_steps >= max_steps:
            stop_reason = "max_steps"
            break
        if usage["total_tokens"] >= token_budget:
            stop_reason = "token_budget"
            break

    preamble = "".join(answer_parts).strip()
    answer = preamble
    suggested_actions: list[str] = list(record_actions)

    if stop_reason in {"max_steps", "token_budget"}:
        handoff = AGENT_HANDOFF_TEXT
        answer = f"{answer}\n\n{handoff}" if answer else handoff
        pending_messages.append(AIMessage(content=handoff))
        _emit(emit, {"type": "token", "delta": ("\n\n" if answer_parts else "") + handoff})
        for action in ("transfer_human", "create_ticket"):
            if action not in suggested_actions:
                suggested_actions.append(action)
    elif stop_reason == "empty_response" or not answer:
        # 最后一步空回复时，优先用最后一条工具结果拼一句人话；
        # 没有工具结果才用固定兜底。不要把「我查一下」当前言留在答案里。
        fallback = fallback_answer(last_tool_result) if last_tool_result else AGENT_EMPTY_TEXT
        answer = f"{preamble}\n\n{fallback}" if preamble else fallback
        pending_messages.append(AIMessage(content=fallback))
        _emit(emit, {"type": "token", "delta": ("\n\n" if preamble else "") + fallback})
        if "transfer_human" not in suggested_actions:
            suggested_actions.append("transfer_human")
        stop_reason = "empty_response"

    citation_evidence = tuple(evidence) if evidence else faq_evidence
    citations = citations_for_answer(answer, citation_evidence)

    return AgentOutcome(
        answer=answer,
        pending_messages=pending_messages,
        tool_traces=tool_traces,
        tool_records=tool_records,
        citations=citations,
        suggested_actions=suggested_actions,
        agent_steps=agent_steps,
        llm_calls=llm_calls,
        token_usage=usage,
        stop_reason=stop_reason,
    )


async def _run_suggest_actions(
    tools_by_name: dict[str, BaseTool], args: dict[str, Any], call_id: str
) -> tuple[str, dict[str, Any]]:
    """执行控制面工具；失败也只回一条结构化错误，不打断循环。"""
    tool = tools_by_name.get("suggest_actions")
    if tool is None:  # pragma: no cover - 工具集构造保证存在
        content = json.dumps({"ok": False, "error": "suggest_actions 未装配"}, ensure_ascii=False)
        return content, {
            "name": "suggest_actions",
            "args": args,
            "tool_call_id": call_id,
            "content": content,
            "traces": [],
        }
    try:
        content = str(await tool.ainvoke(args))
    except Exception as exc:  # noqa: BLE001 - 控制面工具失败不该毁掉整轮
        logger.warning("suggest_actions 参数非法", exc_info=True)
        content = json.dumps(
            {"ok": False, "error": f"suggest_actions 调用失败：{exc}"}, ensure_ascii=False
        )
    return content, {
        "name": "suggest_actions",
        "args": args,
        "tool_call_id": call_id,
        "content": content,
        "traces": [],
    }
