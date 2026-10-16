"""单轮工具编排 —— 本章的核心用例。

## 精确语义（spec 4.1）

```
构建消息（历史来自库 + 工具版 System + 本轮）
  ↓
model.bind_tools(五工具).ainvoke(messages)     ← 第 1 次调用，非流式，只为拿 tool_calls
  ├─ 无 tool_calls → 拿到的文本直接流式吐出（本轮无工具徽章）
  └─ 有 tool_calls → **只取 [0]**
        emit status {tool, args, label}        ← 工具执行**之前**，让用户立刻看到在查
        execute_tool(...)                      ← 参数校验 + 超时 + 重试（永不抛）
        messages += [AIMessage(tool_calls), ToolMessage(结果, tool_call_id)]
        model.astream(messages)                ← 第 2 次调用，流式
        └─ 一个非空 delta 都没有 → fallback_answer(结果)
落库四行 → emit tools（徽章的权威来源）
```

## 「只调一次就收敛」是结构性保证

第 2 次调用之后，本模块**没有任何检查 ``tool_calls`` 的分支，也没有循环**。
所以「模型想连调两次」不可能发生 —— 这比用 ``max_iterations=1`` 之类的运行时刹车
更明确：刹车是「运行时才发现要停」，结构是「根本走不到那里」。

## 为什么第 1 次调用不用流式

工具调用块的增量合并是 LangChain 侧的行为（跨 chunk 拼 ``index``、拼
``function.arguments`` 的分片）。把它和我们的编排混在一起，会让「工具到底选了没有」
变得难测，而且不同网关的分片行为还不一致。用一次非流式调用换确定的
``tool_calls``，第 2 次再用流式 —— 用户要的逐 token 就在那一段。
代价是首 token 延迟多一个 RTT，记为 ch03 的可优化项。

## 空内容的两种含义（容易搞错）

- 第 1 次调用 ``content`` 为空 **+ 有 tool_calls** → **正常**，它本来只为拿工具选择。
- 第 2 次调用没有任何非空 delta → 模型说不出话，用 ``fallback_answer`` 兜底。
- 第 1 次调用既没文本也没 tool_calls → 上游真的返回了空内容，抛
  ``UpstreamEmptyResponse``（沿用 ch01 的判定）。

照搬 ch01 的「空内容即错误」会让每一次工具调用都误判成上游故障，
而错误信息还指向上游 —— 排查方向会完全跑偏。
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from app.chat import ChatMeta, _message_text
from app.context import build_context
from app.db.repositories import Repositories
from app.errors import UpstreamEmptyResponse
from app.prompts import build_tool_chat_messages
from app.tools.executor import ToolResult, execute_tool, to_tool_message_content
from app.tools.registry import TOOL_LABELS, get_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolEvent:
    """编排产出的事件。HTTP 层把它映射成 SSE 帧。

    分三种：``status``（工具执行前，给人看的进行中提示）、
    ``token``（逐字增量）、``tools``（本轮工具轨迹摘要，徽章的权威来源）。
    """

    kind: Literal["status", "token", "tools"]
    payload: dict[str, Any]


#: 工具结果里值得对用户复述的字段，按优先级排列。
#: 兜底话术只挑这些，不把整坨 JSON 念出来。
_SPEAKABLE_FIELDS: tuple[tuple[str, str], ...] = (
    ("status", "状态"),
    ("carrier", "承运商"),
    ("waybill_no", "运单号"),
    ("eta", "预计到达"),
    ("count", "匹配条数"),
    ("name", "名称"),
    ("price_yuan", "价格"),
    ("stock", "库存"),
    ("amount_yuan", "金额"),
    ("ticket_no", "工单号"),
)


def fallback_answer(result: ToolResult) -> str:
    """第 2 次调用说不出话时，用工具结果拼一句人话。

    **为什么必须兜底**：什么都不发的话气泡里是一片空白（前端会显示
    「客服没有返回内容」），而工具其实成功了 —— 用户会以为系统坏了。
    实测 `deepseek-flash` 在单轮编排下就会这样：拿到工具结果后只回一个空文本。

    失败的结果**不能说成查到了** —— 那是编造（违反行为约束第 1 条）。
    """
    if not result.ok:
        return (
            f"抱歉，{_tool_label(result.tool)}的时候没能取到数据（{result.error}）。"
            "我先帮你登记一下，稍后让同事核实后回复你，好吗？"
        )

    payload = _as_mapping(result.data)
    if payload is None:
        return f"我帮你{_tool_label(result.tool)}了：{result.data}"

    parts: list[str] = []
    for key, label in _SPEAKABLE_FIELDS:
        if key in payload and payload[key] not in (None, ""):
            parts.append(f"{label}：{payload[key]}")

    if not parts:
        return f"我帮你查到了：{json.dumps(payload, ensure_ascii=False)}"

    return f"帮你查到了 —— {'；'.join(parts)}。"


def _tool_label(tool: str) -> str:
    """把「正在查询物流…」变成「查询物流」，用于拼句子。"""
    return TOOL_LABELS.get(tool, f"调用 {tool}…").rstrip("…")


def _as_mapping(data: Any) -> dict[str, Any] | None:
    """尽量把工具数据看成字典；不是就不猜。"""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


async def astream_chat_with_tools(
    *,
    session_id: str,
    message: str,
    model: BaseChatModel,
    repositories: Repositories,
    budget_tokens: int,
    model_name: str,
    timeout: float = 3.0,
    max_retries: int = 1,
    meta_out: dict | None = None,
    executor: Callable[..., Any] | None = None,
) -> AsyncIterator[ToolEvent]:
    """跑一轮「模型定工具 → 执行 → 回灌 → 流式收敛」，并落库。

    :param repositories: 当次请求的仓库。历史回读与落库都经它。
    :param meta_out: 可选输出参数（约定键 ``"meta"``），在开始产流**之前**写入
        ``ChatMeta``。异步生成器在第一次 ``__anext__`` 前不执行任何代码，
        HTTP 层无法从返回值拿元信息 —— 这是 ch01 就定下的妥协。
    :param executor: 工具执行器，默认 ``app.tools.executor.execute_tool``。
        可注入以便离线测试（**注意**：不要用 monkeypatch 改模块全局名，
        原因见 executor.py 的 ``resolve`` 参数文档）。
    :raises UpstreamEmptyResponse: 第 1 次调用既无文本也无工具调用。
    :raises Exception: 上游故障原样冒给 HTTP 层，但**只落库用户那一轮**。
    """
    run_tool = executor if executor is not None else execute_tool

    history = await repositories.messages.history_as_messages(session_id)
    rendered = build_tool_chat_messages(history, message, session_id)
    ctx = build_context(
        history=rendered[1:-1],
        question=rendered[-1],
        system_message=rendered[0],
        budget_tokens=budget_tokens,
    )

    if meta_out is not None:
        meta_out["meta"] = ChatMeta(session_id=session_id, model=model_name, context=ctx)

    tools = get_tools(repositories)
    parts: list[str] = []
    tool_summaries: list[dict[str, Any]] = []
    #: 落库用的那一行的完整信息。与 ``tool_summaries`` 分开：那个是给前端/日志看的摘要
    #: （``ToolResult.summary()``），不该为了落库再塞进 ``tool_call_id`` 与正文。
    persisted_tool_rows: list[dict[str, Any]] = []

    try:
        # -- 第 1 次调用：非流式，只为拿确定的 tool_calls -------------------
        first = await model.bind_tools(tools).ainvoke(ctx.messages)
        calls = list(getattr(first, "tool_calls", None) or [])

        if not calls:
            text = _message_text(first)
            if not text:
                raise UpstreamEmptyResponse(
                    "上游返回了空内容（既没有文本，也没有工具调用）。"
                )
            parts.append(text)
            yield ToolEvent("token", {"delta": text})
        else:
            # **单轮**：只认第一个。模型偶尔会一次给多个，多执行的后果是产生
            # 孤儿 ToolMessage（回灌只需要一个），上游会直接拒绝这段历史。
            chosen = calls[0]
            if len(calls) > 1:
                logger.warning(
                    "模型一轮返回了 %d 个工具调用，只执行第一个（单轮策略）：已忽略 %s",
                    len(calls),
                    [c.get("name") for c in calls[1:]],
                )

            tool_name = str(chosen.get("name", ""))
            tool_args = dict(chosen.get("args") or {})
            call_id = str(chosen.get("id", ""))
            yield ToolEvent(
                "status",
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "label": TOOL_LABELS.get(tool_name, f"正在调用 {tool_name}…"),
                },
            )

            result = await run_tool(
                tool_name,
                tool_args,
                repositories=repositories,
                timeout=timeout,
                max_retries=max_retries,
            )
            tool_summaries.append(result.summary())
            tool_content = to_tool_message_content(result)
            persisted_tool_rows.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "tool_call_id": call_id,
                    "content": tool_content,
                }
            )

            follow_up: list[BaseMessage] = [
                *ctx.messages,
                AIMessage(content="", tool_calls=[chosen]),
                ToolMessage(content=tool_content, tool_call_id=call_id),
            ]

            # -- 第 2 次调用：流式。之后**不再**检查 tool_calls，也没有循环。
            async for chunk in model.bind_tools(tools).astream(follow_up):
                delta = _message_text(chunk)
                if delta:  # 空串 delta 是真实上游的行为（流结束时常见）
                    parts.append(delta)
                    yield ToolEvent("token", {"delta": delta})

            if not parts:
                # 模型说不出话，用工具结果兜底 —— 不能让用户看到空气泡。
                fallback = fallback_answer(result)
                parts.append(fallback)
                yield ToolEvent("token", {"delta": fallback})
    except BaseException:
        # 失败时只落用户那一轮（沿用 ch01「不落半截回答」）。
        # 落半截或落孤立的 tool 行都会污染下一轮上下文。
        await _persist(repositories, session_id, message, [], None)
        raise

    await _persist(
        repositories,
        session_id,
        message,
        parts,
        persisted_tool_rows[0] if persisted_tool_rows else None,
    )

    yield ToolEvent("tools", {"tools": tool_summaries})


async def complete_chat_with_tools(
    *,
    session_id: str,
    message: str,
    model: BaseChatModel,
    repositories: Repositories,
    budget_tokens: int,
    model_name: str,
    timeout: float = 3.0,
    max_retries: int = 1,
    executor: Callable[..., Any] | None = None,
) -> tuple[str, ChatMeta]:
    """非流式版本：跑同一套单轮编排，把 token 事件拼成完整回复。

    **为什么非流式也要走编排**：否则两条路的历史来源会不一致 ——
    非流式读 ch01 的内存会话，流式读数据库。同一个 ``session_id``
    在两条路上看到不同的历史，是最容易产出的「偶发答非所问」。

    与流式版本共用同一个 ``astream_chat_with_tools``，所以工具调用、
    落库、失败语义完全一致；差别只是这里不逐块产出，而是收齐。

    :return: ``(完整回复, ChatMeta)``。调用方（HTTP 层）直接可用。
    :raises UpstreamEmptyResponse: 上游返回空内容（与流式路径同样的判定）。
    """
    meta_out: dict = {}
    parts: list[str] = []

    async for event in astream_chat_with_tools(
        session_id=session_id,
        message=message,
        model=model,
        repositories=repositories,
        budget_tokens=budget_tokens,
        model_name=model_name,
        timeout=timeout,
        max_retries=max_retries,
        meta_out=meta_out,
        executor=executor,
    ):
        if event.kind == "token":
            parts.append(event.payload.get("delta", ""))

    meta = meta_out.get("meta")
    if meta is None:  # pragma: no cover - 编排总会写 meta
        raise RuntimeError("编排未产出 meta，无法构造响应")

    return "".join(parts), meta


async def _persist(
    repositories: Repositories,
    session_id: str,
    message: str,
    answer_parts: Sequence[str],
    tool_row: dict[str, Any] | None,
) -> None:
    """把这一轮落库。

    无工具：``user`` → ``assistant``(答案) 两行。
    有工具：``user`` → ``assistant``(带 tool_calls) → ``tool`` → ``assistant``(答案) 四行。
    失败路径由调用方传空的 ``answer_parts``，于是只落 ``user`` 一行。

    ``tool_row`` 同时携带申请单与结果 —— 少了任何一行，
    下次重建历史时 ``ToolMessage`` 就找不到对应的申请单，上游会拒绝这段历史。
    """
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")

    await repositories.messages.append(conversation_id, role="user", content=message)

    if tool_row is not None:
        await repositories.messages.append(
            conversation_id,
            role="assistant",
            content="",
            tool_calls=[
                {
                    "name": tool_row["name"],
                    "args": tool_row["args"],
                    "id": tool_row["tool_call_id"],
                }
            ],
        )
        await repositories.messages.append(
            conversation_id,
            role="tool",
            content=tool_row["content"],
            tool_call_id=tool_row["tool_call_id"],
        )

    if answer_parts:
        await repositories.messages.append(
            conversation_id, role="assistant", content="".join(answer_parts)
        )
