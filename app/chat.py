"""对话用例 —— 把历史、Prompt 模板、上下文裁剪和模型调用编排在一起。

本模块只产出**纯文本 delta**，不关心 SSE 编码：那是 ``api.py`` 的职责。
这样编排逻辑可以完全离线测试（注入假模型），传输细节也能单独测试。

## 异常语义

``astream_chat`` 是异步生成器，上游出错时**不吞异常**，让它向上冒给 ``api.py``
去转成 ``event: error``。但无论成功还是异常，**用户的问题与已收到的部分回复
都要落库**（用 ``finally`` 保证）——否则上游挂一次，这一轮对话就从历史里凭空
消失了，下一轮模型会以为用户没说过话。
"""

from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.context import ContextResult, build_context
from app.memory import InMemorySessionStore
from app.prompts import build_chat_messages


@dataclass(frozen=True)
class ChatMeta:
    """一轮对话的元信息。"""

    session_id: str
    model: str
    context: ContextResult


def _prepare(
    session_id: str,
    message: str,
    store: InMemorySessionStore,
    budget_tokens: int,
) -> tuple[list, ContextResult]:
    """取历史 → 渲染模板 → 裁剪。顺序固定，不要调换。

    用 ``build_chat_messages`` 渲染出 ``[System, *history, Human(本轮)]`` 之后，
    把 System 与本轮问题**摘出来**交给 ``build_context`` 裁剪历史 —— 这样
    「本轮问题参与裁剪」这个坑在结构上就不可能发生：裁剪函数的入参里根本没有
    本轮问题参与预算的路径，它是裁剪完后才被接上去的。
    """
    history = store.get(session_id)
    rendered = build_chat_messages(history, message)
    ctx = build_context(
        history=rendered[1:-1],
        question=rendered[-1],
        system_message=rendered[0],
        budget_tokens=budget_tokens,
    )
    return ctx.messages, ctx


async def astream_chat(
    session_id: str,
    message: str,
    store: InMemorySessionStore,
    model: BaseChatModel,
    budget_tokens: int,
    model_name: str,
    meta_out: dict | None = None,
) -> AsyncIterator[str]:
    """流式对话，逐块产出**非空**文本 delta。

    ``meta_out``：可选的输出参数（约定键 ``"meta"``）。因为异步生成器在第一次
    ``__anext__`` 之前不会执行任何代码，HTTP 层无法从返回值拿到元信息（比如裁剪
    统计来填 SSE 的 ``done`` 事件），所以这里在开始产流**之前**把 ``ChatMeta``
    写进去。这是刻意的妥协，避免为此再造一个包装类型。
    """
    messages, ctx = _prepare(session_id, message, store, budget_tokens)

    if meta_out is not None:
        meta_out["meta"] = ChatMeta(session_id=session_id, model=model_name, context=ctx)

    parts: list[str] = []
    try:
        async for chunk in model.astream(messages):
            delta = chunk.content
            # 空串 delta 是真实上游的行为（流结束时常见），
            # 发出去会让客户端多渲染一个空事件。
            if isinstance(delta, str) and delta:
                parts.append(delta)
                yield delta
    finally:
        store.append(
            session_id,
            [HumanMessage(message), AIMessage("".join(parts))],
        )


async def complete_chat(
    session_id: str,
    message: str,
    store: InMemorySessionStore,
    model: BaseChatModel,
    budget_tokens: int,
    model_name: str,
) -> tuple[str, ChatMeta]:
    """非流式对话，返回完整回复与元信息。"""
    messages, ctx = _prepare(session_id, message, store, budget_tokens)

    response = await model.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)

    store.append(session_id, [HumanMessage(message), AIMessage(text)])

    meta = ChatMeta(session_id=session_id, model=model_name, context=ctx)
    return text, meta
