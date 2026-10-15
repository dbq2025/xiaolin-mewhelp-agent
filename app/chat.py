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
from app.errors import UpstreamEmptyResponse
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


def _content_to_text(content: object) -> str:
    """把消息内容或内容块统一成文本。

    ``content`` 有两种形态：裸字符串，或 content-block 列表
    （``[{"type": "text", "text": "..."}]``）。不统一处理的话，流式路径会把
    列表形态的块**整块丢掉**，非流式路径却 ``str()`` 成 Python repr —— 同一个
    上游响应在两条路径上得到完全不同的结果。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)
    return str(content)


def _message_text(message: object) -> str:
    """从一条消息取文本，优先用 langchain 自己的 ``.text`` 访问器。"""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return _content_to_text(getattr(message, "content", None))


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

    :raises UpstreamEmptyResponse: 上游一个可用文本块都没给。**不能**把空内容
        当成正常回答落库 —— 否则下一轮模型会以为客服刚说了句空话。
    """
    messages, ctx = _prepare(session_id, message, store, budget_tokens)

    if meta_out is not None:
        meta_out["meta"] = ChatMeta(session_id=session_id, model=model_name, context=ctx)

    parts: list[str] = []
    try:
        async for chunk in model.astream(messages):
            delta = _message_text(chunk)
            # 空串 delta 是真实上游的行为（流结束时常见），
            # 发出去会让客户端多渲染一个空事件。
            if delta:
                parts.append(delta)
                yield delta

        if not parts:
            raise UpstreamEmptyResponse(
                "上游返回了空内容（流式响应里没有任何可用文本块）。"
            )
    except BaseException:
        # 失败时只保留用户那一轮，不落库半截或空的 AI 回答
        # （半截回答会污染下一轮上下文，让模型以为它已经答过了）。
        store.append(session_id, [HumanMessage(message)])
        raise

    store.append(session_id, [HumanMessage(message), AIMessage("".join(parts))])


async def complete_chat(
    session_id: str,
    message: str,
    store: InMemorySessionStore,
    model: BaseChatModel,
    budget_tokens: int,
    model_name: str,
) -> tuple[str, ChatMeta]:
    """非流式对话，返回完整回复与元信息。

    :raises UpstreamEmptyResponse: 上游返回了空内容（与流式路径同样的判定）。
    """
    messages, ctx = _prepare(session_id, message, store, budget_tokens)

    response = await model.ainvoke(messages)
    text = _message_text(response)

    if not text:
        store.append(session_id, [HumanMessage(message)])
        raise UpstreamEmptyResponse("上游返回了空内容（非流式响应里没有文本）。")

    store.append(session_id, [HumanMessage(message), AIMessage(text)])

    meta = ChatMeta(session_id=session_id, model=model_name, context=ctx)
    return text, meta
