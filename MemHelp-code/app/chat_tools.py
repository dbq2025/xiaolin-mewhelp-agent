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
from dataclasses import dataclass, replace
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
from app.knowledge.citations import (
    Evidence,
    evidence_from_payload,
    evidence_payload,
    parse_citations,
)
from app.knowledge.pipeline import RetrievalOutcome
from app.tools.knowledge import AnswerContextLike
from app.prompts import REFUSAL_TEXT, build_tool_chat_messages
from app.tools.executor import ToolResult, execute_tool, to_tool_message_content
from app.tools.registry import TOOL_LABELS, get_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolEvent:
    """编排产出的事件。HTTP 层把它映射成 SSE 帧。

    分四种：``status``（工具执行前，给人看的进行中提示）、``token``（逐字增量）、
    ``tools``（本轮工具轨迹摘要，徽章的权威来源）、``citations``（本轮引用证据快照）。
    """

    kind: Literal["status", "token", "tools", "citations"]
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


def _first_call_only(first: AIMessage, chosen: dict[str, Any]) -> AIMessage:
    """把第 1 次调用的响应裁剪成「只含第一个工具调用」，其余字段**原样保留**。

    两个方向都要照顾：

    1. **保留** ``additional_kwargs`` 等字段。思考模型（``deepseek-flash``）的
       ``reasoning_content`` 就住在那里，第二次请求必须带回，否则上游 400：
       ``The reasoning_content in the thinking mode must be passed back to the API.``
       所以不能用 ``AIMessage(content="", tool_calls=[chosen])`` 重建一条 ——
       那会把除 ``tool_calls`` 之外的一切都丢掉，表现为**间歇性**失败。
    2. **裁掉**多余的 ``tool_calls``。模型偶尔一轮给多个调用，而单轮策略只执行第一个；
       若把其余调用一起放回消息里，上游会认为我们漏了它们的 ``ToolMessage``
       （报错大意是「每个 tool_call 都必须有对应的 tool 响应」）。

    返回一条**新**消息（浅拷贝原字段），不改动上游返回的对象。
    """
    if len(first.tool_calls) <= 1:
        return first
    return first.model_copy(update={"tool_calls": [chosen]})


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


def _refusal_of(result: ToolResult) -> tuple[str, str] | None:
    """工具是否报告「证据不足，必须拒答」。

    返回 ``(source, reason)``：``source`` 直接作为 ``low_confidence_questions.source``
    的 ENUM 值，所以只能是 ``retrieval_low_conf`` / ``self_check``。
    """
    payload = _as_mapping(result.data)
    if not payload or not payload.get("refused"):
        return None
    source = str(payload.get("refusal_source") or "self_check")
    reason = str(payload.get("refusal_reason") or "证据不足以回答这个问题")
    return source, reason


async def _evidence_of(result: ToolResult, repositories: Repositories) -> tuple[Evidence, ...]:
    """从 ``query_faq`` 的出参还原证据。字段缺失/形状不对时返回空 —— 不抛。

    编排层只负责「把编号映射回 chunk」，它不该因为工具多/少一个字段就炸掉整轮对话。

    出参里被截断的那些（``truncated: true``）要**回库取全文**：给模型看的是节选，
    但用户点开引用时看到的必须是原文，否则「点开看来源」这件事就名不副实。
    """
    payload = _as_mapping(result.data)
    if not payload:
        return ()
    items = list(payload.get("items") or [])
    try:
        evidence = evidence_from_payload(items)
    except (KeyError, TypeError, ValueError):
        logger.warning("query_faq 出参里的证据形状不对，本轮不提供引用", exc_info=True)
        return ()

    truncated_ids = [
        item.get("chunk_id")
        for item in items
        if item.get("truncated") and item.get("chunk_id") is not None
    ]
    if not truncated_ids:
        return evidence

    try:
        wanted = [int(chunk_id) for chunk_id in truncated_ids]
        rows = {row.id: row.answer for row in await repositories.knowledge.get_many(wanted)}
    except Exception:  # noqa: BLE001 - 取全文失败不该毁掉整轮回答
        logger.warning("引用原文回库补全失败，退回节选", exc_info=True)
        return evidence

    return tuple(
        replace(item, quote=rows[item.chunk_id]) if item.chunk_id in rows else item
        for item in evidence
    )


async def astream_chat_with_tools(
    *,
    session_id: str,
    message: str,
    model: BaseChatModel,
    repositories: Repositories,
    retriever: AnswerContextLike | None = None,
    budget_tokens: int,
    model_name: str,
    timeout: float = 3.0,
    max_retries: int = 1,
    meta_out: dict | None = None,
    executor: Callable[..., Any] | None = None,
) -> AsyncIterator[ToolEvent]:
    """跑一轮「模型定工具 → 执行 → 回灌 → 流式收敛」，并落库。

    :param repositories: 当次请求的仓库。历史回读与落库都经它。
    :param retriever: 当次请求的向量检索器（ch03 起 ``query_faq`` 依赖它）。
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

    tools = get_tools(repositories, retriever)
    parts: list[str] = []
    tool_summaries: list[dict[str, Any]] = []
    #: ch04：本轮引用的证据快照（落库 + 送前端）。无工具调用时保持空列表。
    citations: list[dict[str, Any]] = []
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
                retriever=retriever,
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
                    # 给界面看的完整摘要（含耗时与成败）。**与 tool_calls 分开存** ——
                    # 后者要当模型历史回灌，字段必须严格是 {name, args, id}；
                    # 我实测过往它里面加 duration_ms 会让 AIMessage 构造直接抛
                    # TypeError: tool_call() got an unexpected keyword argument。
                    "traces": [result.summary()],
                }
            )

            # -- ch04：证据不足 → **短路**，不再调用模型 ------------------
            #
            # 验收标准要求「问库里没有的内容必须得到明确拒答」，那就不能把
            # 这件事交给模型的情绪与判断。走到这里时检索已经明确说「答不了」，
            # 直接吐固定话术，并把原话落进低置信度池。
            refusal = _refusal_of(result) if tool_name == "query_faq" else None
            if refusal is not None:
                source, reason = refusal
                # 入池是**旁路**：池子写不进去也要把拒答话术给到用户。
                # 反过来（先写池、失败就抛）会让「数据库抖一下」变成 500，
                # 而验收 4 要求的是「明确拒答」这件用户可见的事。
                try:
                    await _record_low_confidence(
                        repositories, session_id, message, source, reason
                    )
                except Exception:  # noqa: BLE001 - 入池失败不该吞掉拒答
                    logger.exception("低置信度入池失败（已忽略，仍然给出拒答）")
                parts.append(REFUSAL_TEXT)
                yield ToolEvent("token", {"delta": REFUSAL_TEXT})
                await _persist(
                    repositories,
                    session_id,
                    message,
                    parts,
                    persisted_tool_rows[0],
                    citations=[],
                )
                yield ToolEvent("tools", {"tools": tool_summaries})
                return

            # **原样复用第 1 次调用返回的那条 assistant 消息**，不要重建一条。
            #
            # 踩过的坑：``deepseek-flash`` 是思考模型，它的响应里除了 ``tool_calls``
            # 还带 ``reasoning_content``（LangChain 放在 ``additional_kwargs`` 里）。
            # 第二次请求必须把思考内容一并带回，否则上游直接 400：
            #   "The `reasoning_content` in the thinking mode must be passed back to the API."
            #
            # 而「重建一条 AIMessage(content="", tool_calls=[...])」会丢掉那个字段，
            # 于是**间歇性**失败 —— 只在模型真的产生了思考内容时炸，
            # 于是它看起来像随机故障。ch01 的结构化提取也踩过同一个坑。
            #
            # 只取第一个工具调用时，仍要把整条消息放回去：``tool_calls`` 里若留了
            # 未执行的调用，上游会认为我们漏了对应的 ``ToolMessage``。
            assistant_message = _first_call_only(first, chosen)

            follow_up: list[BaseMessage] = [
                *ctx.messages,
                assistant_message,
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

            evidence = await _evidence_of(result, repositories)
            citations = _valid_citations("".join(parts), evidence)
    except BaseException:
        # 失败时只落用户那一轮（沿用 ch01「不落半截回答」）。
        # 落半截或落孤立的 tool 行都会污染下一轮上下文。
        #
        # **落库本身失败时不能盖住原始异常**（Review I1，已实测复现）：
        # 上游故障 + 数据库同时不可用时，``_persist`` 会先抛，于是用户拿到的是
        # 数据库错误，而真正的原因是上游 —— 排查方向被彻底带偏。
        # 这里吞掉落库异常并记日志，然后照旧抛**原始**异常。
        try:
            await _persist(repositories, session_id, message, [], None)
        except Exception:  # noqa: BLE001 - 落库失败不该改变对外暴露的故障类型
            logger.exception("上游失败后落库用户轮次也失败了（已忽略，保留原始异常）")
        raise

    await _persist(
        repositories,
        session_id,
        message,
        parts,
        persisted_tool_rows[0] if persisted_tool_rows else None,
        citations=citations,
    )

    # 只有真有引用时才发这个事件：没有引用的普通闲聊多一个空事件没意义。
    if citations:
        yield ToolEvent("citations", {"citations": citations})
    yield ToolEvent("tools", {"tools": tool_summaries})


async def complete_chat_with_tools(
    *,
    session_id: str,
    message: str,
    model: BaseChatModel,
    repositories: Repositories,
    retriever: AnswerContextLike | None = None,
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
        retriever=retriever,
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
    *,
    citations: Sequence[dict[str, Any]] | None = None,
) -> None:
    """把这一轮落库。

    无工具：``user`` → ``assistant``(答案) 两行。
    有工具：``user`` → ``assistant``(带 tool_calls) → ``tool`` → ``assistant``(答案) 四行。
    失败路径由调用方传空的 ``answer_parts``，于是只落 ``user`` 一行。

    ``tool_row`` 同时携带申请单与结果 —— 少了任何一行，
    下次重建历史时 ``ToolMessage`` 就找不到对应的申请单，上游会拒绝这段历史。

    ``citations`` 只挂在**最终答案**那一行：带工具调用的申请单行没有引用可挂，
    挂上去会让「这一轮到底引了什么」变得含糊。
    """
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")

    await repositories.messages.append(conversation_id, role="user", content=message)

    if tool_row is not None:
        await repositories.messages.append(
            conversation_id,
            role="assistant",
            content="",
            # 模型历史：字段必须**只有** name / args / id（多一个键，
            # 回灌时 AIMessage(tool_calls=...) 就会抛 TypeError）。
            tool_calls=[
                {
                    "name": tool_row["name"],
                    "args": tool_row["args"],
                    "id": tool_row["tool_call_id"],
                }
            ],
            # 界面信息：单独一列，含耗时与成败。前端刷新后靠它还原完整徽章。
            tool_traces=tool_row.get("traces") or [],
        )
        await repositories.messages.append(
            conversation_id,
            role="tool",
            content=tool_row["content"],
            tool_call_id=tool_row["tool_call_id"],
        )

    if answer_parts:
        await repositories.messages.append(
            conversation_id,
            role="assistant",
            content="".join(answer_parts),
            citations=citations or None,
        )


def _valid_citations(answer: str, evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    """只把**正文真的引用了**的证据写进引用快照。

    全部证据都存会带来一个副作用：用户点开 ``[3]`` 却发现这条根本没被上面的话用到。
    快照应当回答「这段话依据了哪几条」，而不是「当时手里有什么」。
    """
    if not evidence:
        return []
    parsed = parse_citations(answer, evidence)
    if parsed.dropped:
        logger.warning(
            "正文里有 %d 个角标映射不到本轮证据，已忽略（编号=%s）——"
            "模型多写角标不该让回答消失，但值得看一眼",
            len(parsed.dropped),
            list(parsed.dropped),
        )
    wanted = set(parsed.valid)
    return [item for item in evidence_payload(evidence) if item["n"] in wanted]


async def _record_low_confidence(
    repositories: Repositories,
    session_id: str,
    message: str,
    source: str,
    reason: str,
) -> None:
    """把「答不上来」的原话落进低置信度池。

    先 upsert 会话拿 ``conversation_id``：池子有指向 ``conversations`` 的外键，
    而这一轮的用户行是稍后才写的，不能指望它有 id。
    """
    conversation_id = await repositories.conversations.upsert_by_session(session_id, "anonymous")
    await repositories.low_confidence.add(conversation_id, message, source, reason)
    logger.info("低置信度问题已入池（source=%s）：%s", source, message[:60])
