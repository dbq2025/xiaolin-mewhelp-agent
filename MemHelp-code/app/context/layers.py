"""三层划分与规则截短 —— 「只挪两个 id，不搬数据」的那一层。

## 三层是什么

按离当前这一轮的远近切三段：

| 层 | 范围（按数据库 ``Message.id``） | 处理方式 |
|---|---|---|
| 层 3 | ``id <= summary_upto_msg_id`` | 不进上下文，只由梗概代表 |
| 层 2 | ``summary_upto < id < layer1_from`` | 按规则截短（见下） |
| 层 1 | ``id >= layer1_from_msg_id`` | **原文，一个字都不压** |

两个锚点就是全部状态：``summary_upto_msg_id`` 说摘要覆盖到哪条，
``layer1_from_msg_id`` 说层 1 从哪条起。降级时**只挪这两个 id，不搬任何数据** ——
这就是为什么聊天记录表不需要为了「压缩」而改写历史。

## 规则截短（层 2）

- **用户原话一个字都不动**：用户的请求是事实来源，截短会改变语义
  （「我不想要了」截成「我不想要」）。
- **客服答复只留开头 60 字**：答复的正文通常很长、信息密度低，开头一般已经
  表达了结论；剩下的细节由层 1 或梗概兜底。
- **大块工具结果换成一行标识**：工具结果动辄几千字，对后续推理的价值主要是
  「查过什么、有没有结果」，不是原始 JSON。

层 2 **不改行数**，只改内容 —— 改行数会破坏「一轮 = user + assistant」的对齐，
让层 1 的整轮切分算错。

## 降级：只前进，不后退

层 1 超预算时把 ``layer1_from_msg_id`` 往前移（移向更新的消息），丢最旧的整轮。
三条纪律：

1. **按整轮切**：新起点必须是 user 行。从 assistant 行开始，模型会看到一句没有
   问题的答复。
2. **只前进**：``layer1_from_msg_id >= summary_upto_msg_id + 1`` 必须永远成立，
   否则层 1 会退回摘要区间、和梗概重复。
3. **一整轮都装不下就留着**：宁可超预算，也不能让模型看不见刚发生的事。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages

from app.context.tokens import count_tokens, make_token_counter
from app.db.repositories import MessageRow

LAYER_SUMMARIZED = 3
"""层 3：已被摘要覆盖，不进上下文。"""

LAYER_TRIMMED = 2
"""层 2：按规则截短。"""

LAYER_VERBATIM = 1
"""层 1：原文，一个字都不压。"""

#: 工具结果换成一行标识时的默认上限（字符）。写死是因为它只影响可读性，不影响正确性。
TOOL_MARKER_CHARS = 40


def classify(
    msg_id: int,
    *,
    summary_upto: int | None,
    layer1_from: int | None,
) -> int:
    """这条消息属于哪一层。``None`` 表示「还没有这个边界」。

    判定顺序必须是「先层 3、再层 1、否则层 2」，而且 **``layer1_from is None``
    要整段算层 1**：

    `layer1_from` 表达的是「层 1 从哪条起」，它只在**降级发生过**之后才有值。
    没降级时若把摘要之后的全部内容算成层 2（规则截短），最近几轮的原文就没了——
    「最近的留原文，一个字都不压」这条直接失效。所以层 2 的语义是
    「曾经是层 1、后来被降级推出去的那段」，它天然存在于 ``summary_upto`` 与
    ``layer1_from`` 之间，而不是「摘要之后的一切」。

    ``summary_upto`` 与 ``layer1_from`` 同时存在时，摘要边界优先 —— 它代表
    「这条已经被压成梗概了」，不允许再作为原文出现。
    """
    if summary_upto is not None and msg_id <= summary_upto:
        return LAYER_SUMMARIZED
    if layer1_from is None or msg_id >= layer1_from:
        return LAYER_VERBATIM
    return LAYER_TRIMMED


def _marker(name: str, original_chars: int, *, limit: int = TOOL_MARKER_CHARS) -> str:
    """把工具结果压成一行标识。**保留原名与字数**，这是它唯一的信息价值。"""
    head = f"[工具结果 · {name} · {original_chars} 字已省略]"
    if len(head) <= limit + 20:  # 标识本身很短，不需要再截
        return head
    return head[: limit + 19] + "…"


def rule_trim_row(
    row: MessageRow,
    *,
    assistant_preview_chars: int = 60,
    tool_names: Mapping[str, str] | None = None,
) -> MessageRow:
    """按角色规则截短一条消息，返回**新**行（``MessageRow`` 是 frozen 的）。

    - ``user``：原样返回内容（但仍是新对象，调用方可以放心地当不可变用）
    - ``assistant``：留开头 ``assistant_preview_chars`` 字，超出补 ``…``；空正文保持空
    - ``tool``：换成一行标识，用 ``tool_names`` 里的工具名，查不到就退回 ``tool_call_id``

    为什么空正文必须保持空：纯工具调用那一行 ``AIMessage.content`` 就是空串，
    如果统一补 ``…``，模型会以为客服说了句省略号，而不是「他直接去查了订单」。
    """
    content = row.content or ""

    if row.role == "tool":
        names = tool_names or {}
        name = names.get(row.tool_call_id or "") or row.tool_call_id or "未知工具"
        return replace(row, content=_marker(name, len(content)))

    if row.role == "assistant":
        if not content:
            return replace(row, content="")
        if len(content) > assistant_preview_chars:
            return replace(row, content=content[:assistant_preview_chars] + "…")
        return replace(row, content=content)

    # user：一个字都不动。
    return replace(row, content=row.content)


def tool_names_from_rows(rows: Sequence[MessageRow]) -> dict[str, str]:
    """从 assistant 行的 ``tool_calls`` 里抽出 ``call_id -> 工具名``。

    ``tool`` 行只带 ``tool_call_id``，工具名挂在调用它的那条 assistant 行上。
    先把整段扫一遍再截短，才能给工具结果标上可读的名字。
    """
    mapping: dict[str, str] = {}
    for row in rows:
        for call in row.tool_calls or ():
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            name = call.get("name")
            if call_id and name:
                mapping[str(call_id)] = str(name)
    return mapping


def row_to_message(row: MessageRow) -> BaseMessage:
    """``MessageRow`` → 可直接发给模型的 ``BaseMessage``。

    与 ``app.db.repositories._row_to_message`` 语义一致（那个是内部函数，不能跨包用；
    这里只接受**已经截短**的行，所以不复用它的 tool_call_id 校验 —— 层 2 的 tool 行
    已经被换成一行标识，不需要再对号入座回灌）。
    """
    content = row.content or ""
    if row.role == "user":
        return HumanMessage(content=content)
    if row.role == "assistant":
        if row.tool_calls:
            return AIMessage(content=content, tool_calls=list(row.tool_calls))
        return AIMessage(content=content)
    if row.role == "tool":
        return ToolMessage(content=content, tool_call_id=row.tool_call_id or "")
    if row.role == "system":
        return SystemMessage(content=content)
    raise ValueError(f"未知消息角色 {row.role!r}（只允许 user/assistant/tool/system）")


@dataclass(frozen=True)
class LayerPlan:
    """一次三层规划的结果。**只读快照**，不落库、不改锚点。"""

    rows: list[MessageRow]
    """输入的全部行（按 id 升序），给日志数「一共多少条」用。"""

    layer2: list[MessageRow]
    """层 2：**已按规则截短**的行。"""

    layer1: list[MessageRow]
    """层 1：原文行。"""

    layer1_from_msg_id: int | None
    """本次规划算出的层 1 起点（建议值；调用方决定是否落库）。"""

    summary_upto_msg_id: int | None
    """本次规划看到的摘要边界（原样回显，规划不改它）。"""

    layer1_tokens: int
    """层 1 实占 tokens。"""

    layer2_tokens: int
    """层 2 截短后实占 tokens。"""

    degraded: bool
    """层 1 是否因超预算被降级过。"""

    summary_due: bool
    """层 2 是否已超预算、该起一次后台摘要。"""

    summary_rows: list[MessageRow]
    """该被压成梗概的行（**未截短原文**）；``summary_due=False`` 时为空。"""

    dropped_layer1_rows: int
    """层 1 因降级丢掉的行数。"""


def _to_messages(rows: Sequence[MessageRow]) -> list[BaseMessage]:
    return [row_to_message(row) for row in rows]


def _first_user_index(rows: Sequence[MessageRow]) -> int | None:
    for idx, row in enumerate(rows):
        if row.role == "user":
            return idx
    return None


def plan_layers(
    rows: Sequence[MessageRow],
    *,
    summary_upto: int | None,
    layer1_from: int | None,
    layer1_budget: int,
    layer2_budget: int,
    assistant_preview_chars: int = 60,
    counter: Callable[[Sequence[BaseMessage]], int] | None = None,
) -> LayerPlan:
    """把全部历史行按两个锚点切成三段，并把层 1 裁到预算内。

    :param rows: 全部历史行，按 ``id`` 升序。
    :param summary_upto: ``conversations.summary_upto_msg_id``，``None`` = 还没摘要。
    :param layer1_from: ``conversations.layer1_from_msg_id``，``None`` = 全部算层 1。
    :param layer1_budget: 层 1 的 token 预算。
    :param layer2_budget: 层 2 的 token 预算；超了就 ``summary_due=True``。
    :param assistant_preview_chars: 层 2 里客服答复保留多少字。
    :param counter: 校准计数器；``None`` 时用默认比例。
    """
    rows = list(rows)
    count = counter or make_token_counter()

    if not rows:
        return LayerPlan(
            rows=[],
            layer2=[],
            layer1=[],
            layer1_from_msg_id=None,
            summary_upto_msg_id=summary_upto,
            layer1_tokens=0,
            layer2_tokens=0,
            degraded=False,
            summary_due=False,
            summary_rows=[],
            dropped_layer1_rows=0,
        )

    # 先把「层 2 的起点」定下来：摘要边界之后。这是降级的**下限**，
    # 免得传入的 layer1_from 是脏值（比 summary_upto 还小）时层 1 倒退。
    floor = (summary_upto or 0) + 1 if summary_upto is not None else None
    effective_layer1_from = layer1_from
    if effective_layer1_from is None and summary_upto is not None:
        effective_layer1_from = floor
    if floor is not None and effective_layer1_from is not None:
        effective_layer1_from = max(effective_layer1_from, floor)

    names = tool_names_from_rows(rows)

    layer2: list[MessageRow] = []
    layer1: list[MessageRow] = []
    for row in rows:
        layer = classify(row.id, summary_upto=summary_upto, layer1_from=effective_layer1_from)
        if layer == LAYER_SUMMARIZED:
            continue
        if layer == LAYER_VERBATIM:
            layer1.append(row)
        else:
            layer2.append(rule_trim_row(row, assistant_preview_chars=assistant_preview_chars, tool_names=names))

    # ---- 层 2：按规则截短后计 token，超预算就报「该摘要了」----
    layer2_tokens = count(_to_messages(layer2)) if layer2 else 0

    # ---- 层 1 降级：超预算就把起点往前移，按整轮对齐 ----
    layer1_messages = _to_messages(layer1)
    layer1_tokens = count(layer1_messages)
    degraded = False
    dropped = 0

    if layer1 and layer1_tokens > layer1_budget:
        kept = trim_messages(
            layer1_messages,
            max_tokens=layer1_budget,
            token_counter=count,
            strategy="last",
            start_on="human",
            allow_partial=False,
            include_system=True,
        )
        if kept:
            # trim_messages 保留的一定是后缀；按条数对齐回 MessageRow。
            kept_rows = layer1[len(layer1) - len(kept):]
            dropped = len(layer1) - len(kept_rows)
            degraded = dropped > 0
            layer1 = kept_rows
        else:
            # 一整轮都装不下：留着，宁可超预算也不能让模型看不见刚发生的事。
            degraded = True
            dropped = 0
        layer1_tokens = count(_to_messages(layer1))

    # 按整轮切：起点必须是 user。trim_messages 已经用 start_on="human" 保证了这点，
    # 但「原始 layer1_from 就指在 assistant 上」这种脏数据要在这一步修正。
    first_user = _first_user_index(layer1)
    if first_user is not None and first_user > 0:
        dropped += first_user
        layer1 = layer1[first_user:]
        degraded = True
        layer1_tokens = count(_to_messages(layer1))

    new_layer1_from = layer1[0].id if layer1 else None
    if new_layer1_from is None and effective_layer1_from is not None:
        # 层 1 全空（例如历史还没到边界）：保留原锚点，别把它清成 None。
        new_layer1_from = effective_layer1_from

    summary_due = bool(layer2) and layer2_tokens > layer2_budget
    summary_rows = list(layer2_untrimmed(rows, layer2)) if summary_due else []

    return LayerPlan(
        rows=rows,
        layer2=layer2,
        layer1=layer1,
        layer1_from_msg_id=new_layer1_from,
        summary_upto_msg_id=summary_upto,
        layer1_tokens=layer1_tokens,
        layer2_tokens=layer2_tokens,
        degraded=degraded,
        summary_due=summary_due,
        summary_rows=summary_rows,
        dropped_layer1_rows=dropped,
    )


def layer2_untrimmed(rows: Sequence[MessageRow], layer2: Sequence[MessageRow]) -> list[MessageRow]:
    """把截短过的层 2 换回**原文**行 —— 摘要必须读原文。

    用 ``id`` 建索引而不是按位置对齐：层 2 不改行数，但用 id 更稳，
    将来若加了「合并连续 tool 行」之类的规则也不会错位。
    """
    by_id = {row.id: row for row in rows}
    return [by_id[row.id] for row in layer2 if row.id in by_id]
