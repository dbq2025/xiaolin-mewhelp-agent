"""上下文裁剪 —— 全项目**唯一**知道 token 预算的地方。

本包不碰网络、不碰磁盘，所以可以完全离线测试。

## 顺序是算法的一部分

固定为：``[System] + trim(历史, 预算) + [本轮问题]``

**先裁历史，再拼本轮问题**。如果图省事把本轮问题一起丢进 ``trim_messages``，
预算紧张时用户刚问的那句话会被当成「最旧的消息」裁掉，模型就会答非所问 ——
这是本章最容易踩的坑，``tests/test_context.py`` 有一条测试专门钉住它。

## 预算的统计范围

``budget_tokens`` **只统计历史消息**，不含 System Prompt，也不含本轮问题。
理由：System Prompt 是固定的角色设定，本轮问题必须保留，两者都没有裁剪空间；
拿它们去挤历史只会让语义更差。

## 包结构

- ``tokens``：校准的「字数折 token」口径（ch07）
- ``budget``：从模型窗口倒推三层预算（ch07）
- ``layers``：三层划分与规则截短（ch07）
- ``assemble``：固定顺序拼装（ch07）

``build_context`` 保留在本模块，是 ch01–ch06 的**兼容外壳**：行为逐字节不变，
内部转成单层规划。新代码走 ``assemble.assemble_context``。
"""

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


@dataclass(frozen=True)
class ContextResult:
    """裁剪结果 + 可观测统计。

    统计字段是刻意暴露的：它让「模型到底收到了多少历史」可被断言，也是后续接
    Langfuse 埋点时要上报的指标。

    ch07 追加的字段全部带默认值，所以位置参数构造（含 ``tests/test_context.py``
    与旧调用方）一个都不用改。
    """

    messages: list[BaseMessage]
    """最终要发给模型的消息：``[System, *保留的历史, Human(本轮)]``。"""

    kept_messages: int
    """保留的历史消息条数。"""

    trimmed_messages: int
    """被裁掉的历史消息条数。"""

    budget_tokens: int
    """历史消息的 token 预算。"""

    used_tokens: int
    """保留的历史消息实际占用的近似 token 数。"""

    system_prompt_tokens: int
    """System Prompt 的近似 token 数（不计入预算，仅作观测）。"""

    # ---- ch07 追加（都有默认值，旧调用方不受影响）----

    summary_tokens: int = 0
    """注入的会话梗概占用（条数无关，按 token 计）。"""

    summary_dropped_segments: int = 0
    """因超出 ``SUMMARY_RESERVE_TOKENS`` 而没注入的梗概段数。

    非 0 说明这个会话的梗概总量已经超过固定预留 —— 预算表里的 500 被低估了。
    留这个数字是为了让「超了」可被观测，而不是悄悄丢段。
    """

    layer2_messages: int = 0
    """层 2（规则截短）的消息条数。"""

    layer1_messages: int = 0
    """层 1（原文滑窗）的消息条数。"""

    layer1_from_msg_id: int | None = None
    """层 1 起点锚点（数据库 Message.id）。``None`` = 全部算层 1。"""

    summary_upto_msg_id: int | None = None
    """摘要覆盖到哪条（数据库 Message.id）。``None`` = 还没有摘要。"""

    degraded: bool = False
    """层 1 是否因超预算而被降级过（锚点前移）。"""

    summary_due: bool = False
    """层 2 是否已超预算、需要起一次后台摘要。"""


def build_context(
    history: Sequence[BaseMessage],
    question: BaseMessage,
    system_message: SystemMessage,
    budget_tokens: int,
) -> ContextResult:
    """把历史裁剪到预算内，再把 System 与本轮问题接上去。

    只做裁剪与拼装，**不构造消息内容** —— 模板渲染是 ``app.prompts`` 的职责
    （``build_chat_messages``），这里接收已渲染好的 SystemMessage 与本轮
    HumanMessage。这样 Prompt 只有一处定义，不会在两个模块里各写一份。

    **这是 ch01–ch06 的兼容外壳**（ch07 起）。它走单层规划、只用 approx 口径、
    不碰锚点、不触发摘要：``tests/test_context.py`` 的 13 条用例钉的就是这个行为。
    新代码（图里的 Agent 节点）走 ``app.context.assemble.assemble_context``。
    """
    history = list(history)

    from app.context.assemble import build_legacy_context

    return build_legacy_context(
        history=history,
        question=question,
        system_message=system_message,
        budget_tokens=budget_tokens,
    )


def __getattr__(name: str):
    """``assemble_context`` / ``build_reference_block`` 按需从 ``assemble`` 导入。

    为什么不用模块顶部 ``from app.context.assemble import ...``：``assemble`` 需要
    ``from app.context import ContextResult``，模块顶部互相导入会循环。用
    ``__getattr__``（PEP 562）推迟到第一次访问，两边都能拿到干净的名字。
    """
    if name in {
        "assemble_context",
        "build_legacy_context",
        "build_reference_block",
        "select_summary_texts",
    }:
        from app.context import assemble as _assemble

        return getattr(_assemble, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContextResult",
    "assemble_context",
    "build_context",
    "build_reference_block",
    "select_summary_texts",
]
