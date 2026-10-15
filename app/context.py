"""上下文裁剪 —— 全项目**唯一**知道 token 预算的地方。

本模块不碰网络、不碰磁盘，所以可以完全离线测试。

## 顺序是算法的一部分

固定为：``[System] + trim(历史, 预算) + [本轮问题]``

**先裁历史，再拼本轮问题**。如果图省事把本轮问题一起丢进 ``trim_messages``，
预算紧张时用户刚问的那句话会被当成「最旧的消息」裁掉，模型就会答非所问 ——
这是本章最容易踩的坑，``tests/test_context.py`` 有一条测试专门钉住它。

## 预算的统计范围

``budget_tokens`` **只统计历史消息**，不含 System Prompt，也不含本轮问题。
理由：System Prompt 是固定的角色设定，本轮问题必须保留，两者都没有裁剪空间；
拿它们去挤历史只会让语义更差。
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


def build_context(
    history: Sequence[BaseMessage],
    message: str,
    system_prompt: str,
    budget_tokens: int,
) -> ContextResult:
    """把历史裁剪到预算内，再接上本轮问题，产出最终模型输入。"""
    history = list(history)

    kept = trim_messages(
        history,
        max_tokens=budget_tokens,
        # 官方推荐用于热路径的近似计数：按字符估算，中文场景偏保守（估多），
        # 安全方向；不为它引入 tiktoken 依赖。
        token_counter="approximate",
        # 保留最近的消息，丢最旧的。
        strategy="last",
        # 历史里如果有 system 消息（例如将来的压缩摘要）也保留。
        include_system=True,
        # 保证保留的历史从 human 开始，避免以 ai 消息开头。
        start_on="human",
        # 不切半条消息：宁可少留一条，也不要把一句话截断成半个。
        allow_partial=False,
    )

    messages: list[BaseMessage] = [SystemMessage(system_prompt), *kept, HumanMessage(message)]

    return ContextResult(
        messages=messages,
        kept_messages=len(kept),
        trimmed_messages=len(history) - len(kept),
        budget_tokens=budget_tokens,
        used_tokens=count_tokens_approximately(kept),
        system_prompt_tokens=count_tokens_approximately([SystemMessage(system_prompt)]),
    )
