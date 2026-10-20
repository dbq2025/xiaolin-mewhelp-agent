"""上下文拼装 —— 决定「发给模型的列表长什么样、按什么顺序」。

## 顺序是硬件约束，不是风格偏好

```
[0] SystemMessage(人设 + 红线 + 工具定义)   ← 每轮字节一致，前缀缓存的锚点
[1..] 层 2：规则截短过的中间历史
[..] 层 1：最近几轮原文
[..] HumanMessage(用户当前这句话)
[..] HumanMessage([会话梗概] … [检索证据] …)
```

三条硬约束（``tests/test_ch07_assemble.py`` 逐条钉住）：

1. **SystemMessage 只有第 0 条，且同一会话每轮字节一致。** 梗概与检索证据
   **不许**进 system —— 上游 chat template 会把所有 system 消息上提到最前面
   合并渲染，一旦 system 里掺了每轮都变的内容，工具定义就被挤到可变内容之后，
   前缀缓存整段作废。这个错误不会报错，只会让账单默默翻几倍。
2. **梗概与证据合成一条**，不是两条。它们都是「本轮参考资料」，属于同一次输入。
3. **用户当前这句话永远在梗概/证据之前。** 模型读到最后一条 human 是参考资料
   时，仍能从前一条 human 看到真正要回答的问题。

## 本模块的历史

``build_legacy_context`` 是 ch01–ch06 那个唯一的 ``build_context`` 的原地搬迁，
行为逐字节不变（``tests/test_context.py`` 13 条用例钉死它）。ch07 起真正的
三层拼装走 ``assemble_context``，两者共用 ``ContextResult``。
"""

from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.context import ContextResult
from app.context.tokens import count_tokens


def build_reference_block(*, summary_text: str = "", evidence_text: str = "") -> str:
    """把「会话梗概」与「检索证据」拼成**一条**参考消息的正文。

    为什么要合成一条：它们都是「本轮参考资料」，属于同一次输入的一部分。
    分成两条会让消息列表多出一个语义位置，将来任何按「最后一条 human」定位
    当前问题的代码都得处理两种情况。

    任一段为空就整段省略；两段都空返回空串（调用方据此决定要不要追加这条）。
    """
    parts: list[str] = []
    if summary_text:
        parts.append(f"[会话梗概]\n{summary_text}")
    if evidence_text:
        parts.append(f"[检索证据]\n{evidence_text}")
    return "\n".join(parts)


def assemble_context(
    *,
    system_message: SystemMessage,
    question: BaseMessage,
    layer2: Sequence[BaseMessage] = (),
    layer1: Sequence[BaseMessage] = (),
    summary_text: str = "",
    evidence_text: str = "",
    budget_tokens: int = 0,
    counter: Callable[[Sequence[BaseMessage]], int] = count_tokens,
    summary_upto_msg_id: int | None = None,
    layer1_from_msg_id: int | None = None,
    degraded: bool = False,
    summary_due: bool = False,
    trimmed_messages: int = 0,
) -> ContextResult:
    """按固定顺序拼出「这一轮发给模型的列表」。

    顺序（**不可调换**）::

        [0] SystemMessage（人设 + 红线 + 工具定义）
        [1..] 层 2：规则截短过的中间历史
        [..] 层 1：最近几轮原文
        [..] HumanMessage(用户当前这句话)
        [..] HumanMessage([会话梗概] … [检索证据] …)   ← 仅当有内容时

    三条硬约束见模块 docstring。这个函数**不裁剪层 1** —— 裁剪是
    ``layers.plan_layers`` 的职责，到这里时层 1 已经是裁好的。分开的理由：
    裁剪要用 ``trim_messages`` 的二分，而拼装要让「顺序」可被单独断言。
    """
    layer2 = list(layer2)
    layer1 = list(layer1)

    messages: list[BaseMessage] = [system_message, *layer2, *layer1, question]

    reference = build_reference_block(summary_text=summary_text, evidence_text=evidence_text)
    if reference:
        messages.append(HumanMessage(content=reference))

    kept = layer2 + layer1
    return ContextResult(
        messages=messages,
        kept_messages=len(kept),
        trimmed_messages=trimmed_messages,
        budget_tokens=budget_tokens,
        used_tokens=counter(kept),
        system_prompt_tokens=counter([system_message]),
        summary_tokens=counter([HumanMessage(content=reference)]) if reference else 0,
        layer2_messages=len(layer2),
        layer1_messages=len(layer1),
        layer1_from_msg_id=layer1_from_msg_id,
        summary_upto_msg_id=summary_upto_msg_id,
        degraded=degraded,
        summary_due=summary_due,
    )


def build_legacy_context(
    *,
    history: Sequence[BaseMessage],
    question: BaseMessage,
    system_message: SystemMessage,
    budget_tokens: int,
) -> ContextResult:
    """ch01–ch06 的单层裁剪，逐字节复刻。

    只做裁剪与拼装，**不构造消息内容** —— 模板渲染是 ``app/prompts.py`` 的职责，
    这里接收已渲染好的 SystemMessage 与本轮 HumanMessage。这样 Prompt 只有一处
    定义，不会在两个模块里各写一份。

    刻意保留 ``token_counter="approximate"``：换成校准口径会让同样的历史被裁得
    更狠，ch01–ch06 的 13 条断言（含 ``used_tokens`` 的具体数字）立刻全红。
    新路径用校准口径，旧外壳用 approx，两条互不串。
    """
    history = list(history)

    kept = trim_messages(
        history,
        max_tokens=budget_tokens,
        # 官方推荐用于热路径的近似计数：按字符估算，中文场景偏低估。
        # 旧外壳保留它，只是为了不改 ch01–ch06 的既有断言。
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

    messages: list[BaseMessage] = [system_message, *kept, question]

    return ContextResult(
        messages=messages,
        kept_messages=len(kept),
        trimmed_messages=len(history) - len(kept),
        budget_tokens=budget_tokens,
        used_tokens=count_tokens_approximately(kept),
        system_prompt_tokens=count_tokens_approximately([system_message]),
        # 旧外壳没有分层：保留的历史全算层 1，层 2 / 梗概 / 锚点都是「没发生」。
        layer1_messages=len(kept),
    )


__all__ = ["assemble_context", "build_legacy_context", "build_reference_block"]
