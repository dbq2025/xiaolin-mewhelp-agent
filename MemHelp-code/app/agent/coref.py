"""指代消解节点 —— 本章只做原样透传。

正式版要根据历史把「它 / 那个订单 / 刚才说的」改写成完整指代。本章刻意留一个
独立节点而不是在意图识别里顺手做：这样下一章替换实现时，图结构、State 字段与
日志都不动，只换一个函数体。
"""

from __future__ import annotations

__all__ = ["resolve_coref"]


def resolve_coref(message: str) -> tuple[str, bool]:
    """返回 `(resolved_message, changed)`。

    本章永远返回原话与 `False`；`changed` 进日志，方便以后对比正式版的改写率。
    """
    return message, False
