"""校准的 token 计数口径 —— 全项目**唯一**的「字数折 token」算法。

## 为什么不用 ``count_tokens_approximately``

LangChain 的 ``count_tokens_approximately`` 默认 ``chars_per_token=4.0``，也就是
「4 个字符 = 1 token」。这个口径对英文大体没问题，**对中文严重低估**。实测：

| 文本 | approx 默认口径 | 真实分词（量级） |
|---|---|---|
| 「我上周买的耳机还没发货，订单号 A123456789」 | ~17 | 40–50 |

低估的方向是**危险**的那一侧：预算以为还剩很多空间，实际已经把窗口撑爆。
反过来，如果把预算按「1 汉字 = 1 token」收紧、却继续用 approx 计数，就会
每轮都判超预算、压缩永远触发 —— **预算与计数器必须同源**，只改一个是负收益。

## 为什么是纯函数

``trim_messages`` 内部用**前缀 token 数二分**找可容纳的最大条数，一次裁剪会
调用计数器 O(log n) 次。如果每次调用都去读一遍 ``Settings``，热路径上就是纯浪费，
而且这个模块会变得离不开配置文件、没法独立断言。所以比例全部由调用方注入，
``make_token_counter`` 只是把配置绑成一个闭包。

## 口径定义

```
每条消息 = ceil(汉字/全角标点 × cjk_chars_per_token
              + 其余字符 ÷ other_chars_per_token)
          + message_overhead
```

``message_overhead`` 默认 3：每条消息在 chat template 里还有分隔符与角色标签的
固定开销，与正文长度无关。**role 名本身不计字符** —— 否则同一句正文当 user
与当 ai 会算出不同 token，白白引入不一致。
"""

from __future__ import annotations

import math
import json
from typing import Any, Callable, Sequence

from langchain_core.messages import BaseMessage

#: 一个汉字大致对应几个 token。1.0 是「不低估」的保守取值。
CJK_CHARS_PER_TOKEN_DEFAULT: float = 1.0

#: 非 CJK 字符大致几个字符一个 token。3.5 比常见的 4.0 略保守。
OTHER_CHARS_PER_TOKEN_DEFAULT: float = 3.5

#: 每条消息的固定开销（角色标签、分隔符）。
MESSAGE_OVERHEAD_TOKENS_DEFAULT: int = 3


def is_cjk(ch: str) -> bool:
    """这个字符算不算「中文侧」—— 汉字、CJK 标点、全角符号。

    - ``U+4E00–U+9FFF``：CJK 统一表意文字（常用汉字）
    - ``U+3400–U+4DBF``：CJK 扩展 A
    - ``U+3000–U+303F``：CJK 标点（、。「」等）
    - ``U+FF00–U+FFEF``：全角字符（，。！？：；（）等）

    全角标点必须算进来：中文句子里它们的密度和汉字一样，按 3.5 字符/token
    算会让一整句标点密的中文被腰斩。
    """
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def count_text(
    text: str,
    *,
    cjk_chars_per_token: float = CJK_CHARS_PER_TOKEN_DEFAULT,
    other_chars_per_token: float = OTHER_CHARS_PER_TOKEN_DEFAULT,
) -> int:
    """把一段文本折成 token 数。空串得 0。

    两侧分别累加再一次性向上取整 —— 不是「两边各自取整再相加」，
    否则一段短文本会被多加一次 token。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    total = cjk / cjk_chars_per_token + other / other_chars_per_token
    return math.ceil(total)


def _content_text(content: Any) -> str:
    """把消息正文摊成字符串。

    ``content`` 可能是 ``str``，也可能是多模态的 ``list[dict]``（新版 LangChain
    也用 list 表示 content blocks）。这里只求**稳定**，不求精确：list 走
    ``json.dumps(ensure_ascii=False)``，让「同样的结构算出同样的数」成立。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # 文本块直接取 text，其余（图片等）用 JSON 形态占位。
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_calls_text(message: BaseMessage) -> str:
    """工具调用的名字与参数也是真发出去的内容。

    纯工具调用那行 ``AIMessage.content`` 常常是空串，如果只数 ``content``，
    这类消息会被当成 0 token —— 但模型确实收到了 ``query_order(order_id=…)``。
    """
    calls = getattr(message, "tool_calls", None) or ()
    if not calls:
        return ""
    parts: list[str] = []
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        parts.append(str(name))
        if args:
            parts.append(json.dumps(args, ensure_ascii=False, default=str))
    return "".join(parts)


def count_message(
    message: BaseMessage,
    *,
    cjk_chars_per_token: float = CJK_CHARS_PER_TOKEN_DEFAULT,
    other_chars_per_token: float = OTHER_CHARS_PER_TOKEN_DEFAULT,
    message_overhead: int = MESSAGE_OVERHEAD_TOKENS_DEFAULT,
) -> int:
    """单条消息的 token 数 = 正文 + 工具调用参数 + 固定开销。"""
    text = _content_text(message.content) + _tool_calls_text(message)
    return (
        count_text(
            text,
            cjk_chars_per_token=cjk_chars_per_token,
            other_chars_per_token=other_chars_per_token,
        )
        + message_overhead
    )


def count_tokens(
    messages: Sequence[BaseMessage],
    *,
    cjk_chars_per_token: float = CJK_CHARS_PER_TOKEN_DEFAULT,
    other_chars_per_token: float = OTHER_CHARS_PER_TOKEN_DEFAULT,
    message_overhead: int = MESSAGE_OVERHEAD_TOKENS_DEFAULT,
) -> int:
    """一批消息的 token 数。

    签名刻意与 ``count_tokens_approximately`` 的调用形态对齐
    （``Callable[[list[BaseMessage]], int]``），可以直接喂给
    ``trim_messages(token_counter=...)``。
    """
    return sum(
        count_message(
            message,
            cjk_chars_per_token=cjk_chars_per_token,
            other_chars_per_token=other_chars_per_token,
            message_overhead=message_overhead,
        )
        for message in messages
    )


def make_token_counter(
    *,
    cjk_chars_per_token: float = CJK_CHARS_PER_TOKEN_DEFAULT,
    other_chars_per_token: float = OTHER_CHARS_PER_TOKEN_DEFAULT,
    message_overhead: int = MESSAGE_OVERHEAD_TOKENS_DEFAULT,
) -> Callable[[Sequence[BaseMessage]], int]:
    """把一组比例绑成闭包，交给 ``trim_messages`` 用。

    ``trim_messages`` 会在内部做**前缀 token 数二分**，每次都调这个闭包；
    绑定后不再解析参数，是热路径上最省的一种形态。
    """

    def _counter(messages: Sequence[BaseMessage]) -> int:
        return count_tokens(
            messages,
            cjk_chars_per_token=cjk_chars_per_token,
            other_chars_per_token=other_chars_per_token,
            message_overhead=message_overhead,
        )

    return _counter


def counter_from_settings(settings: Any) -> Callable[[Sequence[BaseMessage]], int]:
    """按配置造计数器。``settings`` 只需要有三个属性，鸭子类型即可。

    单独开这个函数，是为了让「预算」与「计数器」读**同一份**配置：
    只要调用方都从这里取，就不会出现「预算按 4 字符/token、计数按 1 字/token」
    这种只改一个的翻车。
    """
    return make_token_counter(
        cjk_chars_per_token=float(getattr(settings, "cjk_chars_per_token", 1.0)),
        other_chars_per_token=float(getattr(settings, "other_chars_per_token", 3.5)),
        message_overhead=int(getattr(settings, "message_overhead_tokens", 3)),
    )
