"""指代消解 + Query 改写 —— 把「它能退吗」补成可独立阅读的完整问题。

## 什么时候调模型

不是每轮都调：完整问句直接透传，寒暄也直接透传（ch05 的零调用保证不能被吃掉）。
只有出现指代/省略/口语标记，或首轮很短且没有业务名词时才调一次模型。

## 为什么要防幻觉

模型完全可能把「那个订单」补成一个历史里不存在的订单号，而下游退款子流程会拿它
去查订单 —— 用户看到的是一笔别人的订单。所以补全结果里**新出现的订单号/型号**
必须能在「原话 + 历史」里找到；找不到就整句回退原话，宁可少补。

## 失败语义

超时、异常、坏 JSON、空结果一律回退原话（`source="fallback"`），不阻断聊天；
`resolved_message` 永远是非空字符串。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.intent import is_trivial_chitchat
from app.agent.prompts import COREF_PROMPT
from app.chat import _message_text

logger = logging.getLogger(__name__)

__all__ = [
    "CorefResult",
    "CorefResolver",
    "format_history",
    "needs_coref",
    "parse_coref_reply",
]

#: 指代/省略标记。命中任意一个才值得叫模型。
_ANAPHORA_MARKERS: tuple[str, ...] = (
    "它", "他", "她", "这个", "那个", "这单", "那单", "这款", "那款",
    "这台", "那台", "刚才", "上面", "前面", "还有", "再", "呢",
)

#: 口语标记。
_COLLOQUIAL_MARKERS: tuple[str, ...] = (
    "啥", "咋", "多久", "多少", "怎么", "能不能", "可以不可以", "要不要",
)

_MAX_RESOLVED_CHARS = 512
_MAX_HISTORY_CHARS = 200

_ORDER_TOKEN = re.compile(r"(?<!\d)(\d{8,20})(?!\d)")
_MODEL_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-\d{2,}[A-Za-z]?)(?![A-Za-z0-9])")


@dataclass(frozen=True)
class CorefResult:
    """一次指代消解的产物。"""

    resolved: str
    changed: bool
    raw: str
    source: Literal["llm", "passthrough", "fallback"]
    reason: str = ""


def needs_coref(message: str, history: Sequence[BaseMessage]) -> bool:
    """这句话值不值得叫一次模型？"""
    text = (message or "").strip()
    if not text or is_trivial_chitchat(text):
        return False
    if any(marker in text for marker in _ANAPHORA_MARKERS):
        return True
    # 带明确订单号/型号的问题已经足够自包含；不要因为句子里有「多久/多少」
    # 这种疑问词就多付一次改写调用，也避免把完整问句改坏。
    if _ORDER_TOKEN.search(text) or _MODEL_TOKEN.search(text):
        return False
    if any(marker in text for marker in _COLLOQUIAL_MARKERS):
        return True
    return False


def format_history(messages: Sequence[BaseMessage], limit: int) -> str:
    """把最近的 Human/AI 文本拼成 Prompt 里的「历史」。

    忽略 ToolMessage 与带 tool_calls 的中间 assistant：它们对「它指谁」没有帮助，
    还会把工具 JSON 噪音塞进补全 Prompt。每条截断到 200 字。
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                continue
            role = "客服"
        else:
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        lines.append(f"{role}：{text[:_MAX_HISTORY_CHARS]}")

    if limit <= 0:
        return ""
    return "\n".join(lines[-limit:])


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def _loads_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None


def _has_hallucinated_tokens(resolved: str, allowed_text: str) -> bool:
    """补全结果里有没有历史中不存在的订单号/型号。"""
    allowed_lower = allowed_text.lower()
    for token in _ORDER_TOKEN.findall(resolved):
        if token not in allowed_text:
            return True
    for token in _MODEL_TOKEN.findall(resolved):
        if token.lower() not in allowed_lower:
            return True
    return False


def parse_coref_reply(
    reply: str, original: str, history_text: str
) -> tuple[str, bool] | None:
    """解析补全 JSON；结构不对或编造编号时返回 None。"""
    text = _strip_code_fence((reply or "").strip())
    parsed = _loads_json(text)
    if not isinstance(parsed, dict):
        return None

    resolved = parsed.get("resolved")
    if not isinstance(resolved, str):
        return None
    resolved = resolved.strip()
    if not resolved or len(resolved) > _MAX_RESOLVED_CHARS:
        return None

    allowed = f"{original}\n{history_text}"
    if _has_hallucinated_tokens(resolved, allowed):
        logger.warning("指代消解出现历史中不存在的编号，回退原话：%r", resolved[:120])
        return None

    changed = resolved != (original or "").strip()
    return resolved, changed


class CorefResolver:
    """一次指代消解；**永不抛**。"""

    def __init__(
        self,
        model: object,
        *,
        timeout: float = 5.0,
        history_limit: int = 8,
        enabled: bool = True,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._history_limit = history_limit
        self._enabled = enabled

    async def resolve(
        self, message: str, history: Sequence[BaseMessage] = ()
    ) -> CorefResult:
        original = (message or "").strip()
        if not self._enabled or not needs_coref(original, history):
            return CorefResult(
                resolved=original or (message or ""),
                changed=False,
                raw="",
                source="passthrough",
            )

        history_text = format_history(history, self._history_limit)
        user = f"历史：\n{history_text or '（空）'}\n\n当前：{original}"
        raw = ""
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke(  # type: ignore[attr-defined]
                    [SystemMessage(content=COREF_PROMPT), HumanMessage(content=user)]
                ),
                timeout=self._timeout,
            )
            raw = _message_text(response)
            parsed = parse_coref_reply(raw, original, history_text)
            if parsed is not None:
                resolved, changed = parsed
                return CorefResult(
                    resolved=resolved, changed=changed, raw=raw, source="llm"
                )
            reason = "模型输出无法解析或含编造编号"
        except Exception as exc:  # noqa: BLE001 - 补全失败不该打断聊天
            raw = f"{type(exc).__name__}: {exc}"[:200]
            reason = "指代消解调用失败"
            logger.warning("指代消解失败，回退原话（query=%r）", original, exc_info=True)

        return CorefResult(
            resolved=original,
            changed=False,
            raw=raw,
            source="fallback",
            reason=reason,
        )
