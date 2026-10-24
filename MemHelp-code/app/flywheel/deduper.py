"""查重：让模型判断「新问题」与待审队列里已有的是否同一个意思。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.parsing import extract_json, message_text

logger = logging.getLogger(__name__)

__all__ = ["DedupeResult", "find_duplicate", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = """你是客服知识库的去重员。判断「新问题」与候选列表里的哪一条是同一个意思。

判断标准：用户真正想知道的规则相同，就算同义（措辞、语气、举例不同不算差异）。
如果都不确定，返回 null —— 宁可新建一行让人工合并，也不要把不同问题强行并掉。

只输出 JSON 对象，不要 markdown，不要解释：
{"matched_id": 123 或 null, "reason": "一句话理由"}"""


@dataclass(frozen=True)
class DedupeResult:
    matched_id: int | None
    reason: str
    degraded: bool = False


def _candidate_pairs(
    candidates: Sequence[Any],
) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for item in candidates:
        if isinstance(item, (tuple, list)):
            pairs.append((int(item[0]), str(item[1])))
        else:
            pairs.append((int(getattr(item, "id")), str(getattr(item, "normalized_question"))))
    return pairs


async def find_duplicate(
    model: Any,
    *,
    question: str,
    candidates: Sequence[Any],
    timeout: float = 15.0,
) -> DedupeResult:
    """队列为空时不调模型；模型给出候选外的 id 视为坏输出。"""
    pairs = _candidate_pairs(candidates)
    if not pairs:
        return DedupeResult(matched_id=None, reason="待审队列为空")

    listing = "\n".join(f"{review_id}. {text}" for review_id, text in pairs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"新问题：{question}\n\n候选：\n{listing}"),
    ]
    allowed = {review_id for review_id, _ in pairs}
    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
        data = extract_json(message_text(response))
        if data is None:
            raise ValueError("查重结果不是 JSON 对象")
    except Exception:  # noqa: BLE001 - 查重失败按「不重复」处理，缺口不能丢
        logger.warning("查重失败，按不重复处理（question=%r）", question, exc_info=True)
        return DedupeResult(matched_id=None, reason="查重失败，按不重复处理", degraded=True)

    reason = str(data.get("reason") or "").strip()
    raw_id = data.get("matched_id")
    if raw_id is None:
        return DedupeResult(matched_id=None, reason=reason or "模型认为没有重复")
    try:
        matched_id = int(raw_id)
    except (TypeError, ValueError):
        return DedupeResult(
            matched_id=None, reason="模型返回了非法 id，按不重复处理", degraded=True
        )
    if matched_id not in allowed:
        return DedupeResult(
            matched_id=None, reason="模型返回的 id 不在候选中，按不重复处理", degraded=True
        )
    return DedupeResult(matched_id=matched_id, reason=reason or "模型判定为同义问题")
