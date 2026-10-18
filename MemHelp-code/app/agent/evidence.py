"""工具出参 → 证据 → 引用快照。

## 为什么单独一层

`query_faq` 的出参是跨章节契约（ch02 的 `{ok, keyword, count, items[]}`），
Agent 循环不该为了「知道哪条被引用了」去改工具契约。这里把出参还原成
`Evidence`，再用 ch04 的 `parse_citations` 只保留正文真的引用到的编号。

## 真实出参没有 `quote`

ch04 为了压住 executor 的 4000 字上限，把重复的 `quote` 从 `query_faq` 出参里
删掉了，只留 `answer`。所以这里必须容忍 `quote` 缺失，回落到 `answer`；
否则用户点开引用会看到空白来源。
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Sequence

from app.knowledge.citations import (
    Evidence,
    evidence_from_payload,
    evidence_payload,
    parse_citations,
)
from app.tools.executor import ToolResult

logger = logging.getLogger(__name__)

__all__ = ["citations_for_answer", "evidence_from_tool_result"]


def _as_mapping(data: Any) -> dict[str, Any] | None:
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


async def evidence_from_tool_result(
    result: ToolResult, repositories: Any
) -> tuple[Evidence, ...]:
    """从 `query_faq` 的工具结果还原证据；形状不对返回空，不抛。"""
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
        int(item["chunk_id"])
        for item in items
        if item.get("truncated") and item.get("chunk_id") is not None
    ]
    if not truncated_ids:
        return evidence

    try:
        rows = {row.id: row.answer for row in await repositories.knowledge.get_many(truncated_ids)}
    except Exception:  # noqa: BLE001 - 取全文失败不该毁掉整轮回答
        logger.warning("引用原文回库补全失败，退回节选", exc_info=True)
        return evidence

    return tuple(
        replace(item, quote=rows[item.chunk_id]) if item.chunk_id in rows else item
        for item in evidence
    )


def citations_for_answer(answer: str, evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    """只把正文真的引用了的证据写进快照。

    与 ch02 的 `_valid_citations` 同一口径：快照回答「这段话依据了哪几条」，
    不是「当时手里有什么」。越界角标只丢映射、不改正文。
    """
    if not evidence:
        return []
    parsed = parse_citations(answer, evidence)
    if parsed.dropped:
        logger.warning(
            "正文里有 %d 个角标映射不到本轮证据，已忽略（编号=%s）",
            len(parsed.dropped),
            list(parsed.dropped),
        )
    wanted = set(parsed.valid)
    return [item for item in evidence_payload(evidence) if item["n"] in wanted]
