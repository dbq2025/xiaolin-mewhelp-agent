"""退款/售后 Query 扩写 —— 检索侧现查现用，入库侧不拆存。

## 为什么只在检索侧

同一份知识在库里只保留一份；「这个能退吗」扩成「退货时间窗口 / 退货运费谁承担」
只是查询时的临时视图。入库侧拆多份会让同一知识重复计分，更新时多份漂移。

## 强制 JSON

字段只有 `queries` 一个数组；解析器拒绝缺字段/空数组，失败回退原问题。
模型输出里多出来的字段忽略 —— 提示要求不要有，但容忍不是错误。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import QUERY_EXPANSION_PROMPT
from app.chat import _message_text

logger = logging.getLogger(__name__)

__all__ = ["ExpansionResult", "QueryExpander", "parse_expansion_reply"]


@dataclass(frozen=True)
class ExpansionResult:
    """一次扩写的结果。"""

    queries: tuple[str, ...]
    raw: str
    degraded: bool


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


def parse_expansion_reply(
    reply: str, question: str, *, max_queries: int
) -> tuple[str, ...] | None:
    """解析 `{"queries": [...]}`；去空、去重、截断，并保证原问题在结果里。"""
    text = _strip_code_fence((reply or "").strip())
    parsed = _loads_json(text)
    if not isinstance(parsed, dict):
        return None

    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        return None

    queries: list[str] = []
    for item in raw_queries:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value and value not in queries:
            queries.append(value)
    if not queries:
        return None

    original = (question or "").strip()
    if original and original not in queries:
        queries.insert(0, original)

    limit = max(1, int(max_queries))
    return tuple(queries[:limit])


class QueryExpander:
    """一次扩写；**永不抛**。"""

    def __init__(
        self,
        model: object,
        *,
        timeout: float = 5.0,
        max_queries: int = 4,
        enabled: bool = True,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._max_queries = max_queries
        self._enabled = enabled

    async def expand(
        self, question: str, *, order: Mapping[str, object] | None = None
    ) -> ExpansionResult:
        original = (question or "").strip()
        if not original:
            return ExpansionResult(queries=(), raw="", degraded=True)
        if not self._enabled:
            return ExpansionResult(queries=(original,), raw="", degraded=True)

        order_payload = ""
        if order:
            picked = {
                key: order.get(key)
                for key in ("order_id", "product_name", "product_model", "status", "amount_yuan")
                if order.get(key)
            }
            order_payload = json.dumps(picked, ensure_ascii=False)

        user = f"用户问题：{original}"
        if order_payload:
            user += f"\n订单信息：{order_payload}"

        raw = ""
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke(  # type: ignore[attr-defined]
                    [
                        SystemMessage(content=QUERY_EXPANSION_PROMPT),
                        HumanMessage(content=user),
                    ]
                ),
                timeout=self._timeout,
            )
            raw = _message_text(response)
            parsed = parse_expansion_reply(
                raw, original, max_queries=self._max_queries
            )
            if parsed is not None:
                return ExpansionResult(queries=parsed, raw=raw, degraded=False)
        except Exception:  # noqa: BLE001 - 扩写失败回落单查询
            logger.warning("Query 扩写失败，回落原问题（question=%r）", original, exc_info=True)

        return ExpansionResult(queries=(original,), raw=raw, degraded=True)
