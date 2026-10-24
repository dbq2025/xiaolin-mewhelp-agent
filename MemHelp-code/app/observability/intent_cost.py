"""按意图聚合 Langfuse 的 token 与成本。

意图在 trace 里写成 ``tags=["intent:<intent>"]``（见 observability.tracing.tag_intent），
Langfuse metrics API 支持按 tags 维度分组，因此这里只需要把返回行折叠成
「每个意图一行」，不依赖任何本地日志。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import httpx

from app.config import Settings

INTENT_TAG_PREFIX = "intent:"
METRICS_PATH = "/api/public/v2/metrics"


class LangfuseNotConfigured(RuntimeError):
    """没有配置 Langfuse 凭据，无法查询云端。"""


@dataclass(frozen=True)
class IntentCost:
    intent: str
    tokens: int
    cost: float
    observations: int


def _intent_of(tags: Sequence[str] | None) -> str | None:
    for tag in tags or ():
        if isinstance(tag, str) and tag.startswith(INTENT_TAG_PREFIX):
            intent = tag[len(INTENT_TAG_PREFIX):].strip()
            if intent:
                return intent
    return None


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate_intent_costs(rows: Sequence[dict]) -> list[IntentCost]:
    """把 metrics 返回行折叠成按 tokens 降序的意图列表。

    一行可能带多个 tags（例如 ``intent:refund`` + ``env:dev``）；只取第一个
    ``intent:`` 前缀的 tag，没有该前缀的行直接丢弃（不计入「未知意图」）。
    """
    grouped: "OrderedDict[str, dict[str, float]]" = OrderedDict()
    for row in rows:
        intent = _intent_of(row.get("tags"))
        if intent is None:
            continue
        bucket = grouped.setdefault(intent, {"tokens": 0.0, "cost": 0.0, "rows": 0})
        bucket["tokens"] += _number(row.get("sum_totalTokens"))
        bucket["cost"] += _number(row.get("sum_totalCost"))
        bucket["rows"] += 1

    items = [
        IntentCost(
            intent=intent,
            tokens=int(bucket["tokens"]),
            cost=round(bucket["cost"], 6),
            observations=int(bucket["rows"]),
        )
        for intent, bucket in grouped.items()
    ]
    items.sort(key=lambda item: (-item.tokens, item.intent))
    return items


def _configured(settings: Settings) -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_base_url
    )


async def fetch_intent_costs(
    settings: Settings,
    *,
    days: int = 7,
    client: httpx.AsyncClient | None = None,
) -> list[IntentCost]:
    """查 Langfuse metrics API（tags 维度），返回按意图聚合的成本。"""
    if not _configured(settings):
        raise LangfuseNotConfigured("Langfuse 凭据未配置")

    now = datetime.now(timezone.utc)
    query = {
        "view": "observations",
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": (now - timedelta(days=max(1, days))).isoformat(),
        "toTimestamp": now.isoformat(),
    }
    url = f"{settings.langfuse_base_url.rstrip('/')}{METRICS_PATH}"
    auth = httpx.BasicAuth(settings.langfuse_public_key, settings.langfuse_secret_key)

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await http.get(url, params={"query": _json(query)}, auth=auth)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            await http.aclose()

    return aggregate_intent_costs(payload.get("data") or [])


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
