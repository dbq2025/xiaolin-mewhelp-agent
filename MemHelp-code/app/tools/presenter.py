"""工具结果呈现：只挑回答用得上的字段，把内部枚举码翻成人话。

工具（尤其 MCP 外部工具）的原始返回往往又长又带内部字段；直接塞给模型
既浪费上下文，也容易让模型把 ``SHIPPED`` 这种码原样念给用户。这里做两件事：

1. 按工具名取字段白名单（未列出的工具保留全部键，只去掉 ``None`` 与 ``_`` 前缀）；
2. ``status`` 这类枚举码翻译成 ``status_text``，中文原样透传。

``present_result`` 永不抛：解析失败就原样返回。
"""

from __future__ import annotations

import json
from typing import Any

from app.tools.contract import ToolSpec

ORDER_STATUS_TEXT: dict[str, str] = {
    "PENDING_PAYMENT": "待付款",
    "PAID": "待发货",
    "SHIPPED": "已发货",
    "DELIVERED": "已签收",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}

LOGISTICS_STATUS_TEXT: dict[str, str] = {
    "CREATED": "已揽收",
    "PICKED_UP": "已揽收",
    "IN_TRANSIT": "运输中",
    "OUT_FOR_DELIVERY": "派送中",
    "DELIVERED": "已签收",
    "EXCEPTION": "异常",
}

RETURN_STATUS_TEXT: dict[str, str] = {
    "APPLIED": "已申请",
    "REVIEWING": "审核中",
    "APPROVED": "已通过",
    "REFUNDING": "退款中",
    "REFUNDED": "已退款",
    "REJECTED": "已驳回",
}

#: 工具名 → 保留字段。MCP 工具与内置工具共用一张表。
_FIELDS: dict[str, tuple[str, ...]] = {
    "query_order": (
        "order_id", "status", "amount_yuan", "product_name", "created_at",
    ),
    "track_shipment": (
        "order_id", "carrier", "tracking_no", "status", "nodes", "eta",
    ),
    "check_warranty": (
        "order_id", "product_name", "in_warranty", "warranty_until", "policy",
    ),
    "check_return_progress": (
        "order_id", "return_no", "status", "steps", "refund_amount_yuan",
    ),
}

_STATUS_TABLE: dict[str, dict[str, str]] = {
    "query_order": ORDER_STATUS_TEXT,
    "track_shipment": LOGISTICS_STATUS_TEXT,
    "check_return_progress": RETURN_STATUS_TEXT,
}


def _as_mapping(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def present_result(spec: ToolSpec, raw: Any) -> Any:
    """把工具原始返回整理成适合回灌模型的形状。永不抛。"""
    try:
        data = _as_mapping(raw)
        if data is None:
            return raw

        allowed = _FIELDS.get(spec.name)
        picked = {
            key: value
            for key, value in data.items()
            if not key.startswith("_") and value is not None
            and (allowed is None or key in allowed)
        }

        table = _STATUS_TABLE.get(spec.name)
        if table is not None and "status" in picked:
            status = picked["status"]
            picked["status_text"] = (
                table.get(str(status), str(status)) if table else str(status)
            )
        return picked
    except Exception:  # pragma: no cover - 防御性兜底
        return raw


def summarize_result(spec: ToolSpec, data: Any, *, limit: int = 500) -> str:
    """给审计表用的短摘要：中文不转义，超长截断。永不抛。"""
    try:
        presented = present_result(spec, data) if not isinstance(data, dict) else data
        if isinstance(presented, str):
            text = presented
        else:
            text = json.dumps(presented, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - 防御性兜底
        text = str(data)
    if limit > 0 and len(text) > limit:
        return text[: limit - 1] + "…"
    return text
