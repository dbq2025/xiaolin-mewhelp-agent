"""退款/售后子流程的订单槽位与前端卡片数据。

## 为什么订单号是确定性抽取

「缺订单号不让模型猜」：模型可能把「那个订单」补成一个历史里不存在的编号。
这里只用正则从补全后的问题里抽，抽不到就弹选择器；选择器候选也由 session
确定性派生，同一会话刷新/重跑拿到同一组订单。

## 为什么复用 `query_order` 的演示数据

ch02 的三个业务工具本来就是「按编号确定性派生」的演示实现；这里不另造一套订单
数据，而是先选编号、再调 `query_order` / `query_product`，保证卡片上的金额、
状态、商品名与 Agent 后续查到的一致。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from app.tools.business import derive_rng, query_order, query_product

__all__ = [
    "REFUND_REASONS",
    "build_refund_form",
    "extract_order_id",
    "extract_order_tokens",
    "is_general_refund_policy",
    "list_demo_orders",
]

#: 退款原因固定六类。前端下拉只从这里取，后端用同一份校验。
REFUND_REASONS: tuple[str, ...] = (
    "商品质量问题",
    "尺寸/颜色/型号不合适",
    "与描述不符",
    "不想要了（七天无理由）",
    "物流破损/少件",
    "其他",
)

#: 「订单 1001」「订单号：1001」「单号 2024-1001-AB」。
_ORDER_ARG = re.compile(
    r"(?:订单(?:号)?|单号)(?:是|为)?[：:\s]*([A-Za-z0-9-]{4,32})"
)
#: 独立的长数字订单号（8–20 位）。
_ORDER_STANDALONE = re.compile(r"(?<!\d)(\d{8,20})(?!\d)")


#: 通用政策问句的标记词。
_POLICY_MARKERS: tuple[str, ...] = (
    "政策", "规则", "流程", "怎么退", "多久", "条件", "运费", "到账",
    "时效", "七天", "无理由", "需要什么", "材料", "标准",
)
#: 出现这些词说明问题指向「某一单」，不能当通用政策。
_DEICTIC_MARKERS: tuple[str, ...] = (
    "这", "那", "它", "我", "订单", "该", "此", "本单",
)


def is_general_refund_policy(text: str) -> bool:
    """是不是「不指向具体某一单」的通用退款/售后政策问题。

    例：``退货政策是什么``、``退款流程要多久``、``退货运费谁承担`` →
    跳过订单选择器，直接走扩写+政策检索+Agent；``这个能退吗``、
    ``我的订单能退吗`` 仍走订单槽位。
    """
    source = (text or "").strip()
    if not source or extract_order_id(source):
        return False
    if any(marker in source for marker in _DEICTIC_MARKERS):
        return False
    return any(marker in source for marker in _POLICY_MARKERS)


def extract_order_tokens(text: str) -> set[str]:
    """抽取文本里所有「像订单号」的 token，用于 coref 防幻觉的整体比对。

    与 :func:`extract_order_id` 同一套正则：``订单/单号 + 4–32 位字母数字``，
    以及独立 8–20 位数字。coref 不能用子串判断（``20240115`` 是
    ``20240115001`` 的前缀），必须比较 token 集合。
    """
    source = text or ""
    return {
        *(_ORDER_ARG.findall(source)),
        *(_ORDER_STANDALONE.findall(source)),
    }


def extract_order_id(text: str) -> str | None:
    """从问题里抽订单号；抽不到返回 ``None``，绝不让模型猜。"""
    source = text or ""
    for pattern in (_ORDER_ARG, _ORDER_STANDALONE):
        match = pattern.search(source)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _one_order(order_id: str) -> dict[str, Any]:
    order = json.loads(query_order(order_id))
    product = json.loads(query_product(str(order["product_id"])))
    name = str(product.get("name") or "商品")
    model = str(product.get("model") or "")
    title = f"{name} {model}".strip()
    return {
        "order_id": order_id,
        "title": title,
        "amount_yuan": str(order.get("amount_yuan") or ""),
        "status": str(order.get("status") or ""),
        "created_at": str(order.get("created_at") or ""),
        "product_id": str(order.get("product_id") or ""),
        "product_name": name,
        "product_model": model,
    }


def list_demo_orders(session_id: str, count: int = 3) -> list[dict[str, Any]]:
    """按 session 确定性派生订单卡片。

    同一 ``session_id`` 永远拿到同一组订单；不同 session 的编号/商品不同。
    演示实现不查真实电商系统，也不建 orders 表。
    """
    wanted = max(1, int(count))
    rng = derive_rng("demo-orders", session_id or "anonymous")
    seen: set[str] = set()
    order_ids: list[str] = []
    attempts = 0
    while len(order_ids) < wanted and attempts < wanted * 20:
        attempts += 1
        candidate = f"2024{rng.randint(1_000_000, 9_999_999)}"
        if candidate in seen:
            continue
        seen.add(candidate)
        order_ids.append(candidate)

    return [_one_order(order_id) for order_id in order_ids]


def build_refund_form(order: Mapping[str, Any]) -> dict[str, Any]:
    """退款单的前端契约：订单只读、原因固定六类。"""
    name = str(order.get("product_name") or "商品")
    model = str(order.get("product_model") or "")
    amount = str(order.get("amount_yuan") or "")
    label = f"{name} {model}".strip()
    if amount:
        label = f"{label} · ¥{amount}"
    return {
        "order_id": str(order.get("order_id") or ""),
        "order_label": label,
        "reasons": list(REFUND_REASONS),
        "submit_url": "/api/actions/create-refund",
    }
