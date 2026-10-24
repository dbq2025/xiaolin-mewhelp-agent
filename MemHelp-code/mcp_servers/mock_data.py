"""两个业务 MCP Server 的返回数据契约。

这些 Server 是**独立进程**里的 mock：不接真实物流/售后系统，也不建表。
同一个 ``order_id`` 必须每次得到同样的数据，否则模型前后两轮回答会互相矛盾。
种子用 ``hashlib.sha256`` 派生（不是内置 ``hash()``，它每个进程加盐），
所以重启 Server 后结果也不变。
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta

#: 物流轨迹的状态码；与 ``app/tools/presenter.py`` 的物流枚举一致。
LOGISTICS_STATUSES: tuple[str, ...] = (
    "PICKED_UP",
    "IN_TRANSIT",
    "OUT_FOR_DELIVERY",
    "DELIVERED",
)

#: 退货进度状态码；与 presenter 的退货枚举一致。
RETURN_STATUSES: tuple[str, ...] = (
    "REQUESTED",
    "APPROVED",
    "SHIPPED_BACK",
    "RECEIVED",
    "REFUNDED",
    "REJECTED",
)

CARRIERS: tuple[str, ...] = ("顺丰速运", "中通快递", "京东物流", "圆通速递")

PRODUCTS: tuple[str, ...] = ("智能台灯", "无线耳机", "扫地机器人", "机械键盘", "保温杯")


def seed_for(order_id: str) -> random.Random:
    """按订单号派生一个稳定的随机源（跨进程、跨重启一致）。"""
    digest = hashlib.sha256(order_id.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def tracking_no(rng: random.Random) -> str:
    return "SF" + "".join(str(rng.randint(0, 9)) for _ in range(12))


def _day(base: date, offset: int) -> str:
    return (base + timedelta(days=offset)).isoformat()


def shipment_for(order_id: str) -> dict:
    """物流轨迹：节点按状态截断，保证「派送中」不会出现签收节点。"""
    rng = seed_for(order_id)
    status = rng.choice(LOGISTICS_STATUSES)
    carrier = rng.choice(CARRIERS)
    base = date(2026, 10, 20)
    steps = [
        ("已揽收", "包裹已由快递员揽收"),
        ("运输中", "包裹已到达转运中心"),
        ("派送中", "快递员正在派送"),
        ("已签收", "包裹已签收，感谢使用"),
    ]
    reached = LOGISTICS_STATUSES.index(status) + 1
    nodes = [
        {"time": f"{_day(base, index)} 09:{rng.randint(10, 59)}", "text": text}
        for index, (_, text) in enumerate(steps[:reached])
    ]
    return {
        "order_id": order_id,
        "carrier": carrier,
        "tracking_no": tracking_no(rng),
        "status": status,
        "nodes": nodes,
        "eta": _day(base, reached + 1),
    }


def warranty_for(order_id: str) -> dict:
    rng = seed_for(order_id)
    in_warranty = rng.random() < 0.7
    base = date(2026, 10, 24)
    return {
        "order_id": order_id,
        "product_name": rng.choice(PRODUCTS),
        "in_warranty": in_warranty,
        "warranty_until": _day(base, rng.randint(30, 400) * (1 if in_warranty else -1)),
        "policy": (
            "整机保修一年，人为损坏不在保修范围内。"
            if in_warranty
            else "已超出整机保修期，可付费维修。"
        ),
    }


def return_progress_for(order_id: str) -> dict:
    rng = seed_for(order_id)
    status = rng.choice(RETURN_STATUSES)
    reached = RETURN_STATUSES.index(status) + 1
    titles = ["已申请", "已通过审核", "已寄回", "商家已收货", "已退款", "已驳回"]
    base = date(2026, 10, 18)
    steps = [
        {"title": title, "done": index < reached, "time": _day(base, index)}
        for index, title in enumerate(titles)
        if index < 5
    ]
    return {
        "order_id": order_id,
        "return_no": f"R{abs(int(hashlib.sha256(order_id.encode()).hexdigest()[:6], 16)) % 900000 + 100000}",
        "status": status,
        "steps": steps,
        "refund_amount_yuan": round(rng.uniform(29, 899), 2),
    }
