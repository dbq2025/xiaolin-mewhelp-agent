"""三个「演示用」业务工具：查订单 / 查商品 / 查物流。

## 为什么是随机数据

需求写明：真实场景该调公司电商系统与物流系统的 API，本章演示起见
**在工具内部随机生成数据返回，不接真实接口、不建表**。

## 「随机」必须是可复现的

若每次 ``random.random()``，同一个订单号每次返回不同的物流状态 —— 于是：

- 浏览器里看到的结果，与评估集跑出来的结果对不上；
- 留痕里贴的输出**无法复现**，等于没有证据；
- 用户报「物流状态不对」时，我们查不出是哪一次。

所以统一走 ``derive_rng(scope, key)``：种子由「工具名 + 入参」拼成字符串。
``random.Random(字符串)`` 用的是字符串自身的稳定哈希（**不是**内置 ``hash()``，
后者受 ``PYTHONHASHSEED`` 影响、跨进程会变），所以同号永远同结果。

## 本模块不 import 数据库

``tests/test_tool_business.py`` 有一条用例专门钉住这一点。它换来的是：
没有 MySQL 的机器（CI、沙箱、评审者的笔记本）也能跑完这三个工具与整个编排链路的单测。

## 为什么不抛异常

这些工具**永不抛**，失败也返回结构化 JSON。理由在 ``executor.py`` 的模块文档里 ——
简言之：工具失败是业务信息（「查不到」），要回灌给模型让它组织话术，
而不是变成把 SSE 连接打断的异常。
"""

import json
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any

#: 商品分类。客服场景里够用。
_CATEGORIES = ("数码", "家居", "服饰", "美妆", "食品")

#: 商品名素材：按分类给出，避免出现「数码类目下的洗发水」这种滑稽数据。
_PRODUCT_NAMES = {
    "数码": ("无线蓝牙耳机", "机械键盘", "便携充电宝", "智能手表", "降噪耳塞"),
    "家居": ("陶瓷马克杯", "记忆棉靠枕", "香薰蜡烛", "收纳箱", "棉麻桌布"),
    "服饰": ("纯棉短袖", "轻薄羽绒服", "休闲长裤", "针织开衫", "帆布腰带"),
    "美妆": ("保湿面霜", "氨基酸洁面", "防晒喷雾", "口红", "精华液"),
    "食品": ("坚果礼盒", "手冲咖啡豆", "冻干水果干", "蜂蜜", "杂粮米"),
}

_ORDER_STATUSES = ("待付款", "已付款", "已发货", "已完成", "已取消")

_LOGISTICS_STATUSES = ("已揽收", "运输中", "派送中", "已签收")

_CARRIERS = ("顺丰速运", "中通快递", "圆通速递", "京东物流", "韵达快递")

#: 轨迹文案，与状态一一对应（同一状态下的措辞可以随机挑，所以每个状态给多条）。
_TRACE_TEXTS = {
    "已揽收": ("快件已由【{city}仓】揽收", "商品已出库，等待快递揽收"),
    "运输中": ("快件已到达【{city}转运中心】", "快件离开【{city}分拨中心】，发往下一站"),
    "派送中": ("快递员正在派送，请保持电话畅通", "快件已到达【{city}网点】，正在派送"),
    "已签收": ("快件已签收，感谢使用", "已由本人签收"),
}

_CITIES = ("杭州", "上海", "北京", "广州", "成都", "武汉", "西安")


def derive_rng(scope: str, key: str) -> random.Random:
    """按「作用域 + 入参」派生一个**可复现**的随机源。

    ``scope`` 让不同工具拿到互不相关的序列（否则「订单 1001」与「物流 1001」
    会共用同一串随机数，运单号与金额出现可疑的相关性）。

    用字符串构造 ``random.Random`` 而非 ``hash(key)``：内置 ``hash`` 对字符串
    加了每进程随机盐，跨进程结果不同。
    """
    return random.Random(f"{scope}:{key}")


def _yuan(cents: int) -> str:
    """分 → 两位小数的元字符串。

    金额用字符串而不是 float：``0.1 + 0.2`` 那类表示误差出现在给用户看的金额上
    不可接受，而且「39.9 还是 39.90」该由格式化决定，不该由二进制浮点决定。
    """
    return f"{cents / 100:.2f}"


def _dumps(payload: dict[str, Any]) -> str:
    """统一序列化。``ensure_ascii=False`` 让中文在日志与留痕里可读。"""
    return json.dumps(payload, ensure_ascii=False)


def query_order(order_id: str) -> str:
    """按订单号查订单状态。

    演示实现：数据由 ``order_id`` 派生，不查真实电商系统。

    :param order_id: 订单号。
    :return: JSON 字符串，含 ``ok`` / ``order_id`` / ``status`` / ``amount_yuan``
        / ``product_id`` / ``created_at``。
    """
    rng = derive_rng("order", order_id)
    status = rng.choice(_ORDER_STATUSES)
    created = datetime.now(timezone.utc) - timedelta(
        days=rng.randint(1, 60), hours=rng.randint(0, 23)
    )

    return _dumps(
        {
            "ok": True,
            "order_id": order_id,
            "status": status,
            "amount_yuan": _yuan(rng.randint(1900, 89900)),
            "product_id": f"P{rng.randint(1000, 9999)}",
            "created_at": created.isoformat(timespec="seconds"),
        }
    )


def query_product(product_id: str) -> str:
    """按商品编号查商品信息。

    演示实现：数据由 ``product_id`` 派生，不查真实商品库。

    :param product_id: 商品编号。
    :return: JSON 字符串，含 ``ok`` / ``product_id`` / ``name`` / ``price_yuan``
        / ``stock`` / ``category``。
    """
    rng = derive_rng("product", product_id)
    category = rng.choice(_CATEGORIES)
    name = rng.choice(_PRODUCT_NAMES[category])

    return _dumps(
        {
            "ok": True,
            "product_id": product_id,
            "name": name,
            # 带上型号后缀，让不同商品编号看起来确实是不同商品
            "model": f"{rng.choice('ABCDEFGH')}{rng.randint(10, 99)}",
            "price_yuan": _yuan(rng.randint(900, 59900)),
            "stock": rng.randint(0, 500),
            "category": category,
        }
    )


def query_logistics(order_id: str) -> str:
    """按订单号查物流轨迹。

    演示实现：数据由 ``order_id`` 派生，不查真实物流系统。

    轨迹条数与当前状态**联动**：状态越靠后轨迹越长，否则会出现「已签收但只有
    一条揽收记录」这种自相矛盾的数据，模型转述出来就是幻觉。

    :param order_id: 订单号。
    :return: JSON 字符串，含 ``ok`` / ``order_id`` / ``waybill_no`` / ``carrier``
        / ``status`` / ``traces``（按时间升序）/ ``eta``。
    """
    rng = derive_rng("logistics", order_id)
    status = rng.choice(_LOGISTICS_STATUSES)
    carrier = rng.choice(_CARRIERS)
    waybill_no = f"{rng.choice(('SF', 'ZT', 'YT', 'JD', 'YD'))}{rng.randint(10**11, 10**12 - 1)}"

    # 状态在 _LOGISTICS_STATUSES 里的位置决定轨迹条数：靠后的状态必然有更多前置节点。
    reached = _LOGISTICS_STATUSES.index(status) + 1
    trace_count = max(1, reached + rng.randint(0, 1))

    base = datetime.now(timezone.utc) - timedelta(days=rng.randint(1, 6))
    traces: list[dict[str, str]] = []
    for step in range(trace_count):
        stage = _LOGISTICS_STATUSES[min(step, len(_LOGISTICS_STATUSES) - 1)]
        text = rng.choice(_TRACE_TEXTS[stage]).format(city=rng.choice(_CITIES))
        traces.append(
            {
                "time": (base + timedelta(hours=step * rng.randint(3, 12))).isoformat(
                    timespec="seconds"
                ),
                "desc": text,
            }
        )

    eta = date.today() + timedelta(days=rng.randint(0, 3))

    return _dumps(
        {
            "ok": True,
            "order_id": order_id,
            "waybill_no": waybill_no,
            "carrier": carrier,
            "status": status,
            "traces": traces,
            "eta": eta.isoformat(),
        }
    )
