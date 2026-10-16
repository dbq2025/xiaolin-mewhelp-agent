"""三个「演示用」业务工具的测试。

这三个工具（订单 / 商品 / 物流）在真实场景里该调公司电商系统与物流系统的 API，
本章按需求**在工具内部随机生成数据**，不接真实接口、不建表。

「随机」在这里有个硬要求：**必须按入参派生**。若每次调用都 ``random.random()``，
同一个订单号每次返回不同的物流状态 —— 浏览器里看到的结果、评估集跑出来的结果、
留痕里贴的输出，三者对不上账，而且故障不可复现。所以用例第一件事就是钉住可复现性。

**本模块不得 import 数据库**（见最后一条用例）：这条边界让「没有 MySQL 也能跑完
工具与编排的全部单测」成立。
"""

import json

import pytest

from app.tools.business import (
    derive_rng,
    query_logistics,
    query_order,
    query_product,
)

#: 与验收标准 1、评估集固定使用同一个订单号。
ORDER_ID = "1001"


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_same_order_id_yields_identical_result():
    """同号同结果 —— 评估集与浏览器截图能对账的前提。"""
    assert query_logistics(ORDER_ID) == query_logistics(ORDER_ID)
    assert query_order(ORDER_ID) == query_order(ORDER_ID)


def test_same_product_id_yields_identical_result():
    assert query_product("P2001") == query_product("P2001")


def test_different_ids_yield_different_results():
    """派生必须真的用上入参 —— 全都返回同一份数据会让演示看起来很假。"""
    assert query_logistics("1001") != query_logistics("2002")
    assert query_order("1001") != query_order("2002")
    assert query_product("P1") != query_product("P2")


def test_derive_rng_is_stable_and_scope_separated():
    """同一 (scope, key) 稳定；换 scope 就换一串数。

    scope 分开是为了让「订单 1001」与「物流 1001」拿到互不相关的随机序列 ——
    否则两者的运单号、金额会出现可疑的相关性。
    """
    assert derive_rng("order", "1").random() == derive_rng("order", "1").random()
    assert derive_rng("order", "1").random() != derive_rng("logistics", "1").random()


# ---------------------------------------------------------------------------
# 物流
# ---------------------------------------------------------------------------


def test_logistics_payload_shape():
    data = json.loads(query_logistics(ORDER_ID))

    assert data["ok"] is True
    assert data["order_id"] == ORDER_ID
    assert set(data) >= {"order_id", "waybill_no", "carrier", "status", "traces", "eta"}
    assert isinstance(data["waybill_no"], str) and len(data["waybill_no"]) >= 8


def test_logistics_traces_are_ordered_and_non_empty():
    """轨迹必须至少一条且按时间递增。

    「物流到哪了」是验收标准 1 的原话 —— 模型要靠 ``traces`` 才能回答
    「现在到哪了」。空轨迹会让模型只能说「查到了但没详情」。
    """
    data = json.loads(query_logistics(ORDER_ID))
    traces = data["traces"]

    assert 1 <= len(traces) <= 6
    times = [t["time"] for t in traces]
    assert times == sorted(times)
    for trace in traces:
        assert trace["desc"].strip()


def test_logistics_status_is_one_of_the_documented_values():
    data = json.loads(query_logistics(ORDER_ID))
    assert data["status"] in {"已揽收", "运输中", "派送中", "已签收"}


def test_logistics_eta_is_an_iso_date():
    from datetime import date

    data = json.loads(query_logistics(ORDER_ID))
    date.fromisoformat(data["eta"])  # 解析失败会抛，等于断言格式


# ---------------------------------------------------------------------------
# 订单
# ---------------------------------------------------------------------------


def test_order_payload_shape():
    data = json.loads(query_order(ORDER_ID))

    assert data["ok"] is True
    assert data["order_id"] == ORDER_ID
    assert set(data) >= {"order_id", "status", "amount_yuan", "product_id", "created_at"}
    assert data["status"] in {"待付款", "已付款", "已发货", "已完成", "已取消"}


def test_order_amount_is_a_two_decimal_string():
    """金额用**字符串**而不是 float。

    浮点数 ``0.1 + 0.2`` 那种表示误差出现在给用户看的金额上是不可接受的，
    而且 ``39.9`` 与 ``39.90`` 在展示层的区别需要由格式化决定，不是由二进制浮点决定。
    """
    data = json.loads(query_order(ORDER_ID))
    amount = data["amount_yuan"]

    assert isinstance(amount, str)
    whole, _, cents = amount.partition(".")
    assert whole.isdigit() and len(cents) == 2


def test_order_created_at_is_an_iso_datetime():
    from datetime import datetime

    data = json.loads(query_order(ORDER_ID))
    datetime.fromisoformat(data["created_at"])


# ---------------------------------------------------------------------------
# 商品
# ---------------------------------------------------------------------------


def test_product_payload_shape():
    data = json.loads(query_product("P2001"))

    assert data["ok"] is True
    assert data["product_id"] == "P2001"
    assert set(data) >= {"product_id", "name", "price_yuan", "stock", "category"}
    assert isinstance(data["stock"], int) and data["stock"] >= 0


def test_product_name_is_not_empty():
    """名字空的商品在客服对话里毫无意义 —— 模型只能回「查到了但没名字」。"""
    assert json.loads(query_product("P2001"))["name"].strip()


# ---------------------------------------------------------------------------
# 边界与分层
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_id_is_rejected_with_a_clear_error(bad):
    """空 id 要在工具内就拒掉。

    放过去的话模型会拿到一份「订单 的物流」这种荒谬数据，然后一本正经地回答，
    用户看到的是幻觉而不是错误提示。
    """
    from pydantic import ValidationError

    from app.tools.schemas import QueryOrderArgs

    with pytest.raises(ValidationError):
        QueryOrderArgs(order_id=bad)


def test_business_module_does_not_import_the_database():
    """三个演示工具必须能脱离 MySQL 单独工作。

    不 import ``app.db`` 才能让「没有 MySQL 的机器上跑完整个工具的测试」成立 ——
    这是 CI 与沙箱环境的实际前提。
    """
    import sys

    # 先清掉可能已被其他测试 import 的痕迹，再看 import 本模块会带出什么
    for name in list(sys.modules):
        if name.startswith("app.db"):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("app.tools"):
            del sys.modules[name]

    import app.tools.business  # noqa: F401

    leaked = [name for name in sys.modules if name.startswith("app.db")]
    assert leaked == [], f"app.tools.business 不应带出数据库模块：{leaked}"
