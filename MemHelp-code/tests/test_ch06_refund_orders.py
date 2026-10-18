"""ch06 Task 7：订单号抽取、演示订单卡片与固定退款原因。"""

from __future__ import annotations

from app.agent.refund_orders import (
    REFUND_REASONS,
    build_refund_form,
    extract_order_id,
    list_demo_orders,
)


def test_extract_order_id_patterns():
    assert extract_order_id("订单 20240115001 的物流到哪了") == "20240115001"
    assert extract_order_id("订单号：1001") == "1001"
    assert extract_order_id("单号 2024-1001-AB") == "2024-1001-AB"
    assert extract_order_id("这个能退吗") is None


def test_demo_orders_are_stable_and_complete():
    first = list_demo_orders("s-demo", 3)
    second = list_demo_orders("s-demo", 3)
    assert first == second
    assert len(first) == 3
    assert len({order["order_id"] for order in first}) == 3
    for order in first:
        assert order["order_id"] and order["title"] and order["status"]
        assert order["amount_yuan"]
        assert order["product_name"] and order["product_model"]
        assert order["created_at"]


def test_demo_orders_differ_by_session():
    assert list_demo_orders("s-a", 3) != list_demo_orders("s-b", 3)


def test_refund_reasons_are_fixed_six():
    assert REFUND_REASONS == (
        "商品质量问题",
        "尺寸/颜色/型号不合适",
        "与描述不符",
        "不想要了（七天无理由）",
        "物流破损/少件",
        "其他",
    )


def test_build_refund_form_shape():
    form = build_refund_form(
        {
            "order_id": "1001",
            "product_name": "耳机",
            "product_model": "A1",
            "amount_yuan": "99.00",
        }
    )
    assert form["order_id"] == "1001"
    assert form["order_label"] == "耳机 A1 · ¥99.00"
    assert form["reasons"] == list(REFUND_REASONS)
    assert form["submit_url"] == "/api/actions/create-refund"
