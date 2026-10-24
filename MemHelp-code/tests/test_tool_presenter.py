"""ch08 Task 5：结果呈现（字段白名单 + 枚举码翻人话 + 中文不转义）。"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.tools.contract import ToolSpec
from app.tools.presenter import present_result, summarize_result


def _spec(name: str) -> ToolSpec:
    @tool(name, description="测试用工具")
    def _noop(text: str = "") -> str:
        return text

    return ToolSpec(
        name=name,
        description="测试用工具",
        args_schema={"type": "object"},
        source="mcp",
        mcp_server="logistics",
        read_only=True,
        tool=_noop,
    )


def test_order_result_keeps_only_useful_fields() -> None:
    raw = json.dumps(
        {
            "ok": True,
            "order_id": "1001",
            "status": "SHIPPED",
            "amount_yuan": "199.00",
            "product_name": "蓝牙耳机",
            "created_at": "2026-10-01",
            "internal_trace_id": "abc",
            "_debug": {"sql": "select 1"},
            "remark": None,
        },
        ensure_ascii=False,
    )

    data = present_result(_spec("query_order"), raw)

    assert data == {
        "order_id": "1001",
        "status": "SHIPPED",
        "status_text": "已发货",
        "amount_yuan": "199.00",
        "product_name": "蓝牙耳机",
        "created_at": "2026-10-01",
    }


def test_logistics_status_code_is_translated() -> None:
    raw = json.dumps(
        {
            "order_id": "1001",
            "carrier": "顺丰速运",
            "tracking_no": "SF123",
            "status": "IN_TRANSIT",
            "nodes": [{"time": "2026-10-02", "text": "已离开杭州"}],
            "eta": "2026-10-04",
            "raw_response": "x" * 50,
        },
        ensure_ascii=False,
    )

    data = present_result(_spec("track_shipment"), raw)

    assert data["status_text"] == "运输中"
    assert data["nodes"][0]["text"] == "已离开杭州"
    assert "raw_response" not in data


def test_already_chinese_status_is_left_alone() -> None:
    data = present_result(_spec("query_order"), {"order_id": "1", "status": "已完成"})

    assert data["status_text"] == "已完成"


def test_return_progress_uses_its_own_enum_table() -> None:
    data = present_result(
        _spec("check_return_progress"),
        {"order_id": "1", "return_no": "R1", "status": "REFUNDED", "junk": 1},
    )

    assert data == {
        "order_id": "1",
        "return_no": "R1",
        "status": "REFUNDED",
        "status_text": "已退款",
    }


def test_unknown_tool_keeps_all_keys_but_drops_private_ones() -> None:
    data = present_result(_spec("some_new_tool"), {"a": 1, "_b": 2, "c": None})

    assert data == {"a": 1}


def test_non_json_string_is_returned_as_is() -> None:
    assert present_result(_spec("query_order"), "不是 JSON") == "不是 JSON"


def test_summary_keeps_chinese_readable_and_truncates() -> None:
    data = {"order_id": "1001", "status_text": "已发货", "note": "中文保持原样"}

    summary = summarize_result(_spec("query_order"), data, limit=500)

    assert "已发货" in summary
    assert "\\u" not in summary

    short = summarize_result(_spec("query_order"), data, limit=20)
    assert len(short) == 20
    assert short.endswith("…")
