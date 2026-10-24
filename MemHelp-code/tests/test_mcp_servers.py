"""两个业务 MCP Server 的工具契约（直接调函数，不启进程）。"""

from __future__ import annotations

import inspect

from mcp_servers import aftersales_server, logistics_server
from mcp_servers.mock_data import (
    LOGISTICS_STATUSES,
    RETURN_STATUSES,
    seed_for,
    shipment_for,
)


def test_same_order_is_stable_across_calls() -> None:
    assert shipment_for("1001") == shipment_for("1001")
    assert seed_for("1001").random() == seed_for("1001").random()


def test_different_orders_get_different_data() -> None:
    assert shipment_for("1001") != shipment_for("1002")


def test_track_shipment_returns_complete_tracking() -> None:
    data = logistics_server.track_shipment("1001")

    assert data["order_id"] == "1001"
    assert data["status"] in LOGISTICS_STATUSES
    assert data["carrier"]
    assert data["tracking_no"].startswith("SF")
    assert data["nodes"], "轨迹至少有一个节点"
    assert data["eta"]
    # 节点数量必须与状态一致：派送中不能已经出现签收。
    assert 1 <= len(data["nodes"]) <= LOGISTICS_STATUSES.index(data["status"]) + 1


def test_check_warranty_returns_policy() -> None:
    data = aftersales_server.check_warranty("1001")

    assert data["order_id"] == "1001"
    assert isinstance(data["in_warranty"], bool)
    assert data["warranty_until"]
    assert data["policy"]


def test_check_return_progress_returns_steps() -> None:
    data = aftersales_server.check_return_progress("1001")

    assert data["order_id"] == "1001"
    assert data["status"] in RETURN_STATUSES
    assert data["return_no"].startswith("R")
    assert data["steps"]
    assert isinstance(data["refund_amount_yuan"], float)


def test_servers_listen_on_the_configured_ports() -> None:
    assert logistics_server.mcp.settings.port == 8101
    assert aftersales_server.mcp.settings.port == 8102
    assert logistics_server.mcp.name == "memhelp-logistics"
    assert aftersales_server.mcp.name == "memhelp-aftersales"


def test_tools_are_plain_functions_with_schema_hints() -> None:
    """FastMCP 用签名生成 inputSchema；被包装成对象会让 schema 丢失。"""
    assert inspect.isfunction(logistics_server.track_shipment)
    assert inspect.isfunction(aftersales_server.check_warranty)
    assert inspect.isfunction(aftersales_server.check_return_progress)
