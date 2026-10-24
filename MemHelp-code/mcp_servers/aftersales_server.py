"""售后 MCP Server：查在保状态、查退货进度（Streamable HTTP，独立进程）。

启动：``uv run python -m mcp_servers.aftersales_server``
端点：``http://127.0.0.1:8102/mcp``
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.mock_data import return_progress_for, warranty_for

mcp = FastMCP(
    "memhelp-aftersales",
    instructions="客服系统的售后查询：按订单号返回保修状态与退货退款进度。",
    host="127.0.0.1",
    port=8102,
)


@mcp.tool()
def check_warranty(order_id: str) -> dict:
    """按订单号查询商品是否还在保修期内。

    :param order_id: 订单号，例如 ``1001``。
    """
    return warranty_for(order_id)


@mcp.tool()
def check_return_progress(order_id: str) -> dict:
    """按订单号查询退货/退款进度。

    :param order_id: 订单号，例如 ``1001``。
    """
    return return_progress_for(order_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
