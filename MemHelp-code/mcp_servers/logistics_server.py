"""物流 MCP Server：查物流轨迹（Streamable HTTP，独立进程）。

启动：``uv run python -m mcp_servers.logistics_server``
端点：``http://127.0.0.1:8101/mcp``
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from mcp_servers.mock_data import shipment_for

mcp = FastMCP(
    "memhelp-logistics",
    instructions="客服系统的物流查询：按订单号返回承运商、运单号、轨迹节点与预计送达时间。",
    host="127.0.0.1",
    port=8101,
)


@mcp.tool()
def track_shipment(order_id: str) -> dict:
    """按订单号查询物流轨迹。

    :param order_id: 订单号，例如 ``1001``。
    """
    return shipment_for(order_id)


if os.environ.get("MEMHELP_LOGISTICS_EXTRA") == "1":
    # ch08 验收 3 的演示开关：只重启这个 Server，客服系统不重启也能发现新工具。
    @mcp.tool()
    def estimate_delivery(order_id: str) -> dict:
        """按订单号估算还需几天送达（演示用工具）。

        :param order_id: 订单号，例如 ``1001``。
        """
        shipment = shipment_for(order_id)
        eta = shipment["eta"]
        return {
            "order_id": order_id,
            "eta": eta,
            "note": "演示工具：由 MEMHELP_LOGISTICS_EXTRA=1 打开，重启本 Server 才出现。",
        }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
