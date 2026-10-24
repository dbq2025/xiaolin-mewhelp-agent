"""ch08 review C1：MCP provider 必须穿过 runner 的 ``_config`` 到达图。

单元测试直接构造 ``GraphDeps`` 时很容易漏掉 runner 这一层，所以这里从
``create_app`` 出发，把 provider 挂在 ``app.state`` 上，完整走一次聊天，
断言：① 每轮确实触发发现；② 模型能绑定并调用 MCP 工具；③ 审计落成
``tool_source="mcp"``。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from langchain_core.tools import tool

from app import main
from app.tools.contract import ToolSpec
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn


@tool("track_shipment")
def track_shipment(order_id: str) -> str:
    """查物流轨迹。"""
    return json.dumps(
        {"order_id": order_id, "status": "IN_TRANSIT", "carrier": "SF"},
        ensure_ascii=False,
    )


def _spec() -> ToolSpec:
    return ToolSpec(
        name="track_shipment",
        description="查物流轨迹。",
        args_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        source="mcp",
        mcp_server="logistics",
        read_only=True,
        tool=track_shipment,
    )


class FakeProvider:
    """只实现图会用到的两个方法，并记录发现次数。"""

    def __init__(self) -> None:
        self.discoveries = 0

    async def discover(self) -> None:
        self.discoveries += 1

    def cached_specs(self) -> tuple[ToolSpec, ...]:
        return (_spec(),)


def _intent() -> object:
    return text_turn(
        json.dumps({"intent": "logistics", "confidence": 0.9}, ensure_ascii=False)
    )


def test_mcp_provider_reaches_graph_and_agent_calls_mcp_tool(
    test_settings, fake_repositories
):
    import anyio

    provider = FakeProvider()
    model = ScriptedChatModel()
    model.script.extend(
        [
            _intent(),
            tool_turn("track_shipment", {"order_id": "1001"}, call_id="call_mcp"),
            text_turn("已通过物流 MCP 查到：包裹运输中。"),
        ]
    )
    app = main.create_app(model=model, settings=test_settings)
    app.state.mcp_provider = provider

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": "s-mcp-wiring", "message": "我的包裹到哪了"},
    ) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200

    assert provider.discoveries >= 1
    assert "包裹运输中" in text

    rows = anyio.run(fake_repositories.audit.list_recent, 10)
    mcp_rows = [row for row in rows if row.tool_name == "track_shipment"]
    assert mcp_rows, "MCP 工具调用没有落审计"
    assert mcp_rows[0].tool_source == "mcp"
    assert mcp_rows[0].mcp_server == "logistics"
    assert mcp_rows[0].status == "成功"
