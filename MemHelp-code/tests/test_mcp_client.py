"""MCP Client Provider：逐 Server 发现、只读白名单、失败沿用旧缓存。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from app.tools.mcp_client import McpToolProvider, unwrap_mcp_result


@dataclass
class _FakeTool:
    name: str
    description: str
    args_schema: object

    async def ainvoke(self, args):  # pragma: no cover - provider 不调用
        return "{}"


class _OrderArgs(BaseModel):
    order_id: str


class _FakeClient:
    """假的 MultiServerMCPClient：按 server_name 返回预设结果或抛异常。"""

    def __init__(self, per_server: dict[str, object]) -> None:
        self.per_server = per_server
        self.calls: list[str] = []

    async def get_tools(self, server_name: str | None = None):
        assert server_name is not None, "必须逐 Server 拉取，才能隔离失败"
        self.calls.append(server_name)
        value = self.per_server[server_name]
        if isinstance(value, Exception):
            raise value
        return value


def _provider(per_server, *, readonly=("logistics", "aftersales")) -> McpToolProvider:
    client = _FakeClient(per_server)
    connections = {
        name: {"transport": "streamable_http", "url": f"http://{name}/mcp"}
        for name in per_server
    }
    return McpToolProvider(
        connections,
        read_only_servers=set(readonly),
        timeout=1.0,
        client=client,
    )


async def test_discover_maps_source_server_and_schema() -> None:
    provider = _provider(
        {
            "logistics": [
                _FakeTool("track_shipment", "查物流", {"type": "object", "properties": {}})
            ],
            "aftersales": [_FakeTool("check_warranty", "查保修", _OrderArgs)],
        }
    )

    specs = await provider.discover()

    by_name = {spec.name: spec for spec in specs}
    assert set(by_name) == {"track_shipment", "check_warranty"}
    assert by_name["track_shipment"].source == "mcp"
    assert by_name["track_shipment"].mcp_server == "logistics"
    assert by_name["track_shipment"].read_only is True
    assert by_name["track_shipment"].args_schema["type"] == "object"
    # Pydantic 模型要导出成 JSON Schema，执行引擎才校验得了。
    assert by_name["check_warranty"].args_schema["required"] == ["order_id"]


async def test_one_failing_server_does_not_break_the_other() -> None:
    provider = _provider(
        {
            "logistics": RuntimeError("connection refused"),
            "aftersales": [_FakeTool("check_warranty", "查保修", _OrderArgs)],
        }
    )

    specs = await provider.discover()

    assert [spec.name for spec in specs] == ["check_warranty"]


async def test_failed_server_keeps_previous_cache() -> None:
    provider = _provider(
        {"logistics": [_FakeTool("track_shipment", "查物流", {})], "aftersales": []},
    )
    await provider.discover()

    provider._client.per_server["logistics"] = RuntimeError("boom")  # type: ignore[attr-defined]
    specs = await provider.discover()

    assert [spec.name for spec in specs] == ["track_shipment"]


async def test_server_outside_readonly_whitelist_is_rejected_entirely() -> None:
    provider = _provider(
        {
            "logistics": [_FakeTool("track_shipment", "查物流", {})],
            "aftersales": [_FakeTool("check_return_progress", "退货", {})],
            "warehouse": [_FakeTool("update_stock", "写库存", {})],
        },
        readonly=("logistics", "aftersales"),
    )

    specs = await provider.discover()

    assert "update_stock" not in {spec.name for spec in specs}
    assert "warehouse" not in provider._client.calls  # type: ignore[attr-defined]


async def test_tool_without_schema_gets_permissive_object_schema() -> None:
    provider = _provider({"logistics": [_FakeTool("track", "x", None)], "aftersales": []})

    specs = await provider.discover()

    assert specs[0].args_schema == {"type": "object", "properties": {}}


def test_unwrap_mcp_result_reads_content_blocks() -> None:
    raw = [{"type": "text", "text": '{"order_id": "1001", "status": "IN_TRANSIT"}'}]

    text = unwrap_mcp_result(raw)

    assert text == '{"order_id": "1001", "status": "IN_TRANSIT"}'


def test_unwrap_mcp_result_passes_strings_through() -> None:
    assert unwrap_mcp_result('{"ok": true}') == '{"ok": true}'
