"""ch08 Task 1：工具契约与服务端 MCP 配置。"""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.tools import tool

from app.tools.contract import ToolSpec


def _echo() -> object:
    @tool
    def echo(text: str) -> str:
        """回显。"""
        return text

    return echo


def test_spec_is_frozen_and_keeps_schema() -> None:
    spec = ToolSpec(
        name="echo",
        description="回显",
        args_schema={"type": "object"},
        source="builtin",
        mcp_server=None,
        read_only=True,
        tool=_echo(),  # type: ignore[arg-type]
    )
    assert spec.args_schema["type"] == "object"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_settings_mcp_defaults(test_settings) -> None:
    assert test_settings.mcp_enabled is True
    assert test_settings.mcp_logistics_url.endswith(":8101/mcp")
    assert test_settings.mcp_aftersales_url.endswith(":8102/mcp")
    assert "logistics" in test_settings.mcp_readonly_servers
    assert test_settings.mcp_timeout_seconds > 0
