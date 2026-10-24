"""ch08 Task 4：读写权限策略。

规矩：只读工具直接放行；写工具必须由「用户确认」这条外部事实放行，
**不看 MCP Server 自己声明的 annotations**（那是对方写的，不可信）。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools.contract import ToolSpec
from app.tools.permissions import check_permission


def _spec(name: str, *, read_only: bool, source: str = "builtin") -> ToolSpec:
    @tool(name, description="测试用工具")
    def _noop(text: str = "") -> str:
        return text

    return ToolSpec(
        name=name,
        description="测试用工具",
        args_schema={"type": "object"},
        source=source,  # type: ignore[arg-type]
        mcp_server=None,
        read_only=read_only,
        tool=_noop,
    )


def test_read_only_tool_is_allowed_without_confirmation() -> None:
    assert check_permission(_spec("query_order", read_only=True), confirmed=False) is None


def test_write_tool_is_rejected_without_confirmation() -> None:
    error = check_permission(_spec("create_ticket", read_only=False), confirmed=False)

    assert error is not None
    assert "create_ticket" in error
    assert "确认" in error


def test_write_tool_is_allowed_after_confirmation() -> None:
    assert check_permission(_spec("create_ticket", read_only=False), confirmed=True) is None


def test_mcp_write_tool_is_also_gated() -> None:
    """MCP 侧声明的只读与否不作数，仍按本侧 read_only 走同一道闸。"""
    spec = _spec("refund_now", read_only=False, source="mcp")

    assert check_permission(spec, confirmed=False) is not None
    assert check_permission(spec, confirmed=True) is None
