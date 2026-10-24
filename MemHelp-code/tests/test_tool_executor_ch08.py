"""ch08 统一执行引擎：校验 → 权限 → 调用 → 呈现 → 审计。

这些测试与 ch02 的 ``tests/test_tool_executor.py`` 分工不同：那一份钉住
「超时/重试/脱敏」的旧契约，这一份钉住 ch08 新增的四件事 ——
JSON Schema 校验拦下、写操作权限闸、写操作永不重试、每次调用都落审计。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool

from app.tools.contract import ToolSpec
from app.tools.executor import ToolResult, execute_tool
from app.tools.registry import build_registry


def _register(
    registry,
    name: str,
    fn,
    *,
    read_only: bool = True,
    schema: dict | None = None,
) -> ToolSpec:
    """把一个协程登记成工具；schema 缺省为空对象（无参数）。"""
    tool = StructuredTool.from_function(coroutine=fn, name=name, description="测试工具")
    spec = ToolSpec(
        name=name,
        description="测试工具",
        args_schema=schema or {"type": "object", "properties": {}},
        source="builtin",
        mcp_server=None,
        read_only=read_only,
        tool=tool,
    )
    registry.register(spec)
    return spec


async def _echo() -> str:
    return json.dumps({"ok": True}, ensure_ascii=False)


async def _slow() -> str:
    await asyncio.sleep(0.5)
    return "{}"


async def test_unknown_tool_is_failure_and_audited(fake_repositories) -> None:
    registry = build_registry(fake_repositories)

    result = await execute_tool(
        "no_such_tool", {}, registry=registry, repositories=fake_repositories
    )

    assert result.ok is False
    assert result.status == "失败"
    assert result.attempts == 0
    assert "no_such_tool" in (result.error or "")

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.tool_name == "no_such_tool"
    assert row.status == "失败"
    assert row.tool_source == "builtin"


async def test_invalid_args_are_validation_status_and_audited(fake_repositories) -> None:
    registry = build_registry(fake_repositories)

    result = await execute_tool(
        "query_order", {}, registry=registry, repositories=fake_repositories
    )

    assert result.ok is False
    assert result.status == "校验拦下"
    assert result.attempts == 0
    assert "order_id" in (result.error or "")

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.status == "校验拦下"
    assert row.retry_count == 0
    assert "order_id" in (row.error_message or "")


async def test_write_without_confirmation_is_denied_and_audited(
    fake_repositories,
) -> None:
    registry = build_registry(fake_repositories)

    result = await execute_tool(
        "create_ticket",
        {"description": "退款没到账", "ticket_type": "售后", "conversation_id": "s1"},
        registry=registry,
        repositories=fake_repositories,
    )

    assert result.ok is False
    assert result.status == "权限拒绝"
    assert result.attempts == 0
    assert result.error

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.tool_name == "create_ticket"
    assert row.status == "权限拒绝"
    assert row.error_message


async def test_write_timeout_is_not_retried(fake_repositories) -> None:
    registry = build_registry(fake_repositories)
    _register(registry, "slow_write", _slow, read_only=False)

    result = await execute_tool(
        "slow_write",
        {},
        registry=registry,
        repositories=fake_repositories,
        timeout=0.05,
        max_retries=3,
        confirmed=True,
    )

    assert result.ok is False
    assert result.status == "超时"
    assert result.attempts == 1, "写操作超时未必没执行，重复执行比失败更糟"
    assert result.retried is False

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.status == "超时"
    assert row.retry_count == 0
    assert row.duration_ms is not None


async def test_read_timeout_is_retried_once(fake_repositories) -> None:
    registry = build_registry(fake_repositories)
    _register(registry, "slow_read", _slow, read_only=True)

    result = await execute_tool(
        "slow_read",
        {},
        registry=registry,
        repositories=fake_repositories,
        timeout=0.05,
        max_retries=1,
    )

    assert result.status == "超时"
    assert result.attempts == 2
    assert result.retried is True

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.status == "超时"
    assert row.retry_count == 1


async def test_success_is_audited_with_source_and_summary(fake_repositories) -> None:
    registry = build_registry(fake_repositories)
    _register(registry, "echo_tool", _echo)

    result = await execute_tool(
        "echo_tool",
        {},
        registry=registry,
        repositories=fake_repositories,
        tool_call_id="call_9",
        conversation_id=7,
    )

    assert result.ok is True
    assert result.status == "成功"
    assert result.source == "builtin"
    assert result.tool_call_id == "call_9"

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.status == "成功"
    assert row.tool_name == "echo_tool"
    assert row.tool_call_id == "call_9"
    assert row.conversation_id == 7
    assert row.retry_count == 0
    assert row.result_summary


async def test_audit_failure_never_blocks_execution(fake_repositories) -> None:
    class BrokenAudit:
        async def create(self, **kwargs):
            raise RuntimeError("audit db down")

    object.__setattr__(fake_repositories, "audit", BrokenAudit())
    registry = build_registry(fake_repositories)
    _register(registry, "echo_tool", _echo)

    result = await execute_tool(
        "echo_tool", {}, registry=registry, repositories=fake_repositories
    )

    assert result.ok is True


async def test_presenter_whitelists_fields_and_translates_status(
    fake_repositories,
) -> None:
    registry = build_registry(fake_repositories)

    async def shipment() -> str:
        return json.dumps(
            {
                "order_id": "1001",
                "status": "IN_TRANSIT",
                "carrier": "顺丰",
                "secret_token": "sk-live-xxx",
                "eta": "2026-10-26",
            },
            ensure_ascii=False,
        )

    _register(registry, "track_shipment", shipment)

    result = await execute_tool(
        "track_shipment",
        {"order_id": "1001"},
        registry=registry,
        repositories=fake_repositories,
    )

    assert result.ok is True
    payload = json.loads(result.data)
    assert set(payload) == {"order_id", "status", "status_text", "carrier", "eta"}
    assert payload["status_text"] == "运输中"
    assert "sk-live-xxx" not in result.data


async def test_write_confirmed_executes(fake_repositories) -> None:
    registry = build_registry(fake_repositories)
    _register(registry, "write_tool", _echo, read_only=False)

    result = await execute_tool(
        "write_tool",
        {},
        registry=registry,
        repositories=fake_repositories,
        confirmed=True,
    )

    assert result.ok is True
    assert result.status == "成功"


def test_summary_keeps_legacy_keys() -> None:
    result = ToolResult("query_order", {"order_id": "1"}, True, "{}", None, 1, 5)

    assert set(result.summary()) == {
        "name",
        "args",
        "ok",
        "duration_ms",
        "attempts",
        "started_at",
        "result",
        "error",
    }
