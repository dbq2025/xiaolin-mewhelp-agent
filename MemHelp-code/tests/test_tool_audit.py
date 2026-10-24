"""ch08 Task 6：工具审计表、仓库与写入器。

两条硬规矩：
1. 审计不挂外键 —— 工具调用先于会话落库、或会话被清理时，审计仍要留得住；
2. 写审计失败只 warning，绝不能反过来拦住工具执行。
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect

from app.db.models import ToolAuditLog
from app.tools.audit import AuditEntry, write_audit


def _entry(**overrides) -> AuditEntry:
    base = dict(
        tool_name="query_order",
        tool_source="builtin",
        mcp_server=None,
        tool_call_id="call_1",
        conversation_id=None,
        args={"order_id": "A1"},
        result_summary='{"ok": true}',
        status="成功",
        error=None,
        retry_count=2,
        duration_ms=37,
    )
    base.update(overrides)
    return AuditEntry(**base)


def test_model_columns_match_the_ddl() -> None:
    table = ToolAuditLog.__table__

    assert table.name == "tool_audit_logs"
    assert set(table.c.keys()) == {
        "id",
        "conversation_id",
        "tool_call_id",
        "tool_name",
        "tool_source",
        "mcp_server",
        "arguments",
        "result_summary",
        "status",
        "error_message",
        "retry_count",
        "duration_ms",
        "created_at",
    }
    assert table.c.tool_name.nullable is False
    assert table.c.tool_source.nullable is False
    assert table.c.status.nullable is False
    assert table.c.retry_count.nullable is False
    assert set(table.c.status.type.enums) == {
        "成功", "失败", "超时", "校验拦下", "权限拒绝",
    }
    assert set(table.c.tool_source.type.enums) == {"builtin", "mcp"}


def test_audit_table_has_no_foreign_keys() -> None:
    table = ToolAuditLog.__table__

    assert table.c.conversation_id.foreign_keys == set()
    assert not any(fk for col in table.c for fk in col.foreign_keys)
    assert {ix.name for ix in table.indexes} == {
        "idx_audit_conversation_id",
        "idx_audit_tool_name",
        "idx_audit_status",
    }


async def test_audit_row_persists_all_fields(fake_repositories) -> None:
    await fake_repositories.audit.create(
        tool_name="query_order",
        tool_source="builtin",
        mcp_server=None,
        tool_call_id="call_1",
        conversation_id=None,
        arguments={"order_id": "A1"},
        result_summary='{"ok": true}',
        status="成功",
        error_message=None,
        retry_count=2,
        duration_ms=37,
    )

    rows = await fake_repositories.audit.list_recent(limit=1)

    assert len(rows) == 1
    row = rows[0]
    assert row.tool_name == "query_order"
    assert row.tool_source == "builtin"
    assert row.tool_call_id == "call_1"
    assert row.arguments == {"order_id": "A1"}
    assert row.result_summary == '{"ok": true}'
    assert row.status == "成功"
    assert row.retry_count == 2
    assert row.duration_ms == 37
    assert row.created_at is not None


async def test_audit_keeps_mcp_server_and_null_conversation(fake_repositories) -> None:
    await write_audit(
        fake_repositories,
        _entry(
            tool_name="track_shipment",
            tool_source="mcp",
            mcp_server="logistics",
            status="失败",
            error="连接超时",
        ),
    )

    row = (await fake_repositories.audit.list_recent(limit=1))[0]

    assert row.mcp_server == "logistics"
    assert row.conversation_id is None
    assert row.error_message == "连接超时"


async def test_write_audit_swallows_sync_failure() -> None:
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("db down")

    await write_audit(Boom(), _entry())  # 不抛


async def test_write_audit_swallows_async_failure() -> None:
    class Boom:
        async def create(self, **kwargs):
            raise RuntimeError("db down")

    await write_audit(Boom(), _entry())  # 不抛


async def test_write_audit_accepts_repository_container(fake_repositories) -> None:
    await write_audit(fake_repositories, _entry(status="权限拒绝", error="未确认"))

    row = (await fake_repositories.audit.list_recent(limit=1))[0]
    assert row.status == "权限拒绝"
    assert row.error_message == "未确认"
