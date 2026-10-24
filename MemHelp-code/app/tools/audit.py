"""工具审计写入器。

**审计失败不许反过来拦工具执行**：审计是旁路留痕，不是业务前置条件。
所以这里把所有异常吞成一条 warning —— 数据库写不进去时，用户该拿到的
工具结果仍然要拿到。
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from app.tools.contract import ToolSource, ToolStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """一条工具调用的留痕。字段名对齐 ``tool_audit_logs`` 的列语义。"""

    tool_name: str
    tool_source: ToolSource
    status: ToolStatus
    mcp_server: str | None = None
    tool_call_id: str | None = None
    conversation_id: int | None = None
    args: dict[str, Any] | None = None
    result_summary: str | None = None
    error: str | None = None
    retry_count: int = 0
    duration_ms: int | None = None


def _audit_repository(repositories: Any) -> Any:
    """既接受 ``Repositories`` 容器，也接受单独的仓库对象（便于测试替身）。"""
    return getattr(repositories, "audit", repositories)


async def write_audit(repositories: Any, entry: AuditEntry) -> None:
    """写一条审计；任何失败只 warning。永不抛。"""
    try:
        repo = _audit_repository(repositories)
        outcome = repo.create(
            tool_name=entry.tool_name,
            tool_source=entry.tool_source,
            mcp_server=entry.mcp_server,
            tool_call_id=entry.tool_call_id,
            conversation_id=entry.conversation_id,
            arguments=entry.args,
            result_summary=entry.result_summary,
            status=entry.status,
            error_message=entry.error,
            retry_count=entry.retry_count,
            duration_ms=entry.duration_ms,
        )
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:  # noqa: BLE001 - 审计失败只记日志
        logger.warning("写工具审计失败（已忽略）：%s: %s", type(exc).__name__, exc)
