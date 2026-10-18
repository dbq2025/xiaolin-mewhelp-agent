"""LangGraph checkpointer 工厂。

## 为什么生产用 SQLite，而不是 MySQL

LangGraph 官方只提供内存 / SQLite / Postgres 等 saver，没有官方 MySQL saver。
用户选定「官方 SqliteSaver 落本地文件」：图状态（messages / intent / evidence /
actions…）落 `data/checkpoints.sqlite`，重启后仍在；MySQL 仍是聊天流水、工具轨迹
与页面回读的权威副本。两者职责不同，不是「同一个会话存两遍」的重复实现。

## 为什么测试用 InMemorySaver

测试并行/重复跑时不应该在仓库里留下 sqlite 文件，也不该让上一个用例的 thread 状态
漏进下一个用例。`InMemorySaver` 与 SQLite saver 实现同一套 BaseCheckpointSaver
协议，图代码完全不用感知差异。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import PROJECT_ROOT

__all__ = ["memory_checkpointer", "open_checkpointer", "resolve_checkpoint_path"]


def resolve_checkpoint_path(path: str | Path) -> Path:
    """把配置里的相对路径按项目根解析，与当前工作目录无关。

    ``:memory:`` 是 SQLite 的特殊值，原样返回字符串让调用方识别。
    """
    if str(path) == ":memory:":
        return Path(":memory:")
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@asynccontextmanager
async def open_checkpointer(path: str | Path) -> AsyncIterator[BaseCheckpointSaver]:
    """打开一个 SQLite checkpointer，退出时关闭连接。

    ``AsyncSqliteSaver.from_conn_string`` 是异步上下文管理器；``setup()`` 建
    checkpoint 表（官方文档要求首次运行调用）。父目录在打开前创建，避免
    「文件不存在时 sqlite 不建目录」的启动失败。
    """
    resolved = resolve_checkpoint_path(path)
    if str(resolved) == ":memory:":
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            await saver.setup()
            yield saver
        return

    resolved.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(resolved)) as saver:
        await saver.setup()
        yield saver


def memory_checkpointer() -> InMemorySaver:
    """测试用：进程内、不落盘的 checkpointer。"""
    return InMemorySaver()
