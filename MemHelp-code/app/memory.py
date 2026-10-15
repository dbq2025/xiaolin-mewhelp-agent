"""会话存储 —— 全项目**唯一**知道「历史存在哪」的地方。

本章用进程内内存字典，够 curl 演示用。刻意做成一个只暴露
``get`` / ``append`` / ``clear`` / ``sessions`` 的小接口：第二章要换成 Redis
或数据库时，只重写这一个文件，``chat.py`` 与 ``api.py`` 不用动。

本模块**不碰网络、不碰磁盘**，纯 Python，所以可以完全离线测试。
"""

from collections import defaultdict
from functools import lru_cache
from uuid import uuid4

from langchain_core.messages import BaseMessage


def new_session_id() -> str:
    """生成一个短会话 ID（12 位十六进制，够用且便于 curl 里手写）。"""
    return uuid4().hex[:12]


class InMemorySessionStore:
    """``session_id -> 消息列表`` 的进程内存储。

    **不做并发同步**：本章是单进程、单事件循环，规格里已写明；
    多进程/多线程场景留给第二章的持久化实现。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = defaultdict(list)

    def get(self, session_id: str) -> list[BaseMessage]:
        """返回会话历史的**副本**。

        返回副本是刻意的：调用方（比如裁剪逻辑）拿到列表后可能会原地改动，
        若直接交出内部列表，那些改动会悄悄写回存储。
        """
        return list(self._sessions.get(session_id, ()))

    def append(self, session_id: str, messages: list[BaseMessage]) -> None:
        """把消息追加到会话末尾，保持顺序。"""
        self._sessions[session_id].extend(messages)

    def clear(self, session_id: str) -> None:
        """清空某个会话（不影响其他会话）。"""
        self._sessions.pop(session_id, None)

    def sessions(self) -> list[str]:
        """当前有历史的会话 ID 列表（调试/运维用）。"""
        return list(self._sessions)


@lru_cache
def get_session_store() -> InMemorySessionStore:
    """进程内单例。FastAPI 依赖注入与测试都会拿到同一个实例。"""
    return InMemorySessionStore()
