"""MCP 工具发现：把外部 Server 的工具接进同一张工具表。

## 信任边界

MCP Server 是**别人写的进程**：它的工具描述、``annotations``（例如自称
``readOnlyHint``）都是对方说了算，不能当作权限依据。这里的规则是：

- 只有配置里明确列进 ``mcp_readonly_servers`` 的 Server 才会被接入；
- 接入的工具一律 ``read_only=True``，写操作闸门只认我们这侧的
  ``ToolSpec.read_only`` 与 ``confirmed``，不看 Server 声明。

不在白名单的 Server 整个跳过并 warning —— 宁可少接一个工具，也不能让
一个自称只读的外部写接口绕过确认流。

## 失败与缓存

``discover()`` 逐个 Server 拉取，一个 Server 连不上不影响另一个。
成功的 Server 用新清单替换，失败的 Server 沿用上一轮缓存：
聊天不能因为某个 MCP 进程没起来就整体不可用。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

from app.tools.contract import ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["McpToolProvider", "connections_from_settings", "unwrap_mcp_result"]

#: 没有 schema 的工具用这个兜底：空对象 = 不限制参数，但校验器仍会跑。
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def connections_from_settings(settings: Any) -> dict[str, dict[str, Any]]:
    """从 ``Settings`` 拼出 ``MultiServerMCPClient`` 的连接表。

    只有 ``mcp_enabled`` 打开时才返回内容；关闭时调用方拿到空表，
    Provider 发现不到任何工具，聊天照常工作。
    """
    if not getattr(settings, "mcp_enabled", False):
        return {}
    return {
        "logistics": {
            "transport": "streamable_http",
            "url": settings.mcp_logistics_url,
            "timeout": settings.mcp_timeout_seconds,
        },
        "aftersales": {
            "transport": "streamable_http",
            "url": settings.mcp_aftersales_url,
            "timeout": settings.mcp_timeout_seconds,
        },
    }


def readonly_servers_from_settings(settings: Any) -> set[str]:
    raw = getattr(settings, "mcp_readonly_servers", "") or ""
    return {part.strip() for part in raw.split(",") if part.strip()}


def _schema_of(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return dict(_EMPTY_SCHEMA)
    if isinstance(schema, dict):
        return schema
    exporter = getattr(schema, "model_json_schema", None)
    if callable(exporter):
        try:
            return exporter()
        except Exception:  # noqa: BLE001 - 外部 schema 坏了也不能炸
            return dict(_EMPTY_SCHEMA)
    return dict(_EMPTY_SCHEMA)


def unwrap_mcp_result(raw: Any) -> Any:
    """把 MCP 的 content blocks 摊平成文本。

    ``langchain-mcp-adapters`` 返回的是 ``response_format="content_and_artifact"``
    的结果 —— ``ainvoke`` 拿到的是 ``[{"type": "text", "text": "{...}"}]``，
    不是裸字符串。不摊平的话，呈现层会把这个 list 原样塞给模型。
    """
    if not isinstance(raw, list):
        return raw
    texts = [
        block.get("text", "")
        for block in raw
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not texts:
        return raw
    return "\n".join(texts)


class McpToolProvider:
    """进程级单例：缓存发现结果，每轮由 ``ToolRegistry.refresh_mcp`` 调用。"""

    def __init__(
        self,
        connections: dict[str, dict[str, Any]],
        *,
        read_only_servers: set[str],
        timeout: float = 3.0,
        client: Any | None = None,
    ) -> None:
        self._connections = dict(connections)
        self._read_only_servers = set(read_only_servers)
        self._timeout = timeout
        self._client = client
        self._cache: tuple[ToolSpec, ...] = ()

    def _get_client(self) -> Any:
        if self._client is None:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(self._connections)
        return self._client

    def cached_specs(self) -> tuple[ToolSpec, ...]:
        return self._cache

    async def discover(self) -> tuple[ToolSpec, ...]:
        """拉取所有白名单 Server 的工具；尽量多成功，绝不整体抛。"""
        if not self._connections:
            self._cache = ()
            return self._cache

        try:
            client = self._get_client()
        except Exception as exc:  # noqa: BLE001 - 构造失败沿用旧缓存
            logger.warning("ch08 MCP 客户端构造失败，沿用旧清单：%s", exc)
            return self._cache

        previous_by_server: dict[str, list[ToolSpec]] = {}
        for spec in self._cache:
            previous_by_server.setdefault(spec.mcp_server or "", []).append(spec)

        collected: list[ToolSpec] = []
        succeeded = 0
        for server in sorted(self._connections):
            if server not in self._read_only_servers:
                logger.warning(
                    "ch08 MCP Server %s 不在只读白名单，整站拒绝接入", server
                )
                continue
            try:
                tools = await asyncio.wait_for(
                    client.get_tools(server_name=server),
                    timeout=max(self._timeout * 2, 1.0),
                )
            except Exception as exc:  # noqa: BLE001 - 单个 Server 失败不影响其它
                logger.warning("ch08 MCP Server %s 发现失败，沿用旧清单：%s", server, exc)
                collected.extend(previous_by_server.get(server, ()))
                continue

            succeeded += 1
            for tool in tools or ():
                collected.append(self._to_spec(server, tool))

        if succeeded == 0 and previous_by_server:
            # 全军覆没：整体沿用旧缓存，避免一次网络抖动把工具表清空。
            logger.warning("ch08 所有 MCP Server 发现失败，沿用上一轮工具清单")
            return self._cache

        self._cache = tuple(sorted(collected, key=lambda spec: spec.name))
        return self._cache

    def _to_spec(self, server: str, tool: Any) -> ToolSpec:
        return ToolSpec(
            name=str(getattr(tool, "name", "") or ""),
            # 描述是对方写的，只用于给模型看，不参与任何权限判断。
            description=str(getattr(tool, "description", "") or ""),
            args_schema=_schema_of(tool),
            source="mcp",
            mcp_server=server,
            read_only=True,
            tool=tool,
        )
