"""工具注册中心：内置工具与 MCP 工具登记在同一张表里。

## 一张表，两种来源

- **内置**：服务启动/每轮构造时由 ``app/tools/builtin.py`` 登记，依赖闭包捕获；
- **MCP**：由 ``McpToolProvider`` 从各 Server 动态发现，每轮对话前刷新。

Agent 绑定工具时只看 ``snapshot()``，不关心来源。执行引擎要审计时再读
``spec.source`` / ``spec.mcp_server``，所以「新工具接进来」不需要改核心代码。

## 重名规则

内置优先。MCP 工具与内置重名时**丢弃 MCP 那个**并 warning —— 否则模型会
看到两个同名工具，绑定列表里后者覆盖前者，行为随发现顺序漂移。

## 每轮构造

``ToolRegistry`` 不是模块级单例：内置工具要绑当轮 ``repositories``，
MCP 清单要每轮刷新。进程级只缓存 ``McpToolProvider`` 的发现结果。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from langchain_core.tools import BaseTool

from app.db.repositories import Repositories
from app.tools.builtin import BUILTIN_TOOL_NAMES, build_builtin_tools
from app.tools.contract import ToolSpec
from app.tools.knowledge import AnswerContextLike

logger = logging.getLogger(__name__)

__all__ = [
    "TOOL_LABELS",
    "TOOL_NAMES",
    "ToolRegistry",
    "build_registry",
    "get_tool",
    "get_tools",
]

#: SSE 的 ``status`` 帧用的中文短语。
#:
#: 它在工具**执行之前**就发给前端，让用户立刻看到「在查数据」而不是干等。
#: 以省略号结尾表示进行中。键必须与内置工具名逐一对应；MCP 工具由 provider
#: 按工具名补标签，缺省时前端显示工具名本身。
TOOL_LABELS: dict[str, str] = {
    "query_order": "正在查询订单…",
    "query_product": "正在查询商品…",
    "query_faq": "正在检索常见问题…",
    "create_ticket": "正在创建工单…",
}

#: 内置工具名的权威顺序（也是它们提供给模型的顺序）。
TOOL_NAMES: tuple[str, ...] = BUILTIN_TOOL_NAMES


class ToolRegistry:
    """一轮对话里的工具表：内置 + 已发现的 MCP 工具。"""

    def __init__(
        self,
        repositories: Repositories,
        retriever: AnswerContextLike | None = None,
        *,
        mcp_provider: object | None = None,
    ) -> None:
        self._mcp_provider = mcp_provider
        self._builtin: dict[str, ToolSpec] = {}
        self._mcp: dict[str, ToolSpec] = {}
        self._lock = asyncio.Lock()
        for spec in build_builtin_tools(repositories, retriever):
            self.register(spec)

    # -- 查询 ---------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        return self._builtin.get(name) or self._mcp.get(name)

    def snapshot(self) -> tuple[ToolSpec, ...]:
        """按名字排序的稳定快照 —— 绑定给模型时顺序不随发现顺序漂移。"""
        merged = {**self._mcp, **self._builtin}
        return tuple(merged[name] for name in sorted(merged))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    # -- 登记 ---------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """登记一条工具定义；重名抛 ``ValueError``（内置优先，不静默覆盖）。"""
        if self.get(spec.name) is not None:
            raise ValueError(f"工具名重复：{spec.name}")
        target = self._builtin if spec.source == "builtin" else self._mcp
        target[spec.name] = spec

    def _merge_mcp(self, specs: Iterable[ToolSpec]) -> None:
        """合并一批 MCP 规格；与内置重名的跳过并 warning。"""
        merged: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._builtin:
                logger.warning("ch08 MCP 工具与内置重名，已跳过：%s", spec.name)
                continue
            merged[spec.name] = spec
        self._mcp = merged

    async def refresh_mcp(self) -> None:
        """每轮对话前刷新 MCP 清单。

        发现失败时保留上一轮缓存（聊天不能因为没有 MCP 而整体失败），
        这正是 ``McpToolProvider.discover`` 的契约。
        """
        provider = self._mcp_provider
        if provider is None:
            return
        try:
            specs = await provider.discover()  # type: ignore[attr-defined]
        except Exception as exc:  # provider 自己也不该抛，这里再兜一层
            logger.warning("ch08 MCP 工具刷新失败，沿用旧清单：%s", exc)
            return
        async with self._lock:
            self._merge_mcp(specs)


def build_registry(
    repositories: Repositories,
    retriever: AnswerContextLike | None = None,
    *,
    mcp_provider: object | None = None,
    enable_mcp: bool = False,
) -> ToolRegistry:
    """构造本轮工具表；``enable_mcp=True`` 时合并 provider 的缓存清单。

    不在构造时发网络请求 —— 发现是 ``refresh_mcp()`` 的事，这里只吃缓存，
    让「构造注册表」保持同步、可在测试里随便调。
    """
    registry = ToolRegistry(repositories, retriever, mcp_provider=mcp_provider)
    if enable_mcp and mcp_provider is not None:
        cached = getattr(mcp_provider, "cached_specs", lambda: ())()
        registry._merge_mcp(cached)
    return registry


# -- 兼容旧调用点 ------------------------------------------------------
#
# ch01 的 ``app/chat_tools.py`` 与若干旧测试仍按「拿 BaseTool 列表」使用。
# 它们只关心内置工具，给一层薄包装即可；新代码一律用 ``ToolRegistry``。


def get_tools(
    repositories: Repositories, retriever: AnswerContextLike | None = None
) -> list[BaseTool]:
    """内置工具对象列表（不含 MCP），兼容旧调用点。"""
    tools = [spec.tool for spec in build_builtin_tools(repositories, retriever)]
    names = [tool.name for tool in tools]
    if tuple(names) != TOOL_NAMES:
        raise RuntimeError(f"工具注册表与实现不一致：注册 {TOOL_NAMES}，实际 {tuple(names)}")
    return tools


def get_tool(
    name: str, repositories: Repositories, retriever: AnswerContextLike | None = None
) -> BaseTool | None:
    """按名字取一个内置工具；未知名字返回 ``None``。"""
    for candidate in get_tools(repositories, retriever):
        if candidate.name == name:
            return candidate
    return None
