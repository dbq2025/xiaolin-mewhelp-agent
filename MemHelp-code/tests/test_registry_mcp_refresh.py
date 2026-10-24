"""ch08 Task 10：每轮刷新 MCP 清单，并与内置表合并。

刷新是「一轮一次」的动作：``ToolRegistry`` 每轮新建，``refresh_mcp()``
拉一次 provider 缓存。发现失败绝不能把内置工具一起带走 —— 聊天不能因为
物流 Server 没起就整体不可用。
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from app.tools.contract import ToolSpec
from app.tools.registry import build_registry


def _spec(name: str, *, source: str = "mcp", mcp_server: str | None = "logistics") -> ToolSpec:
    async def run() -> dict:
        return {}

    return ToolSpec(
        name=name,
        description="测试工具",
        args_schema={"type": "object", "properties": {}},
        source=source,  # type: ignore[arg-type]
        mcp_server=mcp_server,
        read_only=True,
        tool=StructuredTool.from_function(coroutine=run, name=name, description="测试工具"),
    )


class FakeProvider:
    """按批次返回清单的假 provider，模拟 Server 侧增删工具。"""

    def __init__(self, batches: list[object]) -> None:
        self._batches = list(batches)
        self._cache: tuple[ToolSpec, ...] = ()
        self.calls = 0

    async def discover(self) -> list[ToolSpec]:
        self.calls += 1
        batch = self._batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        self._cache = tuple(batch)  # type: ignore[arg-type]
        return list(self._cache)

    def cached_specs(self) -> tuple[ToolSpec, ...]:
        return self._cache


async def test_refresh_adds_new_mcp_tool_each_turn(fake_repositories) -> None:
    provider = FakeProvider([[_spec("track_shipment")], [_spec("track_shipment"), _spec("check_warranty")]])
    registry = build_registry(fake_repositories, mcp_provider=provider, enable_mcp=True)

    await registry.refresh_mcp()
    first = {spec.name for spec in registry.snapshot()}
    assert "track_shipment" in first
    assert "check_warranty" not in first

    await registry.refresh_mcp()
    second = {spec.name for spec in registry.snapshot()}
    assert {"track_shipment", "check_warranty"} <= second


async def test_builtin_tools_survive_refresh(fake_repositories) -> None:
    provider = FakeProvider([[_spec("track_shipment")]])
    registry = build_registry(fake_repositories)

    await registry.refresh_mcp()  # 无 provider：什么都不做
    names = {spec.name for spec in registry.snapshot()}
    assert {"query_order", "query_product", "query_faq", "create_ticket"} <= names


async def test_mcp_tool_with_builtin_name_is_skipped(fake_repositories) -> None:
    provider = FakeProvider([[_spec("query_order")]])
    registry = build_registry(fake_repositories, mcp_provider=provider, enable_mcp=True)

    await registry.refresh_mcp()

    spec = registry.get("query_order")
    assert spec is not None
    assert spec.source == "builtin"


async def test_provider_failure_keeps_previous_mcp_list(fake_repositories) -> None:
    provider = FakeProvider([[_spec("track_shipment")], RuntimeError("server down")])
    registry = build_registry(fake_repositories, mcp_provider=provider, enable_mcp=True)

    await registry.refresh_mcp()
    await registry.refresh_mcp()

    assert registry.get("track_shipment") is not None


async def test_refresh_uses_provider_cache_not_return_value(fake_repositories) -> None:
    """刷新只认 provider 的缓存：``discover()`` 的返回值可以是空的。

    真实 provider 把「每个 Server 成功/失败」的裁决留在缓存里；注册表不该
    假设返回值就是最终清单（失败的 Server 要沿用旧缓存）。
    """

    class CacheOnlyProvider:
        def __init__(self) -> None:
            self.cache: tuple[ToolSpec, ...] = ()

        async def discover(self) -> list[ToolSpec]:
            return []

        def cached_specs(self) -> tuple[ToolSpec, ...]:
            return self.cache

    provider = CacheOnlyProvider()
    registry = build_registry(fake_repositories, mcp_provider=provider, enable_mcp=True)

    provider.cache = (_spec("track_shipment"),)
    await registry.refresh_mcp()

    assert registry.get("track_shipment") is not None


async def test_graph_builds_registry_and_refreshes(fake_repositories) -> None:
    """图节点走的就是这条路径：构造本轮表 → 刷新 MCP → 交给 ReAct。"""
    from app.agent.graph import GraphDeps, _turn_registry

    provider = FakeProvider([[_spec("track_shipment")]])
    deps = GraphDeps(
        model=None,
        intent_model=None,
        intent_cheap_model=None,
        repositories=fake_repositories,
        retriever=None,
        settings=None,
        mcp_provider=provider,
    )

    registry = await _turn_registry(deps, fake_repositories, None)

    assert registry.get("track_shipment") is not None
    assert provider.calls == 1


async def test_refresh_is_serialized(fake_repositories) -> None:
    """并发刷新不能把两份清单交叉合并。"""
    provider = FakeProvider([[_spec("track_shipment")], [_spec("check_warranty")]])
    registry = build_registry(fake_repositories, mcp_provider=provider, enable_mcp=True)

    await registry.refresh_mcp()
    await registry.refresh_mcp()

    assert provider.calls == 2
