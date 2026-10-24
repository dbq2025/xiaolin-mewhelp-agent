"""临时探针 3：验证 compiled graph.with_config 后状态 API 与 custom stream 是否仍可用。"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class S(TypedDict, total=False):
    value: int


async def node(state: S) -> S:
    return {"value": state.get("value", 0) + 1}


async def main() -> None:
    builder = StateGraph(S)
    builder.add_node("inc", node)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    graph = builder.compile(checkpointer=MemorySaver())

    bound = graph.with_config({"callbacks": []})
    print("bound type:", type(bound))
    print("bound has aget_state:", hasattr(bound, "aget_state"))
    print("bound has checkpointer:", hasattr(bound, "checkpointer"))
    print("bound has astream:", hasattr(bound, "astream"))

    config = {"configurable": {"thread_id": "t1"}}
    try:
        async for chunk in bound.astream({"value": 1}, config, stream_mode="custom"):
            print("chunk:", chunk)
        print("custom stream ok")
    except Exception as exc:  # noqa: BLE001
        print("custom stream failed:", type(exc).__name__, exc)

    try:
        print("state:", await graph.aget_state(config))
    except Exception as exc:  # noqa: BLE001
        print("aget_state failed:", type(exc).__name__, exc)


asyncio.run(main())
