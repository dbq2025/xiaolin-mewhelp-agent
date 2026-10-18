"""ch06 Task 0 探针：用真实 SQLite checkpointer 验证 interrupt/resume。

不是产品代码，只回答一个问题：LangGraph 1.2.15 的 `interrupt()` 在
`AsyncSqliteSaver` + `astream(stream_mode="custom")` 下暂停/续跑时，
`get_stream_writer()` 的自定义事件、`snapshot.next`、`snapshot.tasks[].interrupts`
是不是按官方文档工作。跑完删掉探针库，不污染 data/。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.checkpoint import open_checkpointer


class ProbeState(TypedDict, total=False):
    message: str
    order_id: str
    done: str


def prepare(state: ProbeState) -> dict:
    writer = get_stream_writer()
    writer({"type": "order_picker", "orders": ["A1", "B2"]})
    return {"message": state.get("message", "")}


def await_order(state: ProbeState) -> dict:
    if state.get("order_id"):
        return {}
    picked = interrupt({"kind": "order_picker", "orders": ["A1", "B2"]})
    return {"order_id": picked["order_id"]}


def finish(state: ProbeState) -> dict:
    writer = get_stream_writer()
    writer({"type": "token", "delta": "已选择 " + state.get("order_id", "")})
    return {"done": "yes"}


def build_graph(saver: BaseCheckpointSaver):
    builder = StateGraph(ProbeState)
    builder.add_node("prepare", prepare)
    builder.add_node("await_order", await_order)
    builder.add_node("finish", finish)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "await_order")
    builder.add_edge("await_order", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=saver)


async def main() -> None:
    path = Path(__file__).resolve().parent.parent / "data" / "probe-ch06.sqlite"
    path.unlink(missing_ok=True)
    config = {"configurable": {"thread_id": "probe-ch06"}}

    async with open_checkpointer(path) as saver:
        graph = build_graph(saver)

        first = []
        async for event in graph.astream({"message": "这单能退吗"}, config, stream_mode="custom"):
            first.append(event)

        snapshot = await graph.aget_state(config)
        pending = next(
            task.interrupts[0].value for task in snapshot.tasks if task.interrupts
        )

        print("picker_event:", first)
        print("next_is_await_order:", snapshot.next == ("await_order",))
        print("interrupt_orders:", pending.get("orders"))
        assert first == [{"type": "order_picker", "orders": ["A1", "B2"]}]
        assert snapshot.next == ("await_order",)
        assert pending["orders"] == ["A1", "B2"]

        resumed = []
        async for event in graph.astream(
            Command(resume={"order_id": "A1"}), config, stream_mode="custom"
        ):
            resumed.append(event)

        final = await graph.aget_state(config)
        print("resume_events:", resumed)
        print("final_order:", final.values.get("order_id"))
        assert resumed == [{"type": "token", "delta": "已选择 A1"}]
        assert final.values.get("order_id") == "A1"
        assert final.values.get("done") == "yes"

    path.unlink(missing_ok=True)
    print("probe: OK")


if __name__ == "__main__":
    asyncio.run(main())
