"""ch05 Task 1：配置项与 checkpointer 工厂。

这里只验证「工厂能打开、状态能读回、测试默认值正确」，不重复 LangGraph 自己的
持久化测试；真正的跨轮恢复由 Task 8 的图测试覆盖。
"""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.checkpoint import memory_checkpointer, open_checkpointer


class _State(TypedDict):
    value: int
    messages: Annotated[list, lambda left, right: left + right]


async def _tiny_graph(checkpointer):
    def bump(state: _State) -> dict:
        return {"value": state.get("value", 0) + 1}

    builder = StateGraph(_State)
    builder.add_node("bump", bump)
    builder.add_edge(START, "bump")
    builder.add_edge("bump", END)
    return builder.compile(checkpointer=checkpointer)


async def test_open_checkpointer_persists_state_across_invocations():
    async with open_checkpointer(":memory:") as saver:
        graph = await _tiny_graph(saver)
        config = {"configurable": {"thread_id": "t1"}}
        await graph.ainvoke({"value": 1}, config)
        snapshot = await graph.aget_state(config)
        assert snapshot.values["value"] == 2


def test_memory_checkpointer_is_in_memory_saver():
    assert isinstance(memory_checkpointer(), InMemorySaver)


def test_settings_have_ch05_defaults(test_settings):
    assert test_settings.agent_max_steps == 6
    assert test_settings.agent_token_budget == 6000
    assert test_settings.checkpoint_db_path == "data/checkpoints.sqlite"
    # 测试默认关掉文件日志，避免每个用例往仓库里写 JSONL。
    assert test_settings.turn_log_path == ""
