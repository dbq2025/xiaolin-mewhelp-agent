"""Agent 的工具集：注册中心里的工具 + 一个控制面 `suggest_actions`。

## `create_ticket` 为什么留在工具集里

ch05 时它被**剔除**，只能由前端按钮触发。ch08 的需求改成「客户明确要求建工单
就走确认流」：模型可以发起，但**不直接执行** —— ReAct 循环只把这次调用记进
``pending_ticket_call``，由图节点 ``ticket_await_confirm`` 用 LangGraph
``interrupt()`` 把预览推给前端，用户点「确认提交」才落库。

所以它必须让模型看见（否则模型不知道有这个能力），但执行权在图上，不在模型手里。

## `suggest_actions` 不是业务工具

它只把「我想建议用户转人工/建工单」写进本轮列表，返回一句确认；不碰数据库、
不发请求。它是 Agent 与前端按钮之间的**控制面**信号。
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.agent.state import SuggestedAction
from app.tools.registry import ToolRegistry

__all__ = ["SuggestActionsArgs", "agent_tools", "make_suggest_actions_tool"]

#: 退款子流程里要隐藏的订单查询工具（订单快照已由系统锁死）。
_ORDER_TOOLS = ("query_order", "track_shipment")


class SuggestActionsArgs(BaseModel):
    """给模型看的建议动作参数。"""

    actions: list[Literal["transfer_human", "create_ticket"]] = Field(
        min_length=1,
        description=(
            "要建议给用户的动作。transfer_human=转人工客服，"
            "create_ticket=创建一个工单。只记录建议，不执行。"
        ),
    )
    reason: str = Field(
        default="",
        description="为什么建议这些动作（一句话，进日志，不直接展示给用户）。",
    )


def make_suggest_actions_tool(record: list[SuggestedAction]) -> BaseTool:
    """构造一个把动作写进 `record` 的控制面工具。

    `record` 是本轮的可变列表；工具只追加、去重，不做任何业务写操作。
    """

    @tool("suggest_actions", args_schema=SuggestActionsArgs)
    def suggest_actions(actions: list[str], reason: str = "") -> str:
        """建议用户转人工客服或创建工单。

        **只记录建议，不执行任何动作**。用户在页面上自己点按钮才会真正转接/建单。
        当问题需要人工介入、反复沟通未解决、或用户明确要求人工时使用它。
        一次可以把两个动作都给用户，也可以只给一个。
        """
        for action in actions:
            if action not in record:
                record.append(action)  # type: ignore[arg-type]
        labels = {
            "transfer_human": "转人工",
            "create_ticket": "建工单",
        }
        rendered = "、".join(labels.get(action, action) for action in record)
        return json.dumps(
            {
                "ok": True,
                "recorded": list(record),
                "message": f"已记录建议：{rendered}（仅记录，未执行；由用户选择）",
                "reason": reason,
            },
            ensure_ascii=False,
        )

    return suggest_actions


def agent_tools(
    registry: ToolRegistry,
    *,
    include_faq: bool,
    include_order: bool = True,
    record_actions: list[SuggestedAction],
) -> list[BaseTool]:
    """本轮 Agent 可用的工具。

    :param registry: 当轮工具表（内置 + MCP）。
    :param include_faq: 业务数据出口为 True（Agent 可自己查政策）；
        知识出口为 False（已经强制预检索，避免二次检索与引用编号冲突）。
    :param include_order: 退款子流程传 False：订单快照已由系统查好并锁死，
        再给订单/物流查询会让模型拿另一个编号再查一次，出现
        「系统快照 A 单、Agent 自报 B 单」。商品查询仍保留。
    :param record_actions: 控制面工具写入的动作列表。
    """
    tools = [spec.tool for spec in registry.snapshot()]
    if not include_faq:
        tools = [tool for tool in tools if tool.name != "query_faq"]
    if not include_order:
        tools = [tool for tool in tools if tool.name not in _ORDER_TOOLS]
    tools.append(make_suggest_actions_tool(record_actions))
    return tools
