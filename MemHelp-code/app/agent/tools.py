"""Agent 的工具集：复用第 2 章查询工具 + 一个控制面 `suggest_actions`。

## 为什么 `create_ticket` 必须被剔除

需求 8：转人工与建工单是**两件事**，都交用户在前端自选，后端不自动执行。
如果把 `create_ticket` 留在 Agent 工具集里，模型迟早会「顺手」把工单建了 ——
用户还没点按钮，票已经写了，这违背需求，而且很难在事后区分是模型建的还是用户建的。

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
from app.db.repositories import Repositories
from app.tools.knowledge import AnswerContextLike
from app.tools.registry import get_tools

__all__ = ["SuggestActionsArgs", "agent_tools", "make_suggest_actions_tool"]


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
    repositories: Repositories,
    retriever: AnswerContextLike | None,
    *,
    include_faq: bool,
    record_actions: list[SuggestedAction],
) -> list[BaseTool]:
    """本轮 Agent 可用的工具。

    :param include_faq: 业务数据出口为 True（Agent 可自己查政策）；
        知识出口为 False（已经强制预检索，避免二次检索与引用编号冲突）。
    :param record_actions: 控制面工具写入的动作列表。
    """
    tools = [tool for tool in get_tools(repositories, retriever) if tool.name != "create_ticket"]
    if not include_faq:
        tools = [tool for tool in tools if tool.name != "query_faq"]
    tools.append(make_suggest_actions_tool(record_actions))
    return tools
