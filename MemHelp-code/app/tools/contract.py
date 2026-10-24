"""ch08：工具的注册契约。

一个工具进注册中心只需要这一张「身份卡」：名字、用途描述、JSON Schema
参数定义，外加来源与读写属性。内置工具与 MCP 工具共用同一形状，
执行引擎因此不必关心工具从哪来。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import BaseTool

ToolSource = Literal["builtin", "mcp"]
ToolStatus = Literal["成功", "失败", "超时", "校验拦下", "权限拒绝"]

# 审计表的状态枚举（与 sql/ch08-ddl.sql 一致）。放在这里，执行引擎、
# 取消建单的补写路径、测试都引用同一份，避免文案漂移。
TOOL_STATUSES: tuple[ToolStatus, ...] = ("成功", "失败", "超时", "校验拦下", "权限拒绝")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """注册中心里的一条工具定义。

    ``args_schema`` 是**已导出的 JSON Schema**（不是 Pydantic 模型）：内置工具
    由 ``model_json_schema()`` 导出，MCP 工具直接用 Server 的 ``inputSchema``。
    ``presenter`` 是可选的结果呈现器，签名 ``(raw) -> data``；为空时结果原样返回。
    """

    name: str
    description: str
    args_schema: dict[str, Any]
    source: ToolSource
    mcp_server: str | None
    read_only: bool
    tool: BaseTool
    presenter: Callable[[Any], Any] | None = field(default=None)
