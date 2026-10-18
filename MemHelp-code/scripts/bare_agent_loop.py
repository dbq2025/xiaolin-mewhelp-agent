"""祛魅热身：不用 LangGraph，手写一遍最裸的 Agent 循环。

## 它到底是什么

```
messages = [Human]
while 没超过最大步数:
    ai = model.bind_tools(tools).ainvoke(messages)
    messages.append(ai)
    if ai 没有 tool_calls:
        return ai 的文本
    tool = tools[ai.tool_calls[0].name]
    result = tool.ainvoke(ai.tool_calls[0].args)
    messages.append(ToolMessage(result))
```

没有图、没有 checkpoint、没有状态机。它就是一个**带工具的循环**。
`app/agent/react.py` 是同一个循环加上流式、token 预算、建议动作与落库边界。

## 为什么不进线上链路

热身代码的价值是「看清结构」，不是多一条维护路径。线上只保留
`app/agent/react.py` 这一套；这里用脚本运行，测试只钉住循环语义。

运行（真实模型需要 `.env`）：

```powershell
uv run --directory MemHelp-code python scripts/bare_agent_loop.py --message "订单 1001 到哪了"
# 不联网看结构：
uv run --directory MemHelp-code python scripts/bare_agent_loop.py --fake
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from app.tools.business import query_order as _query_order

__all__ = ["BareResult", "query_order_tool", "run_bare_agent"]


@tool("query_order")
def query_order_tool(order_id: str) -> str:
    """按订单号查询订单状态、金额与下单时间。"""
    return _query_order(order_id)


@dataclass
class BareResult:
    """一轮裸循环的产物。"""

    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)


def _first_call_only(message: AIMessage, chosen: dict[str, Any]) -> AIMessage:
    if len(message.tool_calls) <= 1:
        return message
    return message.model_copy(update={"tool_calls": [chosen]})


async def run_bare_agent(
    model: Any,
    user_message: str,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 5,
    verbose: bool = True,
) -> BareResult:
    """最裸的 ReAct 循环：有工具调用就执行并回灌，没有就收敛。"""
    by_name = {item.name: item for item in tools}
    messages: list[BaseMessage] = [HumanMessage(content=user_message)]
    tool_calls: list[str] = []
    steps = 0

    while steps < max_steps:
        response = await model.bind_tools(tools).ainvoke(messages)
        messages.append(response)
        calls = list(getattr(response, "tool_calls", None) or [])

        if not calls:
            return BareResult(
                answer=response.content if isinstance(response.content, str) else str(response.content),
                messages=messages,
                steps=steps,
                tool_calls=tool_calls,
            )

        chosen = calls[0]
        name = str(chosen.get("name", ""))
        args = dict(chosen.get("args") or {})
        call_id = str(chosen.get("id", ""))
        tool_calls.append(name)
        if verbose:
            print(f"[step {steps + 1}] -> {name}({json.dumps(args, ensure_ascii=False)})")

        # 只执行第一个调用，并把 assistant 消息裁到只剩它 —— 与正式版同一口径。
        messages[-1] = _first_call_only(response, chosen)
        target = by_name.get(name)
        if target is None:
            observation = json.dumps({"ok": False, "error": f"未知工具 {name}"}, ensure_ascii=False)
        else:
            observation = str(await target.ainvoke(args))
        messages.append(ToolMessage(content=observation, tool_call_id=call_id))
        steps += 1

    return BareResult(
        answer=(
            f"已达到最大步数 {max_steps}，先停在这里。"
            "真实链路里这时会给用户转人工/建单的建议。"
        ),
        messages=messages,
        steps=steps,
        tool_calls=tool_calls,
    )


class _DemoModel:
    """`--fake` 用：第一步查订单，第二步收敛。"""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: Sequence[BaseTool]) -> "_DemoModel":
        return self

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                        "id": "call_demo_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="订单 1001 已发货，正在派送，预计明天送达。")


async def _main() -> int:
    parser = argparse.ArgumentParser(description="手写 Agent 循环热身")
    parser.add_argument("--message", default="订单 1001 到哪了")
    parser.add_argument("--fake", action="store_true", help="不联网，用本地脚本模型")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    if args.fake:
        model: Any = _DemoModel()
    else:
        from app.config import get_settings
        from app.llm import get_chat_model

        model = get_chat_model(get_settings())

    result = await run_bare_agent(
        model, args.message, [query_order_tool], max_steps=args.max_steps
    )
    print("\n=== 最终答案 ===")
    print(result.answer)
    print(f"\n工具调用：{result.tool_calls}；步数：{result.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
