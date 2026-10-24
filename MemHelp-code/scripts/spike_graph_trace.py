"""临时探针 4：LangGraph 编译期回调是否把模型 span 挂到手动根 span 下，标签是否落到 trace。"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TypedDict

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.environ["LANGFUSE_BASE_URL"],
    tracing_enabled=True,
)

model = FakeListChatModel(responses=["pong"])


class S(TypedDict, total=False):
    answer: str


async def node(state: S) -> S:
    # 模拟「意图分类之后才知道 intent」：在节点里给当前 trace 打 tag。
    with propagate_attributes(tags=["intent:chitchat"], metadata={"intent": "chitchat"}):
        reply = await model.ainvoke("ping")
    return {"answer": reply.content}


async def main() -> None:
    builder = StateGraph(S)
    builder.add_node("agent", node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile().with_config({"callbacks": [CallbackHandler()]})

    with langfuse.start_as_current_observation(name="spike-graph-turn", input={"q": "ping"}) as span:
        result = await graph.ainvoke({"answer": ""}, {"configurable": {"thread_id": "spike-graph"}})
        span.update(output=result, metadata={"intent": "chitchat", "turn_tokens": 0})
        trace_id = span.trace_id
    print("trace:", trace_id, "answer:", result)
    langfuse.flush()
    time.sleep(8)

    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    r = httpx.get(
        f"{base}/api/public/v2/observations", params={"traceId": trace_id, "limit": 50}, auth=auth, timeout=20
    )
    print("obs:", r.status_code)
    for row in r.json().get("data", []):
        print(" ", row["type"], row["name"], "parent=", row["parentObservationId"])

    q = {
        "view": "observations",
        "metrics": [{"measure": "totalTokens", "aggregation": "sum"}],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": "2026-10-01T00:00:00Z",
        "toTimestamp": "2026-12-31T00:00:00Z",
    }
    r = httpx.get(f"{base}/api/public/v2/metrics", params={"query": __import__("json").dumps(q)}, auth=auth, timeout=20)
    print("tags metrics:", r.status_code, r.text[:800])


asyncio.run(main())
