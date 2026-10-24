"""临时探针：验证 Langfuse SDK v5 的 span/回调/metadata 行为（跑完即删）。"""

from __future__ import annotations

import json
import os
import time

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.environ["LANGFUSE_BASE_URL"],
    tracing_enabled=True,
)

print("SDK version:", __import__("langfuse").__version__)
print("client attrs:", [a for a in dir(langfuse) if "trace" in a.lower() or "observation" in a.lower()])

with langfuse.start_as_current_observation(
    name="spike-turn",
    input={"question": "spike"},
    metadata={"intent": "refund", "seed": True},
) as span:
    print("span attrs:", [a for a in dir(span) if "trace" in a.lower() or "update" in a.lower()])
    with span.start_as_current_observation(
        name="spike-generation", as_type="generation", model="spike-model", input="hi"
    ) as gen:
        gen.update(output="hello", usage_details={"input": 11, "output": 22, "total": 33})
    span.update(metadata={"intent": "refund", "turn_tokens": 33})
    trace_id = span.trace_id

print("trace_id:", trace_id)
langfuse.flush()
time.sleep(6)

import httpx

auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
query = {
    "view": "observations",
    "metrics": [
        {"measure": "totalTokens", "aggregation": "sum"},
        {"measure": "totalCost", "aggregation": "sum"},
    ],
    "dimensions": [{"field": "metadata.intent"}],
    "filters": [],
    "fromTimestamp": "2026-10-01T00:00:00Z",
    "toTimestamp": "2026-12-31T00:00:00Z",
}
r = httpx.get(f"{base}/api/public/v2/metrics", params={"query": json.dumps(query)}, auth=auth, timeout=20)
print("metrics status:", r.status_code)
print("metrics body:", r.text[:2000])

r2 = httpx.get(f"{base}/api/public/v2/observations", params={"limit": 5}, auth=auth, timeout=20)
print("observations status:", r2.status_code)
print("observations body:", r2.text[:2000])
