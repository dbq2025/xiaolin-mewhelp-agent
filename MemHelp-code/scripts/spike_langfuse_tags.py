"""临时探针 2：验证 tags 维度能否按意图聚合 token（跑完即删）。"""

from __future__ import annotations

import json
import os
import time

import httpx
from langfuse import Langfuse, propagate_attributes

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.environ["LANGFUSE_BASE_URL"],
    tracing_enabled=True,
)

with langfuse.start_as_current_observation(
    name="spike-tagged-turn", input={"question": "tagged"}, metadata={"intent": "refund"}
) as span:
    with propagate_attributes(tags=["intent:refund"], metadata={"intent": "refund"}):
        with span.start_as_current_observation(
            name="tagged-generation", as_type="generation", model="spike-model", input="hi"
        ) as gen:
            gen.update(output="hello", usage_details={"input": 100, "output": 200, "total": 300})
    span.update(output="done", metadata={"intent": "refund", "turn_tokens": 300})
    trace_id = span.trace_id

langfuse.flush()
time.sleep(8)

auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")


def metrics(dimensions, filters=None):
    query = {
        "view": "observations",
        "metrics": [{"measure": "totalTokens", "aggregation": "sum"}],
        "dimensions": [{"field": d} for d in dimensions],
        "filters": filters or [],
        "fromTimestamp": "2026-10-01T00:00:00Z",
        "toTimestamp": "2026-12-31T00:00:00Z",
    }
    r = httpx.get(f"{base}/api/public/v2/metrics", params={"query": json.dumps(query)}, auth=auth, timeout=20)
    print(f"metrics {dimensions} -> {r.status_code}: {r.text[:800]}")


metrics(["tags"])
metrics(["tags"], [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}])

# 取根 observation 详情，看 metadata 是否可读、usage 是否在根上
r = httpx.get(f"{base}/api/public/v2/observations/{trace_id}", auth=auth, timeout=20)
print("obs-by-trace status:", r.status_code, r.text[:1200])

# traces v2？
for path in (f"/api/public/v2/traces/{trace_id}", f"/api/public/v2/traces?limit=3"):
    r = httpx.get(f"{base}{path}", auth=auth, timeout=20)
    print(path, r.status_code, r.text[:800])
