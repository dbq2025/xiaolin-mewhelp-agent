"""ch09 Task 10：按意图聚合 Langfuse token / 成本。"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.observability.intent_cost import (
    IntentCost,
    LangfuseNotConfigured,
    aggregate_intent_costs,
    fetch_intent_costs,
)
from tests.fakes import FakeChatModel


def _settings(test_settings, **overrides):
    base = {
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
        "langfuse_base_url": "https://jp.cloud.langfuse.com",
    }
    base.update(overrides)
    return test_settings.model_copy(update=base)


def test_aggregate_sums_by_intent_tag():
    rows = [
        {"tags": ["intent:refund"], "sum_totalTokens": 300, "sum_totalCost": 0.03},
        {"tags": ["intent:refund", "env:dev"], "sum_totalTokens": 200, "sum_totalCost": 0.02},
        {"tags": [], "sum_totalTokens": 99, "sum_totalCost": 0.01},
        {"tags": ["env:prod"], "sum_totalTokens": 50, "sum_totalCost": 0.005},
    ]

    items = aggregate_intent_costs(rows)

    assert [item.intent for item in items] == ["refund"]
    assert items[0].tokens == 500
    assert items[0].cost == pytest.approx(0.05)


def test_aggregate_sorts_by_tokens_desc_and_keeps_cost():
    rows = [
        {"tags": ["intent:chitchat"], "sum_totalTokens": 20, "sum_totalCost": 0.001},
        {"tags": ["intent:refund"], "sum_totalTokens": 90, "sum_totalCost": 0.009},
        {"tags": ["intent:refund"], "sum_totalTokens": 10, "sum_totalCost": 0.001},
    ]

    items = aggregate_intent_costs(rows)

    assert [item.intent for item in items] == ["refund", "chitchat"]
    assert items[0] == IntentCost(intent="refund", tokens=100, cost=0.01, observations=2)


async def test_fetch_intent_costs_calls_metrics_with_tags_dimension(test_settings):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["query"] = json.loads(request.url.params["query"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"tags": ["intent:refund"], "sum_totalTokens": 300, "sum_totalCost": 0.03},
                    {"tags": ["intent:chitchat"], "sum_totalTokens": 40, "sum_totalCost": 0.004},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        items = await fetch_intent_costs(_settings(test_settings), days=3, client=client)

    assert seen["path"] == "/api/public/v2/metrics"
    assert seen["auth"].startswith("Basic ")
    query = seen["query"]
    assert query["view"] == "observations"
    assert query["dimensions"] == [{"field": "tags"}]
    assert {"measure": "totalTokens", "aggregation": "sum"} in query["metrics"]
    assert {"measure": "totalCost", "aggregation": "sum"} in query["metrics"]
    assert "fromTimestamp" in query and "toTimestamp" in query
    assert [item.intent for item in items] == ["refund", "chitchat"]


async def test_fetch_intent_costs_without_credentials_raises(test_settings):
    with pytest.raises(LangfuseNotConfigured):
        await fetch_intent_costs(
            test_settings.model_copy(update={"langfuse_public_key": ""}),
            days=7,
        )


def test_api_intent_cost_returns_items_and_total(test_settings, monkeypatch):
    async def fake_fetch(settings, *, days=7, client=None):
        return [
            IntentCost(intent="refund", tokens=300, cost=0.03, observations=2),
            IntentCost(intent="chitchat", tokens=100, cost=0.01, observations=1),
        ]

    monkeypatch.setattr("app.api.fetch_intent_costs", fake_fetch)
    client = TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=_settings(test_settings))
    )

    resp = client.get("/api/intent-cost", params={"days": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_tokens"] == 400
    assert body["days"] == 3
    assert body["items"][0] == {
        "intent": "refund",
        "tokens": 300,
        "cost": 0.03,
        "observations": 2,
    }


def test_api_intent_cost_without_credentials_is_empty(test_settings):
    client = TestClient(
        app_main.create_app(
            model=FakeChatModel(chunks=["好"]),
            settings=test_settings.model_copy(
                update={"langfuse_public_key": "", "langfuse_secret_key": ""}
            ),
        )
    )

    resp = client.get("/api/intent-cost")

    assert resp.status_code == 200
    assert resp.json() == {"configured": False, "days": 7, "items": [], "total_tokens": 0}
