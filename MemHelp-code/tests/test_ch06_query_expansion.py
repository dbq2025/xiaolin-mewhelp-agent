"""ch06 Task 5：退款/售后 Query 扩写（强制 JSON queries）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.query_expansion import QueryExpander, parse_expansion_reply


class _Model:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.last_messages = None

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_messages = messages
        return AIMessage(content=self.reply)


def test_parse_queries_and_dedupe():
    queries = parse_expansion_reply(
        '{"queries":["退货时间窗口","退货运费谁承担","退货时间窗口"]}',
        "这个能退吗",
        max_queries=4,
    )
    assert queries == ("这个能退吗", "退货时间窗口", "退货运费谁承担")


def test_parse_rejects_missing_or_empty():
    assert parse_expansion_reply('{"q":["a"]}', "问题", max_queries=4) is None
    assert parse_expansion_reply('{"queries":[]}', "问题", max_queries=4) is None


def test_parse_caps_max_and_keeps_original():
    queries = parse_expansion_reply(
        '{"queries":["a","b","c","d","e"]}', "原问题", max_queries=3
    )
    assert queries is not None
    assert len(queries) == 3
    assert "原问题" in queries


def test_parse_tolerates_code_fence():
    queries = parse_expansion_reply(
        '```json\n{"queries":["退货条件","退货运费"]}\n```',
        "能退吗",
        max_queries=4,
    )
    assert queries == ("能退吗", "退货条件", "退货运费")


async def test_expand_includes_order_context():
    model = _Model('{"queries":["退货时间窗口","退货运费谁承担"]}')
    result = await QueryExpander(model).expand(
        "这个能退吗", order={"order_id": "20240115001", "product_name": "耳机"}
    )
    assert result.degraded is False
    assert len(result.queries) == 3
    assert model.calls == 1
    assert "20240115001" in model.last_messages[-1].content


async def test_expand_falls_back_on_bad_json():
    result = await QueryExpander(_Model("不是 JSON")).expand("这个能退吗")
    assert result.queries == ("这个能退吗",)
    assert result.degraded is True


async def test_expand_disabled_skips_model():
    model = _Model('{"queries":["a","b"]}')
    result = await QueryExpander(model, enabled=False).expand("这个能退吗")
    assert result.queries == ("这个能退吗",)
    assert model.calls == 0
