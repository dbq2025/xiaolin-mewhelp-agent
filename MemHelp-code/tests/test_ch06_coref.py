"""ch06 Task 3：指代消解触发、解析、防幻觉与回退。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.coref import CorefResolver, format_history, needs_coref, parse_coref_reply


class _Model:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        return AIMessage(content=self.reply)


def test_complete_question_passthrough():
    assert needs_coref("退货政策是什么", []) is False
    assert needs_coref("你好", []) is False


def test_anaphora_needs_coref():
    history = [
        HumanMessage(content="订单 20240115001 的物流到哪了"),
        AIMessage(content="运输中。"),
    ]
    assert needs_coref("那它能退吗", history) is True


def test_colloquial_needs_coref():
    assert needs_coref("钱啥时候回来", []) is True


def test_ellipsis_with_na_needs_coref():
    history = [HumanMessage(content="订单 20240115001 的物流到哪了")]
    assert needs_coref("那现在到哪了", history) is True


def test_explicit_id_makes_question_self_contained():
    assert needs_coref("XH-3000A 续航多久", []) is False
    assert needs_coref("订单 20240115001 多少钱", []) is False


def test_parse_rejects_missing_resolved():
    assert parse_coref_reply('{"changed": true}', "原话", "") is None


def test_parse_tolerates_code_fence():
    parsed = parse_coref_reply(
        '```json\n{"resolved":"订单 20240115001 能退吗","changed":true}\n```',
        "那它能退吗",
        "订单 20240115001 的物流到哪了",
    )
    assert parsed == ("订单 20240115001 能退吗", True)


async def test_resolver_uses_history():
    history = [
        HumanMessage(content="订单 20240115001 的物流到哪了"),
        AIMessage(content="运输中。"),
    ]
    model = _Model('{"resolved":"订单 20240115001 能退吗","changed":true}')
    result = await CorefResolver(model).resolve("那它能退吗", history)
    assert result.resolved == "订单 20240115001 能退吗"
    assert result.changed is True
    assert result.source == "llm"
    assert model.calls == 1


async def test_hallucinated_order_number_falls_back():
    model = _Model('{"resolved":"订单 20249999999 能退吗","changed":true}')
    result = await CorefResolver(model).resolve("那它能退吗", [])
    assert result.resolved == "那它能退吗"
    assert result.source == "fallback"


async def test_timeout_falls_back():
    class _Slow:
        async def ainvoke(self, *args, **kwargs):
            import asyncio

            await asyncio.sleep(1)
            return AIMessage(content="{}")

    result = await CorefResolver(_Slow(), timeout=0.01).resolve("钱啥时候回来", [])
    assert result.resolved == "钱啥时候回来"
    assert result.source == "fallback"


def test_format_history_ignores_tool_and_limits():
    messages = [
        HumanMessage(content="a"),
        ToolMessage(content="tool", tool_call_id="1"),
        AIMessage(content="b"),
        HumanMessage(content="c"),
    ]
    text = format_history(messages, limit=2)
    assert "tool" not in text
    assert text.count("\n") <= 2
