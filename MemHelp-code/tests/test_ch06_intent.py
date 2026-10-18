"""ch06 Task 4：八类意图、other 兜底与成本档升级。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.intent import IntentClassifier, heuristic_intent, parse_intent_reply


class _Model:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        return AIMessage(content=self.replies.pop(0))


def test_parse_other():
    result = parse_intent_reply('{"intent":"other","confidence":0.42}')
    assert result is not None
    assert result.intent == "other"


def test_parse_accepts_chinese_other_alias():
    result = parse_intent_reply('{"intent":"其他","confidence":0.4}')
    assert result is not None and result.intent == "other"


def test_heuristic_default_is_other():
    assert heuristic_intent("这个东西怎么弄") == "other"
    assert heuristic_intent("我要投诉并退款") == "complaint"


async def test_quality_mode_only_calls_primary():
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    result = await IntentClassifier(primary).classify("这个能退吗")
    assert result.source == "llm"
    assert result.llm_calls == 1
    assert primary.calls == 1


async def test_cost_mode_escalates_low_confidence():
    cheap = _Model('{"intent":"logistics","confidence":0.3}')
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    result = await IntentClassifier(
        primary, cheap_model=cheap, cost_mode=True, confidence_threshold=0.6
    ).classify("那它能退吗")
    assert result.intent == "refund_return"
    assert result.source == "llm_escalated"
    assert result.llm_calls == 2


async def test_cost_mode_accepts_high_confidence_cheap():
    cheap = _Model('{"intent":"order","confidence":0.95}')
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    result = await IntentClassifier(
        primary, cheap_model=cheap, cost_mode=True
    ).classify("订单 1001 多少钱")
    assert result.intent == "order"
    assert result.source == "llm_cheap"
    assert primary.calls == 0


async def test_bad_cheap_json_escalates():
    cheap = _Model("我觉得是物流")
    primary = _Model('{"intent":"logistics","confidence":0.9}')
    result = await IntentClassifier(
        primary, cheap_model=cheap, cost_mode=True
    ).classify("到哪了")
    assert result.source == "llm_escalated"
    assert result.intent == "logistics"


async def test_bad_primary_falls_back_to_heuristic():
    primary = _Model("nonsense")
    result = await IntentClassifier(primary).classify("我要投诉")
    assert result.source == "heuristic"
    assert result.intent == "complaint"


async def test_cheap_exception_escalates():
    class _Boom:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("cheap down")

    primary = _Model('{"intent":"order","confidence":0.9}')
    result = await IntentClassifier(
        primary, cheap_model=_Boom(), cost_mode=True
    ).classify("订单 1001 多少钱")
    assert result.source == "llm_escalated"
    assert result.intent == "order"


def test_intent_prompt_has_other_and_examples():
    from app.agent.prompts import INTENT_PROMPT

    assert "other" in INTENT_PROMPT
    assert "你们公司股票能买吗" in INTENT_PROMPT
    assert "示例" in INTENT_PROMPT or "用户：" in INTENT_PROMPT
