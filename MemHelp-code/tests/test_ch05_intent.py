"""ch05 Task 4：意图分类的解析、快速通道、启发式兜底与模型异常降级。"""

import pytest
from langchain_core.messages import AIMessage

from app.agent.intent import (
    IntentClassifier,
    heuristic_intent,
    is_trivial_chitchat,
    parse_intent_reply,
)


class _StaticModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def ainvoke(self, *args, **kwargs):
        return AIMessage(content=self.reply)


class _BoomModel:
    async def ainvoke(self, *args, **kwargs):
        raise RuntimeError("upstream down")


def test_parse_intent_json_tolerates_code_fence():
    result = parse_intent_reply('```json\n{"intent":"logistics","confidence":0.91}\n```')
    assert result is not None
    assert result.intent == "logistics"
    assert result.confidence == pytest.approx(0.91)
    assert result.source == "llm"


def test_parse_intent_accepts_extra_prose_around_json():
    result = parse_intent_reply('分类结果：{"intent":"complaint","confidence":0.8} 以上。')
    assert result is not None and result.intent == "complaint"


def test_parse_invalid_returns_none():
    assert parse_intent_reply("我觉得是物流") is None
    assert parse_intent_reply('{"intent":"nonsense","confidence":0.9}') is None


def test_parse_accepts_chinese_alias():
    result = parse_intent_reply('{"intent":"退款退货","confidence":0.7}')
    assert result is not None and result.intent == "refund_return"


def test_trivial_chitchat_is_exact_not_prefix():
    assert is_trivial_chitchat("你好")
    assert is_trivial_chitchat("在吗？")
    assert is_trivial_chitchat(" 谢谢！ ")
    # 寒暄后面带真问题，绝不能走快速通道
    assert not is_trivial_chitchat("在吗，退货政策是什么")
    assert not is_trivial_chitchat("你好，我的订单 1001 到哪了")


def test_heuristic_priority_complaint_over_refund():
    assert heuristic_intent("收到的杯子是碎的，我要投诉并退款") == "complaint"


def test_heuristic_separates_order_and_logistics():
    assert heuristic_intent("订单 1001 多少钱") == "order"
    assert heuristic_intent("订单 1001 的物流到哪了") == "logistics"


def test_heuristic_default_is_other_not_chitchat():
    assert heuristic_intent("这个东西怎么弄") == "other"


async def test_classifier_uses_model_json():
    result = await IntentClassifier(_StaticModel('{"intent":"product_consult","confidence":0.88}')).classify(
        "XH-3000A 续航多久"
    )
    assert result.intent == "product_consult"
    assert result.source == "llm"


async def test_classifier_fast_path_skips_model():
    class _Exploding:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("寒暄快速通道不该调模型")

    result = await IntentClassifier(_Exploding()).classify("你好")
    assert result.intent == "chitchat"
    assert result.source == "rule"


async def test_classifier_falls_back_on_upstream_error():
    result = await IntentClassifier(_BoomModel()).classify("我要投诉")
    assert result.intent == "complaint"
    assert result.source == "heuristic"
    assert "upstream" in result.raw
