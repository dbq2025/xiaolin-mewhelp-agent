"""客服 Prompt 测试。

按用户要求，纯 Prompt 的产出不走 TDD：**行为是否真的守住**由评估集
（`evals/service_prompt_cases.md` + `run_evals.py`）验证，这里只钉住
**离线可测的结构**：

- 模板渲染出的消息序列与角色正确；
- 7 条行为约束条款确实**逐字**存在于 System Prompt 里（否则评估清单里判的
  是一份不存在的规则）。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts import (
    BEHAVIOR_RULES,
    EXTRACT_PROMPT,
    SERVICE_SYSTEM_PROMPT,
    build_chat_messages,
)


def test_build_chat_messages_order_is_system_history_human():
    history = [HumanMessage("上轮问题"), AIMessage("上轮回答")]
    msgs = build_chat_messages(history, "本轮问题")
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == SERVICE_SYSTEM_PROMPT
    assert [type(m) for m in msgs[1:]] == [HumanMessage, AIMessage, HumanMessage]
    assert msgs[-1].content == "本轮问题"


def test_build_chat_messages_with_empty_history():
    msgs = build_chat_messages([], "你好")
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)


def test_behavior_rules_are_seven_and_all_appear_in_system_prompt():
    assert len(BEHAVIOR_RULES) == 7
    for rule in BEHAVIOR_RULES:
        assert rule in SERVICE_SYSTEM_PROMPT


def test_system_prompt_covers_the_red_lines():
    """三条红线必须有明文：不编造、不越权承诺、不索要验证码。"""
    for keyword in ("不编造", "不能承诺", "不索要验证码"):
        assert keyword in SERVICE_SYSTEM_PROMPT


def test_system_prompt_defines_the_customer_service_role():
    for keyword in ("小美", "客服"):
        assert keyword in SERVICE_SYSTEM_PROMPT


def test_extract_prompt_renders_text_variable():
    rendered = EXTRACT_PROMPT.format(text="订单 A123 要退款")
    assert "订单 A123 要退款" in rendered


def test_extract_prompt_forbids_fabrication():
    rendered = EXTRACT_PROMPT.format(text="x")
    for keyword in ("不要推测", "null"):
        assert keyword in rendered
