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
    msgs = EXTRACT_PROMPT.format_messages(text="订单 A123 要退款")
    assert any("订单 A123 要退款" in str(m.content) for m in msgs)


def test_extract_prompt_forbids_fabrication():
    msgs = EXTRACT_PROMPT.format_messages(text="x")
    body = "\n".join(str(m.content) for m in msgs)
    for keyword in ("不要推测", "null"):
        assert keyword in body


# ---------------------------------------------------------------------------
# ch04：引用规范 / 负面知识 / 拒答话术
# ---------------------------------------------------------------------------


def test_tool_prompt_has_eight_behavior_rules():
    from app.prompts import BEHAVIOR_RULES_WITH_TOOLS, TOOL_SYSTEM_PROMPT_TEMPLATE

    assert len(BEHAVIOR_RULES_WITH_TOOLS) == 8
    for rule in BEHAVIOR_RULES_WITH_TOOLS:
        assert rule in TOOL_SYSTEM_PROMPT_TEMPLATE


def test_tool_prompt_forbids_promising_times():
    from app.prompts import TOOL_SYSTEM_PROMPT_TEMPLATE

    assert "不承诺到账时间与送达时点" in TOOL_SYSTEM_PROMPT_TEMPLATE


def test_citation_rules_are_in_the_tool_prompt():
    from app.prompts import CITATION_RULES, TOOL_SYSTEM_PROMPT_TEMPLATE

    for rule in CITATION_RULES:
        assert rule in TOOL_SYSTEM_PROMPT_TEMPLATE
    assert "[n]" in TOOL_SYSTEM_PROMPT_TEMPLATE


def test_negative_knowledge_lines_are_in_the_tool_prompt():
    from app.prompts import NEGATIVE_KNOWLEDGE, TOOL_SYSTEM_PROMPT_TEMPLATE

    for item in NEGATIVE_KNOWLEDGE:
        assert item in TOOL_SYSTEM_PROMPT_TEMPLATE


def test_tool_prompt_keeps_the_system_info_last():
    """引用规范插在「系统信息」之前 —— session_id 那段必须仍在末尾。"""
    from app.prompts import TOOL_SYSTEM_PROMPT_TEMPLATE

    assert TOOL_SYSTEM_PROMPT_TEMPLATE.index("## 引用规范") < TOOL_SYSTEM_PROMPT_TEMPLATE.index(
        "## 系统信息"
    )


def test_refusal_text_states_three_things_and_has_no_citation():
    from app.prompts import REFUSAL_TEXT

    assert "知识库" in REFUSAL_TEXT and "没有" in REFUSAL_TEXT
    assert "转人工" in REFUSAL_TEXT
    assert "[" not in REFUSAL_TEXT  # 没有证据可引，带角标会让用户以为下面有出处
