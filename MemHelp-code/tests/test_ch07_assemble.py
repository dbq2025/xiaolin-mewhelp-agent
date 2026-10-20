"""拼装顺序是硬件约束，不是风格偏好。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context import assemble_context, build_context

SYSTEM = "你是客服小美。"


def _msg(i, role, text):
    return (HumanMessage if role == "user" else AIMessage)(text, id=str(i))


def test_order_is_system_layer2_layer1_question_reference():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer2=[_msg(1, "user", "旧问题"), _msg(2, "assistant", "旧回答…")],
        layer1=[_msg(3, "user", "近问题"), _msg(4, "assistant", "近回答")],
        question=HumanMessage("当前这句"),
        summary_text="用户问过耳机",
        evidence_text="证据1",
    )
    contents = [m.content for m in out.messages]
    assert contents[0] == SYSTEM
    assert contents[1].startswith("旧问题")
    assert contents[2].startswith("旧回答")
    assert contents[3].startswith("近问题")
    assert contents[4] == "近回答"
    assert contents[5] == "当前这句"
    assert "用户问过耳机" in contents[6] and "证据1" in contents[6]


def test_only_one_system_message_and_it_is_first():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer1=[_msg(1, "user", "q"), _msg(2, "assistant", "a")],
        question=HumanMessage("当前"),
        summary_text="梗概",
        evidence_text="证据",
    )
    systems = [m for m in out.messages if isinstance(m, SystemMessage)]
    assert len(systems) == 1
    assert isinstance(out.messages[0], SystemMessage)


def test_summary_and_evidence_are_merged_into_one_message():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("当前"),
        summary_text="梗概",
        evidence_text="证据",
    )
    refs = out.messages[-1]
    assert isinstance(refs, HumanMessage)
    assert "梗概" in refs.content and "证据" in refs.content


def test_no_reference_message_when_both_empty():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("当前"),
    )
    assert len(out.messages) == 2
    assert out.messages[-1].content == "当前"


def test_question_comes_before_reference():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("当前"),
        summary_text="梗概",
    )
    assert out.messages[-2].content == "当前"
    assert out.messages[-1].content != "当前"


def test_system_content_is_identical_across_two_turns():
    """Review Focus 4：system 一旦掺进可变内容，前缀缓存整段作废。"""
    first = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("第一轮"),
    )
    second = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer1=[_msg(1, "user", "第一轮"), _msg(2, "assistant", "答")],
        question=HumanMessage("第二轮"),
        summary_text="一段梗概",
        evidence_text="一条证据",
    )
    assert first.messages[0].content == second.messages[0].content


def test_system_content_identical_even_with_different_evidence():
    """证据每轮都变；只要它不进 system，前缀就不会作废。"""
    a = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("q1"),
        evidence_text="[证据1] 退货新规",
    )
    b = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("q2"),
        evidence_text="[证据1] 完全不同的政策",
    )
    assert a.messages[0].content == b.messages[0].content


def test_counts_report_each_layer():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer2=[_msg(1, "user", "a")],
        layer1=[_msg(2, "user", "b")],
        question=HumanMessage("c"),
    )
    assert out.layer2_messages == 1
    assert out.layer1_messages == 1
    assert out.kept_messages == 2
    assert out.trimmed_messages == 0


def test_reference_block_labels_both_parts():
    from app.context import build_reference_block

    block = build_reference_block(summary_text="梗概内容", evidence_text="证据内容")
    assert block.index("会话梗概") < block.index("梗概内容")
    assert block.index("梗概内容") < block.index("检索证据")
    assert block.index("检索证据") < block.index("证据内容")


def test_reference_block_omits_empty_parts():
    from app.context import build_reference_block

    assert build_reference_block() == ""
    assert build_reference_block(summary_text="只有梗概") == "[会话梗概]\n只有梗概"
    assert build_reference_block(evidence_text="只有证据") == "[检索证据]\n只有证据"


def test_legacy_shell_still_uses_approximate_counting():
    """兼容外壳必须逐字节复刻 ch01 行为：used_tokens 仍是 approx 口径。"""
    from langchain_core.messages.utils import count_tokens_approximately

    out = build_context(
        history=[HumanMessage("历史问题0"), AIMessage("历史回答0")],
        question=HumanMessage("本轮"),
        system_message=SystemMessage(SYSTEM),
        budget_tokens=2000,
    )
    assert out.used_tokens == count_tokens_approximately(out.messages[1:-1])
    assert out.budget_tokens == 2000


def test_legacy_shell_reports_zero_for_new_fields():
    """新字段在旧外壳里必须是「没发生」的样子，不能让调用方误判。"""
    out = build_context(
        history=[HumanMessage("q"), AIMessage("a")],
        question=HumanMessage("本轮"),
        system_message=SystemMessage(SYSTEM),
        budget_tokens=2000,
    )
    assert out.layer2_messages == 0
    assert out.layer1_messages == 2
    assert out.summary_upto_msg_id is None
    assert out.layer1_from_msg_id is None
    assert out.degraded is False
    assert out.summary_due is False
    assert out.summary_tokens == 0


def test_reference_text_is_not_a_system_message():
    """梗概 + 证据合成的那条必须是 Human —— 上游模板会把 system 全部上提合并。"""
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("当前"),
        summary_text="梗概",
        evidence_text="证据",
    )
    for msg in out.messages[1:]:
        assert not isinstance(msg, SystemMessage)


def test_layer1_is_kept_verbatim_in_output():
    """层 1 不做任何改写：内容原样进 messages。"""
    long_text = "我上周买的耳机还没发货" * 20
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer1=[HumanMessage(long_text)],
        question=HumanMessage("当前"),
    )
    assert out.messages[1].content == long_text


# ---- 梗概注入的预算上限（select_summary_texts）----


def test_summary_texts_within_budget_are_all_used():
    from app.context.assemble import select_summary_texts

    kept, dropped = select_summary_texts(["第一段", "第二段"], budget_tokens=500)
    assert kept == ["第一段", "第二段"]
    assert dropped == 0


def test_summary_texts_keep_head_and_tail_and_drop_the_middle():
    """最早那段记着最早的事实，最近那段接着现在的对话，中间的多半是过程。"""
    from app.context.assemble import select_summary_texts

    texts = [f"第{i}段" + "内容" * 40 for i in range(1, 7)]
    kept, dropped = select_summary_texts(texts, budget_tokens=200)

    assert dropped > 0
    assert kept[0] == texts[0]
    assert kept[-1] == texts[-1]
    assert kept == [texts[0], *texts[len(texts) - (len(kept) - 1):]]
    assert len(kept) + dropped == len(texts)


def test_single_oversized_summary_is_truncated_not_dropped():
    """只有一段且超预算时截短它 —— 宁可少几个字，也不能把唯一早期事实丢掉。"""
    from app.context.assemble import select_summary_texts

    kept, dropped = select_summary_texts(["订" * 2000], budget_tokens=100)

    assert dropped == 0
    assert len(kept) == 1
    assert kept[0].endswith("…")
    assert len(kept[0]) < 2000


def test_summary_texts_respect_the_budget_after_selection():
    from app.context.assemble import select_summary_texts
    from app.context.tokens import count_tokens

    texts = [f"第{i}段" + "内容" * 60 for i in range(1, 9)]
    kept, _ = select_summary_texts(texts, budget_tokens=300)
    total = count_tokens([HumanMessage(content="\n".join(kept))])

    assert total <= 300


def test_empty_summary_texts_select_nothing():
    from app.context.assemble import select_summary_texts

    assert select_summary_texts([], budget_tokens=500) == ([], 0)
    assert select_summary_texts(["", "   "], budget_tokens=500) == ([], 0)


def test_zero_budget_drops_everything_rather_than_injecting_free_text():
    from app.context.assemble import select_summary_texts

    kept, dropped = select_summary_texts(["第一段", "第二段"], budget_tokens=0)
    assert kept == []
    assert dropped == 2


def test_assemble_reports_dropped_summary_segments():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        question=HumanMessage("当前"),
        summary_text="梗概",
        summary_dropped_segments=2,
    )
    assert out.summary_dropped_segments == 2
