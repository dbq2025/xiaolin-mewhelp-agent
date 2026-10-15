"""上下文裁剪测试。

两条 Review Focus 钉在这里：
- **裁剪顺序**：先裁历史、再拼本轮问题。顺序反了，预算紧张时用户刚问的那句话
  会被裁掉，模型答非所问。
- **预算约束**：裁剪后实际占用不能超预算（近似计数偏保守，是安全方向）。

注意 `build_context` 只负责裁剪与拼装，不构造消息内容 —— 模板渲染是
`app.prompts.build_chat_messages` 的职责。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.context import build_context

SYSTEM = "你是客服小美。"


def _history(pairs: int = 5, prefix: str = "历史") -> list:
    """造 `pairs` 轮问答。"""
    msgs: list = []
    for i in range(pairs):
        msgs.append(HumanMessage(f"{prefix}问题{i}"))
        msgs.append(AIMessage(f"{prefix}回答{i}"))
    return msgs


def _build(history, question: str, budget: int):
    """按新契约调用：已渲染的 System / Human 消息 + 预算。"""
    return build_context(
        history=history,
        question=HumanMessage(question),
        system_message=SystemMessage(SYSTEM),
        budget_tokens=budget,
    )


def test_message_order_is_system_then_history_then_current_question():
    msgs = _build(_history(2), "本轮问题", 2000).messages
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == SYSTEM
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content == "本轮问题"


def test_current_question_survives_a_tiny_budget():
    """Review Focus 1：先裁历史再拼本轮，问题永不被裁掉。"""
    result = _build(_history(50), "这句话绝不能丢", 1)
    assert result.messages[-1].content == "这句话绝不能丢"
    assert isinstance(result.messages[0], SystemMessage)


def test_tight_budget_drops_oldest_and_keeps_newest():
    history = _history(20)
    result = _build(history, "本轮", 60)
    assert result.trimmed_messages > 0
    kept = result.messages[1:-1]
    assert kept, "预算 60 应至少保住最近一条历史"
    assert kept[-1].content == history[-1].content
    assert all(m.content not in ("历史问题0", "历史回答0") for m in kept)


def test_no_trimming_when_history_fits():
    result = _build(_history(2), "本轮", 2000)
    assert result.trimmed_messages == 0
    assert result.kept_messages == 4


def test_trimmed_and_kept_account_for_all_history_messages():
    result = _build(_history(20), "本轮", 60)
    assert result.trimmed_messages + result.kept_messages == 40


def test_kept_history_starts_on_human_message():
    result = _build(_history(20), "本轮", 60)
    kept = result.messages[1:-1]
    if kept:
        assert isinstance(kept[0], HumanMessage)


def test_used_tokens_within_budget_when_possible():
    """Review Focus 3：裁剪后历史不能超预算。"""
    result = _build(_history(20), "短问题", 120)
    assert result.used_tokens <= 120


def test_total_prompt_tokens_stay_within_budget_plus_overhead():
    """更贴近真实约束的一条：**总量**（System + 历史 + 本轮）要有界。

    单看 ``used_tokens <= budget`` 近乎恒真（``trim_messages`` 自己就保证了），
    真正会撑爆上下文窗口的是三者之和。这里把上界写成
    ``budget + system + question``，超出即说明预算语义被改坏了。
    """
    from langchain_core.messages.utils import count_tokens_approximately

    system = SystemMessage(SYSTEM)
    question = HumanMessage("短问题")
    result = _build(_history(50), "短问题", 200)

    total = count_tokens_approximately(result.messages)
    upper_bound = (
        200
        + count_tokens_approximately([system])
        + count_tokens_approximately([question])
    )
    assert total <= upper_bound, (
        f"总量 {total} 超出上界 {upper_bound} —— 裁剪没有把历史压到预算内"
    )


def test_tiny_budget_reports_empty_history_without_raising():
    """预算被压到 1 时不能抛异常，历史被裁空、仍能基于本轮作答。"""
    result = _build(_history(20), "问题", 1)
    assert result.kept_messages == 0
    assert result.trimmed_messages == 40
    assert len(result.messages) == 2


def test_empty_history_yields_system_and_question_only():
    result = _build([], "你好", 2000)
    assert len(result.messages) == 2
    assert result.kept_messages == 0
    assert result.trimmed_messages == 0
    assert result.used_tokens == 0


def test_system_prompt_tokens_reported_and_excluded_from_budget():
    result = _build(_history(20), "问题", 60)
    assert result.system_prompt_tokens == count_tokens_approximately([SystemMessage(SYSTEM)])
    assert result.used_tokens <= 60


def test_allow_partial_is_false_so_no_half_messages():
    """不切半条消息：保留的每条历史都必须原样来自输入。"""
    history = _history(20)
    originals = {m.content for m in history}
    result = _build(history, "问题", 60)
    for msg in result.messages[1:-1]:
        assert msg.content in originals


def test_budget_tokens_is_echoed_back():
    result = _build(_history(2), "问题", 777)
    assert result.budget_tokens == 777
