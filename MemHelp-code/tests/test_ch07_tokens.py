"""校准 token 口径。

为什么不能用 ``count_tokens_approximately`` 的默认口径：实测 50 字中文（28 个汉字）
只算 17 tokens，真实分词约 40–50 —— 低估近 3 倍，方向是危险的那一侧（会撑爆窗口）。

本模块是**纯函数**，不读 ``Settings``：参数全部由调用方注入。原因见
``app/context/tokens.py`` 的模块 docstring —— ``trim_messages`` 的二分里会反复
调用它，每次读一遍配置是纯粹的浪费，而且会让它没法在测试里脱离配置独立断言。
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.context.tokens import count_text, count_tokens, make_token_counter


def test_chinese_costs_about_one_token_per_char():
    assert count_text("你好世界") == 4  # 4 汉字 × 1.0


def test_ascii_costs_chars_over_3_5_rounded_up():
    assert count_text("hello world") == 4  # ceil(11 / 3.5) = 4


def test_mixed_text_adds_both_sides():
    assert count_text("你好abc") == 3  # 2 + ceil(3/3.5)=1


def test_fullwidth_punctuation_counts_as_cjk():
    """全角标点按 1 token 算 —— 中文句子里它们和汉字一样密。"""
    assert count_text("你好，世界。") == 6


def test_empty_text_is_zero():
    assert count_text("") == 0


def test_message_overhead_is_per_message():
    one = count_tokens([HumanMessage("你好世界")])
    two = count_tokens([HumanMessage("你好世界"), AIMessage("你好世界")])
    assert one == 4 + 3
    assert two == 2 * (4 + 3)


def test_role_name_is_not_counted():
    """role 名不计字符：否则同一句正文当 user 与当 ai 会算出不同 token。"""
    assert count_tokens([HumanMessage("你好世界")]) == count_tokens([AIMessage("你好世界")])


def test_non_string_content_does_not_raise():
    """多模态 content 是 list；这里只要求稳定，不要求精确。"""
    msg = HumanMessage(content=[{"type": "text", "text": "你好"}])
    assert count_tokens([msg]) >= 1


def test_make_token_counter_matches_count_tokens():
    msgs = [HumanMessage("你好世界"), AIMessage("hello world")]
    assert make_token_counter()(msgs) == count_tokens(msgs)


def test_task_tool_call_arguments_are_counted():
    """带 tool_calls 的 AI 消息，参数也是真发出去的内容，不能当它不存在。"""
    bare = AIMessage("")
    with_calls = AIMessage(
        "",
        tool_calls=[
            {
                "name": "query_order",
                "args": {"order_id": "A123456789"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    assert count_tokens([with_calls]) > count_tokens([bare])


def test_calibrated_counter_is_stricter_than_approximate_for_chinese():
    """同一条中文，校准口径必须比 approx 大 —— 这是「不低估」的最低要求。"""
    from langchain_core.messages.utils import count_tokens_approximately

    msgs = [HumanMessage("我上周买的耳机还没发货，订单号 A123456789")]
    assert count_tokens(msgs) > count_tokens_approximately(msgs)


def test_custom_ratios_are_honoured():
    """比例可调：CJK 1.0 / 其他 3.5 是默认，但必须能被配置覆盖。

    ``chars_per_token`` 是「几个字符合一个 token」——**调大它，token 变少**。
    这个方向很容易写反，所以两个方向都断言。
    """
    assert count_text("你好世界", cjk_chars_per_token=0.5) == 8   # 4 字 ÷ 0.5
    assert count_text("你好世界", cjk_chars_per_token=2.0) == 2   # 4 字 ÷ 2
    assert count_text("abcdefg", other_chars_per_token=1.0) == 7
    assert count_text("abcdefg", other_chars_per_token=7.0) == 1


def test_counter_is_usable_as_trim_messages_token_counter():
    """签名必须满足 ``Callable[[list[BaseMessage]], int]``，否则 trim_messages 用不了。"""
    from langchain_core.messages.utils import trim_messages

    msgs = [HumanMessage("问题" * 10), AIMessage("回答" * 10)] * 5
    kept = trim_messages(
        msgs,
        max_tokens=100,
        token_counter=make_token_counter(),
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    assert 0 < len(kept) < len(msgs)
    assert make_token_counter()(kept) <= 100


def test_empty_list_is_zero():
    assert count_tokens([]) == 0
