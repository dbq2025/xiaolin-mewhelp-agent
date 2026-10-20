"""三层划分 + 规则截短。全部纯函数，不碰 DB、不碰网络。

这一层的全部职责是「按两个锚点把历史切成三段」，以及「层 2 按规则截短」。
它**不写库、不改锚点**——只报告建议的新锚点值，由调用方决定是否落库。
"""

from datetime import datetime, timezone

from app.context.layers import (
    LAYER_SUMMARIZED,
    LAYER_TRIMMED,
    LAYER_VERBATIM,
    classify,
    plan_layers,
    rule_trim_row,
)


def _row(i, role, content, **over):
    from app.db.repositories import MessageRow

    over.setdefault("tool_calls", None)
    over.setdefault("tool_call_id", None)
    over.setdefault("created_at", datetime.now(timezone.utc))
    return MessageRow(id=i, role=role, content=content, **over)


def _turns(n, *, start=1, user_chars=20, ai_chars=20):
    rows, i = [], start
    for k in range(n):
        rows.append(_row(i, "user", "问" * user_chars + str(k)))
        i += 1
        rows.append(_row(i, "assistant", "答" * ai_chars + str(k)))
        i += 1
    return rows


# ---------------------------------------------------------------- 分层判定


def test_classify_three_layers():
    assert classify(5, summary_upto=5, layer1_from=10) == LAYER_SUMMARIZED
    assert classify(6, summary_upto=5, layer1_from=10) == LAYER_TRIMMED
    assert classify(9, summary_upto=5, layer1_from=10) == LAYER_TRIMMED
    assert classify(10, summary_upto=5, layer1_from=10) == LAYER_VERBATIM


def test_classify_with_null_anchors_puts_everything_in_layer1():
    assert classify(1, summary_upto=None, layer1_from=None) == LAYER_VERBATIM


def test_classify_with_only_summary_anchor():
    """有摘要边界、还没降级过：摘要覆盖的进层 3，其余都是原文。

    层 2 只在「降级把 layer1_from 推出去」之后才存在。没有降级时把摘要之后
    的一切算层 2，最近几轮就会被规则截短 —— 「最近的留原文」直接失效。
    """
    assert classify(3, summary_upto=3, layer1_from=None) == LAYER_SUMMARIZED
    assert classify(4, summary_upto=3, layer1_from=None) == LAYER_VERBATIM


def test_classify_with_only_layer1_anchor():
    assert classify(3, summary_upto=None, layer1_from=4) == LAYER_TRIMMED
    assert classify(4, summary_upto=None, layer1_from=4) == LAYER_VERBATIM


# ---------------------------------------------------------------- 规则截短


def test_rule_trim_keeps_user_text_verbatim():
    long = "我上周买的耳机还没发货" * 20
    assert rule_trim_row(_row(1, "user", long)).content == long


def test_rule_trim_shortens_assistant_to_preview():
    row = _row(2, "assistant", "好" * 200)
    out = rule_trim_row(row, assistant_preview_chars=60)
    assert out.content == "好" * 60 + "…"
    assert len(out.content) == 61


def test_rule_trim_leaves_short_assistant_untouched():
    row = _row(2, "assistant", "好的呢")
    assert rule_trim_row(row, assistant_preview_chars=60).content == "好的呢"


def test_rule_trim_keeps_empty_assistant_body_empty():
    """纯工具调用那行正文是空串，截短不能把它变成「…」。"""
    assert rule_trim_row(_row(3, "assistant", "")).content == ""


def test_rule_trim_replaces_tool_result_with_one_line_marker():
    row = _row(4, "tool", "很长的工具结果" * 100, tool_call_id="call_abc")
    out = rule_trim_row(row)
    assert out.content.startswith("[工具结果 · ")
    assert "call_abc" in out.content
    assert "字已省略]" in out.content
    assert len(out.content) < 60


def test_rule_trim_marks_tool_marker_with_tool_name_when_available():
    """有工具名就用工具名，比一串 call_x8f3 可读得多。

    工具名不在 tool 行上（那行只有 ``tool_call_id``），它挂在调用它的 assistant 行
    的 ``tool_calls`` 里。所以由调用方把这张表传进来。
    """
    row = _row(4, "tool", "x" * 500, tool_call_id="call_abc")
    out = rule_trim_row(row, tool_names={"call_abc": "query_order"})
    assert "query_order" in out.content
    assert "call_abc" not in out.content


def test_rule_trim_falls_back_to_call_id_when_name_unknown():
    row = _row(4, "tool", "x" * 500, tool_call_id="call_abc")
    assert "call_abc" in rule_trim_row(row, tool_names={}).content


def test_rule_trim_does_not_mutate_the_input_row():
    row = _row(2, "assistant", "好" * 200)
    rule_trim_row(row)
    assert row.content == "好" * 200


def test_rule_trim_preserves_id_and_role():
    row = _row(7, "assistant", "好" * 200)
    out = rule_trim_row(row)
    assert (out.id, out.role) == (7, "assistant")


# ---------------------------------------------------------------- 层 1 规划


def test_plan_with_no_anchors_returns_everything_in_layer1():
    rows = _turns(3)
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=None, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.layer2 == []
    assert [r.id for r in plan.layer1] == [r.id for r in rows]
    assert plan.degraded is False and plan.summary_due is False


def test_plan_splits_by_anchors():
    rows = _turns(5)  # id 1..10，奇数 user、偶数 assistant
    plan = plan_layers(
        rows, summary_upto=4, layer1_from=7, layer1_budget=99999, layer2_budget=99999
    )
    assert [r.id for r in plan.layer2] == [5, 6]
    assert [r.id for r in plan.layer1] == [7, 8, 9, 10]


def test_boundary_on_an_assistant_row_is_corrected_forward():
    """锚点指在 assistant 上（脏数据或历史遗留）时，层 1 要挪到下一轮 user。

    从 assistant 行开始，模型会看到一句没有对应问题的答复 —— 这是最容易
    让模型答非所问的输入形态。
    """
    rows = _turns(5)
    plan = plan_layers(
        rows, summary_upto=4, layer1_from=8, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.layer1[0].role == "user"
    assert plan.layer1_from_msg_id == 9
    assert plan.degraded is True


def test_plan_keeps_summarized_rows_out_of_both_layers():
    """层 3 不进上下文，但它仍在 plan.rows 里（日志要能数出「早期有多少条」）。"""
    rows = _turns(5)
    plan = plan_layers(
        rows, summary_upto=4, layer1_from=7, layer1_budget=99999, layer2_budget=99999
    )
    assert [r.id for r in plan.rows] == list(range(1, 11))
    assert not any(r.id <= 4 for r in plan.layer1 + plan.layer2)


def test_layer1_over_budget_moves_the_anchor_forward():
    rows = _turns(10)  # 20 条
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=1, layer1_budget=60, layer2_budget=99999
    )
    assert plan.degraded is True
    assert plan.dropped_layer1_rows > 0
    assert plan.layer1_from_msg_id > 1
    # 按整轮切：新的层 1 第一条必须是 user
    assert plan.layer1[0].role == "user"
    assert plan.layer1[-1].id == rows[-1].id


def test_layer1_degradation_only_moves_forward():
    """Review Focus 1：降级只前进不后退，绝不回到摘要区间里。"""
    rows = _turns(10)
    plan = plan_layers(
        rows, summary_upto=8, layer1_from=9, layer1_budget=60, layer2_budget=99999
    )
    assert plan.layer1_from_msg_id >= 9


def test_layer1_never_starts_before_the_summary_anchor():
    """即使传入的 layer1_from 比 summary_upto 还小（脏数据），也不能倒退回来。"""
    rows = _turns(10)
    plan = plan_layers(
        rows, summary_upto=8, layer1_from=1, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.layer1_from_msg_id >= 9
    assert all(r.id > 8 for r in plan.layer1)


def test_single_turn_that_does_not_fit_is_kept_and_flagged():
    rows = [_row(1, "user", "问" * 5000), _row(2, "assistant", "答" * 5000)]
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=None, layer1_budget=10, layer2_budget=99999
    )
    assert plan.degraded is True
    # 宁可超，也不让模型看不到刚发生的事
    assert [r.id for r in plan.layer1] == [1, 2]


def test_multiple_turns_that_do_not_fit_keep_the_last_turn_and_report_drops():
    """预算连最后一整轮都装不下时：只留最后一轮，且 dropped 必须如实计数。

    这条守的是可观测性：如果这里「保留全部层 1」并让 dropped=0，调用方报给
    前端的 trimmed 会恒为 0，降级在日志和界面上都彻底不可见。
    """
    rows = _turns(10)  # 10 轮 / 20 条，每条都远超预算
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=None, layer1_budget=10, layer2_budget=99999
    )
    assert plan.degraded is True
    assert plan.dropped_layer1_rows > 0
    # 只留最后一轮：从最后一条 user 开始
    assert plan.layer1[0].role == "user"
    assert plan.layer1[-1].id == rows[-1].id
    assert len(plan.layer1) == len(rows) - plan.dropped_layer1_rows



def test_layer1_within_budget_is_not_degraded():
    rows = _turns(3)
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=1, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.degraded is False
    assert plan.dropped_layer1_rows == 0


def test_layer1_tokens_are_counted():
    rows = _turns(3)
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=1, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.layer1_tokens > 0
    assert plan.layer1_tokens <= 99999


# ---------------------------------------------------------------- 层 2 规划


def test_layer2_over_budget_is_due_for_summary():
    rows = _turns(5)  # 层 2 = id 1..8
    plan = plan_layers(
        rows, summary_upto=0, layer1_from=9, layer1_budget=99999, layer2_budget=50
    )
    assert plan.summary_due is True
    assert [r.id for r in plan.summary_rows] == list(range(1, 9))
    assert plan.layer2_tokens > 50


def test_layer2_within_budget_is_not_due():
    plan = plan_layers(
        _turns(5), summary_upto=0, layer1_from=9, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.summary_due is False
    assert plan.summary_rows == []


def test_layer2_empty_is_never_due():
    rows = _turns(3)
    plan = plan_layers(
        rows, summary_upto=None, layer1_from=None, layer1_budget=99999, layer2_budget=1
    )
    assert plan.summary_due is False


def test_layer2_tokens_are_counted_after_rule_trimming():
    """截短必须让层 2 明显变小，否则「按规则截短」没生效。"""
    rows = [_row(1, "user", "问"), _row(2, "assistant", "答" * 400)]
    full = plan_layers(
        rows, summary_upto=0, layer1_from=3, layer1_budget=99999, layer2_budget=99999
    )
    trimmed = plan_layers(
        rows,
        summary_upto=0,
        layer1_from=3,
        layer1_budget=99999,
        layer2_budget=99999,
        assistant_preview_chars=10,
    )
    assert trimmed.layer2_tokens < full.layer2_tokens


def test_summary_rows_cover_the_whole_layer2():
    """summary_rows 就是「这一批要压成一段」的行，不能只给一部分。"""
    rows = _turns(5)
    plan = plan_layers(
        rows, summary_upto=0, layer1_from=9, layer1_budget=99999, layer2_budget=50
    )
    assert [r.id for r in plan.summary_rows] == [r.id for r in plan.layer2]


def test_summary_rows_are_the_untrimmed_rows():
    """摘要要读原文，不能读截短版——否则梗概里只剩客服的开头 60 字。"""
    rows = [_row(1, "user", "问"), _row(2, "assistant", "答" * 400)]
    plan = plan_layers(
        rows, summary_upto=0, layer1_from=3, layer1_budget=99999, layer2_budget=1
    )
    assert plan.summary_rows[1].content == "答" * 400


# ---------------------------------------------------------------- 边界


def test_empty_rows_are_handled():
    plan = plan_layers([], summary_upto=None, layer1_from=None, layer1_budget=100, layer2_budget=100)
    assert plan.rows == [] and plan.layer1 == [] and plan.layer2 == []
    assert plan.layer1_from_msg_id is None
    assert plan.degraded is False and plan.summary_due is False


def test_reported_anchor_is_the_first_layer1_row_id():
    rows = _turns(5)
    plan = plan_layers(
        rows, summary_upto=4, layer1_from=7, layer1_budget=99999, layer2_budget=99999
    )
    assert plan.layer1_from_msg_id == 7
    assert plan.summary_upto_msg_id == 4
