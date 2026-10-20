"""预算倒推：演示配置必须逐项对上验收第 2 条的数字。

这组断言是本章的「数字契约」。滑窗 5650 / 层1 3954 / 层2 1695 不是随手取的，
是从 ``MODEL_CONTEXT_WINDOW=18000`` 一路扣下来的结果；任何一项固定开销改了口径，
这里就会红。
"""

from app.config import Settings
from app.context.budget import (
    INSUFFICIENT_BUDGET_PREFIX,
    compute_budget,
    format_budget_report,
)


def _settings(**over):
    data = dict(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    data.update(over)
    return Settings(_env_file=None, **data)


def _demo(**over):
    """验收第 2 条点名的那组演示配置。"""
    base = dict(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        tool_result_max_tokens=1200,
        agent_max_steps=3,
        rag_rerank_top_n=5,
    )
    base.update(over)
    return compute_budget(_settings(**base))


def test_demo_configuration_reproduces_the_documented_numbers():
    r = _demo()
    assert (r.output_reserve, r.react_peak) == (2000, 5600)
    assert r.safety_margin == 1800
    assert (r.system_reserve, r.evidence_reserve, r.summary_reserve) == (1200, 1250, 500)
    assert r.overhead == 12350
    assert r.window_available == 5650
    assert r.desired_history == 8000
    assert r.history_budget == 5650
    assert r.layer2_budget == 1695
    assert r.layer1_budget == 3954
    assert r.ok is True


def test_default_window_is_healthy_and_keeps_the_desired_turn_count():
    r = compute_budget(_settings())
    assert r.ok is True
    # 默认 agent_max_steps = 6（不能改，ch05 的断言钉着它）
    assert r.react_peak == 2000 + 6 * 1200
    assert r.safety_margin == 3200
    assert r.window_available == 32000 - 20600
    # 默认窗口匀得出来的比「想留住的 20 轮」多 → 取小的那个（desired）
    assert r.window_available > r.desired_history
    assert r.history_budget == r.desired_history == 8000


def test_too_small_window_reports_insufficient_budget():
    """验收第 2 条括号里说的「光调窗口不够」：这里窗口被固定开销吃光。"""
    r = _demo(model_context_window=8000)
    assert r.ok is False
    assert r.reason.startswith(INSUFFICIENT_BUDGET_PREFIX)
    assert r.window_available <= 0
    assert r.history_budget == 0


def test_budget_that_cannot_fit_a_single_turn_is_not_ok():
    """第二类不 OK：窗口匀得出来，但连一轮都装不下。

    演示配置下固定开销 10550 + ceil(0.1W)。W=12000 时段 250 token —— 正数，
    但装不下一轮 400，属于「滑窗会归零」那种不健康。
    """
    r = _demo(model_context_window=12000)
    assert r.window_available == 250
    assert r.ok is False
    assert r.reason.startswith(INSUFFICIENT_BUDGET_PREFIX)


def test_history_budget_never_below_one_turn_when_ok():
    for window in (12000, 12800, 14000, 15000, 18000, 32000):
        r = _demo(model_context_window=window)
        if r.ok:
            assert r.history_budget >= r.per_turn_floor
        else:
            assert r.reason.startswith(INSUFFICIENT_BUDGET_PREFIX)


def test_layer_budgets_split_the_history_budget():
    r = _demo()
    assert r.layer1_budget + r.layer2_budget + 1 == r.history_budget


def test_layer_budgets_are_integer_and_positive_for_the_demo_config():
    r = _demo()
    assert isinstance(r.layer1_budget, int) and isinstance(r.layer2_budget, int)
    assert r.layer1_budget > 0 and r.layer2_budget > 0


def test_safety_margin_rounds_up():
    r = _demo(model_context_window=18001)
    assert r.safety_margin == 1801  # ceil(1800.1)


def test_format_budget_report_contains_the_key_numbers():
    line = format_budget_report(_demo())
    for piece in ("window=18000", "history=5650", "layer1=3954", "layer2=1695"):
        assert piece in line


def test_report_is_frozen():
    """预算报告是纯数据快照：被下游改掉会让「同一轮内自相矛盾」。"""
    r = _demo()
    with __import__("pytest").raises(Exception):
        r.history_budget = 1  # type: ignore[misc]
