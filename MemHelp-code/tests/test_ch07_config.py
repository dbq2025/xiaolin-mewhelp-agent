"""ch07 新增配置键：存在性、默认值、别名兼容、取值约束。

两条「别名」用例是本章最容易翻车的地方：``MAX_AGENT_STEPS`` 与 ``AGENT_MAX_STEPS``
都要能读，而 ``agent_max_steps`` 的默认值**一个字都不能改**（6）——
``tests/test_ch05_config_checkpoint.py:45`` 钉着它。做法是 ``AliasChoices``，
不是改默认值。
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _base(**over):
    data = dict(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    data.update(over)
    return Settings(_env_file=None, **data)


def test_ch07_defaults():
    s = _base()
    assert s.model_context_window == 32000
    assert s.max_output_tokens == 4000
    assert s.max_user_input_tokens == 2000
    assert s.tool_result_max_tokens == 1200
    assert s.safety_margin_ratio == 0.10
    assert s.system_prompt_reserve_tokens == 1200
    assert s.evidence_tokens_per_item == 250
    assert s.summary_reserve_tokens == 500
    assert s.history_desired_turns == 20
    assert s.history_tokens_per_turn == 400
    assert s.layer1_ratio == 0.7
    assert s.layer2_ratio == 0.3
    assert s.assistant_preview_chars == 60
    assert s.summary_enabled is True
    assert s.summary_model == ""
    assert s.summary_timeout_seconds == 20
    assert s.summary_max_tokens == 300
    assert s.cjk_chars_per_token == 1.0
    assert s.other_chars_per_token == 3.5
    assert s.message_overhead_tokens == 3
    assert s.app_log_path == "log/app.log"


def test_max_agent_steps_accepts_both_env_names(monkeypatch):
    """新名 MAX_AGENT_STEPS 与旧名 AGENT_MAX_STEPS 都要能读；默认仍是 6。"""
    assert _base().agent_max_steps == 6

    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    assert _base().agent_max_steps == 3

    monkeypatch.delenv("MAX_AGENT_STEPS")
    monkeypatch.setenv("AGENT_MAX_STEPS", "4")
    assert _base().agent_max_steps == 4


def test_rerank_top_k_accepts_both_env_names(monkeypatch):
    """RERANK_TOP_K（ch07 新名）与 RAG_RERANK_TOP_N（ch03 旧名）互通，不能是两个独立默认值。"""
    assert _base().rag_rerank_top_n == 10

    monkeypatch.setenv("RERANK_TOP_K", "5")
    assert _base().rag_rerank_top_n == 5

    monkeypatch.delenv("RERANK_TOP_K")
    monkeypatch.setenv("RAG_RERANK_TOP_N", "7")
    assert _base().rag_rerank_top_n == 7


def test_aliased_fields_still_accept_python_keywords():
    """别名不能把「按字段名构造」这条路堵死 —— 测试与脚本大量这么用。"""
    assert _base(agent_max_steps=3).agent_max_steps == 3
    assert _base(rag_rerank_top_n=5).rag_rerank_top_n == 5


@pytest.mark.parametrize(
    "bad",
    [
        {"model_context_window": 0},
        {"max_output_tokens": 0},
        {"safety_margin_ratio": 1.5},
        {"safety_margin_ratio": -0.1},
        {"layer1_ratio": 0.0},
        {"layer1_ratio": 1.5},
        {"layer2_ratio": 0.0},
        {"cjk_chars_per_token": 0.0},
        {"other_chars_per_token": 0.0},
        {"assistant_preview_chars": 0},
        {"history_desired_turns": 0},
        {"history_tokens_per_turn": 0},
    ],
)
def test_ch07_budget_settings_reject_invalid_values(bad):
    with pytest.raises(ValidationError):
        _base(**bad)


def test_budget_ratios_must_sum_to_one():
    """0.7 + 0.3 是「层 1 拿七成、层 2 拿三成」的前提，不能配出两成或十二成。"""
    with pytest.raises(ValidationError):
        _base(layer1_ratio=0.5, layer2_ratio=0.2)
