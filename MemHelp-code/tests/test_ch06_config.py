"""ch06 Task 1：意图/指代/扩写配置默认值与模型名覆盖。"""

from app.config import Settings
from app.llm import get_chat_model


def _settings(**overrides) -> Settings:
    base = dict(
        openai_base_url="https://example.com/v1",
        openai_api_key="sk-test",
        openai_model="base-model",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_ch06_config_defaults():
    s = _settings()
    assert s.intent_model == ""
    assert s.intent_cheap_model == ""
    assert s.intent_cost_mode is False
    assert s.intent_confidence_threshold == 0.6
    assert s.intent_timeout == 5.0
    assert s.coref_enabled is True
    assert s.coref_timeout == 5.0
    assert s.coref_history_messages == 8
    assert s.query_expansion_enabled is True
    assert s.query_expansion_timeout == 5.0
    assert s.query_expansion_max_queries == 4
    assert s.refund_order_count == 3


def test_chat_model_name_can_be_overridden():
    model = get_chat_model(_settings(), model_name="intent-big")
    assert model.model_name == "intent-big"


def test_chat_model_defaults_to_openai_model():
    model = get_chat_model(_settings())
    assert model.model_name == "base-model"


def test_coref_history_messages_is_bounded():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(coref_history_messages=0)
