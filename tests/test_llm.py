from app.config import Settings
from app.llm import get_chat_model


def _settings() -> Settings:
    return Settings(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="deepseek-chat",
    )


def test_get_chat_model_wires_model_name():
    model = get_chat_model(_settings())
    assert model.model_name == "deepseek-chat"


def test_get_chat_model_wires_custom_base_url():
    model = get_chat_model(_settings())
    assert "gw.example.com" in str(model.openai_api_base)


def test_get_chat_model_uses_settings_api_key():
    model = get_chat_model(_settings())
    assert model.openai_api_key.get_secret_value() == "sk-test"


def test_only_llm_module_imports_langchain_openai():
    """全局约束：只有 app/llm.py 可以 import langchain_openai。"""
    from pathlib import Path

    offenders = []
    for path in Path("app").rglob("*.py"):
        if path.as_posix() == "app/llm.py":
            continue
        if "langchain_openai" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert offenders == []
