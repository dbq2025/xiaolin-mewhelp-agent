"""配置校验测试。

刻意不使用 pytest 的 ``tmp_path``：沙箱下新建目录会被收回权限（见 conftest.py）。
需要「从 .env 文件读」的场景改用已存在且可写的目录，写完即删。
"""

import pytest
from pydantic import ValidationError

from app.config import Settings

ENV_FIXTURE_DIR = ".uv-cache"


def _write_env(tmp_name: str, body: str):
    """把一段 .env 内容写进既有可写目录，返回 path 对象。"""
    from pathlib import Path

    path = Path(ENV_FIXTURE_DIR) / tmp_name
    path.write_text(body, encoding="utf-8")
    return path


def test_settings_read_from_env_file():
    env = _write_env(
        "t-read.env",
        "OPENAI_BASE_URL=https://gw.example.com/v1\n"
        "OPENAI_API_KEY=sk-test\n"
        "OPENAI_MODEL=deepseek-chat\n"
        "CONTEXT_TOKEN_BUDGET=1234\n",
    )
    try:
        s = Settings(_env_file=env)
    finally:
        env.unlink(missing_ok=True)

    assert s.openai_base_url == "https://gw.example.com/v1"
    assert s.openai_model == "deepseek-chat"
    assert s.context_token_budget == 1234
    assert s.request_timeout == 60.0


def test_settings_strip_surrounding_whitespace():
    s = Settings(
        openai_base_url="  https://gw.example.com/v1  ",
        openai_api_key="  sk-test  ",
        openai_model="  deepseek-chat  ",
    )
    assert s.openai_base_url == "https://gw.example.com/v1"
    assert s.openai_api_key == "sk-test"
    assert s.openai_model == "deepseek-chat"


@pytest.mark.parametrize(
    "overrides",
    [
        {"openai_api_key": "   "},
        {"openai_model": ""},
        {"openai_base_url": "gw.example.com"},
        {"context_token_budget": 0},
        {"context_token_budget": -1},
    ],
    ids=["blank-key", "blank-model", "scheme-missing", "budget-zero", "budget-negative"],
)
def test_settings_reject_invalid_values(overrides):
    base = {
        "openai_base_url": "https://gw.example.com/v1",
        "openai_api_key": "sk-test",
        "openai_model": "m",
    }
    base.update(overrides)
    with pytest.raises(ValidationError):
        Settings(**base)


def test_settings_require_all_three_openai_fields(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
