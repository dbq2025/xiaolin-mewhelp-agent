"""配置校验测试。

刻意不使用 pytest 的 ``tmp_path``：沙箱下新建目录会被收回权限（见 conftest.py）。
需要「从 .env 文件读」的场景改用**自建**的可写目录，写完即删 —— 目录必须自建，
不能假定已存在（例如 ``.uv-cache/`` 被 gitignore，干净克隆上根本没有）。

同理，这些测试断言的是「``_env_file`` 里的值生效」。pydantic-settings 的优先级是
**OS 环境变量 > .env 文件**，所以必须先把宿主环境里的 ``OPENAI_*`` 清掉，否则在
CI 或任何导出过这些变量的机器上会得到与预期不符的结果。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

#: 放临时 .env 的目录（自建，不依赖工作区里已有目录）
ENV_FIXTURE_DIR = Path(".uv-cache")


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    """避免宿主的 OPENAI_* 环境变量盖过测试自己写的 .env 文件。"""
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "CONTEXT_TOKEN_BUDGET",
        "REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_env(tmp_name: str, body: str) -> Path:
    """把一段 .env 内容写进自建的可写目录，返回 path 对象。"""
    ENV_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = ENV_FIXTURE_DIR / tmp_name
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
