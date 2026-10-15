"""配置的唯一来源。

除本模块外，任何地方都不得直接读 ``os.environ`` 或 ``.env``。

字段名刻意用 ``openai_*`` 而不是 ``model_*``：``model_`` 是 pydantic 的保留
命名空间，用 ``model`` 作字段名会触发保护性告警。pydantic-settings 默认
大小写不敏感地匹配环境变量名，所以 ``OPENAI_BASE_URL`` 会落到
``openai_base_url`` 上。
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 ``.env`` / 环境变量读取的全部运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = Field(
        ...,
        min_length=1,
        description="上游 OpenAI 协议端点，例如 https://gateway.example.com/v1",
    )
    openai_api_key: str = Field(..., min_length=1, description="上游密钥。")
    openai_model: str = Field(..., min_length=1, description="模型名，换模型只改这一项。")

    context_token_budget: int = Field(
        default=2000,
        ge=1,
        description="多轮历史裁剪的 token 预算。只统计历史消息，不含 System Prompt 与本轮问题。",
    )
    request_timeout: float = Field(default=60.0, gt=0, description="单次上游请求超时（秒）。")

    @field_validator("openai_base_url", "openai_api_key", "openai_model", mode="before")
    @classmethod
    def _strip_strings(cls, value: object) -> object:
        """去掉误粘贴进来的首尾空白，避免密钥带空格这种难查的 401。"""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("openai_base_url", mode="after")
    @classmethod
    def _require_http_scheme(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("openai_base_url 必须以 http:// 或 https:// 开头")
        return value


@lru_cache
def get_settings() -> Settings:
    """进程内单例。配置缺失会在**首次调用时**立即报错，而不是拖到请求处理中。"""
    return Settings()
