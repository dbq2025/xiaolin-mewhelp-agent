"""配置的唯一来源。

除本模块外，任何地方都不得直接读 ``os.environ`` 或 ``.env``。

字段名刻意用 ``openai_*`` 而不是 ``model_*``：``model_`` 是 pydantic 的保留
命名空间，用 ``model`` 作字段名会触发保护性告警。pydantic-settings 默认
大小写不敏感地匹配环境变量名，所以 ``OPENAI_BASE_URL`` 会落到
``openai_base_url`` 上。
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 项目根目录（``app/`` 的上一级）。
#:
#: ``.env`` 的路径**必须绝对**：pydantic-settings 的相对路径是按**当前工作目录**
#: 解析的，而本仓库的布局是「仓库根 + 代码在 MemHelp-code/ 子目录」（见
#: ``agent.md``）。如果这里写 ".env"，从仓库根运行 uvicorn 就会找不到配置 ——
#: 实测过：报 3 个 Field required。用 ``__file__`` 定位则与 CWD 无关。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """从 ``.env`` / 环境变量读取的全部运行配置。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
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

    # -- ch02：MySQL ------------------------------------------------------
    #
    # 默认值指向 docker-compose.yml 里那套开发库，让「本地起得来」成为默认行为；
    # 生产部署必须显式覆盖（默认口令只配用于本地演示）。
    mysql_host: str = Field(default="127.0.0.1", min_length=1, description="MySQL 主机。")
    mysql_port: int = Field(default=3306, ge=1, le=65535, description="MySQL 端口。")
    mysql_user: str = Field(default="mewhelp", min_length=1, description="MySQL 用户名。")
    mysql_password: str = Field(default="mewhelp", description="MySQL 密码。")
    mysql_database: str = Field(default="mewhelp", min_length=1, description="MySQL 库名。")

    # -- ch02：工具基础设施 ------------------------------------------------
    tool_timeout_seconds: float = Field(
        default=3.0, gt=0, description="单次工具执行超时（秒）。"
    )
    tool_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="仅对瞬时错误的重试次数（不含首次尝试）。业务性失败与参数错误不重试。",
    )

    # -- ch03：RAG（向量语义检索）----------------------------------------
    #
    # 嵌入走云端（OpenAI 兼容 /embeddings）。上游 DeepSeek 没有这个端点，
    # 所以这里需要**独立**的 base_url / api_key / model 三项。
    milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        min_length=1,
        description="Milvus 连接地址（gRPC）。",
    )
    milvus_collection: str = Field(
        default="knowledge", min_length=1, description="Milvus 集合名。"
    )
    embedding_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        min_length=1,
        description="嵌入服务的 OpenAI 兼容根地址（实际请求 {base}/embeddings）。",
    )
    embedding_api_key: str = Field(
        default="",
        description=(
            "嵌入服务的 API Key（硅基流动账户 Key）。"
            "**允许为空**：离线测试与 --status 不该因为没密钥就崩；"
            "真正发请求前由 require_embedding_key() 报错。"
        ),
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        min_length=1,
        description="嵌入模型名。服务商会不定期调整可用模型，所以不写死在代码里。",
    )
    embedding_dim: int = Field(
        default=1024, ge=1, description="向量维度。BGE-M3 dense 输出固定 1024。"
    )
    embedding_timeout: float = Field(
        default=30.0, gt=0, description="单次嵌入请求超时（秒）。"
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=256, description="单次请求最多带多少条文本。"
    )
    embedding_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="仅对 429 与 5xx 的重试次数（不含首次）。4xx 不重试。",
    )
    rag_top_k: int = Field(
        default=5, ge=1, le=50, description="检索返回的最大条数（也是 query_faq 的取数上限）。"
    )
    rag_warmup: bool = Field(
        default=True,
        description="启动时预热嵌入服务与 Milvus 集合。见 create_app 的 rag_warmup 参数。",
    )
    chunk_max_chars: int = Field(
        default=600, ge=1, description="单个切分块的最大字符数（表格按行切时可不遵守）。"
    )
    chunk_overlap: int = Field(
        default=80, ge=0, description="同一节内相邻块的重叠字符上限；按句回退，绝不切半句。"
    )
    qa_batch_size: int = Field(
        default=8, ge=1, le=64, description="对话挖知识时每批喂给 LLM 的会话数。"
    )
    qa_dedupe_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="问答对语义去重的余弦阈值：≥ 该值判为同义问法。",
    )

    def require_embedding_key(self) -> str:
        """返回嵌入服务密钥；没配就抛可读错误。

        为什么不在 ``Settings`` 上设成必填：``EMBEDDING_API_KEY`` 只有「真正要
        发嵌入请求」时才需要。设成必填会让离线单测、``--status``、健康检查
        一起因为一个环境差异而崩 —— 而那不是配置错误，是这台机器还没填密钥。

        为什么不用 ``OPENAI_API_KEY``：那是聊天上游（DeepSeek）的密钥，
        两个服务商的额度与权限是分开的，混用会把「聊天能用但检索 401」
        变成一条极难查的线索。
        """
        if not self.embedding_api_key:
            raise ValueError(
                "缺少 EMBEDDING_API_KEY（硅基流动账户 API Key）。"
                "请在 MemHelp-code/.env 里设置，例如 EMBEDDING_API_KEY=sk-xxx；"
                "或参考 .env.example。"
            )
        return self.embedding_api_key

    def embedding_url(self) -> str:
        """``{EMBEDDING_BASE_URL}/embeddings``，去掉 base 末尾多余的斜杠。"""
        return f"{self.embedding_base_url.rstrip('/')}/embeddings"

    def database_url(self) -> str:
        """拼出 SQLAlchemy 用的 MySQL 连接串。

        这是全项目**唯一**生成连接串的地方（配置单一来源）。

        ``quote_plus`` 转义用户名与密码：密码里出现 ``@`` ``:`` ``/`` 时不转义会让
        URL 被解析到错误的 host —— 表现为「连不上数据库」，根因却在一个密码字符上，
        极难查。``charset=utf8mb4`` 必须带上，否则驱动按 latin1 收发，中文变问号。

        返回的连接串内含明文密码，**不得**写进日志或响应体。
        """
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @field_validator(
        "openai_base_url",
        "openai_api_key",
        "openai_model",
        "mysql_host",
        "mysql_user",
        "mysql_password",
        "mysql_database",
        mode="before",
    )
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
