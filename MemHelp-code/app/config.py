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

from pydantic import AliasChoices, Field, field_validator, model_validator
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

    # -- ch08：业务 MCP Server -------------------------------------------
    #
    # 两个 Server 都是本仓库自带的独立进程（mcp_servers/），默认端口 8101/8102。
    # 关掉 mcp_enabled 后客服系统只保留内置工具，用于断网/离线演示。
    mcp_enabled: bool = Field(default=True, description="是否接入业务 MCP Server。")
    mcp_logistics_url: str = Field(
        default="http://127.0.0.1:8101/mcp",
        min_length=1,
        description="物流 MCP Server 的 Streamable HTTP 地址。",
    )
    mcp_aftersales_url: str = Field(
        default="http://127.0.0.1:8102/mcp",
        min_length=1,
        description="售后 MCP Server 的 Streamable HTTP 地址。",
    )
    mcp_readonly_servers: str = Field(
        default="logistics,aftersales",
        description="只读 Server 白名单（逗号分隔）。不在此列的 Server 整站拒调。",
    )
    mcp_timeout_seconds: float = Field(
        default=3.0, gt=0, description="MCP 工具调用与工具发现的超时（秒）。"
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

    # -- ch04：混合检索 + 重排 -------------------------------------------------
    #
    # 重排走另一家的 /rerank（本机没有 GPU，也不下模型）。默认复用嵌入那家
    # （硅基流动）的账户，所以 api_key 默认留空、由 require_rerank_key() 回落。
    rerank_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        min_length=1,
        description="重排服务的根地址（实际请求 {base}/rerank）。",
    )
    rerank_api_key: str = Field(
        default="",
        description=(
            "重排服务的 API Key。**允许为空**：空则回落到 ``EMBEDDING_API_KEY``"
            "（同一家账户）；两者都空时重排降级为 RRF 序，不让检索失败。"
        ),
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        min_length=1,
        description="重排模型名。本章按用户指定固定为 bge-reranker-v2-m3。",
    )
    rerank_timeout: float = Field(
        default=10.0, gt=0, description="单次重排请求超时（秒）。"
    )
    rag_hybrid_top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="混合检索中**每一路**（dense / BM25）各召回多少条。",
    )
    rag_candidate_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="RRF 融合后送进重排的候选数。比 rerank_top_n 大，给重排留跨路候选。",
    )
    rag_rerank_top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        validation_alias=AliasChoices("RAG_RERANK_TOP_N", "RERANK_TOP_K"),
        description=(
            "重排后保留、并作为证据编号的条数。ch07 起也叫 RERANK_TOP_K（预算里"
            "按这个条数扣证据开销），两个环境变量名都能读。"
        ),
    )
    rag_min_rerank_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "检索层低置信阈值：最高重排分低于它就拒答并写低置信度池。"
            "默认值用评估集的正负样本标定，见 dev-notes/ch04-eval-report.md。"
        ),
    )
    rag_query_rewrite: bool = Field(
        default=True,
        description="口语问法是否调模型归一改写。关掉只走静态表（排障用）。",
    )
    rag_rewrite_timeout: float = Field(
        default=3.0, gt=0, description="单次 Query 改写请求超时（秒）。"
    )
    evidence_confidence_model: str = Field(
        default="evals/ch09_confidence_model.json",
        description=(
            "ch09 正式证据置信度模型的 JSON artifact 路径（相对项目根）。"
            "文件缺失或损坏时闸退回 ch05 的 RAG_MIN_RERANK_SCORE 阈值版。"
        ),
    )
    knowledge_self_check: bool = Field(
        default=True,
        description=(
            "ch09：置信度闸通过后、进 Agent 前是否再让模型自评一次「证据够不够」。"
            "生产默认开；单元测试关掉，避免每次图测试都多一次模型调用。"
        ),
    )
    self_check_timeout: float = Field(
        default=5.0, gt=0, description="生成前自评的单次超时（秒）。"
    )
    flywheel_enabled: bool = Field(
        default=True,
        description=(
            "ch09：入池后是否自动跑「标准化 → 查重 → 待审队列」。"
            "生产默认开；离线单测关掉，避免多调几次模型。"
        ),
    )
    flywheel_normalize_timeout: float = Field(
        default=20.0, gt=0, description="问题标准化模型调用超时（秒）。"
    )
    flywheel_dedupe_timeout: float = Field(
        default=15.0, gt=0, description="查重模型调用超时（秒）。"
    )

    # -- ch05：Workflow 编排 + 主 Agent -------------------------------------
    checkpoint_db_path: str = Field(
        default="data/checkpoints.sqlite",
        min_length=1,
        description=(
            "LangGraph checkpointer 的 SQLite 文件路径。相对路径按项目根解析；"
            "测试会注入 InMemorySaver，不落盘。"
        ),
    )
    agent_max_steps: int = Field(
        default=6,
        ge=1,
        le=12,
        validation_alias=AliasChoices("AGENT_MAX_STEPS", "MAX_AGENT_STEPS"),
        description=(
            "ReAct 最多执行几轮工具调用（不含最终收敛的那次模型调用）。"
            "ch07 起也叫 MAX_AGENT_STEPS（预算里按这个步数扣单轮峰值），"
            "两个环境变量名都能读；**默认值不变**，ch05 的既有断言依赖它。"
        ),
    )
    agent_token_budget: int = Field(
        default=6000,
        ge=500,
        description="单轮 Agent 累计 token 预算（按 usage_metadata.total_tokens 求和）。",
    )
    turn_log_path: str = Field(
        default="logs/ch05-turns.jsonl",
        description=(
            "每轮结构化日志的 JSONL 路径。**允许为空**：空串表示只打 logger，"
            "不写文件（测试默认如此）。"
        ),
    )

    # -- ch09：Langfuse 观测 -------------------------------------------------
    langfuse_enabled: bool = Field(
        default=True,
        description="总开关；false 或缺少任一连接项时，tracing 全降级为 no-op。",
    )
    langfuse_public_key: str = Field(default="", description="Langfuse 项目 public key。")
    langfuse_secret_key: str = Field(default="", description="Langfuse 项目 secret key。")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse 地址；自部署时换成自己的 URL。",
    )
    langfuse_environment: str = Field(
        default="development",
        description="trace 的环境标签（production / staging / development）。",
    )

    # -- ch09：评估流水线 ---------------------------------------------------

    eval_schedule_enabled: bool = Field(
        default=True,
        description="是否在进程内定时跑 ch04 评估集并写 eval_runs。",
    )
    eval_schedule_interval_hours: float = Field(
        default=24.0,
        gt=0,
        description="定时评估间隔（小时）；默认每天一轮。",
    )
    eval_script_path: str = Field(
        default="evals/run_ch04_evals.py",
        description="评估脚本相对项目根目录的路径。",
    )
    eval_generation_limit: int = Field(
        default=0,
        ge=0,
        description=(
            "生成段每桶取多少条；0 表示全部。生成模型偶发卡住时可设小值，"
            "检索段仍跑全量，报告里会注明。"
        ),
    )

    # -- ch06：正式版分流器 -------------------------------------------------
    intent_model: str = Field(
        default="",
        description=(
            "意图识别的主模型名。空串表示回落到 OPENAI_MODEL；"
            "换更强模型只改这一项，不改代码。"
        ),
    )
    intent_cheap_model: str = Field(
        default="",
        description=(
            "成本档的小模型名。空串表示成本档不启用；"
            "只有 INTENT_COST_MODE=true 时才先调它。"
        ),
    )
    intent_cost_mode: bool = Field(
        default=False,
        description="true 时走「小模型先判 → 低置信度升级主模型」；默认质量档。",
    )
    intent_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="成本档升级阈值：小模型置信度低于它时用主模型重判。",
    )
    intent_timeout: float = Field(
        default=5.0, gt=0, description="单次意图分类调用超时（秒）。"
    )
    coref_enabled: bool = Field(
        default=True,
        description="指代消解是否启用；关掉只走确定性透传（排障用）。",
    )
    coref_timeout: float = Field(
        default=5.0, gt=0, description="单次指代消解调用超时（秒）。"
    )
    coref_history_messages: int = Field(
        default=8,
        ge=1,
        le=20,
        description="送入指代消解的最近消息条数（只取 Human/AI 文本）。",
    )
    query_expansion_enabled: bool = Field(
        default=True,
        description="退款/售后子流程是否做 Query 扩写；关掉直接单查询。",
    )
    query_expansion_timeout: float = Field(
        default=5.0, gt=0, description="单次 Query 扩写调用超时（秒）。"
    )
    query_expansion_max_queries: int = Field(
        default=4,
        ge=2,
        le=6,
        description="扩写最多保留几条查询；去重后截断。",
    )
    refund_order_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="订单选择器候选数（按 session 确定性派生）。",
    )

    # -- ch07：会话上下文管理（三层 + 预算倒推 + 异步摘要）------------------
    model_context_window: int = Field(
        default=32000,
        ge=1000,
        description=(
            "模型上下文窗口（token）。默认 32000 要能装下 20 轮——验收要求"
            "「默认窗口下聊二十轮不触发任何降级和摘要」。"
        ),
    )
    max_output_tokens: int = Field(
        default=4000,
        ge=1,
        description="输出预留：模型单次要往外吐的 token 上限，从窗口里先扣掉。",
    )
    max_user_input_tokens: int = Field(
        default=2000,
        ge=1,
        description="单轮用户输入的瞬时峰值（含系统提示之外的注入）。",
    )
    tool_result_max_tokens: int = Field(
        default=1200,
        ge=1,
        description="单次工具结果的峰值 token 数；乘以步数得到单轮 ReAct 峰值。",
    )
    safety_margin_ratio: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description=(
            "安全余量占窗口的比例，用来吸收计数器误差与 chat template 差异。"
            "取 0.10 表示窗口的十分之一定为不可动用。"
        ),
    )
    system_prompt_reserve_tokens: int = Field(
        default=1200,
        ge=1,
        description=(
            "系统提示（人设 + 红线 + 工具定义）的固定估算。实测校准口径下"
            "720 字符约 583 token，带 5 条证据约 711——1200 是含余量的保守值。"
        ),
    )
    evidence_tokens_per_item: int = Field(
        default=250,
        ge=1,
        description="每条检索证据的估算 token 数（合进「检索证据」这项固定开销）。",
    )
    summary_reserve_tokens: int = Field(
        default=500,
        ge=1,
        description="注入的会话梗概（合成一条）的估算 token 数。",
    )
    history_desired_turns: int = Field(
        default=20,
        ge=1,
        description="想留住的轮数。与「每轮稳态占用」相乘得到历史预算的上界。",
    )
    history_tokens_per_turn: int = Field(
        default=400,
        ge=1,
        description=(
            "每轮稳态占用的 tokens。既用来算「想留住多少」，也是启动自检里"
            "「连一轮都装不下」的门槛。"
        ),
    )
    layer1_ratio: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="历史预算里层 1（原文滑窗）占的比例。",
    )
    layer2_ratio: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="历史预算里层 2（规则截短）占的比例。与 layer1_ratio 之和必须是 1。",
    )
    assistant_preview_chars: int = Field(
        default=60,
        ge=1,
        description="层 2 里客服答复只留开头多少字；用户原话一个字都不动。",
    )
    summary_enabled: bool = Field(
        default=True,
        description=(
            "摘要总开关。关掉后层 2 超预算也不起后台任务——只降级不摘要，"
            "方便做对照实验。"
        ),
    )
    summary_model: str = Field(
        default="",
        description="摘要用的模型名；空串表示与主模型相同。",
    )
    summary_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        description="单次摘要任务的上游超时（秒）。",
    )
    summary_max_tokens: int = Field(
        default=300,
        ge=1,
        description="摘要输出上限；正文超出按它换算的字数截断。",
    )
    cjk_chars_per_token: float = Field(
        default=1.0,
        gt=0,
        description=(
            "几个 CJK 字符合成一个 token。**预算与计数器必须读同一个值**——"
            "只改一个的净效果是反的。"
        ),
    )
    other_chars_per_token: float = Field(
        default=3.5,
        gt=0,
        description="几个非 CJK 字符合成一个 token。",
    )
    message_overhead_tokens: int = Field(
        default=3,
        ge=0,
        description="每条消息的固定开销（角色标签、分隔符），与正文长度无关。",
    )
    app_log_path: str = Field(
        default="log/app.log",
        description=(
            "上下文可观测日志（model_ctx / history_ctx / summary 五态）的落盘路径。"
            "相对路径按项目根解析；空串表示只打 logger 不写文件。"
        ),
    )

    # -- ch10：主题分类旁路 ------------------------------------------------
    topic_service_url: str = Field(
        default="http://127.0.0.1:8100",
        description=(
            "ch10：旁路推理 sidecar 的地址。**主应用不装 torch**，"
            "分类请求一律发到这里；连不上就返 503 并提示启动命令。"
        ),
    )
    topic_service_timeout: float = Field(
        default=30.0,
        gt=0,
        description="ch10：单次 sidecar 调用的超时（秒）；一批 100 条约几秒，留足余量。",
    )
    topic_batch_limit: int = Field(
        default=100,
        ge=1,
        description="ch10：一次批跑最多归多少条。批太大显存与请求时长都不好控。",
    )
    topic_batch_min_pending: int = Field(
        default=50,
        ge=0,
        description=(
            "ch10：水位线。积压少于它时前端提示「先攒一攒」，**只提示不拦截** ——"
            "拦下来会让调试时没法跑小批。"
        ),
    )
    topic_model_dir: str = Field(
        default="models/topics",
        description=(
            "ch10：模型产物目录（相对项目根）。sidecar 取其中最近一次 run；"
            "也可用 TOPIC_MODEL_DIR 环境变量直接指到某个 run。"
        ),
    )

    @model_validator(mode="after")
    def _layer_ratios_must_sum_to_one(self) -> "Settings":
        """层 1 + 层 2 必须刚好把历史预算分完。

        允许 1e-6 的浮点误差：``0.7 + 0.3`` 在二进制浮点里未必精确等于 1，
        这不影响 ``round()`` 的取整结果。
        """
        total = self.layer1_ratio + self.layer2_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"LAYER1_RATIO + LAYER2_RATIO 必须等于 1，当前 {self.layer1_ratio} + "
                f"{self.layer2_ratio} = {total}"
            )
        return self

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

    def require_rerank_key(self) -> str:
        """返回重排服务密钥；自己的没配就用嵌入那家的（同一账户）。

        为什么允许回落：硅基流动一个账户同时供 ``/embeddings`` 与 ``/rerank``，
        让用户为同一把 Key 填两遍只会制造「填了一处、另一处 401」的排查成本。

        为什么两者都空时不抛在构造期：与 ``require_embedding_key`` 同理 ——
        离线测试与 ``--status`` 不该因为没配 Key 就崩。真到要发请求时由
        调用方决定是降级（重排可降级）还是报错。
        """
        key = self.rerank_api_key or self.embedding_api_key
        if not key:
            raise ValueError(
                "缺少 RERANK_API_KEY（硅基流动账户 API Key）。"
                "可以留空复用 EMBEDDING_API_KEY，或单独设置；"
                f"请检查 {ENV_FILE}，或参考 .env.example。"
            )
        return key

    def rerank_url(self) -> str:
        """``{RERANK_BASE_URL}/rerank``，去掉 base 末尾多余的斜杠。"""
        return f"{self.rerank_base_url.rstrip('/')}/rerank"

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
        "intent_model",
        "intent_cheap_model",
        "mysql_host",
        "mysql_user",
        "mysql_password",
        "mysql_database",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_base_url",
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
