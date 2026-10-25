"""四张表的 ORM 映射 —— ``sql/ch02-ddl.sql`` 的 Python 投影。

**本模块只定义表，不写任何查询**。查询一律在 ``app/db/repositories.py``：
那条边界让「SQL 写在哪」有唯一答案，也让本文件能当 DDL 的对照表来读。

## 与 DDL 对齐的硬要求

每一列的名字、类型长度、可空性、ENUM 取值都必须与 ``sql/ch02-ddl.sql`` 一致。
差一个字母（``session_id`` → ``sessionId``）在本层毫无症状，到真库上表现为
``Unknown column``，且只在跑到那条路径时才炸。所以：

- 列名集合由 ``tests/test_db_models.py`` 逐个钉住；
- 跨数据库的逐列对齐由 ``tests/test_db_mysql.py`` 在真实 MySQL 上验。

## 为什么 ENUM 用 ``SAEnum(*值, name=...)``

DDL 用的是 MySQL 原生 ``ENUM``。ORM 若只在类型标注里写 ``Literal`` 或裸 ``str``，
Python 侧就毫无约束 —— 非法值会一路走到 ``INSERT`` 才被数据库拒绝，错误信息是
驱动的 ``Data truncated for column``，把「调用方拼错字符串」伪装成「数据库故障」。
用 ``SAEnum`` 后，非法取值在**会话 flush 时**就报 ``LookupError``，错误信息带列名。

``name=`` 是给约束/类型起的标识（MySQL 下不参与 DDL，因为原生 ENUM 不需要约束名），
SQLite 等不支持的方言下 ``native_enum=False`` 才会用它建 CHECK。
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

#: 自增主键的类型。
#:
#: MySQL 上用 ``BIGINT``（与 DDL 一致）；SQLite 上**必须**退化成 ``INTEGER`` ——
#: SQLite 只把 ``INTEGER PRIMARY KEY`` 当作 rowid 别名来自增，``BIGINT PRIMARY KEY``
#: 不会自增，插入时报 ``NOT NULL constraint failed: conversations.id``。
#: 用 ``with_variant`` 表达这个方言差异，而不是让测试去迁就生产类型
#: （那样测的就不是真东西了）。
_AUTOINC_PK = BigInteger().with_variant(Integer, "sqlite")

#: 会话处理状态。取值与 DDL 的 ENUM 字面量逐字相同。
CONVERSATION_STATUSES: tuple[str, ...] = ("进行中", "已转人工", "已结束")

#: 消息角色。刻意对齐 Chat Completions 协议的 role 命名。
MESSAGE_ROLES: tuple[str, ...] = ("user", "assistant", "tool")

#: 工单类型。
TICKET_TYPES: tuple[str, ...] = ("售后", "投诉", "咨询")

#: 工单处理状态。
TICKET_STATUSES: tuple[str, ...] = ("待处理", "已处理")

#: ch03：chunk 的向量化状态。双写幂等与「中断后续跑」都靠它。
#:
#: - ``pending``：原文已写入 MySQL，但 Milvus 里还没有（或回填没做完）对应的向量；
#: - ``done``：向量已写入 Milvus 且 ``vector_id`` 已回填。
VECTORIZE_STATUSES: tuple[str, ...] = ("pending", "done")

#: ch03：chunk 的来源类型。
#:
#: 用户 DDL 的注释写的是「faq / policy / manual 等」，``mined`` 是本章新增的第 4 个
#: 取值 —— 挖矿产物要能和文档类 FAQ 在元数据上区分开，否则排查召回时
#: 分不清「这条是文档里的还是对话里挖的」。
CONTENT_TYPES: tuple[str, ...] = ("faq", "policy", "manual", "mined")

#: ch03：对话抽取暂存表的状态。#:
#: **真库是 5 值超集**（实测 ``SHOW CREATE TABLE``：后续章节加了
#: ``approved`` / ``rejected``）。ORM 必须带上这两个值：只写 3 值的话，读到别人
#: 写进去的 ``approved`` 行会在加载时抛 ``LookupError``，表现为「暂存表偶尔读不出来」。
#: 本章只写前三个值。
STAGING_STATUSES: tuple[str, ...] = (
    "extracted",
    "kept",
    "discarded",
    "approved",
    "rejected",
)


class Conversation(Base):
    """会话壳：一通对话的统一身份，``messages`` / ``tickets`` 都引用它。"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    #: 对外会话标识。聊天页的 sessionStorage 与 ``/api/chat/sessions/{id}/messages``
    #: 用的都是它 —— 所以它必须是唯一键，否则同一字符串会对应两行会话。
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(*CONVERSATION_STATUSES, name="conversation_status"),
        nullable=False,
        server_default=CONVERSATION_STATUSES[0],
    )
    #: ch07：梗概覆盖到哪条 ``messages.id``（含）。``NULL`` = 还没有任何梗概。
    #:
    #: 两个锚点是**三层结构的唯一边界**：降级/摘要只改这两个整数，不搬一行数据。
    #: 这样「历史分层」是一次纯读操作，不受搬数据的中断影响 —— 改到一半崩了，
    #: 大不了下次重算，绝不会出现「消息在两张表里各有一半」。
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: ch07：层 1（原文区）从哪条 ``messages.id`` 起（含）。``NULL`` = 还没降级过。
    #:
    #: 恒有 ``layer1_from_msg_id >= summary_upto_msg_id + 1``（层 1 不落进摘要区间）。
    #: 这条不变式由 ``ConversationRepository.set_anchors`` 在应用侧强制。
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: 由**数据库**填时间，不用应用进程的时钟：容器里两边时钟可能不同步，
    #: 混用会让「按时间排序」出现负间隔。
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_user_id", "user_id"),)


class Message(Base):
    """消息流水：一通会话底下挂 N 条。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(SAEnum(*MESSAGE_ROLES, name="message_role"), nullable=False)
    #: assistant 纯工具调用时正文为空，所以可空。
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: assistant 消息带的工具调用申请单，形如
    #: ``[{"name": ..., "args": {...}, "id": "call_x"}]``。
    #: 用 JSON 类型而非 Text：Python 侧直接存取 list/dict，不必每次手工编解码
    #: （那种地方迟早有人漏一次）。
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    #: tool 消息对应的申请单 id，回灌时对号入座。
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: **给界面看的**工具轨迹（含耗时、成败），由我们自己定格式。
    #:
    #: 为什么不并进 ``tool_calls``：那一列会被原样塞回
    #: ``AIMessage(tool_calls=...)`` 当作**模型历史**，而 LangChain 对 tool_call
    #: 的字段是严格的 —— 实测多一个 ``duration_ms`` 会直接抛
    #: ``TypeError: tool_call() got an unexpected keyword argument 'duration_ms'``。
    #: 所以模型历史保持干净的 ``{name, args, id}``，界面信息另开一列。
    tool_traces: Mapped[list | None] = mapped_column(JSON, nullable=True)
    #: ch04：本轮引用证据的**快照**，形如
    #: ``[{"n":1,"chunk_id":3,"section_path":"…","category":"…","quote":"…"}]``。
    #:
    #: 为什么存快照而不是只存 chunk_id：chunk 日后会被重切/删除，审计时要能看到
    #: **当时**喂给模型的到底是什么文本；只留 id 的话，一次重建就把证据变成悬空指针。
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    #: ch05：本轮建议给用户的动作选项（转人工 / 建工单）。
    #:
    #: 形状是 **列表 of 对象**，例如
    #: ``[{"action": "transfer_human", "ticket_type": "投诉"}]``。
    #: 存动作而不是「按钮已点状态」：点击状态是浏览器侧的一次性 UX，不该冒充
    #: 业务事实；工单是否真的创建，以 ``tickets`` 表为准。
    actions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_conversation_id", "conversation_id"),)


class ConversationSummary(Base):
    """ch07：会话梗概，**一段一行**。

    ## 为什么单开一张表

    三层结构的第三层是「早期历史压成一段段梗概」。这些梗概有三个性质，决定了
    它们既不能塞进 ``conversations`` 的一列、也不能塞进 ``messages``：

    1. **只追加**。每次后台摘要任务把一批消息压成**新的一段**，旧段一个字不改。
       塞进 ``conversations.summary`` 一列就只剩最后一次结果，旧段被覆盖 —— 而
       旧段恰恰是「这个事实已经被有损压过一次」的证据，丢了它就无法解释
       「为什么模型只记得订单号、不记得原话」。
    2. **不参与合并**。新段只总结**待压缩的那批消息**，旧段只作为背景给模型看。
       若每轮把「旧梗概 + 新消息」再压一次，同一个事实会被反复有损压缩，最后
       退化成一句「用户有售后需求」—— 订单号、手机号全丢。
    3. **不是对话消息**。``messages`` 表的每一行都会被
       ``MessageRepository.history_as_messages`` 回灌成 ``BaseMessage``；把梗概
       写进去，模型会把梗概当成「自己刚说过的话」，且 ``role`` 枚举里也没有
       一个诚实的取值可用。

    ## 字段职责

    ``segment_no`` 从 1 起、按会话内递增，配 ``UNIQUE(conversation_id, segment_no)``
    是**多进程并发写**的兜底：应用侧的 in-flight 集合只在单进程内有效，两个进程
    同时压同一段时，数据库唯一键才是真正收口的地方（冲突方放弃并留痕，不重试改号）。

    ``source_tokens`` 记录压之前那批消息的 token 估算，``elapsed_ms`` 记录本次
    耗时 —— 这两个数是「摘要没阻塞用户回复」在日志之外的第二份证据。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False
    )
    #: 段号，会话内从 1 递增。
    segment_no: Mapped[int] = mapped_column(Integer, nullable=False)
    #: 本段覆盖的消息区间 ``[from_msg_id, upto_msg_id]``（闭区间，对应 ``messages.id``）。
    from_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: 梗概正文。目标长度几十到一两百字，只写事实与诉求。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: 本段压了多少条消息。
    msg_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    #: 压之前的 token 估算（用与预算同一套校准口径）。
    source_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    #: 生成本段的模型名。``NULL`` = 用了默认模型或未记录。
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: 后台任务的耗时（毫秒）。用来证明摘要确实没有挡在用户回复的路径上。
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("conversation_id", "segment_no", name="uq_summary_segment"),
        # 索引名必须**全库唯一**：MySQL 的索引名是按表隔离的，但 SQLite 是
        # 按 database 隔离 —— 与 messages 表重名时 ``create_all`` 直接报
        # ``index idx_conversation_id already exists``。所以这里带 ``summary_`` 前缀。
        Index("idx_summary_conversation_id", "conversation_id", "segment_no"),
    )


class Faq(Base):
    """FAQ 问答对：``query_faq`` 的数据源。

    ch03 起检索改走向量库，这张表退居原始录入 —— 但本章的关键词
    ``LIKE`` 查询正是在它上面留下「漏召回」的可复现基线。
    """

    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_category", "category"),)


class Ticket(Base):
    """人工工单：``create_ticket`` 落地。

    ``ticket_no`` 是业务主键（不是自增 id）—— 它是给人念的编号，如 ``T20260701008``。
    """

    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(
        SAEnum(*TICKET_TYPES, name="ticket_type"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        SAEnum(*TICKET_STATUSES, name="ticket_status"),
        nullable=False,
        server_default=TICKET_STATUSES[0],
    )
    #: ch05：用户点「建工单」时带的一次性键。唯一索引保证重试不会建出第二张。
    #: 可空：ch02 的模型直接调 create_ticket 时不需要它。
    idempotency_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_tickets_conversation_id", "conversation_id"),
        Index("uq_tickets_idempotency_key", "idempotency_key", unique=True),
    )


class KnowledgeChunk(Base):
    """知识库 chunk —— **原文权威源**（``sql/ch03-ddl.sql`` 的 Python 投影）。

    ## 三格进向量，其余只存

    `category` + `questions` + `answer` 拼成一段文本做向量化（拼接函数在
    ``app/knowledge/chunker.py::render_embed_text``，建库与检索必须共用它）。
    `section_path` / `content_type` / `is_key_clause` / `prev_chunk_id` /
    `next_chunk_id` 是**元数据，只存不进向量** —— 所以 Milvus 集合里只有
    ``id + vector``（见 spec §3.1），检索拿 id 回这张表取原文。

    ## 双写状态机

    ``vectorize_status`` 是「原文已写、向量未写」与「两库都写好」的唯一判据。
    续跑只拣 ``pending``（或 ``vector_id IS NULL``）的块，靠确定性主键
    ``str(id)`` 覆盖写 Milvus，所以重跑不会产生重复向量。
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    #: 问法或本节标题；多个问法用换行分隔（挖矿合并同义问法时会用到）。
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    #: 含当前标题的完整路径，供溯源与调试；不进向量。
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: 只认显式标记（标题「（关键条款）」或正文 `<!-- key-clause -->`），不靠关键词猜。
    is_key_clause: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("0")
    )
    #: 同文档内相邻块的指针；外键删除时置空，不级联删邻居。
    #: 约束在 ``__table_args__`` 的 ForeignKeyConstraint 里声明（要具名，与 DDL 对齐），
    #: 这里**不能**再写单独的 ForeignKey —— 会生成重复约束。
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: Milvus 集合 ``knowledge`` 里的主键。本章用确定性主键 ``str(id)``。
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        SAEnum(*VECTORIZE_STATUSES, name="vectorize_status"),
        nullable=False,
        server_default=VECTORIZE_STATUSES[0],
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # 名字刻意叫 idx_chunks_category 而不是 DDL 里的 idx_category：
        # **SQLite 的索引名是库级全局的**，`faq` 表已经有 idx_category，同名会
        # 在建表时直接报 "index idx_category already exists"。MySQL 的索引名只在
        # 表内唯一，两种名字都合法 —— ch02 的 `tickets` 用 idx_tickets_conversation_id
        # 避开 `messages` 的同名索引，就是同一条约定。
        Index("idx_chunks_category", "category"),
        Index("idx_vectorize_status", "vectorize_status"),
        # 与 DDL 一致：SQLAlchemy 在列上单独写 ForeignKey 时不会再生成具名约束，
        # 而真库的约束名是 fk_chunks_prev / fk_chunks_next。用 ForeignKeyConstraint
        # 把名字固定下来，`test_db_mysql.py` 才能逐名对齐。
        ForeignKeyConstraint(
            ["prev_chunk_id"],
            ["knowledge_chunks.id"],
            name="fk_chunks_prev",
            ondelete="SET NULL",
        ),
        ForeignKeyConstraint(
            ["next_chunk_id"],
            ["knowledge_chunks.id"],
            name="fk_chunks_next",
            ondelete="SET NULL",
        ),
    )


class QaStaging(Base):
    """历史对话抽出的问答对 —— 离线中转暂存表。

    ## 为什么要先暂存

    抽取是**分批**喂 LLM 的（一批几个会话，防串味）；去重要在**全部批次跑完之后**
    做（逐批去重会让「第 2 批里和第 1 批同义的问法」重复入库）。暂存表把
    「抽出来」与「判重入库」这两步解耦，中间可以人工看、可以调阈值重去重。

    ## 溯源

    ``source_ref`` 存来源会话的 ``session_id``，``batch_no`` 存抽取批次号。
    两者都不入最终知识库，只用于「这条是哪批、从哪个对话挖出来的」。
    """

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(*STAGING_STATUSES, name="qa_staging_status"),
        nullable=False,
        server_default=STAGING_STATUSES[0],
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_batch_no", "batch_no"),
        Index("idx_status", "status"),
    )


# ---------------------------------------------------------------------------
# ch04：低置信度问题池 + 编造个案台账
# ---------------------------------------------------------------------------

#: 低置信度问题的入池入口。取值与 ``sql/ch04-ddl.sql`` 的 ENUM 逐字相同。
#:
#: - ``retrieval_low_conf``：检索 0 命中，或最高分低于 ``RAG_MIN_RERANK_SCORE``；
#: - ``self_check``：生成前模型自评「证据不够答」；
#: - ``user_feedback``：用户反馈未解决（留给后续章节，本章不写这个值）。
LOW_CONFIDENCE_SOURCES: tuple[str, ...] = (
    "retrieval_low_conf",
    "self_check",
    "user_feedback",
)

#: 编造个案台账的处置状态（中文取值，与 DDL 一致）。
FAITH_CASE_STATUSES: tuple[str, ...] = ("未解决", "已解决", "无需解决")


class LowConfidenceQuestion(Base):
    """低置信度问题池 —— 客服「答不上来」的原话沉淀。

    这张表是数据飞轮的入口：一条原话 + 它为什么答不上来，比一句
    「今天拒答了 12 次」有用得多。``reason`` 留的是复盘材料，不是给用户看的。

    ``conversation_id`` 可空：池子允许收进来路不明的问题（人工补录、离线评测），
    有来源会话时才填 —— 但填了就必须指向真实会话（外键）。
    """

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: 用户原话（带情绪口语），**不做归一化** —— 池子的价值就在于保留原始说法。
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        SAEnum(*LOW_CONFIDENCE_SOURCES, name="low_confidence_source"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: ch09：落池当轮的召回片段快照（Top-N 的原文 + 得分），人工审核的依据。
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    #: ch09：归并到的待审行；查重命中时回填，驳回/删除后置空。
    matched_review_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_lcq_conversation",
        ),
        ForeignKeyConstraint(
            ["matched_review_id"],
            ["review_queue.id"],
            name="fk_lcq_review",
            ondelete="SET NULL",
        ),
        Index("idx_lcq_source", "source"),
        Index("idx_lcq_created_at", "created_at"),
        # SQLite 的索引名是库级唯一的，故与 MySQL DDL 的 ``idx_matched_review_id`` 区分。
        Index("idx_lcq_matched_review", "matched_review_id"),
    )


#: 待审队列的审核状态。与 ``sql/ch09-ddl.sql`` 的 ENUM 逐字相同。
REVIEW_STATUSES: tuple[str, ...] = ("待审", "通过", "驳回")

#: 评估记录的触发方式。
EVAL_TRIGGERED_BY: tuple[str, ...] = ("定时", "手动")


class ReviewQueue(Base):
    """标准化问题 + 建议答案的待审队列 —— 一行一个**去重后**的缺口。

    同一意图的问题反复入池只累加 ``occurrence_count``，不新增行；被归并的
    原始提问留在 ``low_confidence_questions``，用 ``matched_review_id`` 指回来。
    """

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    #: 模型改写后的标准 FAQ 式问题（去重与审核都以它为准）。
    normalized_question: Mapped[str] = mapped_column(String(512), nullable=False)
    #: 模型给的示例答案，审核人可改；通过时写入 ``approved_answer``。
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    review_status: Mapped[str] = mapped_column(
        SAEnum(*REVIEW_STATUSES, name="review_status"),
        nullable=False,
        server_default=REVIEW_STATUSES[0],
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_review_queue_status", "review_status"),)


class EvalRun(Base):
    """一轮自动化评估的分数快照 —— 一行一轮，按时间连成趋势。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(
        SAEnum(*EVAL_TRIGGERED_BY, name="eval_triggered_by"),
        nullable=False,
        server_default=EVAL_TRIGGERED_BY[0],
    )
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    #: ``{"retrieval": {...}, "generation": {...}}``，键与 ch04 报告一致。
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_eval_runs_created_at", "created_at"),)


class FaithCase(Base):
    """编造个案台账 —— 忠实度裁判判出的每条编造，一题一行长期管理。

    ## 为什么不是「报告里列一下」

    报告是产物，重跑一次就被覆盖。个案的价值在跨轮追溯：同一道题反复被判编造，
    说明库里那一格一直没补对。所以同一题不新增行，只累加 ``seen_count``、
    刷新最近一次的答案/理由/角标快照。

    ## 复发要退回未解决

    已标「已解决」的题又被判出来，是**复发**而不是新问题：状态退回「未解决」，
    让它重新出现在待处理列表里。``resolved_at`` 仍保留 —— 那张时间戳正是
    「上次解决过、这次又犯了」的证据。
    """

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    #: 评估集题号（如 ``A43``），一题一行。
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, server_default="hybrid_rerank")
    #: 被判编造的那版生成答案原文。
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    #: 裁判给的理由（编在哪一句）。
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: 这一轮喂给模型的 Top-K 证据**全集**快照（答案里的 ``[n]`` 就是它的序号）。
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        SAEnum(*FAITH_CASE_STATUSES, name="faith_case_status"),
        nullable=False,
        server_default=FAITH_CASE_STATUSES[0],
    )
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    #: 处置说明：标已解决要写清怎么解决，标无需解决要写清为什么不用改。
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("eval_id", name="uk_eval_id"),
        Index("idx_faith_status", "status"),
        Index("idx_faith_last_seen_at", "last_seen_at"),
    )

# ---------------------------------------------------------------------------
# ch08：工具审计
# ---------------------------------------------------------------------------

#: 审计状态。与 ``sql/ch08-ddl.sql`` 的 ENUM 逐字相同（顺序也一致）。
TOOL_AUDIT_STATUSES: tuple[str, ...] = ("成功", "失败", "超时", "校验拦下", "权限拒绝")


class ToolAuditLog(Base):
    """每次工具调用一条，**包括**被权限拒绝、被校验拦下的那些。

    刻意不挂外键：工具调用可能发生在会话行落库之前（或会话被清理之后），
    审计不能因此写不进去 —— 它的职责是「留痕」，不是「参与业务一致性」。
    """

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    #: 所属会话；调用发生时可能还没有会话行，所以可空。
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: 模型给的调用 id，用来把审计行与那一轮的工具消息对上。
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_source: Mapped[str] = mapped_column(
        SAEnum("builtin", "mcp", name="tool_audit_source"), nullable=False
    )
    #: 来源是 MCP 时记 Server 名（logistics / aftersales …）。
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        SAEnum(*TOOL_AUDIT_STATUSES, name="tool_audit_status"), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_audit_conversation_id", "conversation_id"),
        Index("idx_audit_tool_name", "tool_name"),
        Index("idx_audit_status", "status"),
    )


# ---------------------------------------------------------------------------
# ch10：主题分类结果
# ---------------------------------------------------------------------------


class TopicClassification(Base):
    """旁路分类器对低置信度问题的一次归类结果。

    ``labels`` 存**中文类目名**（按类目表顺序），不存编号：编号只在训练脚本
    里当下标用，落库后要能被人直接读。多标签直接存 JSON 数组 —— 另开一张
    关联表会把「按标签计数」变成一次 join 聚合，而这张表只服务于分布统计。

    ``question_id`` 上有唯一约束，语义是「重跑覆盖，不是追加」：批跑从上次
    水位重来（sidecar 崩了、进程被中断）时，没有这个约束同一道题会被算两次，
    把主题分布整体带偏，而且带偏的方向取决于中断了几次。
    """

    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(_AUTOINC_PK, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("low_confidence_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    labels: Mapped[list] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("question_id", name="uk_question_id"),)
