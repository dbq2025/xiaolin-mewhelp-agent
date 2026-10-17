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

#: ch03：对话抽取暂存表的状态。
#:
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_conversation_id", "conversation_id"),)


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_tickets_conversation_id", "conversation_id"),)


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
