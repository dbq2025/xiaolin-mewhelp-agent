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
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
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
