"""ORM 映射测试。

**这一层测的不是 SQL，是「映射与 DDL 是否逐字对齐」**：
``sql/ch02-ddl.sql`` 是建表的唯一真相，ORM 只是它的 Python 投影。差一个字母
（``session_id`` 拼成 ``sessionId``）在单测里毫无症状，到真库上表现为
``Unknown column`` —— 所以列名集合要逐个钉住。

真正的「DDL ↔ ORM 跨数据库对齐」由 ``tests/test_db_mysql.py`` 在真实 MySQL 上验，
这里只保证模型自身的形状（不能连库，沙箱里也没有 MySQL）。
"""


def test_metadata_has_exactly_four_ch02_tables():
    """多一张少一张都要红 —— 表名漂移会让 compose 灌好的数据白费。"""
    import app.db.models  # noqa: F401 - 必须先 import 才会注册进 metadata
    from app.db.base import Base

    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversation_session_id_is_unique_and_not_nullable():
    """``session_id`` 是前后端契约的落点：必须非空且唯一。

    不唯一的话，同一字符串会对应两行会话，历史与工具轨迹随机挂到其中一行 ——
    表现为「刷新后聊天记录少了一半」，且不可复现。

    唯一性在 SQLAlchemy 里可能落成 **UniqueConstraint**（``unique=True`` 于列上）
    或**唯一索引**（DDL 里写的是 ``UNIQUE KEY``）。两种形态在这里都认，
    否则换个写法就会误红 —— 而真库上的实际形态由 ``test_db_mysql.py`` 兜底。
    """
    from sqlalchemy import UniqueConstraint

    from app.db.models import Conversation

    table = Conversation.__table__
    col = table.c.session_id
    assert col.nullable is False
    assert col.type.length == 64

    unique_constraints = [
        [c.name for c in con.columns]
        for con in table.constraints
        if isinstance(con, UniqueConstraint)
    ]
    unique_indexes = [
        [c.name for c in ix.columns] for ix in table.indexes if ix.unique
    ]
    assert ["session_id"] in unique_constraints + unique_indexes


def test_message_columns_match_ddl():
    """列名必须与 sql/ch02-ddl.sql 的 messages 表逐字相同。"""
    from app.db.models import Message

    assert set(Message.__table__.c.keys()) == {
        "id",
        "conversation_id",
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "tool_traces",
        "created_at",
    }


def test_conversation_columns_match_ddl():
    from app.db.models import Conversation

    assert set(Conversation.__table__.c.keys()) == {
        "id",
        "session_id",
        "user_id",
        "status",
        "created_at",
        "updated_at",
    }


def test_ticket_columns_match_ddl():
    """工单号是业务主键（不是自增 id）—— 这是 DDL 里刻意的设计。"""
    from app.db.models import Ticket

    assert set(Ticket.__table__.c.keys()) == {
        "ticket_no",
        "conversation_id",
        "description",
        "ticket_type",
        "status",
        "created_at",
    }
    assert Ticket.__table__.c.ticket_no.primary_key is True
    assert Ticket.__table__.c.ticket_no.type.length == 32


def test_faq_columns_match_ddl():
    from app.db.models import Faq

    assert set(Faq.__table__.c.keys()) == {
        "id",
        "question",
        "answer",
        "category",
        "created_at",
        "updated_at",
    }
    assert Faq.__table__.c.question.type.length == 512
    assert Faq.__table__.c.category.type.length == 64


def test_enum_columns_carry_the_exact_ddl_values():
    """DDL 用的是 MySQL ENUM；ORM 侧必须同样受限。

    只在 Python 侧写个 Literal 是不够的 —— 那样「模型给了非法工单类型」会一路走到
    INSERT 才被数据库拒绝，错误信息是驱动的 ``Data truncated``。
    """
    from sqlalchemy import Enum as SAEnum

    from app.db.models import Conversation, Message, Ticket

    assert isinstance(Conversation.__table__.c.status.type, SAEnum)
    assert set(Conversation.__table__.c.status.type.enums) == {"进行中", "已转人工", "已结束"}
    assert isinstance(Message.__table__.c.role.type, SAEnum)
    assert set(Message.__table__.c.role.type.enums) == {"user", "assistant", "tool"}
    assert isinstance(Ticket.__table__.c.ticket_type.type, SAEnum)
    assert set(Ticket.__table__.c.ticket_type.type.enums) == {"售后", "投诉", "咨询"}
    assert set(Ticket.__table__.c.status.type.enums) == {"待处理", "已处理"}


def test_enum_constants_are_exported_as_tuple_constants():
    """Task 3 要用这些常量在入口校验 role，Task 8 要用它们做参数 Schema。"""
    from app.db.models import (
        CONVERSATION_STATUSES,
        MESSAGE_ROLES,
        TICKET_STATUSES,
        TICKET_TYPES,
    )

    assert MESSAGE_ROLES == ("user", "assistant", "tool")
    assert TICKET_TYPES == ("售后", "投诉", "咨询")
    assert TICKET_STATUSES == ("待处理", "已处理")
    assert CONVERSATION_STATUSES == ("进行中", "已转人工", "已结束")


def test_tool_calls_is_json_and_nullable():
    """assistant 纯文本消息没有工具调用，这一列必须可空。

    Python 侧要能直接存取 list/dict（不是字符串），否则每次读写都要手工 json 编解码，
    迟早有人在某条路径上忘记。
    """
    from sqlalchemy import JSON

    from app.db.models import Message

    col = Message.__table__.c.tool_calls
    assert col.nullable is True
    assert isinstance(col.type, JSON)
    assert Message.__table__.c.tool_call_id.nullable is True
    assert Message.__table__.c.content.nullable is True


def test_created_at_uses_database_clock_not_the_app_process():
    """时间戳由**数据库**填，不走应用进程。

    应用进程的时钟可能与库不同步（容器里尤其常见），两边混用会让「按时间排序的消息」
    出现负间隔。测试里断言存在 ``server_default``，不去比具体函数。
    """
    from app.db.models import Conversation, Message

    assert Conversation.__table__.c.created_at.server_default is not None
    assert Message.__table__.c.created_at.server_default is not None


def test_foreign_keys_point_at_conversations():
    from app.db.models import Message, Ticket

    for model in (Message, Ticket):
        fks = list(model.__table__.c.conversation_id.foreign_keys)
        assert len(fks) == 1
        assert fks[0].target_fullname == "conversations.id"
