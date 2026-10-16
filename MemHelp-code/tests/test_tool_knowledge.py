"""``query_faq`` 与 ``create_ticket`` 的测试。

这两个工具与 Task 5 的三个不同：它们**真的连数据**（``faq`` 表查、``tickets`` 表写）。
所以这里注入内存假仓库 —— 既不依赖 MySQL，也不依赖 SQLAlchemy 的查询形状
（那是 ``tests/test_db_repositories.py`` 的职责）。

本文件要钉住两件容易做错的事：

1. **查不到不是错误**（验收标准 3 的预期形态）。漏召回必须表现为
   ``ok=true, count=0``，让模型如实说「没查到」；若做成 ``ok=false``，
   模型会去道歉或编造，而这**不是**我们想观察的基线。
2. **转义不重复**。``LIKE`` 通配符转义是仓库层的职责（见 repositories.py），
   工具层原样传关键词。两边都转会把 ``100%`` 变成 ``100\\\\%``，反而查不到。
"""

import json

import pytest

from app.tools.knowledge import query_faq
from app.tools.ticketing import create_ticket


# ---------------------------------------------------------------------------
# 假仓库
# ---------------------------------------------------------------------------


class FakeFaqRepo:
    """记录收到的关键词，并按子串匹配假装成数据库。"""

    def __init__(self, rows=()) -> None:
        self.rows = list(rows)
        self.last_keyword: str | None = None
        self.last_limit: int | None = None

    async def search(self, keyword, limit=5):
        from app.db.repositories import FaqHit

        self.last_keyword = keyword
        self.last_limit = limit
        return [
            FaqHit(question=q, answer=a, category=c)
            for (q, a, c) in self.rows
            if keyword in q or keyword in a
        ][:limit]


class FakeTicketRepo:
    def __init__(self, ticket_no="T20260701001") -> None:
        self.created: list[tuple[int, str, str]] = []
        self._ticket_no = ticket_no

    async def create(self, conversation_id, description, ticket_type, *, day=None):
        self.created.append((conversation_id, description, ticket_type))
        return self._ticket_no


class FakeConversationRepo:
    def __init__(self, conversation_id=7) -> None:
        self.calls: list[tuple[str, str]] = []
        self._id = conversation_id

    async def upsert_by_session(self, session_id, user_id):
        self.calls.append((session_id, user_id))
        return self._id


# ---------------------------------------------------------------------------
# query_faq
# ---------------------------------------------------------------------------


async def test_query_faq_returns_hits_with_questions_and_answers():
    repo = FakeFaqRepo(
        [("退货政策是什么？", "签收后 7 天内无理由退货。", "售后")]
    )
    out = json.loads(await query_faq("退货", faq_repo=repo))

    assert out["ok"] is True
    assert out["count"] == 1
    assert out["items"][0]["question"] == "退货政策是什么？"
    assert out["items"][0]["answer"] == "签收后 7 天内无理由退货。"
    assert out["items"][0]["category"] == "售后"


async def test_query_faq_reports_zero_hits_without_raising():
    """验收标准 3：查不到是**预期结果**，必须是 ``ok=true`` + ``count=0``。

    做成 ``ok=false`` 的话，模型收到的信号是「工具坏了」，于是它会道歉或编造，
    而不是如实说「FAQ 里没有这条」—— 那样漏召回就观察不到了。
    """
    out = json.loads(await query_faq("邮费是多少", faq_repo=FakeFaqRepo([])))

    assert out["ok"] is True
    assert out["count"] == 0
    assert out["items"] == []
    assert out["keyword"] == "邮费是多少"


async def test_query_faq_passes_keyword_through_untouched():
    """转义是仓库层的职责 —— 工具层重复转义会让本来能命中的词查不到。"""
    repo = FakeFaqRepo([])
    await query_faq("100%纯棉", faq_repo=repo)

    assert repo.last_keyword == "100%纯棉"


async def test_query_faq_strips_surrounding_whitespace():
    """模型偶尔会把关键词带空白一起传过来（尤其从整句里截取时）。"""
    repo = FakeFaqRepo([])
    await query_faq("  退货  ", faq_repo=repo)

    assert repo.last_keyword == "退货"


async def test_query_faq_uses_a_bounded_limit():
    """结果条数必须有上限：命中 200 条 FAQ 会把上下文预算一次吃光。"""
    repo = FakeFaqRepo([(f"退货 {i}", "答案", "售后") for i in range(50)])
    out = json.loads(await query_faq("退货", faq_repo=repo))

    assert out["count"] <= 5
    assert repo.last_limit == 5


async def test_query_faq_lets_repository_failure_propagate():
    """仓库炸了必须让异常冒出去，由 ``executor.py`` 分类。

    不能在工具里吞成「空结果」：那样「基础设施真的坏了」与「业务上没查到」
    会变成同一个信号（``ok=true, count=0``），于是数据库挂了的时候，
    模型会一本正经地告诉用户「我们 FAQ 里没有这条」。
    """

    class Exploding:
        async def search(self, keyword, limit=5):
            raise RuntimeError("connection reset by peer")

    with pytest.raises(RuntimeError):
        await query_faq("退货", faq_repo=Exploding())


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------


async def test_create_ticket_returns_ticket_no_and_persists():
    tickets = FakeTicketRepo("T20260701001")
    conversations = FakeConversationRepo(7)

    out = json.loads(
        await create_ticket(
            description="杯子碎了",
            ticket_type="售后",
            conversation_id="sess-a",
            ticket_repo=tickets,
            conversation_repo=conversations,
        )
    )

    assert out["ok"] is True
    assert out["ticket_no"] == "T20260701001"
    assert out["ticket_type"] == "售后"
    assert tickets.created == [(7, "杯子碎了", "售后")]


async def test_create_ticket_maps_session_id_to_conversation_id():
    """``session_id`` 是字符串（前端给的），表里要的是数字外键。

    映射必须在工具内完成：让模型去猜数字 id 是不现实的，而模型一定会猜错。
    """
    conversations = FakeConversationRepo(42)
    await create_ticket(
        description="要投诉",
        ticket_type="投诉",
        conversation_id="sess-xyz",
        ticket_repo=FakeTicketRepo(),
        conversation_repo=conversations,
    )

    assert conversations.calls == [("sess-xyz", "anonymous")]


async def test_create_ticket_creates_the_conversation_if_missing():
    """第一句就要求建工单时，会话可能还不存在 —— upsert 必须能建出来，
    否则外键会指向不存在的行（真库上直接报外键约束失败）。"""
    conversations = FakeConversationRepo(42)
    await create_ticket(
        description="咨询",
        ticket_type="咨询",
        conversation_id="brand-new",
        ticket_repo=FakeTicketRepo(),
        conversation_repo=conversations,
    )

    assert conversations.calls == [("brand-new", "anonymous")]


@pytest.mark.parametrize("bad_type", ["退款", "shipping", ""])
async def test_create_ticket_rejects_type_outside_the_ddl_enum(bad_type):
    """非法类型要在调用仓库之前就拦下。

    传到仓库层才炸的话，错误信息来自 SQLAlchemy 或数据库驱动，
    对模型毫无可操作性；在这里拦下可以明确告诉它允许的三个取值。
    """
    with pytest.raises(ValueError, match="ticket_type"):
        await create_ticket(
            description="描述",
            ticket_type=bad_type,
            conversation_id="sess-a",
            ticket_repo=FakeTicketRepo(),
            conversation_repo=FakeConversationRepo(),
        )


async def test_create_ticket_rejects_blank_description():
    with pytest.raises(ValueError, match="description"):
        await create_ticket(
            description="   ",
            ticket_type="售后",
            conversation_id="sess-a",
            ticket_repo=FakeTicketRepo(),
            conversation_repo=FakeConversationRepo(),
        )


async def test_create_ticket_lets_repository_failure_propagate():
    """同上：基础设施故障必须让执行器分类，不能在这里被吞成「成功」。"""

    class Exploding:
        async def create(self, conversation_id, description, ticket_type, *, day=None):
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        await create_ticket(
            description="杯子碎了",
            ticket_type="售后",
            conversation_id="sess-a",
            ticket_repo=Exploding(),
            conversation_repo=FakeConversationRepo(),
        )


# ---------------------------------------------------------------------------
# 分层约束
# ---------------------------------------------------------------------------


def test_knowledge_and_ticketing_import_only_the_repository_layer():
    """这两个工具可以 import ``app.db.repositories``（仓库协议），
    但不得 import ``app.db.models`` 或 ``app.db.session`` —— 它们该面对业务语义方法，
    而不是 ORM 与连接装配。"""
    import app.tools.knowledge as knowledge
    import app.tools.ticketing as ticketing

    for module in (knowledge, ticketing):
        source = open(module.__file__, encoding="utf-8").read()
        assert "app.db.models" not in source, module.__name__
        assert "app.db.session" not in source, module.__name__
