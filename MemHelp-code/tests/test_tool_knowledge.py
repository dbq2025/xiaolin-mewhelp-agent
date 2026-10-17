"""``query_faq``（向量检索版）与 ``create_ticket`` 的测试。

``query_faq`` 的内核在 ch03 从「关键词 LIKE」换成了「向量检索」，但它的**出参契约
一字未变** —— 本文件的前几条用例就是在钉这件事：前端徽章、评估集、工具 Schema
全都依赖 ``{ok, keyword, count, items:[{question, answer, category}]}`` 这个形状。

要钉住的四件容易做错的事：

1. **出参形状不许变**（多一个 ``score`` 键就是改契约）；
2. **查不到不是错误**：必须是 ``ok=true, count=0``，让模型如实说「没查到」；
3. **异常不许吞**：向量库挂了要冒出去由 executor 分类，否则「基础设施坏了」
   与「业务上没查到」会变成同一个信号；
4. **整句原话进检索**：截成名词会丢掉「换说法」的语义信息。
"""

import json

import pytest

from app.knowledge.retrieval import KnowledgeHit
from app.tools.knowledge import MAX_FAQ_ITEMS, query_faq
from app.tools.ticketing import create_ticket


# ---------------------------------------------------------------------------
# 假依赖
# ---------------------------------------------------------------------------


class FakeRetriever:
    """记录收到的查询，并按给定的命中列表返回（不排序、不过滤）。"""

    def __init__(self, hits=()) -> None:
        self.hits = list(hits)
        self.calls: list[tuple[str, int]] = []

    async def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


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
    retriever = FakeRetriever([KnowledgeHit("退货政策是什么？", "签收后 7 天内无理由退货。", "售后", 0.91)])

    out = json.loads(await query_faq("退货", retriever=retriever))

    assert out["ok"] is True
    assert out["count"] == 1
    assert set(out["items"][0]) == {"question", "answer", "category"}
    assert out["items"][0]["question"] == "退货政策是什么？"
    assert out["items"][0]["answer"] == "签收后 7 天内无理由退货。"
    assert out["items"][0]["category"] == "售后"


async def test_query_faq_does_not_leak_the_similarity_score():
    """相似度只用于日志/调试。多一个键就是改契约。"""
    retriever = FakeRetriever([KnowledgeHit("q", "a", "c", 0.99)])

    out = json.loads(await query_faq("退货", retriever=retriever))

    assert "score" not in out["items"][0]


async def test_query_faq_reports_zero_hits_without_raising():
    """查不到是**正常业务结果**：必须是 ``ok=true`` + ``count=0``。

    做成 ``ok=false`` 的话，模型收到的信号是「工具坏了」，于是它会道歉或编造，
    而不是如实说「库里没有这条」。
    """
    out = json.loads(await query_faq("邮费是多少", retriever=FakeRetriever()))

    assert out["ok"] is True
    assert out["count"] == 0
    assert out["items"] == []
    assert out["keyword"] == "邮费是多少"


async def test_query_faq_passes_the_whole_question_through():
    """整句原话进向量检索 —— 截成名词会丢掉「换说法」的语义。"""
    retriever = FakeRetriever()
    await query_faq("买了七天还能退吗", retriever=retriever)

    assert retriever.calls == [("买了七天还能退吗", MAX_FAQ_ITEMS)]


async def test_query_faq_strips_surrounding_whitespace():
    """模型偶尔会把整句带空白一起传过来。"""
    retriever = FakeRetriever()
    await query_faq("  退货  ", retriever=retriever)

    assert retriever.calls[0][0] == "退货"


async def test_query_faq_skips_the_retriever_for_blank_keywords():
    """纯空白不调检索器（省一次模型前向），但仍返回合法 JSON。"""
    retriever = FakeRetriever([KnowledgeHit("q", "a", "c", 1.0)])

    out = json.loads(await query_faq("   ", retriever=retriever))

    assert out["count"] == 0
    assert retriever.calls == []


async def test_query_faq_uses_a_bounded_limit():
    """结果条数必须有上限：命中 50 条会把上下文预算一次吃光。"""
    retriever = FakeRetriever([KnowledgeHit(f"q{i}", "答案", "售后", 1.0) for i in range(50)])

    out = json.loads(await query_faq("退货", retriever=retriever))

    assert out["count"] == MAX_FAQ_ITEMS
    assert retriever.calls[0][1] == MAX_FAQ_ITEMS


async def test_query_faq_lets_retriever_failure_propagate():
    """检索炸了必须让异常冒出去，由 ``executor.py`` 分类。

    不能在工具里吞成「空结果」：那样「Milvus 真的坏了」与「业务上没查到」
    会变成同一个信号，于是向量库挂了的时候，模型会一本正经地告诉用户
    「我们库里没有这条」。
    """

    class Exploding:
        async def search(self, query, top_k=5):
            raise RuntimeError("connection reset by peer")

    with pytest.raises(RuntimeError):
        await query_faq("退货", retriever=Exploding())


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
