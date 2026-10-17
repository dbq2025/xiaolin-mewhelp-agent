"""``query_faq``（混合检索版）与 ``create_ticket`` 的测试。

``query_faq`` 的内核在 ch03 从「关键词 LIKE」换成向量检索，ch04 又换成
「改写 + 双路召回 + RRF + 重排」；但它的**出参契约只增不改** ——
``{ok, keyword, count, items:[...]}`` 必须还在，前端徽章、历史评估集、工具 Schema
全都依赖这个形状。

要钉住的五件容易做错的事：

1. **ch02 的键不许丢、不许改语义**；新字段只能加；
2. **``items`` 的顺序是首尾编排**（否则模型对中段注意力最弱这件事就白做了），
   而每项的 ``n`` 仍是相关度序号；
3. **查不到不是错误**：``count=0`` 是正常业务结果；
4. **异常不许吞**：检索挂了要冒出去由 executor 分类；
5. **拒答时 items 为空**：给了证据模型就可能硬答。
"""

import json

import pytest

from app.knowledge.citations import build_evidence
from app.knowledge.hybrid import RetrievedChunk
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from app.tools.knowledge import query_faq
from app.tools.ticketing import create_ticket


# ---------------------------------------------------------------------------
# 假依赖
# ---------------------------------------------------------------------------


class FakePipeline:
    """记录收到的查询，返回预设的 ``RetrievalOutcome``。"""

    def __init__(self, outcome: RetrievalOutcome | None = None, *, error: Exception | None = None):
        self.outcome = outcome or _outcome()
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def answer_context(self, query, *, category=None):
        self.calls.append((query, category))
        if self.error is not None:
            raise self.error
        return self.outcome


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


def _chunk(chunk_id: int, answer: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        question=f"问法{chunk_id}",
        answer=answer,
        category="售后政策",
        section_path=f"售后手册 / 第{chunk_id}节",
        content_type="manual",
        score=score,
        source="rerank",
    )


def _outcome(*chunks: RetrievedChunk) -> RetrievalOutcome:
    return RetrievalOutcome(
        query=UnderstoodQuery(original="q", standard="q", bm25_text="q"),
        evidence=build_evidence(chunks),
    )


def _refused(source: str = "self_check", reason: str = "证据不足") -> RetrievalOutcome:
    return RetrievalOutcome(
        query=UnderstoodQuery(original="q", standard="q", bm25_text="q"),
        evidence=(),
        refused=True,
        refusal_source=source,
        refusal_reason=reason,
    )


# ---------------------------------------------------------------------------
# query_faq：契约与顺序
# ---------------------------------------------------------------------------


async def test_query_faq_keeps_the_ch02_keys():
    retriever = FakePipeline(_outcome(_chunk(1, "签收后 7 天内无理由退货。")))

    out = json.loads(await query_faq("退货", retriever=retriever))

    assert {"ok", "keyword", "count", "items"} <= set(out)
    assert out["ok"] is True
    assert out["count"] == 1
    assert out["items"][0]["question"] == "问法1"
    assert out["items"][0]["answer"] == "签收后 7 天内无理由退货。"
    assert out["items"][0]["category"] == "售后政策"


async def test_query_faq_adds_citation_fields():
    retriever = FakePipeline(_outcome(_chunk(5, "满 99 包邮")))

    item = json.loads(await query_faq("运费", retriever=retriever))["items"][0]

    assert item["n"] == 1
    assert item["chunk_id"] == 5
    assert item["section_path"] == "售后手册 / 第5节"
    assert item["quote"] == "满 99 包邮"


async def test_query_faq_does_not_leak_scores():
    """分数只用于日志/报告。多一个键就是改契约。"""
    retriever = FakePipeline(_outcome(_chunk(1, "a", score=0.99)))

    item = json.loads(await query_faq("q", retriever=retriever))["items"][0]

    assert "score" not in item


async def test_items_are_in_head_tail_order_but_numbers_stay_by_relevance():
    """实现「最相关的放首尾」的位置，同时保证 n 仍是相关度序号。"""
    chunks = [_chunk(index, f"答案{index}") for index in range(1, 11)]
    retriever = FakePipeline(_outcome(*chunks))

    items = json.loads(await query_faq("q", retriever=retriever))["items"]

    assert [item["n"] for item in items] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]


async def test_query_faq_reports_zero_hits_as_success():
    retriever = FakePipeline(_outcome())

    out = json.loads(await query_faq("邮费是多少", retriever=retriever))

    assert out["ok"] is True
    assert out["count"] == 0
    assert out["items"] == []
    assert out["refused"] is False


# ---------------------------------------------------------------------------
# query_faq：入参与拒答
# ---------------------------------------------------------------------------


async def test_query_faq_passes_the_whole_question_and_category():
    retriever = FakePipeline(_outcome(_chunk(1, "a")))

    await query_faq("买了七天还能退吗", retriever=retriever, category="售后政策")

    assert retriever.calls == [("买了七天还能退吗", "售后政策")]


async def test_query_faq_strips_whitespace_and_skips_blank():
    retriever = FakePipeline(_outcome(_chunk(1, "a")))

    await query_faq("  退货  ", retriever=retriever)
    assert retriever.calls[0][0] == "退货"

    out = json.loads(await query_faq("   ", retriever=retriever))
    assert out["count"] == 0
    assert len(retriever.calls) == 1  # 空白没有多调一次


async def test_refusal_returns_no_items_but_keeps_the_reason():
    retriever = FakePipeline(_refused("self_check", "证据只讲流程"))

    out = json.loads(await query_faq("赔付上限多少", retriever=retriever))

    assert out["refused"] is True
    assert out["count"] == 0
    assert out["items"] == []
    assert out["refusal_source"] == "self_check"
    assert out["refusal_reason"] == "证据只讲流程"


async def test_query_faq_lets_retriever_failure_propagate():
    """检索炸了必须让异常冒出去，由 ``executor.py`` 分类。

    不能在工具里吞成「空结果」：那样「Milvus/重排真的坏了」与「业务上没查到」
    会变成同一个信号，于是向量库挂了的时候，模型会一本正经地告诉用户
    「我们库里没有这条」。
    """
    retriever = FakePipeline(error=RuntimeError("connection reset by peer"))

    with pytest.raises(RuntimeError):
        await query_faq("退货", retriever=retriever)


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
