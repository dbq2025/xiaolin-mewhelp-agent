"""仓库层测试 —— 跑在 SQLite 内存库上（见 conftest.py 的 ``sqlite_session``）。

**这一层负责「SQL 只出现在一个文件里」这条约束的落实**：所有查询形状在这里被钉住，
上层（工具、编排）只面对业务语义方法，不写 SQL。

真实 MySQL 上的行为差异（ENUM、``LIKE`` 排序规则、连接池）由
``tests/test_db_mysql.py`` 兜底 —— 本文件全绿不代表真库全绿，这一点是刻意的：
本层要能在**没有 MySQL** 的环境里跑完。
"""

import pytest


# ---------------------------------------------------------------------------
# 会话 upsert
# ---------------------------------------------------------------------------


async def test_upsert_by_session_is_idempotent(sqlite_session):
    """同一 session_id 第 2 轮必须复用同一个 id，不能新建行。

    新建的话，第 2 轮的消息挂到一个**新会话**上，而 tickets 仍指向旧会话 ——
    表现为「聊了几轮之后历史只剩最后一轮」，且外键指向漂移。
    """
    from app.db.models import Conversation
    from app.db.repositories import ConversationRepository

    repo = ConversationRepository(sqlite_session)
    first = await repo.upsert_by_session("sess-a", "anonymous")
    second = await repo.upsert_by_session("sess-a", "anonymous")

    assert first == second
    assert sqlite_session.query(Conversation).count() == 1


async def test_upsert_by_session_returns_the_primary_key(sqlite_session):
    from app.db.models import Conversation
    from app.db.repositories import ConversationRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s1", "anonymous")
    row = sqlite_session.query(Conversation).one()
    assert cid == row.id


async def test_upsert_by_session_keeps_sessions_separate(sqlite_session):
    from app.db.repositories import ConversationRepository

    repo = ConversationRepository(sqlite_session)
    a = await repo.upsert_by_session("sess-a", "anonymous")
    b = await repo.upsert_by_session("sess-b", "anonymous")
    assert a != b


# ---------------------------------------------------------------------------
# 消息流水
# ---------------------------------------------------------------------------


async def test_append_and_list_preserves_order_and_tool_columns(sqlite_session):
    """四行的顺序与工具列都必须能原样取回 —— 这是「工具轨迹落库」的最小证据。"""
    from app.db.repositories import ConversationRepository, MessageRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    repo = MessageRepository(sqlite_session)

    await repo.append(cid, role="user", content="订单 1001 的物流到哪了")
    await repo.append(
        cid,
        role="assistant",
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    await repo.append(cid, role="tool", content='{"ok":true}', tool_call_id="c1")
    await repo.append(cid, role="assistant", content="已发货")

    rows = await repo.list_for_conversation(cid)
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[1].content == ""
    assert rows[2].tool_call_id == "c1"


async def test_append_rejects_unknown_role_at_the_door(sqlite_session):
    """非法 role 必须在入口就炸。

    靠 MySQL 的 ENUM 兜会拖到 flush/commit 才失败，错误信息是驱动层的
    ``Data truncated for column 'role'`` —— 把「调用方拼错字符串」伪装成
    「数据库故障」，两者排查方向完全不同。
    """
    from app.db.repositories import ConversationRepository, MessageRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    with pytest.raises(ValueError, match="system"):
        await MessageRepository(sqlite_session).append(cid, role="system", content="x")


async def test_find_conversation_id_returns_none_for_unknown_session(sqlite_session):
    """第一次对话时会话还不存在 —— 回读历史必须能表达「还没有」。"""
    from app.db.repositories import MessageRepository

    assert await MessageRepository(sqlite_session).find_conversation_id("never-seen") is None


async def test_history_as_messages_rebuilds_langchain_objects(sqlite_session):
    """编排层（Task 9）的输入。tool 行必须带回自己的 tool_call_id，否则回灌错位。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.db.repositories import ConversationRepository, MessageRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    repo = MessageRepository(sqlite_session)
    await repo.append(cid, role="user", content="问")
    await repo.append(
        cid,
        role="assistant",
        content="",
        tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "c1"}],
    )
    await repo.append(cid, role="tool", content='{"ok":true}', tool_call_id="c1")
    await repo.append(cid, role="assistant", content="答")

    rebuilt = await repo.history_as_messages("s")

    assert [type(m) for m in rebuilt] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert rebuilt[0].content == "问"
    assert rebuilt[1].tool_calls[0]["name"] == "query_faq"
    assert rebuilt[2].tool_call_id == "c1"
    assert rebuilt[3].content == "答"


async def test_history_as_messages_is_empty_for_unknown_session(sqlite_session):
    from app.db.repositories import MessageRepository

    assert await MessageRepository(sqlite_session).history_as_messages("nope") == []


# ---------------------------------------------------------------------------
# FAQ 关键词查询
# ---------------------------------------------------------------------------


async def test_faq_search_finds_by_question_and_by_answer(sqlite_session):
    """验收 2 的数据路径：关键词要能在**问题**和**答案**两处命中。"""
    from app.db.models import Faq
    from app.db.repositories import FaqRepository

    sqlite_session.add(
        Faq(question="退货政策是什么？", answer="签收后 7 天内无理由退货。", category="售后")
    )
    sqlite_session.add(
        Faq(question="发票怎么开？", answer="退货时可一并申请发票重开。", category="发票")
    )
    sqlite_session.commit()

    repo = FaqRepository(sqlite_session)
    assert [f.question for f in await repo.search("退货")] == [
        "退货政策是什么？",
        "发票怎么开？",
    ]


async def test_faq_search_returns_empty_for_unmatched_keyword(sqlite_session):
    """验收 3 的预期形态：查不到是**空列表**，不是异常。

    工具层要把它转成 ``ok=true, count=0``，让模型如实回答「没查到」，
    而不是让它以为工具坏了。
    """
    from app.db.models import Faq
    from app.db.repositories import FaqRepository

    sqlite_session.add(
        Faq(question="退货运费谁承担？", answer="质量问题由平台承担。", category="售后")
    )
    sqlite_session.commit()

    assert await FaqRepository(sqlite_session).search("邮费") == []


async def test_faq_search_respects_limit(sqlite_session):
    from app.db.models import Faq
    from app.db.repositories import FaqRepository

    for i in range(5):
        sqlite_session.add(Faq(question=f"退货问题 {i}", answer="x", category="售后"))
    sqlite_session.commit()

    assert len(await FaqRepository(sqlite_session).search("退货", limit=2)) == 2


def test_escape_like_neutralises_wildcards():
    """Review Focus 5：``%`` 与 ``_`` 必须当字面量。

    不转义的话，用户问「100%纯棉」会命中**全表**（``%`` 通配一切），
    于是模型拿着 5 条毫不相关的 FAQ 去回答 —— 而且看起来「工具是成功的」。
    """
    from app.db.repositories import escape_like

    assert escape_like("100%纯棉") == "100\\%纯棉"
    assert escape_like("a_b") == "a\\_b"
    assert escape_like("c\\d") == "c\\\\d"


async def test_faq_search_treats_wildcards_as_literals(sqlite_session):
    """转义要真的接进查询里 —— 只测 ``escape_like`` 不够。"""
    from app.db.models import Faq
    from app.db.repositories import FaqRepository

    sqlite_session.add(Faq(question="普通问题", answer="普通答案", category="售后"))
    sqlite_session.add(Faq(question="100%纯棉说明", answer="全棉材质", category="售后"))
    sqlite_session.commit()

    repo = FaqRepository(sqlite_session)
    assert [f.question for f in await repo.search("%")] == ["100%纯棉说明"]


# ---------------------------------------------------------------------------
# 工单
# ---------------------------------------------------------------------------


async def test_ticket_no_follows_the_documented_format(sqlite_session):
    """形如 ``T20260701008``：T + 8 位日期 + 3 位当日序号。"""
    from datetime import date

    from app.db.repositories import ConversationRepository, TicketRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    repo = TicketRepository(sqlite_session)
    ticket_no = await repo.create(cid, "杯子碎了", "售后", day=date(2026, 7, 1))

    assert ticket_no == "T20260701001"


async def test_ticket_sequence_increments_within_the_same_day(sqlite_session):
    from datetime import date

    from app.db.repositories import ConversationRepository, TicketRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    repo = TicketRepository(sqlite_session)
    first = await repo.create(cid, "问题一", "咨询", day=date(2026, 7, 1))
    second = await repo.create(cid, "问题二", "投诉", day=date(2026, 7, 2))

    assert first == "T20260701001"
    # 换一天，序号重新从 1 开始（不是全局流水号）
    assert second == "T20260702001"


async def test_ticket_row_is_persisted_with_its_type_and_default_status(sqlite_session):
    from datetime import date

    from app.db.models import Ticket
    from app.db.repositories import ConversationRepository, TicketRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    await TicketRepository(sqlite_session).create(cid, "杯子碎了", "售后", day=date(2026, 7, 1))

    row = sqlite_session.query(Ticket).one()
    assert row.ticket_no == "T20260701001"
    assert row.conversation_id == cid
    assert row.description == "杯子碎了"
    assert row.ticket_type == "售后"
    assert row.status == "待处理"


async def test_ticket_rejects_type_outside_the_ddl_enum(sqlite_session):
    """非法工单类型要在写库前就被拦下。"""
    from datetime import date

    from app.db.repositories import ConversationRepository, TicketRepository

    cid = await ConversationRepository(sqlite_session).upsert_by_session("s", "anonymous")
    with pytest.raises(ValueError, match="退款"):
        await TicketRepository(sqlite_session).create(cid, "描述", "退款", day=date(2026, 7, 1))


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------


def test_repositories_container_is_synchronous_assembly(sqlite_session):
    """``from_session`` 是纯装配（不碰 IO），所以是同步的 —— 不是 async。"""
    from app.db.repositories import Repositories

    repos = Repositories.from_session(sqlite_session)
    for name in ("conversations", "messages", "faq", "tickets", "knowledge", "qa_staging"):
        assert getattr(repos, name) is not None


# ---------------------------------------------------------------------------
# ch03：知识库 chunk（双写状态机）
# ---------------------------------------------------------------------------


class _Chunk:
    """最小 chunk 形状（duck typing）—— 仓库层不该 import 切分器的类型。

    ``ordinal`` 是切分器给的文档内序号；仓库层只用它来决定「同一篇里的先后」，
    不落库（DDL 没有这一列）。
    """

    category = "售后"
    questions = "能退吗"
    answer = "7 天无理由"
    section_path = "售后 / 退货"
    content_type = "faq"
    is_key_clause = False
    ordinal = 0


async def test_upsert_chunk_is_idempotent_by_natural_key(sqlite_session):
    """同一 (content_type, section_path, questions) 再写一次不能新增行。

    这是「建库重跑不翻倍」的第一道闸：没有它，`--rebuild` 之外的任何重跑都会
    让库里的块数翻倍，而检索结果看起来还「挺正常」。
    """
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    id1, outcome1 = await repo.upsert_chunk(_Chunk())
    id2, outcome2 = await repo.upsert_chunk(_Chunk())

    assert (outcome1, outcome2) == ("created", "unchanged")
    assert id2 == id1
    assert await repo.count_by_status() == {"pending": 1, "done": 0}


async def test_upsert_chunk_resets_status_when_answer_changes(sqlite_session):
    """原文改了 → 旧向量作废，必须回到 pending 等着重算。

    不重置的话，改了答案的块会一直用旧向量被检索到 —— 用户看到的是**旧答案**，
    而库里明明已经是新答案，排查方向会完全跑偏。
    """
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    cid, _ = await repo.upsert_chunk(_Chunk())
    await repo.mark_done([(cid, str(cid))])
    assert await repo.count_by_status() == {"pending": 0, "done": 1}

    class _Changed(_Chunk):
        answer = "7 天无理由，需保持完好"

    cid2, outcome = await repo.upsert_chunk(_Changed())

    assert cid2 == cid and outcome == "updated"
    assert await repo.count_by_status() == {"pending": 1, "done": 0}
    rows = await repo.list_pending()
    assert rows[0].answer == "7 天无理由，需保持完好"
    assert rows[0].vector_id is None


async def test_list_pending_returns_only_pending_and_respects_limit(sqlite_session):
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    ids = []
    for i in range(3):
        class _Ci(_Chunk):
            questions = f"问法{i}"
            ordinal = i

        cid, _ = await repo.upsert_chunk(_Ci())
        ids.append(cid)
    await repo.mark_done([(ids[0], str(ids[0]))])

    assert [r.id for r in await repo.list_pending()] == [ids[1], ids[2]]
    assert [r.id for r in await repo.list_pending(limit=1)] == [ids[1]]

    # 「状态是 done 但 vector_id 为空」= 回填丢了一半，也必须被拣出来重算。
    from sqlalchemy import text

    sqlite_session.execute(
        text("UPDATE knowledge_chunks SET vectorize_status='done' WHERE id = :i"),
        {"i": ids[2]},
    )
    sqlite_session.commit()
    assert ids[2] in [r.id for r in await repo.list_pending()]


async def test_link_neighbours_sets_prev_and_next(sqlite_session):
    """文档内顺序由调用方给（切分器算出的 ordinal），仓库层只连指针。

    跨文档不连：不同章节「首尾相连」会让检索命中后顺着链走到无关章节。
    """
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    a, _ = await repo.upsert_chunk(_Chunk())

    class _B(_Chunk):
        questions = "第二问"
        ordinal = 1

    b, _ = await repo.upsert_chunk(_B())
    await repo.link_neighbours([a, b])

    rows = {r.id: r for r in await repo.get_many([a, b])}
    assert rows[a].prev_chunk_id is None and rows[a].next_chunk_id == b
    assert rows[b].prev_chunk_id == a and rows[b].next_chunk_id is None


async def test_get_many_preserves_nothing_and_returns_empty_for_unknown(sqlite_session):
    """``get_many`` 只是批量取行，顺序由调用方按向量分数重排（见 retrieval）。"""
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    a, _ = await repo.upsert_chunk(_Chunk())

    assert [r.id for r in await repo.get_many([a])] == [a]
    assert await repo.get_many([9999]) == []


async def test_truncate_clears_all_rows(sqlite_session):
    """``--rebuild`` 用；只清表不清 Milvus 会留下孤儿向量（见 ingest.rebuild）。"""
    from app.db.repositories import KnowledgeChunkRepository

    repo = KnowledgeChunkRepository(sqlite_session)
    await repo.upsert_chunk(_Chunk())
    await repo.truncate()
    assert await repo.count_by_status() == {"pending": 0, "done": 0}


# ---------------------------------------------------------------------------
# ch03：问答暂存表
# ---------------------------------------------------------------------------


async def test_qa_staging_marks_and_counts(sqlite_session):
    from app.db.repositories import QaStagingRepository

    repo = QaStagingRepository(sqlite_session)
    await repo.add_rows(
        [
            {"batch_no": "b1", "source_ref": "s1", "question": "邮费？", "answer": "满 99 包邮"},
            {"batch_no": "b1", "source_ref": "s2", "question": "能退吗？", "answer": "七天无理由"},
        ]
    )
    assert await repo.count_by_status() == {"extracted": 2}
    assert await repo.known_sources() == {"s1", "s2"}

    rows = await repo.list_by_status("extracted")
    await repo.mark([rows[1].id], "discarded")
    assert await repo.count_by_status() == {"extracted": 1, "discarded": 1}


async def test_qa_staging_updates_merged_questions(sqlite_session):
    """同义问法合并后要能回写 ``questions``（多行）。"""
    from app.db.repositories import QaStagingRepository

    repo = QaStagingRepository(sqlite_session)
    await repo.add_rows([{"batch_no": "b", "source_ref": "s", "question": "邮费是多少", "answer": "a"}])
    row = (await repo.list_by_status("extracted"))[0]
    await repo.update_questions(row.id, "邮费是多少\n寄回去要我付邮费吗")
    assert (await repo.list_by_status("extracted"))[0].question == "邮费是多少\n寄回去要我付邮费吗"


async def test_faq_repository_list_all_returns_every_row(sqlite_session):
    """存量 FAQ 要能整表读出来灌进知识库（content_type='faq'）。"""
    from app.db.models import Faq
    from app.db.repositories import FaqRepository

    sqlite_session.add_all(
        [
            Faq(question="退货政策", answer="7 天无理由", category="售后"),
            Faq(question="运费怎么算", answer="满 99 包邮", category="物流"),
        ]
    )
    sqlite_session.commit()

    rows = await FaqRepository(sqlite_session).list_all()
    assert [r.question for r in rows] == ["退货政策", "运费怎么算"]
    assert rows[1].category == "物流"


async def test_list_sessions_returns_ids_in_creation_order(sqlite_session):
    """挖矿要先枚举「有哪些会话」——没有这个方法，定时任务只能一张表一张表地摸。"""
    from app.db.repositories import ConversationRepository

    repo = ConversationRepository(sqlite_session)
    await repo.upsert_by_session("s-1", "anonymous")
    await repo.upsert_by_session("s-2", "anonymous")

    rows = await repo.list_sessions()
    assert [r.session_id for r in rows] == ["s-1", "s-2"]
    assert [r.session_id for r in await repo.list_sessions(limit=1)] == ["s-1"]

