"""真实 MySQL 上的集成验证。

**默认跳过**，只在 ``MEMHELP_TEST_MYSQL=1`` 时运行::

    uv run --directory MemHelp-code python scripts/migrate_existing.py   # 复用既有库时先迁移
    $env:MEMHELP_TEST_MYSQL = "1"
    uv run --directory MemHelp-code python -m pytest tests/test_db_mysql.py -v

## 为什么必须有这个文件

``sql/ch02-ddl.sql`` 是建表的真相，ORM 只是它的 Python 投影。
单测跑在 SQLite 上（``native_enum`` 会退化成 VARCHAR+CHECK），
所以**跨方言的差异在那里测不出来**：

- 列名差一个字母 → 真库报 ``Unknown column``；
- 有 ``NOT NULL`` 且无默认值的列而 ORM 不知道 → INSERT 直接失败；
- 中文 ENUM 值被 latin1 double-encode → 存进去是乱码，读出来对不上。

这三类都是「单测全绿、真库炸」的典型。本文件是唯一能拦住它们的地方。

## 两种目标库（默认对准第二种）

1. **本章 compose 起的新库**（宿主 3307、``mewhelp/mewhelp``）：
   结构与 ``sql/ch02-ddl.sql`` 逐列一致。
2. **复用的既有库**（宿主 3306、``root/root``）—— **默认**：
   这是本机实际的形态。它是**后续章节演进过的**库，于是有两处已知分歧，
   下面用 ``KNOWN_SCHEMA_DIVERGENCES`` 显式记录，测试按「对齐现实」来断言，
   而不是假装分歧不存在。

用环境变量可切换目标：``MEMHELP_TEST_MYSQL_PORT`` / ``_USER`` / ``_PASSWORD`` /
``_DATABASE`` / ``_HOST``。
"""

import os

import pytest
from sqlalchemy import inspect, text

from app.config import Settings
from app.db.models import Base

#: 复用既有库时**已知且刻意保留**的结构分歧。
#:
#: 这些不是缺陷，而是「本章的 ORM 面对一个演进了的库」的事实。把它们写下来，
#: 是为了让后来者一眼看到「为什么断言与 DDL 不完全一样」，
#: 也让分歧变化时（比如又加了列）测试能立刻反映出来。
KNOWN_SCHEMA_DIVERGENCES: dict[str, dict[str, object]] = {
    "conversations": {
        "extra_columns": {"summary", "summary_upto_msg_id"},
        "note": "后续章节加的会话摘要字段；两列都可空，不影响本章 ORM 的插入",
    },
    "tickets": {
        "extra_enum_values": {"退款"},
        "note": "真库的 ticket_type 比本章多一个『退款』；本章的工具只会产出三个值",
    },
    "qa_extraction_staging": {
        "extra_enum_values": {"approved", "rejected"},
        "note": "后续章节加的人工审核状态；ORM 的 SAEnum 必须包含它们，否则读到即 LookupError",
    },
}


def _connection_defaults() -> dict[str, object]:
    """真库连接的默认值 —— **默认对准「复用的既有库」**（宿主 3306 / root）。"""
    return {
        "host": os.getenv("MEMHELP_TEST_MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MEMHELP_TEST_MYSQL_PORT", "3306")),
        "user": os.getenv("MEMHELP_TEST_MYSQL_USER", "root"),
        "password": os.getenv("MEMHELP_TEST_MYSQL_PASSWORD", "root"),
        "database": os.getenv("MEMHELP_TEST_MYSQL_DATABASE", "mewhelp"),
    }


DEFAULTS = _connection_defaults()

pytestmark = pytest.mark.skipif(
    os.getenv("MEMHELP_TEST_MYSQL") != "1",
    reason="需要真实 MySQL：设 MEMHELP_TEST_MYSQL=1 并先起库（或先跑迁移脚本）",
)


def _settings() -> Settings:
    return Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        mysql_host=str(DEFAULTS["host"]),
        mysql_port=int(DEFAULTS["port"]),  # type: ignore[arg-type]
        mysql_user=str(DEFAULTS["user"]),
        mysql_password=str(DEFAULTS["password"]),
        mysql_database=str(DEFAULTS["database"]),
    )


@pytest.fixture(scope="module")
def engine():
    """连真库。连不上就让整个模块报错 —— 跳过会掩盖问题。"""
    from app.db.session import create_engine_from

    eng = create_engine_from(_settings())
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# 结构对齐：ORM 需要的列必须都在真库里
# ---------------------------------------------------------------------------


def test_every_orm_table_exists_in_the_real_database(engine):
    real = set(inspect(engine).get_table_names())
    expected = set(Base.metadata.tables)
    assert expected <= real, f"真库缺表：{expected - real}"


@pytest.mark.parametrize("table_name", sorted(Base.metadata.tables))
def test_every_orm_column_exists_in_the_real_database(engine, table_name):
    """**整个文件最重要的一条** —— 它拦住 ``Unknown column``。

    这里断言的是「ORM 的每一列在真库里都存在」，而**不是**「两边列完全相等」。
    为什么方向是单向的：复用既有库时，真库会**多**出后续章节的列
    （``conversations.summary`` 等）—— 那些列可空，本章的 ORM 不碰它们也不受影响。
    反过来（ORM 有而真库没有）才是致命的：查询与插入都会直接失败。
    """
    real = {col["name"] for col in inspect(engine).get_columns(table_name)}
    expected = {col.name for col in Base.metadata.tables[table_name].columns}
    missing = expected - real
    assert not missing, f"{table_name} 在真库缺列（会报 Unknown column）：{missing}"


@pytest.mark.parametrize("table_name", sorted(KNOWN_SCHEMA_DIVERGENCES))
def test_documented_schema_divergences_still_hold(engine, table_name):
    """已知分歧必须还在 —— 分歧消失（或又多了）时这条会红，提醒更新文档。

    没有这条，``KNOWN_SCHEMA_DIVERGENCES`` 会慢慢变成一份过时的谎话。
    """
    spec = KNOWN_SCHEMA_DIVERGENCES[table_name]
    real = {col["name"] for col in inspect(engine).get_columns(table_name)}
    orm = {col.name for col in Base.metadata.tables[table_name].columns}

    extra = real - orm
    documented = set(spec.get("extra_columns", set()))  # type: ignore[arg-type]
    assert extra == documented, (
        f"{table_name} 的实际多余列 {extra} 与记录的分歧 {documented} 不一致"
    )


def test_conversations_session_id_is_unique_in_the_real_database(engine):
    """``session_id`` 的唯一键必须真的存在。

    它是 ``upsert_by_session`` 幂等的前提：唯一键缺失时并发会插出两行同键会话。
    复用既有库时这一列由 ``sql/ch02-migrate-existing.sql`` 补上。
    """
    indexes = inspect(engine).get_indexes("conversations")
    unique_index_columns = [tuple(ix["column_names"]) for ix in indexes if ix.get("unique")]
    constraints = inspect(engine).get_unique_constraints("conversations")
    constraint_columns = [tuple(c["column_names"]) for c in constraints]

    assert ("session_id",) in unique_index_columns + constraint_columns, (
        "session_id 没有唯一键 —— 复用既有库时请先跑 scripts/migrate_existing.py"
    )


def test_session_id_column_is_not_nullable_in_the_real_database(engine):
    """必须 NOT NULL：可空 + 唯一键时，MySQL 允许**多个 NULL**，
    唯一性在 NULL 上形同虚设（同一「空会话键」可以有很多行）。"""
    columns = {col["name"]: col for col in inspect(engine).get_columns("conversations")}
    assert not columns["session_id"]["nullable"]


def test_real_database_uses_utf8mb4(engine):
    """中文 ENUM 值被 latin1 double-encode 之后读出来是乱码且对不上。"""
    with engine.connect() as conn:
        charset = conn.execute(
            text(
                "SELECT DEFAULT_CHARACTER_SET_NAME FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME = DATABASE()"
            )
        ).scalar()
    assert charset == "utf8mb4", f"库默认字符集是 {charset}，中文会乱码"


def test_engine_connects_with_the_configured_charset(engine):
    """连接层也要 utf8mb4：库对了而连接是 latin1，写进去的中文一样坏。"""
    assert engine.url.query.get("charset") == "utf8mb4"


def test_chinese_enum_values_round_trip(engine):
    """写一个中文 ENUM 值再读回来，必须逐字相等。

    比只看库的默认字符集更硬 —— 它覆盖「库对了但连接层错了」这类问题。
    """
    from sqlalchemy.orm import Session

    from app.db.models import Conversation

    probe_session = f"charset-probe-{os.getpid()}"
    with Session(engine) as session:
        session.query(Conversation).filter(Conversation.session_id == probe_session).delete()
        session.commit()

        row = Conversation(session_id=probe_session, user_id="probe")
        session.add(row)
        session.commit()
        probe_id = row.id
        session.expunge_all()

        read_back = session.get(Conversation, probe_id)
        assert read_back is not None
        assert read_back.session_id == probe_session
        assert read_back.status == "进行中", f"中文 ENUM 读回来是 {read_back.status!r}"

        session.delete(read_back)
        session.commit()


def test_ticket_type_enum_accepts_the_values_this_chapter_uses(engine):
    """本章的工具只会产出 售后/投诉/咨询 —— 真库的 ENUM 必须接受这三个。

    真库可能**多**接受别的值（既有库里有『退款』，见 KNOWN_SCHEMA_DIVERGENCES）——
    那是超集，不影响本章。但本章用的三个值一个都不能缺，
    缺了就是在「参数校验通过、写库被拒」这种最难查的半对齐状态。
    """
    column_type = next(
        col["type"] for col in inspect(engine).get_columns("tickets") if col["name"] == "ticket_type"
    )
    real_values = {getattr(value, "__str__", lambda: value)() for value in column_type.enums}

    from app.db.models import TICKET_TYPES

    missing = set(TICKET_TYPES) - real_values
    assert not missing, f"真库的 ticket_type ENUM 不接受本章要用的值：{missing}"


# ---------------------------------------------------------------------------
# 验收标准 3 的基线（跑在**真实语料**上）
# ---------------------------------------------------------------------------


def test_faq_table_has_corpus_to_search(engine):
    """语料表不能是空的 —— 空表会让「查不到」这个结论毫无意义。"""
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM faq")).scalar()
    assert count and count > 0, "faq 表没有语料，验收标准 2/3 都无从谈起"


def test_acceptance_criterion_3_baseline_holds_in_the_real_database(engine):
    """**验收标准 3 的基线**：真实语料里不含「邮费」字面量。

    这条一旦红，说明语料里出现了含「邮费」的条目 ——
    「邮费是多少」不再漏召回，ch03 的「改造前基线」就没了。
    这是**故意保留的能力缺口**，不是 bug。
    """
    with engine.connect() as conn:
        hits = conn.execute(
            text(
                "SELECT COUNT(*) FROM faq "
                "WHERE question LIKE '%邮费%' OR answer LIKE '%邮费%'"
            )
        ).scalar()
    assert hits == 0, f"真实语料里有 {hits} 条含「邮费」的 FAQ，验收标准 3 的基线被破坏"


# ---------------------------------------------------------------------------
# 仓库层在真库上的行为（ENUM / 排序规则 / 自增都与 SQLite 不同）
# ---------------------------------------------------------------------------


async def test_repository_upsert_is_idempotent_on_the_real_database(engine):
    """SQLite 上的幂等不能代表 MySQL 上的幂等（隔离级别与自增语义都不同）。"""
    from sqlalchemy.orm import Session

    from app.db.repositories import ConversationRepository

    probe_session = f"mysql-upsert-probe-{os.getpid()}"
    with Session(engine) as session:
        repo = ConversationRepository(session)
        first = await repo.upsert_by_session(probe_session, "probe")
        second = await repo.upsert_by_session(probe_session, "probe")
        assert first == second

        session.execute(
            text("DELETE FROM conversations WHERE session_id = :sid"), {"sid": probe_session}
        )
        session.commit()


async def test_message_round_trip_on_the_real_database(engine):
    """消息流水（含 JSON 的 tool_calls 与 tool_call_id）在真库上要能原样存取。

    JSON 列与 ENUM 的方言差异都在这条里被跑到。
    """
    from sqlalchemy.orm import Session

    from app.db.repositories import ConversationRepository, MessageRepository, Repositories

    probe_session = f"mysql-msg-probe-{os.getpid()}"
    with Session(engine) as session:
        repos = Repositories.from_session(session)
        cid = await repos.conversations.upsert_by_session(probe_session, "probe")
        await repos.messages.append(cid, role="user", content="订单 1001 的物流到哪了")
        await repos.messages.append(
            cid,
            role="assistant",
            content="",
            tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        await repos.messages.append(cid, role="tool", content='{"ok":true}', tool_call_id="c1")
        await repos.messages.append(cid, role="assistant", content="已发货")

        rebuilt = await repos.messages.history_as_messages(probe_session)
        assert rebuilt[1].tool_calls[0]["name"] == "query_logistics"
        assert rebuilt[2].tool_call_id == "c1"

        session.execute(text("DELETE FROM messages WHERE conversation_id = :cid"), {"cid": cid})
        session.execute(
            text("DELETE FROM conversations WHERE session_id = :sid"), {"sid": probe_session}
        )
        session.commit()


async def test_faq_search_works_on_the_real_corpus(engine):
    """真库上的 ``LIKE`` 走 utf8mb4 的排序规则，与 SQLite 规则不同。

    「能查到」与「查不到」两种结果都要在真库上成立。
    """
    from sqlalchemy.orm import Session

    from app.db.repositories import FaqRepository

    with Session(engine) as session:
        repo = FaqRepository(session)
        # 真实语料里必有「退货」相关的条目（否则语料本身有问题，上面的用例会红）
        assert await repo.search("退货"), "「退货」在真实语料里应能命中"
        assert await repo.search("邮费") == [], "「邮费」必须命中 0 条（验收标准 3）"


async def test_query_faq_hydrates_chunk_text_from_the_real_database(engine):
    """端到端（真 MySQL）：向量只存 id，原文回 ``knowledge_chunks`` 取。

    ch02 那条「工具查真库语料时『邮费』返回 0 条」的用例在本章被**取代** ——
    工具已不再查 ``faq`` 表，漏召回正是本章要修的东西。这里验证的是新链路里
    唯一的 SQL 那一跳（``WHERE id IN (...)``）在真库上成立：列类型、utf8mb4、
    以及 ``questions`` 里换行分隔的多问法都能原样取回。
    """
    import json

    from sqlalchemy.orm import Session

    from app.db.models import KnowledgeChunk
    from app.db.repositories import KnowledgeChunkRepository
    from app.knowledge.retrieval import KnowledgeRetriever
    from app.knowledge.vector_store import VectorHit
    from app.tools.knowledge import query_faq

    marker = "ch03-it-邮费-mark"
    with Session(engine) as session:
        row = KnowledgeChunk(
            category="物流",
            questions=marker,
            answer="满 99 元包邮，未满收 10 元运费。",
            section_path="物流 / 运费",
            content_type="faq",
            is_key_clause=False,
        )
        session.add(row)
        session.commit()
        chunk_id = int(row.id)

        class _Embedder:
            async def embed(self, texts):
                return [[0.1] * 4 for _ in texts]

            async def warmup(self) -> None:
                return None

        class _Store:
            async def search(self, vector, top_k):
                return [VectorHit(str(chunk_id), 0.93)]

        try:
            retriever = KnowledgeRetriever(
                _Embedder(), _Store(), KnowledgeChunkRepository(session)
            )
            payload = json.loads(await query_faq("邮费是多少", retriever=retriever))
        finally:
            session.query(KnowledgeChunk).filter(KnowledgeChunk.id == chunk_id).delete()
            session.commit()

    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["items"][0]["question"] == marker
    assert "99" in payload["items"][0]["answer"]


def test_existing_conversations_all_have_a_session_key(engine):
    """既有会话行都该有会话键。

    复用既有库时 ``session_id`` 是新增列，历史行需要回填
    （``scripts/migrate_existing.py`` 会赋 ``demo-seed-<id>``）。
    留空的话这些会话永远无法通过接口访问，而症状是「查历史返回空」——
    看起来像消息丢了，其实是会话键没对上。
    """
    with engine.connect() as conn:
        blanks = conn.execute(
            text("SELECT COUNT(*) FROM conversations WHERE session_id IS NULL OR session_id = ''")
        ).scalar()
    assert blanks == 0, f"有 {blanks} 行会话没有 session_id，历史将无法通过接口回读"


# ---------------------------------------------------------------------------
# ch03：两张新表在真库上的列对齐、ENUM 超集与自引用外键删除
# ---------------------------------------------------------------------------


def test_ch03_tables_match_the_orm_columns(engine):
    """真库的列集合必须与 ORM 一致 —— 差一列在单测里毫无症状。"""
    from sqlalchemy import inspect

    from app.db.models import KnowledgeChunk, QaStaging

    inspector = inspect(engine)
    for model in (KnowledgeChunk, QaStaging):
        actual = {column["name"] for column in inspector.get_columns(model.__tablename__)}
        expected = {column.name for column in model.__table__.columns}
        assert actual == expected, model.__tablename__


async def test_staging_status_round_trips_all_five_values(engine):
    """真库 ENUM 是 5 值超集，ORM 必须能读能写。

    只写 3 值的话，读到后续章节写入的 ``approved`` 会抛 ``LookupError`` ——
    症状是「暂存表偶尔读不出来」，极难查。
    """
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from app.db.models import STAGING_STATUSES, QaStaging

    marker = "ch03-it-staging"
    with Session(engine) as session:
        try:
            for status in STAGING_STATUSES:
                session.add(
                    QaStaging(
                        batch_no=marker,
                        source_ref=marker,
                        question=f"问法-{status}",
                        answer="答案",
                        status=status,
                    )
                )
            session.commit()

            # 直接从库里读回来：SAEnum 不认的值会在这里抛 LookupError
            values = session.execute(
                text(
                    "SELECT DISTINCT status FROM qa_extraction_staging WHERE batch_no = :b"
                ),
                {"b": marker},
            ).scalars().all()
            assert set(values) == set(STAGING_STATUSES)
        finally:
            session.execute(
                text("DELETE FROM qa_extraction_staging WHERE batch_no = :b"),
                {"b": marker},
            )
            session.commit()


async def test_deleting_linked_chunks_works_against_self_referencing_foreign_keys(engine):
    """删掉互相引用的两块，在真库上必须成立。

    `knowledge_chunks` 有两个指向自己的外键（`ON DELETE SET NULL`）；SQLite 默认
    不强制外键，所以「单测绿、真库炸」的风险正好落在这里 —— 这正是
    ``--rebuild``（整表 DELETE）会走到的路径。

    **刻意不调 ``truncate()``**：那会清空这台共享库上的真实知识库。这里用
    同一张表、同一类互相引用的行、同一个删除路径（``delete_ids``）来验证，
    既能证明外键行为，又不动别人的数据。
    """
    from sqlalchemy.orm import Session

    from app.db.models import KnowledgeChunk
    from app.db.repositories import KnowledgeChunkRepository

    marker = "ch03-it-delete"
    with Session(engine) as session:
        repo = KnowledgeChunkRepository(session)
        first = KnowledgeChunk(
            category=marker, questions=f"{marker}-1", answer="甲", section_path=marker,
            content_type="manual", is_key_clause=False,
        )
        second = KnowledgeChunk(
            category=marker, questions=f"{marker}-2", answer="乙", section_path=marker,
            content_type="manual", is_key_clause=False,
        )
        session.add_all([first, second])
        session.commit()
        first.next_chunk_id = second.id
        second.prev_chunk_id = first.id
        session.commit()

        try:
            deleted = await repo.delete_ids([first.id, second.id])
            assert deleted == 2
            assert session.query(KnowledgeChunk).filter(
                KnowledgeChunk.category == marker
            ).count() == 0
        finally:
            session.rollback()
