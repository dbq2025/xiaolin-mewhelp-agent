"""离线建库编排：MySQL 记「待向量化」→ Milvus 拿 vector_id 回填 → 状态转「已向量化」。

这一层要钉住的不是「能跑通」，而是**挂了能重跑**：

- 块写进 MySQL 而向量没写 → 仍是 ``pending``，下次被拣起；
- 向量写进 Milvus 而回填没做 → 仍是 ``pending``，重写用同一主键覆盖，**不产生重复**；
- 两库都写成功 → ``done``，重跑跳过。

测试用 SQLite + 假嵌入器 + 假向量库，完全离线；真机路径由
``scripts/build_knowledge.py`` 的演示命令覆盖（见 spec §10）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.repositories import KnowledgeChunkRepository
from app.knowledge.ingest import (
    IngestReport,
    ingest_faq_table,
    ingest_markdown,
    rebuild,
    vectorize_pending,
)


def _repo(session) -> KnowledgeChunkRepository:
    return KnowledgeChunkRepository(session)


class _FakeEmbedder:
    """按文本长度给确定性向量，方便断言「写了几条」而不用管内容。"""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.warmup_calls = 0

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[0.01 * (i + 1)] * 4 for i, _ in enumerate(texts)]

    async def warmup(self) -> None:
        self.warmup_calls += 1


class _FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}
        self.upsert_calls = 0
        self.reset_calls = 0
        self.ensure_calls = 0

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def upsert(self, ids, vectors) -> None:
        self.upsert_calls += 1
        for row_id, vector in zip(ids, vectors):
            self.rows[row_id] = list(vector)  # 同主键覆盖，模拟 Milvus upsert

    async def search(self, vector, top_k):
        return []

    async def count(self) -> int:
        return len(self.rows)

    async def reset(self) -> None:
        self.reset_calls += 1
        self.rows.clear()


@pytest.fixture
def sample_doc():
    """写进**已存在**的目录，用完即删（沙箱不给用 tmp_path）。"""
    path = Path(__file__).resolve().parent / "_tmp_ch03_doc.md"
    path.write_text(
        "# 售后手册\n\n## 运费\n\n满 99 包邮。\n\n## 时效\n\n48 小时内发货。\n",
        encoding="utf-8",
    )
    yield path
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


async def test_ingest_creates_pending_then_vectorizes(sqlite_session, sample_doc):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()

    report = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder
    )

    assert isinstance(report, IngestReport)
    assert report.docs == 1
    assert report.chunks_total == report.chunks_created == 2
    assert report.vectors_written == 2
    assert await repo.count_by_status() == {"pending": 0, "done": 2}
    assert await store.count() == 2


async def test_embed_text_uses_the_three_labelled_slots(sqlite_session, sample_doc):
    """嵌入的文本必须走 ``render_embed_text``（分类/问法/答案），不能只喂正文。"""
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()

    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    joined = "\n".join(text for batch in embedder.batches for text in batch)
    assert "分类：售后手册" in joined
    assert "问法：运费" in joined
    assert "答案：满 99 包邮。" in joined


async def test_ingest_is_idempotent_on_rerun(sqlite_session, sample_doc):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    first = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder
    )

    second = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder
    )

    assert (second.chunks_created, second.chunks_updated) == (0, 0)
    assert second.chunks_skipped == first.chunks_total
    assert second.vectors_written == 0
    assert await store.count() == first.chunks_total


async def test_changed_answer_resets_to_pending_and_is_revectorized(
    sqlite_session, sample_doc
):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    sample_doc.write_text(
        "# 售后手册\n\n## 运费\n\n满 199 元包邮。\n\n## 时效\n\n48 小时内发货。\n",
        encoding="utf-8",
    )
    report = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder
    )

    assert report.chunks_updated == 1
    assert report.vectors_written == 1
    assert await repo.count_by_status() == {"pending": 0, "done": 2}


# ---------------------------------------------------------------------------
# 中断与续跑（验收标准 2 的核心）
# ---------------------------------------------------------------------------


async def test_interruption_after_chunks_written_leaves_pending(sqlite_session, sample_doc):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()

    report = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder, vector_limit=1
    )

    assert report.chunks_created == 2
    assert report.vectors_written == 1
    assert await repo.count_by_status() == {"pending": 1, "done": 1}
    assert await store.count() == 1

    # 重跑：只补那 1 条，不重复建块
    second = await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder
    )
    assert second.chunks_created == 0
    assert second.vectors_written == 1
    assert await repo.count_by_status() == {"pending": 0, "done": 2}
    assert await store.count() == 2


async def test_vectors_written_twice_do_not_duplicate(sqlite_session, sample_doc):
    from sqlalchemy import text

    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    # 模拟「Milvus 写成功但回填丢失」：状态被改回 pending
    sqlite_session.execute(
        text("UPDATE knowledge_chunks SET vectorize_status='pending', vector_id=NULL")
    )
    sqlite_session.commit()
    before = store.upsert_calls

    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    assert store.upsert_calls > before  # 确实重写了
    assert await store.count() == 2  # 但同主键覆盖，行数不变


async def test_vectorize_pending_with_zero_limit_writes_nothing(sqlite_session, sample_doc):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    assert await vectorize_pending(chunk_repo=repo, vector_store=store, embedder=embedder) == 0


# ---------------------------------------------------------------------------
# 结构：前后块指针
# ---------------------------------------------------------------------------


async def test_prev_next_links_follow_document_order(sqlite_session, sample_doc):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()

    await ingest_markdown(
        [sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder, vector_limit=0
    )

    rows = await repo.list_pending()
    assert rows[0].prev_chunk_id is None
    assert rows[0].next_chunk_id == rows[1].id
    assert rows[1].prev_chunk_id == rows[0].id
    assert rows[1].next_chunk_id is None


async def test_links_are_not_crossed_between_documents(sqlite_session):
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    a = Path(__file__).resolve().parent / "_tmp_ch03_a.md"
    b = Path(__file__).resolve().parent / "_tmp_ch03_b.md"
    a.write_text("# 甲\n\n## 甲一\n\n内容一。\n", encoding="utf-8")
    b.write_text("# 乙\n\n## 乙一\n\n内容二。\n", encoding="utf-8")
    try:
        await ingest_markdown(
            [a, b], chunk_repo=repo, vector_store=store, embedder=embedder, vector_limit=0
        )
        rows = {r.section_path: r for r in await repo.list_pending()}
        assert rows["甲 / 甲一"].next_chunk_id is None
        assert rows["乙 / 乙一"].prev_chunk_id is None
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 存量 FAQ 与重建
# ---------------------------------------------------------------------------


async def test_ingest_faq_table_maps_rows_to_chunks(sqlite_session, fake_repositories):
    from app.db.models import Faq

    sqlite_session.add_all(
        [
            Faq(question="退货政策", answer="7 天无理由", category="售后"),
            Faq(question="运费怎么算", answer="满 99 包邮", category="物流"),
        ]
    )
    sqlite_session.commit()
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()

    report = await ingest_faq_table(
        faq_repo=fake_repositories.faq,
        chunk_repo=repo,
        vector_store=store,
        embedder=embedder,
        vector_limit=0,  # 只建块，方便检查元数据映射
    )

    assert report.chunks_total == 2
    rows = await repo.list_pending()
    assert {r.content_type for r in rows} == {"faq"}
    assert {r.questions for r in rows} == {"退货政策", "运费怎么算"}
    assert {r.category for r in rows} == {"售后", "物流"}
    assert all(r.section_path.startswith("平台FAQ / ") for r in rows)


async def test_rebuild_clears_vectors_before_rows(sqlite_session, sample_doc):
    """先清向量再清表：反了会留下指向已删原文的孤儿向量。"""
    repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
    await ingest_markdown([sample_doc], chunk_repo=repo, vector_store=store, embedder=embedder)

    await rebuild(chunk_repo=repo, vector_store=store)

    assert store.reset_calls == 1
    assert await store.count() == 0
    assert await repo.count_by_status() == {"pending": 0, "done": 0}


async def test_multi_chunk_section_does_not_collapse_in_the_database(sqlite_session):
    """同一个 section 切成的多块必须是多行 —— 不能被自然键合并成一行。

    这条是**真事故的回填**：大表格被切成两块，两块共用
    ``(content_type, section_path, questions)``，于是第二块覆盖了第一块，
    表格后半截静默消失；建库报告只显示 ``created`` 比 ``total`` 少，不报错。
    修法是切分器给同节的第 2、3…块在 ``section_path`` 后加 ``#n``。
    """
    path = Path(__file__).resolve().parent / "_tmp_ch03_table.md"
    rows = "\n".join(f"| P{i} | {i} 元 |" for i in range(1, 26))
    path.write_text(
        f"# 价格表\n\n## 运费\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}\n", encoding="utf-8"
    )
    try:
        repo, store, embedder = _repo(sqlite_session), _FakeStore(), _FakeEmbedder()
        report = await ingest_markdown(
            [path],
            chunk_repo=repo,
            vector_store=store,
            embedder=embedder,
            max_chars=300,
            overlap=0,
            vector_limit=0,
        )
    finally:
        path.unlink(missing_ok=True)

    assert report.chunks_total == report.chunks_created > 1
    rows_in_db = await repo.list_pending()
    assert len(rows_in_db) == report.chunks_total
    # 两个块都还在，没有任何一行被覆盖掉
    assert all("| 商品 | 价格 |" in row.answer for row in rows_in_db)
