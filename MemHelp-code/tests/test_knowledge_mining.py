"""对话挖知识：分批抽取 → 暂存 → 整体去重 → 入库。

## 为什么要分两段

- **分批抽取**：一次喂太多会话，模型会把 A 会话的答案接到 B 会话的问题上（串味）。
- **整体去重**：逐批去重会让「第 2 批里与第 1 批同义的问法」重复入库。
  所以抽取按批写暂存表，去重在**全部批次跑完之后**对全量做。

本文件的假对象：假 LLM（按 prompt 里的 `### 会话 <id>` 标记回填来源）、
假嵌入器（同话题同向量）、真仓库（SQLite 内存库）。
"""

from __future__ import annotations

import re

import pytest

from app.db.repositories import Repositories
from app.knowledge.mining import (
    DedupeReport,
    ExtractedQa,
    ExtractionBatch,
    mine_conversations,
    normalise_question,
    dedupe_and_ingest,
)


class _FakeStructured:
    """按 prompt 里的 `### 会话 <id>` 标记回填 session_id —— 顺带钉住 prompt 格式。"""

    def __init__(self, pairs_factory) -> None:
        self._pairs_factory = pairs_factory

    async def ainvoke(self, messages):
        # messages 是 [SystemMessage]；直接读 content，**不要** str() 整条
        # （repr 会把换行变成字面 \n，正则就会把后续内容一起吃进 session_id）。
        first = messages[0] if isinstance(messages, (list, tuple)) else messages
        text = getattr(first, "content", str(first))
        match = re.search(r"### 会话 (\S+)", text)
        session_id = match.group(1) if match else ""
        return ExtractionBatch(pairs=self._pairs_factory(session_id))


class _FakeLlm:
    def __init__(self, pairs_factory) -> None:
        self._pairs_factory = pairs_factory
        self.calls = 0
        self.methods: list[str | None] = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is ExtractionBatch
        # 必须走 tool-calling：json_schema 会被 DeepSeek 网关 400 掉
        self.methods.append(kwargs.get("method"))
        self.calls += 1
        return _FakeStructured(self._pairs_factory)


class _TopicEmbedder:
    """同一话题 → 同一向量（cos=1）；不同话题 → 正交（cos=0）。"""

    async def embed(self, texts):
        return [
            [1.0, 0.0, 0.0, 0.0] if "邮费" in text else [0.0, 1.0, 0.0, 0.0]
            for text in texts
        ]

    async def warmup(self) -> None:
        return None


async def _seed_conversations(session, count: int) -> Repositories:
    repos = Repositories.from_session(session)
    for index in range(count):
        conversation_id = await repos.conversations.upsert_by_session(
            f"demo-ch03-{index:03d}", "u"
        )
        await repos.messages.append(
            conversation_id, role="user", content=f"邮费是多少？第{index}次"
        )
        await repos.messages.append(
            conversation_id, role="assistant", content="满 99 元包邮。"
        )
    return repos


def _one_pair(session_id: str) -> list[ExtractedQa]:
    return [ExtractedQa(session_id=session_id, question="邮费是多少", answer="满 99 元包邮")]


# ---------------------------------------------------------------------------
# 分批抽取
# ---------------------------------------------------------------------------


async def test_batches_are_sized_and_each_gets_a_batch_no(sqlite_session):
    repos = await _seed_conversations(sqlite_session, 10)
    llm = _FakeLlm(_one_pair)

    report = await mine_conversations(
        repositories=repos, llm=llm, embedder=_TopicEmbedder(), batch_size=4
    )

    assert report.batches == 3
    assert llm.calls == 3
    # 必须走 tool-calling（json_schema 会被 DeepSeek 网关 400 掉）
    assert set(llm.methods) == {"function_calling"}
    rows = await repos.qa_staging.list_by_status("extracted")
    assert len(rows) == 3
    assert len({row.batch_no for row in rows}) == 3


async def test_failed_batch_writes_nothing_and_other_batches_continue(sqlite_session):
    calls = {"n": 0}

    class _FlakyLlm(_FakeLlm):
        def with_structured_output(self, schema, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                class _Boom:
                    async def ainvoke(self, messages):
                        raise RuntimeError("upstream 500")

                return _Boom()
            return _FakeStructured(self._pairs_factory)

    repos = await _seed_conversations(sqlite_session, 10)
    report = await mine_conversations(
        repositories=repos, llm=_FlakyLlm(_one_pair), embedder=_TopicEmbedder(), batch_size=4
    )

    assert len(report.failed_batches) == 1
    rows = await repos.qa_staging.list_by_status("extracted")
    assert len(rows) == 2  # 第 1、3 批各 1 行，第 2 批 0 行
    assert report.failed_batches[0] not in {row.batch_no for row in rows}


async def test_empty_llm_result_writes_nothing_and_does_not_crash(sqlite_session):
    repos = await _seed_conversations(sqlite_session, 3)

    report = await mine_conversations(
        repositories=repos, llm=_FakeLlm(lambda sid: []), embedder=_TopicEmbedder(), batch_size=3
    )

    assert report.pairs_written == 0
    assert await repos.qa_staging.count_by_status() == {}


async def test_blank_and_duplicate_pairs_are_filtered_before_staging(sqlite_session):
    def pairs(session_id: str):
        return [
            ExtractedQa(session_id=session_id, question="   ", answer="x"),
            ExtractedQa(session_id=session_id, question="邮费是多少？", answer="满 99 包邮"),
            ExtractedQa(session_id=session_id, question="邮费是多少", answer="满 99 包邮"),
            ExtractedQa(session_id="不存在的会话", question="编造的", answer="编造的"),
        ]

    repos = await _seed_conversations(sqlite_session, 1)
    await mine_conversations(
        repositories=repos, llm=_FakeLlm(pairs), embedder=_TopicEmbedder(), batch_size=1
    )

    rows = await repos.qa_staging.list_by_status("extracted")
    assert len(rows) == 1
    assert rows[0].question == "邮费是多少？"
    assert rows[0].source_ref == "demo-ch03-000"


async def test_second_run_skips_already_mined_sessions(sqlite_session):
    repos = await _seed_conversations(sqlite_session, 2)

    # batch_size=1：保证两个会话都被挖到（假 LLM 每批只回一条，来源是本批第一个会话）
    await mine_conversations(
        repositories=repos, llm=_FakeLlm(_one_pair), embedder=_TopicEmbedder(), batch_size=1
    )
    before = await repos.qa_staging.count_by_status()
    assert before == {"extracted": 2}

    await mine_conversations(
        repositories=repos, llm=_FakeLlm(_one_pair), embedder=_TopicEmbedder(), batch_size=1
    )
    assert await repos.qa_staging.count_by_status() == before

    await mine_conversations(
        repositories=repos,
        llm=_FakeLlm(_one_pair),
        embedder=_TopicEmbedder(),
        batch_size=1,
        force=True,
    )
    assert await repos.qa_staging.count_by_status() != before


# ---------------------------------------------------------------------------
# 整体去重与入库
# ---------------------------------------------------------------------------


async def test_normalise_question_removes_space_punctuation_and_width(sqlite_session):
    assert normalise_question("邮费 是多少？") == normalise_question("邮费是多少")
    assert normalise_question("ＱＱ邮箱") == normalise_question("qq邮箱")


async def test_dedupe_is_literal_then_semantic(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    await repos.qa_staging.add_rows(
        [
            {"batch_no": "b1", "source_ref": "s1", "question": "邮费是多少", "answer": "满 99 包邮"},
            {"batch_no": "b1", "source_ref": "s2", "question": "邮费 是多少？", "answer": "满 99 包邮"},
            {"batch_no": "b2", "source_ref": "s3", "question": "寄回来要我付邮费吗", "answer": "满 99 包邮"},
            {"batch_no": "b2", "source_ref": "s4", "question": "发票怎么开？", "answer": "订单页申请开票"},
        ]
    )

    report = await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())

    assert isinstance(report, DedupeReport)
    assert (report.kept, report.discarded) == (2, 2)

    kept = await repos.qa_staging.list_by_status("kept")
    assert len(kept) == 2
    merged = next(row for row in kept if row.question.startswith("邮费"))
    # 「邮费 是多少？」是**字面**重复，直接丢弃、不并入；「寄回来要我付邮费吗」
    # 是语义同义，才并进问题行。所以是两行，不是三行。
    assert merged.question == "邮费是多少\n寄回来要我付邮费吗"

    assert await repos.knowledge.count_by_status() == {"pending": 2, "done": 0}


async def test_limit_applies_to_unmined_sessions_not_to_the_raw_list(sqlite_session):
    """``--limit`` 必须先过滤「已挖过」再截断。

    反过来的话，定时任务用 ``--limit 200`` 跑两次：第一次挖了前 200 个会话，
    第二次又挑到同一批（全都已知）→ 一个都不剩 → **永远不前进**。
    """
    repos = await _seed_conversations(sqlite_session, 3)
    llm = _FakeLlm(_one_pair)

    first = await mine_conversations(
        repositories=repos, llm=llm, embedder=_TopicEmbedder(), batch_size=1, limit=1
    )
    second = await mine_conversations(
        repositories=repos, llm=llm, embedder=_TopicEmbedder(), batch_size=1, limit=1
    )

    assert first.batches == 1 and second.batches == 1
    rows = await repos.qa_staging.list_by_status("extracted")
    assert {row.source_ref for row in rows} == {"demo-ch03-000", "demo-ch03-001"}


async def test_ingest_happens_before_rows_are_marked_kept(sqlite_session):
    """入库失败时，行必须仍是 ``extracted``，下次能被重试。

    先标 kept 再入库的话，入库中途失败（或进程被杀）会留下「状态说 kept、
    知识库里却没有」的行：普通重跑只读 extracted，它们永远补不回来。
    """
    import dataclasses

    repos = Repositories.from_session(sqlite_session)
    await repos.qa_staging.add_rows(
        [
            {"batch_no": "b1", "source_ref": "s1", "question": "邮费是多少", "answer": "满 99 包邮"},
            {"batch_no": "b1", "source_ref": "s2", "question": "能退吗", "answer": "七天无理由"},
        ]
    )

    class _FailingKnowledge:
        def __init__(self):
            self.calls = 0

        async def upsert_chunk(self, chunk):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("db gone")
            return 1, "created"

    broken = dataclasses.replace(repos, knowledge=_FailingKnowledge())

    with pytest.raises(RuntimeError, match="db gone"):
        await dedupe_and_ingest(repositories=broken, embedder=_TopicEmbedder())

    # 第一行已经入库但**没有**被标 kept；第二行同样还是 extracted —— 重跑会重做
    assert await repos.qa_staging.count_by_status() == {"extracted": 2}


async def test_dedupe_is_idempotent_when_no_extracted_rows(sqlite_session):
    repos = Repositories.from_session(sqlite_session)

    report = await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())

    assert (report.kept, report.discarded, report.ingested) == (0, 0, 0)


async def test_dedupe_compares_against_already_kept_rows(sqlite_session):
    """重复跑挖矿时，新抽出的问答必须与**已有的 kept 行**比较。

    只在本批内比较的话，每一轮都会把同样的知识再加一遍 —— 条数一轮轮涨，
    而每次看起来都「只新增了几条」。
    """
    repos = Repositories.from_session(sqlite_session)
    await repos.qa_staging.add_rows(
        [{"batch_no": "b1", "source_ref": "s1", "question": "邮费是多少", "answer": "满 99 包邮"}]
    )
    await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())
    await repos.knowledge.truncate()  # 模拟 --rebuild 之后
    assert await repos.knowledge.count_by_status() == {"pending": 0, "done": 0}

    # 第二轮又抽出同义问法（LLM 每轮措辞可能不同）
    await repos.qa_staging.add_rows(
        [
            {"batch_no": "b2", "source_ref": "s1", "question": "寄回去要我付邮费吗", "answer": "满 99 包邮"},
            {"batch_no": "b2", "source_ref": "s2", "question": "邮费 是多少？", "answer": "满 99 包邮"},
        ]
    )
    report = await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())

    assert report.kept == 0  # 两条都与已有 kept 行重复
    assert report.discarded == 2
    # 只重放了那一行已有的 kept，没有新增重复知识
    assert await repos.knowledge.count_by_status() == {"pending": 1, "done": 0}


async def test_dedupe_replays_already_kept_rows(sqlite_session):
    """``--rebuild`` 清空知识库后，``--dedupe-only`` 必须能把 kept 行放回去。

    挖矿产物的原始来源是暂存表；不重放的话，重建之后只能 `--force` 重挖
    （重新调 LLM，既不便宜也不保证抽得一样）。
    """
    repos = Repositories.from_session(sqlite_session)
    await repos.qa_staging.add_rows(
        [{"batch_no": "b1", "source_ref": "sess-r", "question": "邮费是多少", "answer": "满 99 包邮"}]
    )
    await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())
    assert await repos.knowledge.count_by_status() == {"pending": 1, "done": 0}

    # 模拟 --rebuild：知识库清空，但暂存表保留（行是 kept）
    await repos.knowledge.truncate()
    assert await repos.knowledge.count_by_status() == {"pending": 0, "done": 0}

    report = await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())

    assert report.ingested == 1
    assert await repos.knowledge.count_by_status() == {"pending": 1, "done": 0}


async def test_mined_chunks_carry_source_metadata(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    await repos.qa_staging.add_rows(
        [{"batch_no": "b1", "source_ref": "sess-x", "question": "邮费是多少", "answer": "满 99 包邮"}]
    )

    await dedupe_and_ingest(repositories=repos, embedder=_TopicEmbedder())

    rows = await repos.knowledge.list_pending()
    assert rows[0].content_type == "mined"
    assert rows[0].category == "对话挖掘"
    assert rows[0].section_path == "对话挖掘 / sess-x"
