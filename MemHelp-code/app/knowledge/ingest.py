"""离线建库编排 —— 双写：MySQL 原文优先，Milvus 向量随后。

## 顺序是有讲究的

```
写 MySQL(pending) → 回填 prev/next → 写 Milvus → 回填 vector_id / done
```

**先 MySQL 后 Milvus**：原文是权威源，块没进库之前不该有向量（否则向量指向一个
不存在的 id）。**先写 Milvus 再回填**：反过来的话，中断会留下「状态说 done、
Milvus 里却没有」的块 —— 那种块永远拣不出来，续跑也补不上。

## 三种中断点都收敛

1. 块写完、向量未写 → ``pending``，下次拣起；
2. 向量写完、回填未做 → 仍是 ``pending``，下次用同一主键覆盖写（不产生重复）；
3. 两库都写成功 → ``done``，下次跳过。

这就是「按主键幂等，挂了能重跑」的全部含义，也是验收标准 2。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.db.repositories import FaqRepository, KnowledgeChunkRepository
from app.knowledge.chunker import render_bm25_text, render_embed_text, split_markdown
from app.knowledge.embedder import Embedder
from app.knowledge.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "IngestReport",
    "content_type_for",
    "ingest_faq_table",
    "ingest_markdown",
    "rebuild",
    "vectorize_ids",
    "vectorize_pending",
]

#: 文件名 → content_type 的约定。写在这里而不是散在 CLI 里，是为了让
#: 「文档叫什么名字」与「落库的元数据」只有一处映射。
_FILENAME_TYPES: tuple[tuple[str, str], ...] = (
    ("退货政策", "policy"),
    ("商品FAQ", "faq"),
    ("售后手册", "manual"),
)

#: 存量 FAQ 表灌进知识库时的章节路径前缀。
FAQ_SECTION_PREFIX = "平台FAQ"


@dataclass
class IngestReport:
    """一次建库的可打印结果。"""

    docs: int = 0
    chunks_total: int = 0
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_skipped: int = 0
    chunks_pruned: int = 0
    vectors_written: int = 0


def content_type_for(path: Path) -> str:
    """按文件名猜 ``content_type``；认不出来就当手册（最宽松的一类）。"""
    for marker, content_type in _FILENAME_TYPES:
        if marker in path.stem:
            return content_type
    return "manual"


class _Row:
    """给 ``upsert_chunk`` 用的最小行形状。

    切分器产出的 ``Chunk`` 也能直接喂进去；这里再包一层是为了让 **FAQ 表**
    这种「本来就是结构化行」的来源复用同一条写库路径。
    """

    __slots__ = (
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "ordinal",
    )

    def __init__(
        self,
        *,
        category: str,
        questions: str,
        answer: str,
        section_path: str,
        content_type: str,
        is_key_clause: bool,
        ordinal: int,
    ) -> None:
        self.category = category
        self.questions = questions
        self.answer = answer
        self.section_path = section_path
        self.content_type = content_type
        self.is_key_clause = is_key_clause
        self.ordinal = ordinal


# ---------------------------------------------------------------------------
# 向量化（续跑的唯一入口）
# ---------------------------------------------------------------------------


async def vectorize_pending(
    *,
    chunk_repo: KnowledgeChunkRepository,
    vector_store: VectorStore,
    embedder: Embedder,
    limit: int | None = None,
    batch_size: int = 32,
) -> int:
    """把 ``pending``（或 ``vector_id`` 为空）的块补上向量，返回本次写入条数。

    :param limit: 本次最多处理多少块。``0`` 表示一块都不处理（演示「只建块」用），
        ``None`` 表示一直补到没有 pending 为止。
    """
    if limit is not None and limit <= 0:
        return 0

    written = 0
    while True:
        remaining = None if limit is None else limit - written
        if remaining is not None and remaining <= 0:
            break

        take = batch_size if remaining is None else min(batch_size, remaining)
        rows = await chunk_repo.list_pending(limit=take)
        if not rows:
            break

        vectors = await embedder.embed([render_embed_text(row) for row in rows])
        if len(vectors) != len(rows):
            # 嵌入器返回条数不对时**必须停**：继续下去会让 upsert 与 mark_done 用
            # 不同的长度，结果是「MySQL 说 done、Milvus 里少几条」——正是续跑机制
            # 应该消灭的那种半对齐状态。
            raise RuntimeError(
                f"嵌入返回条数不符：期望 {len(rows)}，实际 {len(vectors)}"
            )
        ids = [str(row.id) for row in rows]

        # 先写向量，再回填状态。反了就永远补不上（见模块 docstring）。
        #
        # ch04 起写入的是 VectorRecord：除了稠密向量，还要把 BM25 原文与品类
        # 一起落到 Milvus（Milvus 从此是「检索索引副本」，MySQL 仍是原文权威）。
        await vector_store.upsert(
            [
                VectorRecord(
                    id=row_id,
                    dense=vector,
                    text=render_bm25_text(row),
                    category=row.category,
                    content_type=row.content_type,
                )
                for row_id, vector, row in zip(ids, vectors, rows)
            ]
        )
        await chunk_repo.mark_done([(row.id, str(row.id)) for row in rows])
        written += len(rows)
        logger.info("已向量化 %d 块（本次累计 %d）", len(rows), written)

    return written


async def vectorize_ids(
    chunk_ids: Sequence[int],
    *,
    chunk_repo: KnowledgeChunkRepository,
    vector_store: VectorStore,
    embedder: Embedder,
    batch_size: int = 32,
) -> int:
    """只给指定块补向量，返回写入条数（审核通过后的增量路径）。

    与 :func:`vectorize_pending` 共用同一套「先写 Milvus 再回填」顺序；区别只是
    候选集来自主键列表而不是 ``pending`` 状态 —— 审核通过的块刚 ``upsert_chunk``
    完，可能连 pending 都还没被后台任务扫到，不能等下一轮全量续跑。
    """
    ids = [int(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return 0

    written = 0
    for start in range(0, len(ids), max(1, batch_size)):
        batch_ids = ids[start : start + batch_size]
        rows = await chunk_repo.get_many(batch_ids)
        if not rows:
            continue

        vectors = await embedder.embed([render_embed_text(row) for row in rows])
        if len(vectors) != len(rows):
            raise RuntimeError(
                f"嵌入返回条数不符：期望 {len(rows)}，实际 {len(vectors)}"
            )

        await vector_store.upsert(
            [
                VectorRecord(
                    id=str(row.id),
                    dense=vector,
                    text=render_bm25_text(row),
                    category=row.category,
                    content_type=row.content_type,
                )
                for vector, row in zip(vectors, rows)
            ]
        )
        await chunk_repo.mark_done([(row.id, str(row.id)) for row in rows])
        written += len(rows)
        logger.info("已向量化 %d 块（指定 id，累计 %d）", len(rows), written)

    return written


# ---------------------------------------------------------------------------
# 文档建库
# ---------------------------------------------------------------------------


async def _ingest_chunks(
    *,
    chunks: Sequence[_Row],
    chunk_repo: KnowledgeChunkRepository,
    report: IngestReport,
) -> list[int]:
    """写块并回填同篇内的前后指针（外键要求 id 先存在，所以分两步）。"""
    ids: list[int] = []
    for chunk in chunks:
        chunk_id, outcome = await chunk_repo.upsert_chunk(chunk)
        ids.append(chunk_id)
        if outcome == "created":
            report.chunks_created += 1
        elif outcome == "updated":
            report.chunks_updated += 1
        else:
            report.chunks_skipped += 1
    # **总是**回填指针（哪怕只有一块）：单块文档也要把上一轮的 next/prev 清掉，
    # 否则改完文档后尾块还指着已被删除的旧块。
    await chunk_repo.link_neighbours(ids)
    return ids


async def ingest_markdown(
    paths: Iterable[Path],
    *,
    chunk_repo: KnowledgeChunkRepository,
    vector_store: VectorStore,
    embedder: Embedder,
    max_chars: int = 600,
    overlap: int = 80,
    vector_limit: int | None = None,
) -> IngestReport:
    """切分 Markdown 文档、写库并向量化。

    :param vector_limit: 本次最多补多少块的向量（演示中断用）；``None`` 表示全补。
    """
    report = IngestReport()
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        content_type = content_type_for(path)
        chunks = split_markdown(
            text,
            content_type=content_type,
            source=path.name,
            max_chars=max_chars,
            overlap=overlap,
        )
        rows = [
            _Row(
                category=chunk.category,
                questions=chunk.questions,
                answer=chunk.answer,
                section_path=chunk.section_path,
                content_type=chunk.content_type,
                is_key_clause=chunk.is_key_clause,
                ordinal=chunk.ordinal,
            )
            for chunk in chunks
        ]
        ids = await _ingest_chunks(chunks=rows, chunk_repo=chunk_repo, report=report)

        report.chunks_total += len(ids)
        report.docs += 1
        logger.info("%s：%d 块（%s）", path.name, len(ids), content_type)

        # 增量重建：这篇文档里**已经不存在**的旧块要清掉，连同它们的向量。
        # 不清理的话，改过的章节会留下旧答案并继续被检索到 —— 那是面向客户的错答，
        # 而且没有任何报错。
        pruned = await _prune_stale_sections(
            rows=rows, keep_ids=ids, chunk_repo=chunk_repo, vector_store=vector_store
        )
        report.chunks_pruned += pruned

    report.vectors_written = await vectorize_pending(
        chunk_repo=chunk_repo,
        vector_store=vector_store,
        embedder=embedder,
        limit=vector_limit,
    )
    return report


async def _prune_stale_sections(
    *,
    rows: Sequence[_Row],
    keep_ids: Sequence[int],
    chunk_repo: KnowledgeChunkRepository,
    vector_store: VectorStore,
) -> int:
    """删掉同篇文档里本次没有生成的旧块（先删向量、再删行）。"""
    if not rows:
        return 0

    root = rows[0].section_path.split(" / ")[0]
    keep = set(keep_ids)
    stale = [chunk.id for chunk in await chunk_repo.list_by_scope(root) if chunk.id not in keep]
    if not stale:
        return 0

    # 先删向量：反了的话，中断会留下指向已删原文的孤儿向量。
    await vector_store.delete([str(chunk_id) for chunk_id in stale])
    deleted = await chunk_repo.delete_ids(stale)
    logger.info("清理已不存在的旧块 %d 条（%s）", deleted, root)
    return deleted


async def ingest_faq_table(
    *,
    faq_repo: FaqRepository,
    chunk_repo: KnowledgeChunkRepository,
    vector_store: VectorStore,
    embedder: Embedder,
    vector_limit: int | None = None,
) -> IngestReport:
    """把存量 ``faq`` 表灌进知识库（``content_type='faq'``，问法就是原问题的问法）。"""
    report = IngestReport(docs=1)
    rows = [
        _Row(
            category=hit.category,
            questions=hit.question,
            answer=hit.answer,
            section_path=f"{FAQ_SECTION_PREFIX} / {hit.category}",
            content_type="faq",
            is_key_clause=False,
            ordinal=index,
        )
        for index, hit in enumerate(await faq_repo.list_all())
    ]
    ids = await _ingest_chunks(chunks=rows, chunk_repo=chunk_repo, report=report)
    report.chunks_total = len(ids)
    report.vectors_written = await vectorize_pending(
        chunk_repo=chunk_repo,
        vector_store=vector_store,
        embedder=embedder,
        limit=vector_limit,
    )
    return report


# ---------------------------------------------------------------------------
# 重建
# ---------------------------------------------------------------------------


async def rebuild(
    *, chunk_repo: KnowledgeChunkRepository, vector_store: VectorStore
) -> None:
    """清空两库（**先清向量，再清表**）。

    顺序反了的话，如果进程在两步之间被杀，会留下指向已删原文的孤儿向量 ——
    检索命中一个不存在的 id，只能靠「跳过未知 id」兜底，而那会静默少结果。
    """
    await vector_store.reset()
    await chunk_repo.truncate()


async def reindex(
    *, chunk_repo: KnowledgeChunkRepository, vector_store: VectorStore
) -> int:
    """重建向量索引，**保留全部原文**（含挖矿产物）。

    与 :func:`rebuild` 的区别：不 ``truncate``，而是把每块打回 ``pending`` 并清掉
    ``vector_id``，之后 ``vectorize_pending`` 会把它们全部重新写进新集合。

    为什么需要这条路：ch04 的 BM25 字段与 analyzer 只能在**建集合时**声明，
    所以集合必须 drop 重建；但 ``knowledge_chunks`` 里可能躺着从真实对话挖出来的
    问答对（``content_type='mined'``），清表就把它们一起删了 —— 那是最贵的数据
    （要花 LLM 费用重挖，且内容未必一样）。

    :return: 被打回 ``pending`` 的块数。
    """
    await vector_store.reset()
    return await chunk_repo.reset_status()
