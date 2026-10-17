"""建库 CLI —— 离线建库的唯一入口。

## 用法

```powershell
# 全量建库（文档 + 存量 faq 表）
uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table

# 只看状态（不写库）
uv run --directory MemHelp-code python scripts/build_knowledge.py --status

# 演示「故意中断」：只补前 5 块的向量
uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table --rebuild --max-chunks 5

# 再跑一次，把剩下的 pending 补齐
uv run --directory MemHelp-code python scripts/build_knowledge.py --docs knowledge --faq-table
```

## 为什么 `--status` 单独存在

验收标准 2 的判据是「重跑后漏向量化的块被捡起补齐」。肉眼看不到 MySQL 的
`vectorize_status` 与 Milvus 的行数，所以把两侧计数并排打出来：
`done` 与 Milvus 行数相等且 `pending=0` 才算补齐。

## 为什么先校验密钥

切分几百块、写半天 MySQL，最后因为 401 白跑一遍 —— 不如在开跑前就报错。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让脚本能 import app.*（它从 scripts/ 下运行，不在 app 包内）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402
from app.errors import redact  # noqa: E402
from app.knowledge.embedder import get_embedder  # noqa: E402
from app.knowledge.ingest import (  # noqa: E402
    IngestReport,
    ingest_faq_table,
    ingest_markdown,
    rebuild,
    reindex,
    vectorize_pending,
)
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402

logger = logging.getLogger("build_knowledge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线建库：Markdown/FAQ → MySQL(pending) → Milvus → 回填 done"
    )
    parser.add_argument("--docs", type=str, default=None, help="Markdown 目录或单个文件")
    parser.add_argument(
        "--faq-table", action="store_true", help="把存量 faq 表也灌进知识库"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先清空 Milvus 集合与 knowledge_chunks（会丢掉挖矿产物）",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help=(
            "只重建向量索引：drop 集合并把所有块打回 pending，**保留原文**"
            "（含 mined）。schema 升级（如 ch04 的 BM25 字段）时用这个"
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="本次最多补多少块的向量（演示中断用）",
    )
    parser.add_argument("--status", action="store_true", help="只打印两侧计数，不写库")
    return parser


def summarise(db_counts: dict[str, int], milvus_count: int) -> str:
    """把 MySQL 状态计数与 Milvus 行数拼成一行人话，并给出是否一致。"""
    pending = int(db_counts.get("pending", 0))
    done = int(db_counts.get("done", 0))
    consistent = pending == 0 and done == milvus_count
    verdict = "一致" if consistent else "不一致"
    return (
        f"knowledge_chunks: pending={pending} done={done} | milvus={milvus_count} "
        f"→ {verdict}"
        + ("" if consistent else "（pending 未清零或与 Milvus 行数不等）")
    )


def remaining_budget(max_chunks: int | None, written: int) -> int | None:
    """``--max-chunks`` 是**整次运行**的预算，不是每个建库步骤各一份。

    文档建库与 FAQ 建库都要向量化；若各给一次预算，`--max-chunks 5` 会写成 10 条，
    与「本次最多补 5 条」的直觉不符，中断演示的预期数字也会跟着错。
    """
    if max_chunks is None:
        return None
    return max(max_chunks - written, 0)


def _collect_docs(target: str) -> list[Path]:
    path = Path(target)
    if path.is_dir():
        return sorted(path.glob("*.md"))
    return [path]


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine_from(settings)
    factory = session_factory(engine)

    # 先确认密钥（--status 除外）：别等连完 Milvus、切完文档，才因为 401 白跑一遍。
    if not args.status:
        settings.require_embedding_key()

    vector_store = MilvusVectorStore(
        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
    )
    # 重建路径**不能**先 ensure：旧集合（ch03 的 id+vector）会被结构自检拦下，
    # 而那正是本次要替换掉的东西。重建分支会自己 drop 再按新 schema 建。
    if not (args.rebuild or args.reindex):
        await vector_store.ensure_collection()

    if args.status:
        with factory() as session:
            counts = await Repositories.from_session(session).knowledge.count_by_status()
        print(summarise(counts, await vector_store.count()))
        return 0

    embedder = get_embedder(settings)

    with factory() as session:
        repositories = Repositories.from_session(session)

        if args.rebuild and args.reindex:
            print("--rebuild 与 --reindex 不能同时用：一个会清表，一个要保留原文")
            return 1

        if args.rebuild:
            await rebuild(chunk_repo=repositories.knowledge, vector_store=vector_store)
            print("已清空 Milvus 集合与 knowledge_chunks")

        if args.reindex:
            pending = await reindex(
                chunk_repo=repositories.knowledge, vector_store=vector_store
            )
            print(f"已重建集合，{pending} 块打回 pending（原文保留），接下来重新向量化")

        report = IngestReport()
        written_total = 0
        if args.docs:
            docs = _collect_docs(args.docs)
            if not docs:
                print(f"没有找到 Markdown 文件：{args.docs}")
                return 1
            report = await ingest_markdown(
                docs,
                chunk_repo=repositories.knowledge,
                vector_store=vector_store,
                embedder=embedder,
                max_chars=settings.chunk_max_chars,
                overlap=settings.chunk_overlap,
                vector_limit=remaining_budget(args.max_chunks, written_total),
            )
            written_total += report.vectors_written
            print(
                f"文档建库：docs={report.docs} total={report.chunks_total} "
                f"created={report.chunks_created} updated={report.chunks_updated} "
                f"skipped={report.chunks_skipped} pruned={report.chunks_pruned} "
                f"vectors={report.vectors_written}"
            )

        if args.faq_table:
            faq_report = await ingest_faq_table(
                faq_repo=repositories.faq,
                chunk_repo=repositories.knowledge,
                vector_store=vector_store,
                embedder=embedder,
                vector_limit=remaining_budget(args.max_chunks, written_total),
            )
            written_total += faq_report.vectors_written
            print(
                f"FAQ 表建库：total={faq_report.chunks_total} "
                f"created={faq_report.chunks_created} updated={faq_report.chunks_updated} "
                f"vectors={faq_report.vectors_written}"
            )

        if not args.docs and not args.faq_table:
            # 只带 --max-chunks 时：单纯补 pending
            written = await vectorize_pending(
                chunk_repo=repositories.knowledge,
                vector_store=vector_store,
                embedder=embedder,
                limit=args.max_chunks,
            )
            print(f"补向量：vectors={written}")

        counts = await repositories.knowledge.count_by_status()

    print(summarise(counts, await vector_store.count()))
    await embedder.aclose()
    engine.dispose()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    import asyncio

    try:
        return asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - CLI 要把原因打给人看
        # 过一遍 redact：嵌入服务的报错可能回显请求头片段（含密钥）。
        print(f"建库失败：{redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
