"""对话挖知识的定时任务入口（CLI + 外部调度器）。

## 用法

```powershell
# 挖一批（默认 8 个会话一批）
uv run --directory MemHelp-code python scripts/mine_conversations.py --limit 30

# 只重跑去重入库（抽取已跑完、阈值调过了）
uv run --directory MemHelp-code python scripts/mine_conversations.py --dedupe-only

# 看暂存表与入库情况
uv run --directory MemHelp-code python scripts/mine_conversations.py --status

# 重挖（默认跳过已挖过的会话）
uv run --directory MemHelp-code python scripts/mine_conversations.py --force

# 注册为 Windows 计划任务（每天 03:00）
schtasks /create /tn "MewHelpMineKnowledge" /sc daily /st 03:00 ^
  /tr "uv run --directory MemHelp-code python scripts/mine_conversations.py --limit 200"
```

## 为什么是 CLI 而不是常驻定时器

定时任务的价值是「按时跑一次、失败能重跑、日志可查」。把调度交给操作系统的
计划任务/cron，web 进程就只负责服务请求 —— 两种职责分开，重启 web 不会顺带
重跑一遍挖矿，沙箱里也容易演示（跑一次看一次结果）。

## 幂等

已经挖过的会话（`source_ref` 已在暂存表）默认跳过；`--force` 才重挖。
每次运行的批次号是 `ch03-<时间戳>-<序号>`，失败批号会打印出来。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 让脚本能 import app.*（它从 scripts/ 下运行，不在 app 包内）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402
from app.knowledge.embedder import get_embedder  # noqa: E402
from app.knowledge.ingest import vectorize_pending  # noqa: E402
from app.knowledge.mining import dedupe_and_ingest, mine_conversations  # noqa: E402
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402
from app.llm import get_chat_model  # noqa: E402

logger = logging.getLogger("mine_conversations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从历史客服对话挖问答对并入库")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个会话")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="每批喂给 LLM 的会话数（默认取配置）"
    )
    parser.add_argument("--force", action="store_true", help="已挖过的会话也重挖")
    parser.add_argument(
        "--dedupe-only", action="store_true", help="不调 LLM，只对暂存表重跑去重入库"
    )
    parser.add_argument("--status", action="store_true", help="只打印状态，不写库")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine_from(settings)
    factory = session_factory(engine)

    if args.status:
        with factory() as session:
            repositories = Repositories.from_session(session)
            staging = await repositories.qa_staging.count_by_status()
            knowledge = await repositories.knowledge.count_by_status()
        print(f"暂存表：{staging or '（空）'}")
        print(f"知识库：{knowledge}")
        return 0

    # 挖矿要调 LLM（现有 OPENAI_*），去重要调嵌入（需要 EMBEDDING_API_KEY）。
    settings.require_embedding_key()
    llm = get_chat_model(settings)
    embedder = get_embedder(settings)
    vector_store = MilvusVectorStore(
        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
    )

    with factory() as session:
        repositories = Repositories.from_session(session)

        if not args.dedupe_only:
            report = await mine_conversations(
                repositories=repositories,
                llm=llm,
                embedder=embedder,
                batch_size=args.batch_size or settings.qa_batch_size,
                limit=args.limit,
                force=args.force,
            )
            print(
                f"抽取：batches={report.batches} pairs={report.pairs_written} "
                f"skipped_sessions={report.skipped_sessions}"
            )
            if report.failed_batches:
                print(f"失败的批次：{report.failed_batches}", file=sys.stderr)

        dedupe = await dedupe_and_ingest(
            repositories=repositories,
            embedder=embedder,
            threshold=settings.qa_dedupe_threshold,
        )
        print(f"去重入库：kept={dedupe.kept} discarded={dedupe.discarded} ingested={dedupe.ingested}")

        written = await vectorize_pending(
            chunk_repo=repositories.knowledge,
            vector_store=vector_store,
            embedder=embedder,
        )
        print(f"补向量：vectors={written}")

        failed = bool(locals().get("report") and report.failed_batches)

    await embedder.aclose()
    engine.dispose()
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - CLI 要把原因打给人看
        print(f"挖矿失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
