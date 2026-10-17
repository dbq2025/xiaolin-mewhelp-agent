"""打印低置信度问题池里最近的若干条（验收 4 的观察窗口）。

    uv run --directory MemHelp-code python scripts/show_low_confidence.py
    uv run --directory MemHelp-code python scripts/show_low_confidence.py --limit 5 --source self_check

为什么单独一个脚本而不是让人去写 SQL：验收 4 要求「拒答 + 入池」都能看见，
而这个仓库的约定是「查询一律在 repositories」（表结构改了，脚本会跟着报错，
而不是悄悄查一张不存在的表）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402


async def _run(limit: int, source: str | None) -> int:
    settings = get_settings()
    engine = create_engine_from(settings)
    factory = session_factory(engine)
    try:
        with factory() as session:
            repositories = Repositories.from_session(session)
            rows = await repositories.low_confidence.list_recent(limit * 5 if source else limit)
            counts = await repositories.low_confidence.count_by_source()
    finally:
        engine.dispose()

    if source:
        rows = [row for row in rows if row.source == source][:limit]

    if not rows:
        print("池子是空的 —— 先问一句知识库里没有的问题试试。")
        return 0

    print(f"{'id':>5}  {'source':<18} {'created_at':<20} 原话 / 原因")
    print("-" * 100)
    for row in rows:
        created = row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "-"
        question = row.raw_question.replace("\n", " ")[:38]
        reason = (row.reason or "-").replace("\n", " ")[:34]
        print(f"{row.id:>5}  {row.source:<18} {created:<20} {question}  ←  {reason}")

    print("\n按入口统计：", ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="查看低置信度问题池")
    parser.add_argument("--limit", type=int, default=10, help="显示多少条（默认 10）")
    parser.add_argument(
        "--source",
        default=None,
        choices=["retrieval_low_conf", "self_check", "user_feedback"],
        help="只看某个入池入口",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args.limit, args.source))
    except Exception as exc:  # noqa: BLE001 - 脚本要把原因打给人看
        print(f"读取失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
