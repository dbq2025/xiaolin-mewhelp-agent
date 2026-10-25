"""手动跑一轮 ch09 评估并写入 ``eval_runs``。

等价于 ``POST /api/eval-runs/run``，给没有起服务 / 想跑两轮对比时用：

    uv run python scripts/run_eval_round.py --trigger 手动
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.db.repositories import Repositories
from app.db.session import create_engine_from, session_factory
from app.eval_runs import run_evaluation


async def _main(trigger: str) -> int:
    settings = get_settings()
    engine = create_engine_from(settings)
    session = session_factory(engine)()
    try:
        repositories = Repositories.from_session(session)
        run_id = await run_evaluation(
            repositories=repositories, trigger=trigger, settings=settings
        )
    finally:
        session.close()
        engine.dispose()
    print(f"eval_runs 已写入：id={run_id} trigger={trigger}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑一轮 ch04 评估集并写入 eval_runs")
    parser.add_argument("--trigger", choices=("定时", "手动"), default="手动")
    args = parser.parse_args(argv)
    return asyncio.run(_main(args.trigger))


if __name__ == "__main__":
    sys.exit(main())
