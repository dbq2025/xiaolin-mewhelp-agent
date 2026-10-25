"""按意图打印 Langfuse token / 成本排行。

用法：
    uv run python scripts/intent_cost.py --days 7
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.observability.intent_cost import (  # noqa: E402
    LangfuseNotConfigured,
    fetch_intent_costs,
)


def _render(items: list) -> str:
    if not items:
        return "（所选时间窗口内没有带 intent 标签的 trace）"
    lines = [f"{'意图':<16}{'观测组':>6}{'tokens':>10}{'成本(USD)':>12}"]
    lines.append("-" * 46)
    for item in items:
        lines.append(
            f"{item.intent:<16}{item.observations:>6}{item.tokens:>10}{item.cost:>12.6f}"
        )
    lines.append("-" * 46)
    lines.append(
        f"{'合计':<16}{'':>6}{sum(item.tokens for item in items):>10}"
        f"{sum(item.cost for item in items):>12.6f}"
    )
    return "\n".join(lines)


async def _main(days: int) -> int:
    settings = get_settings()
    try:
        items = await fetch_intent_costs(settings, days=days)
    except LangfuseNotConfigured as exc:
        print(f"未配置 Langfuse：{exc}", file=sys.stderr)
        print("请在 .env 设置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL。", file=sys.stderr)
        return 1
    print(_render(items))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按意图统计 Langfuse token 花销")
    parser.add_argument("--days", type=int, default=7, help="回看天数（默认 7）")
    args = parser.parse_args(argv)
    return asyncio.run(_main(max(1, args.days)))


if __name__ == "__main__":
    raise SystemExit(main())
