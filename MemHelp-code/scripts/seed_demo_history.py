"""把 ``sql/ch03-seed-history.sql`` 灌进 MySQL（演示历史会话）。

## 为什么不直接在 README 里写 mysql 客户端命令

本机的 ``mysql`` 客户端不一定在 PATH 里，而且容易踩字符集问题（latin1 客户端把中文
double-encode）。用项目自己的连接配置（``app.config`` 的 ``database_url``）执行，
「连哪个库、用什么字符集」与运行时**完全一致**，少一个会漂移的地方。

脚本本身幂等：SQL 里先 ``DELETE ... WHERE session_id LIKE 'demo-ch03-%'`` 再插入。

## 用法

```powershell
uv run --directory MemHelp-code python scripts/seed_demo_history.py
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能 import app.*（它从 scripts/ 下运行，不在 app 包内）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text  # noqa: E402

from app.config import get_settings  # noqa: E402

DEFAULT_SQL = Path(__file__).resolve().parents[2] / "sql" / "ch03-seed-history.sql"


def _statements(sql: str) -> list[str]:
    """按 ``;`` 切成单条语句，**按行**去掉注释后再判断空段。

    不能「整段以 ``--`` 开头的就丢掉」：文件开头是两行注释 + 紧跟
    ``SET NAMES utf8mb4;``，整段丢弃会让这条语句永远不执行（现在靠连接串的
    charset 顶着，下一个真实语句就没这么幸运了）。
    """
    statements: list[str] = []
    for chunk in sql.split(";"):
        body = "\n".join(
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ).strip()
        if body:
            statements.append(body)
    return statements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="灌入 ch03 演示历史会话")
    parser.add_argument("--sql", default=str(DEFAULT_SQL), help="种子 SQL 路径")
    args = parser.parse_args(argv)

    path = Path(args.sql)
    if not path.is_file():
        print(f"找不到种子 SQL：{path}", file=sys.stderr)
        return 1

    statements = _statements(path.read_text(encoding="utf-8"))
    engine = create_engine(get_settings().database_url())
    try:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
            conversations = conn.execute(
                text("SELECT COUNT(*) FROM conversations WHERE session_id LIKE 'demo-ch03-%'")
            ).scalar()
            messages = conn.execute(
                text(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id IN "
                    "(SELECT id FROM conversations WHERE session_id LIKE 'demo-ch03-%')"
                )
            ).scalar()
    finally:
        engine.dispose()

    print(f"已灌入 demo-ch03-* 会话：conversations={conversations} messages={messages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
