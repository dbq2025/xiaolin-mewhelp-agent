"""对**已存在的** MySQL 库执行 ch02 幂等迁移。

## 为什么需要这个脚本（而不只是 .sql 文件）

``sql/ch02-migrate-existing.sql`` 里用了 ``PREPARE`` / ``EXECUTE``，
那是**服务端**预处理语句。想在一次连接里跑完它，客户端必须能执行多语句
（``CLIENT.MULTI_STATEMENTS``），并且要一路把结果集消费干净 —— 手工在
mysql 客户端里粘贴、或用不支持多语句的驱动，都会中途断掉。

所以这里用 PyMySQL 显式开启多语句、按结果集逐个消费，并且**用数据库自己的
``DATABASE()``**（连接时已选定库），不依赖脚本里写死库名。

## 用法

    # 复用既有库（默认连 127.0.0.1:3306，与 docker-compose 的既有实例一致）
    uv run --directory MemHelp-code python scripts/migrate_existing.py

    # 指定连接（或设 MYSQL_* 环境变量）
    uv run --directory MemHelp-code python scripts/migrate_existing.py \\
        --host 127.0.0.1 --port 3306 --user root --password root --database mewhelp

    # 只看会做什么，不实际执行
    uv run --directory MemHelp-code python scripts/migrate_existing.py --dry-run

脚本**可反复执行**（迁移本身幂等），已迁过的库再跑是空操作。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pymysql

#: 迁移脚本路径：仓库根的 sql/（本文件在 MemHelp-code/scripts/ 下）
MIGRATION_FILE = Path(__file__).resolve().parents[2] / "sql" / "ch02-migrate-existing.sql"


def _split_statements(sql: str) -> list[str]:
    """把 SQL 文本切成可逐条执行的语句。

    **为什么不用简单 ``split(";")``**：本迁移脚本里没有存储过程或字符串字面量里
    的分号，但为了不给自己埋雷，这里仍然按行过滤掉纯注释行，再按分号切。
    切出来的空片段与只剩注释的片段丢弃。
    """
    lines: list[str] = []
    for raw in sql.splitlines():
        stripped = raw.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(raw)
    body = "\n".join(lines)

    statements: list[str] = []
    for chunk in body.split(";"):
        cleaned = chunk.strip()
        if cleaned:
            statements.append(cleaned)
    return statements


def main() -> int:
    parser = argparse.ArgumentParser(description="对既有 MySQL 库执行 ch02 幂等迁移")
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", "root"))
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE", "mewhelp"))
    parser.add_argument("--file", default=str(MIGRATION_FILE))
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的语句")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"[ERR] 找不到迁移脚本：{path}", file=sys.stderr)
        return 2

    statements = _split_statements(path.read_text(encoding="utf-8"))
    print(f"[i] 迁移脚本：{path}")
    print(f"[i] 拆出 {len(statements)} 条语句")
    print(f"[i] 目标：{args.user}@{args.host}:{args.port}/{args.database}")

    if args.dry_run:
        for index, statement in enumerate(statements, 1):
            head = " ".join(statement.split())[:110]
            print(f"  {index:>2}. {head}")
        print("[i] --dry-run：未执行任何语句")
        return 0

    connection = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        autocommit=True,
    )

    try:
        with connection.cursor() as cursor:
            for index, statement in enumerate(statements, 1):
                head = " ".join(statement.split())[:90]
                try:
                    cursor.execute(statement)
                except pymysql.err.MySQLError as exc:
                    print(f"[ERR] 第 {index} 条失败：{head}", file=sys.stderr)
                    print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1

                # 消费该语句可能产生的所有结果集（SELECT 自检语句会返回行）
                while True:
                    if cursor.description:
                        for row in cursor.fetchall():
                            print(f"  [{index}] {row}")
                    if not cursor.nextset():
                        break

        # 迁移后的结构复核 —— 用 information_schema 直接看，不信脚本自检的输出
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COLUMN_NAME, IS_NULLABLE, COLUMN_TYPE
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'conversations' AND COLUMN_NAME = 'session_id'
                """
            )
            column = cursor.fetchone()

            cursor.execute(
                """
                SELECT COUNT(*) FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'conversations'
                  AND INDEX_NAME = 'uk_session_id' AND NON_UNIQUE = 0
                """
            )
            unique_key = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id IS NULL OR session_id = ''"
            )
            blanks = cursor.fetchone()[0]

        print()
        print("[i] 复核：")
        print(f"    conversations.session_id = {column}")
        print(f"    uk_session_id（唯一键） = {'存在' if unique_key else '缺失'}")
        print(f"    空 session_id 的行数 = {blanks}")

        ok = bool(column) and column[1] == "NO" and unique_key == 1 and blanks == 0
        print("[OK] 迁移完成，结构符合本章 ORM 的期望" if ok else "[WARN] 复核未全绿，请看上面的字段")
        return 0 if ok else 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
