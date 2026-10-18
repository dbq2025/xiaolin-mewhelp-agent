"""一次性：给真实 MySQL 补 ch05 的两处结构（幂等）。

1. ``messages.actions`` JSON 列；
2. ``tickets.idempotency_key`` 列 + 唯一键（建单防重）。

用法：`uv run --directory MemHelp-code python scripts/apply_ch05_migration.py`
不打印连接串或密码。
"""

from __future__ import annotations

import sys

import pymysql

from app.config import get_settings


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    return int(cur.fetchone()[0]) > 0


def _index_exists(cur, table: str, index: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        """,
        (table, index),
    )
    return int(cur.fetchone()[0]) > 0


def main() -> int:
    settings = get_settings()
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            if _column_exists(cur, "messages", "actions"):
                print("messages.actions 已存在，跳过")
            else:
                cur.execute("ALTER TABLE messages ADD COLUMN actions JSON NULL AFTER citations")
                print("已添加 messages.actions")

            if _column_exists(cur, "tickets", "idempotency_key"):
                print("tickets.idempotency_key 已存在，跳过")
            else:
                cur.execute(
                    "ALTER TABLE tickets ADD COLUMN idempotency_key VARCHAR(64) NULL AFTER status"
                )
                print("已添加 tickets.idempotency_key")

            if _index_exists(cur, "tickets", "uq_tickets_idempotency_key"):
                print("uq_tickets_idempotency_key 已存在，跳过")
            else:
                cur.execute(
                    "ALTER TABLE tickets ADD UNIQUE KEY uq_tickets_idempotency_key "
                    "(idempotency_key)"
                )
                print("已添加 uq_tickets_idempotency_key")
            return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
