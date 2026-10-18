"""一次性：给真实 MySQL 的 messages 加 actions 列（幂等）。

用法：`uv run --directory MemHelp-code python scripts/apply_ch05_migration.py`
不打印连接串或密码。
"""

from __future__ import annotations

import sys

import pymysql

from app.config import get_settings


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
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'messages'
                  AND COLUMN_NAME = 'actions'
                """
            )
            exists = int(cur.fetchone()[0])
            if exists:
                print("messages.actions 已存在，跳过")
                return 0
            cur.execute("ALTER TABLE messages ADD COLUMN actions JSON NULL AFTER citations")
            print("已添加 messages.actions")
            return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
