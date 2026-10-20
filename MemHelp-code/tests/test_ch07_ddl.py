"""ch07 迁移脚本的静态约束。

这些检查不连库，但能挡住「脚本看起来能跑、真库上炸 1064」这一类问题：
ch02 就因为 ``IF()`` 漏一个引号返工过一次。
"""

from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "sql" / "ch07-ddl.sql"


def test_ddl_exists_and_sets_names():
    text = SQL.read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in text


def test_ddl_is_idempotent():
    text = SQL.read_text(encoding="utf-8")
    assert "information_schema.COLUMNS" in text
    assert "PREPARE stmt FROM @ddl" in text
    assert "DEALLOCATE PREPARE stmt" in text
    # 两个锚点列 + 建表 + 补唯一键 = 至少 4 段可重复执行的 DDL。
    assert text.count("DEALLOCATE PREPARE stmt") >= 4


def test_ddl_adds_the_two_anchors_and_drops_the_old_summary_column():
    text = SQL.read_text(encoding="utf-8")
    assert "summary_upto_msg_id" in text
    assert "layer1_from_msg_id" in text
    assert "CREATE TABLE conversation_summaries" in text
    # 旧双层方案要把整坨摘要塞进 conversations.summary 一列；ch07 起不再落这一列。
    assert "ADD COLUMN summary " not in text


def test_ddl_creates_the_segment_unique_key():
    text = SQL.read_text(encoding="utf-8")
    assert "uq_summary_segment" in text
    assert "UNIQUE KEY" in text


def test_ddl_never_drops_anything():
    """迁移只加不减：真库里可能还留着旧方案的 summary 列，DROP 掉就回滚不了。"""
    code = "\n".join(
        line for line in SQL.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    assert "DROP " not in code.upper()


def test_ddl_quotes_are_balanced():
    """单引号必须成对 —— 踩过 ``IF()`` 漏引号报 1064 指向 ELSE 分支。

    与 ``test_repo_conventions.py`` 里那条不同，这里**按整份脚本**统计而不是逐行：
    建表语句是一段跨多行的字符串字面量，逐行数必然报假警。
    """
    code = "\n".join(
        line for line in SQL.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    assert code.count("'") % 2 == 0


def test_ddl_create_table_statement_is_one_quoted_literal():
    """建表语句必须是**一整段**字符串字面量：``PREPARE`` 只接受单条语句，
    中间断成两个单引号串就变成非法 SQL。"""
    text = SQL.read_text(encoding="utf-8")
    start = text.index("'CREATE TABLE conversation_summaries")
    end = text.index("'SELECT 1'", start)
    body = text[start:end]
    # 只有首尾两个引号；内部再出现一个就是把语句截断了。
    assert body.count("'") == 2
    assert body.startswith("'") and body.rstrip().endswith("',")
