"""``scripts/seed_demo_history.py`` 的 SQL 切分逻辑（离线）。

踩过的坑：最初按「整段以 ``--`` 开头就丢弃」切分，结果文件开头的
``-- 注释`` + ``SET NAMES utf8mb4;`` 落在同一个段里，被整段丢掉 ——
那条语句**从来没执行过**。今天靠连接串的 charset 顶着，下一个共享注释的
真实语句就没这么幸运了。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "seed_demo_history.py"
    spec = importlib.util.spec_from_file_location("seed_demo_history_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seed():
    return _load()


def test_statements_keep_a_statement_that_follows_a_comment(seed):
    sql = "-- 文件头注释\nSET NAMES utf8mb4;\nINSERT INTO t VALUES (1);\n"
    statements = seed._statements(sql)
    assert statements == ["SET NAMES utf8mb4", "INSERT INTO t VALUES (1)"]


def test_statements_drop_pure_comment_chunks(seed):
    sql = "-- 只有注释\n\n-- 还是注释\n"
    assert seed._statements(sql) == []


def test_real_seed_file_keeps_set_names(seed):
    """真实种子文件里 ``SET NAMES utf8mb4`` 必须出现在切分结果中。"""
    sql = (ROOT.parent / "sql" / "ch03-seed-history.sql").read_text(encoding="utf-8")
    statements = seed._statements(sql)
    assert any("SET NAMES utf8mb4" in statement for statement in statements)
    assert sum("INSERT INTO conversations" in s for s in statements) == 30
