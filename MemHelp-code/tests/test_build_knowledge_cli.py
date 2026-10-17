"""建库 CLI 的纯函数测试（离线）。

脚本本身要连 MySQL/Milvus/云端嵌入，不适合在单测里跑；但**参数解析**与
**状态汇总**是纯函数，值得钉住 —— 尤其是汇总那一段：验收标准 2 的判据
（`pending=0` 且 `done == Milvus 行数`）就打印在它里面，写错了会让「补没补齐」
看不出来。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_cli():
    """按路径加载 scripts/build_knowledge.py（它不在 app 包里）。"""
    path = ROOT / "scripts" / "build_knowledge.py"
    spec = importlib.util.spec_from_file_location("build_knowledge_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return _load_cli()


def test_parser_accepts_demo_flags(cli):
    args = cli.build_parser().parse_args(
        ["--docs", "knowledge", "--faq-table", "--max-chunks", "5"]
    )
    assert args.docs == "knowledge"
    assert args.faq_table is True
    assert args.max_chunks == 5


def test_parser_defaults_are_safe(cli):
    args = cli.build_parser().parse_args([])
    assert args.docs is None
    assert args.faq_table is False
    assert args.rebuild is False
    assert args.status is False
    assert args.max_chunks is None


def test_summarise_reports_pending_and_milvus_delta(cli):
    text = cli.summarise({"pending": 2, "done": 8}, 8)
    assert "pending=2" in text and "done=8" in text and "milvus=8" in text
    assert "一致" in text


def test_summarise_flags_mismatch_between_mysql_and_milvus(cli):
    text = cli.summarise({"pending": 0, "done": 8}, 7)
    assert "不一致" in text


def test_summarise_treats_pending_as_inconsistent_even_when_counts_match(cli):
    """还有 pending 就说明没补完 —— 即使 done 数与 Milvus 相等也不能说「一致」。"""
    text = cli.summarise({"pending": 3, "done": 8}, 8)
    assert "不一致" in text


def test_remaining_budget_is_shared_across_build_steps(cli):
    """``--max-chunks`` 是**整次运行**的预算，不是每个建库步骤各一份。

    各给一份的话，`--max-chunks 5` 在「文档 + FAQ 表」两个步骤里会写 10 条，
    中断演示的预期数字（pending = 总数-5）就跟着错。
    """
    assert cli.remaining_budget(None, 7) is None
    assert cli.remaining_budget(5, 0) == 5
    assert cli.remaining_budget(5, 3) == 2
    assert cli.remaining_budget(5, 5) == 0
    assert cli.remaining_budget(5, 9) == 0  # 不会变成负数
