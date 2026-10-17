"""语义检索评估脚本的**判定逻辑**测试（离线）。

`evals/run_rag_evals.py` 要连真库、真 Milvus、真嵌入才能跑；但它的**判定规则**
是纯函数，值得先钉住 —— 判定写错的后果比脚本报错严重得多：它会给出一个
「向量检索更好」的假结论。

同时钉住评估集本身的两条约束：
1. 必须有一条「邮费」换说法用例（验收标准 1 的落点）；
2. 换说法用例与语料**零词面重叠** —— 否则关键词基线也会命中，对照就失去意义。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "evals" / "run_rag_evals.py"
    spec = importlib.util.spec_from_file_location("run_rag_evals_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evals_module():
    return _load_module()


@pytest.fixture(scope="module")
def cases(evals_module):
    return evals_module.load_cases(ROOT / "evals" / "rag_cases.jsonl")


# ---------------------------------------------------------------------------
# 判定规则
# ---------------------------------------------------------------------------


def test_hit_requires_expect_any_and_all_must_contain(evals_module):
    case = {"expect_any": ["包邮", "运费"], "must_contain": ["99"]}

    assert evals_module.hit(case, [SimpleNamespace(answer="满 99 元包邮")]) is True
    assert evals_module.hit(case, [SimpleNamespace(answer="满 199 元包邮")]) is False
    assert evals_module.hit(case, []) is False


def test_hit_is_satisfied_by_any_single_chunk(evals_module):
    """只要**某一条**命中同时满足两个条件就算命中，不能跨条目拼条件。"""
    case = {"expect_any": ["包邮"], "must_contain": ["99"]}
    hits = [
        SimpleNamespace(answer="满 99 元起送"),   # 有 99，没有包邮
        SimpleNamespace(answer="本店包邮"),        # 有包邮，没有 99
    ]
    assert evals_module.hit(case, hits) is False


def test_load_cases_skips_comments_and_blank_lines(evals_module):
    path = ROOT / "evals" / "_tmp_rag_cases.jsonl"
    path.write_text(
        '# 注释行\n\n{"id":"a","query":"q","expect_any":["x"],"must_contain":[]}\n',
        encoding="utf-8",
    )
    try:
        loaded = evals_module.load_cases(path)
        assert [case["id"] for case in loaded] == ["a"]
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 评估集内容约束
# ---------------------------------------------------------------------------


def test_paraphrase_fee_case_exists(cases):
    fee = [case for case in cases if case["id"] == "paraphrase-fee"]
    assert fee, "验收标准 1 的落点必须存在"
    assert fee[0]["query"] == "邮费是多少"
    assert fee[0]["paraphrase"] is True


def test_paraphrase_cases_have_no_literal_overlap_with_the_corpus(cases):
    """换说法用例的关键词，不能在任何知识块原文里出现 —— 否则基线也能命中。"""
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "knowledge").glob("*.md")
    )
    for case in cases:
        if not case.get("paraphrase"):
            continue
        assert case["query"] not in corpus, case["id"]
        assert case["keyword"] not in corpus, case["id"]


def test_cases_have_required_fields(cases):
    assert len(cases) >= 15
    for case in cases:
        assert case["id"] and case["query"] and case["keyword"]
        assert isinstance(case["expect_any"], list) and case["expect_any"]
        assert isinstance(case["must_contain"], list)
        assert isinstance(case["paraphrase"], bool)
