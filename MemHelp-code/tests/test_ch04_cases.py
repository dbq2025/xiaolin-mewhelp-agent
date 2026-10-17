"""ch04 评估集的数据校验（不连库、不调模型）。

## 为什么数据也要有测试

评估集是**度量衡**。金标锚串写错（拼错一个字、指向不存在的块）不会让脚本崩，
只会让 Recall 悄悄变低 —— 那种错最难发现，因为它伪装成「检索不行」。

## 锚串解析用真的切分器

不连 MySQL：直接把 ``knowledge/*.md`` 过一遍 ``split_markdown``，用同样的
``render_bm25_text`` 拼出文本再匹配。切分规则改了，这条测试会跟着红 ——
这正是我们要的。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.knowledge.chunker import render_bm25_text, split_markdown

CASES_FILE = Path("evals/ch04_cases.jsonl")
KNOWLEDGE_DIR = Path("knowledge")

BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for line_no, line in enumerate(
        CASES_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise AssertionError(f"{CASES_FILE.name}:{line_no} 不是合法 JSON：{exc}") from exc
    return cases


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return load_cases()


@pytest.fixture(scope="module")
def chunks() -> list[object]:
    """把全部知识文档切一遍 —— 与建库用的是同一个切分器。"""
    result: list[object] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        result.extend(
            split_markdown(path.read_text(encoding="utf-8"), content_type="faq", source=path.name)
        )
    return result


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


def test_forty_eight_cases(cases):
    assert len(cases) == 48


def test_four_buckets_of_twelve(cases):
    assert Counter(case["bucket"] for case in cases) == {bucket: 12 for bucket in BUCKETS}


def test_each_bucket_has_ten_answer_and_two_refuse(cases):
    for bucket in BUCKETS:
        behaviors = Counter(
            case["behavior"] for case in cases if case["bucket"] == bucket
        )
        assert behaviors == {"answer": 10, "refuse": 2}, f"{bucket}: {behaviors}"


def test_difficulty_uses_three_levels_and_all_present(cases):
    difficulties = {case["difficulty"] for case in cases}
    assert difficulties == {"easy", "medium", "hard"}


def test_ids_are_unique_and_prefixed_by_bucket(cases):
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    for case in cases:
        assert case["id"].startswith(case["bucket"][0])


def test_answer_cases_have_anchors_and_containment(cases):
    for case in cases:
        if case["behavior"] != "answer":
            continue
        assert case["gold_anchors"], f"{case['id']} 缺 gold_anchors"
        assert case["must_contain"], f"{case['id']} 缺 must_contain"


def test_refuse_cases_have_no_gold(cases):
    for case in cases:
        if case["behavior"] == "refuse":
            assert case["gold_anchors"] == []


def test_model_bucket_covers_spelling_variants(cases):
    """Review Focus 1：用户不会照抄语料里的写法（小写、空格）。"""
    queries = [case["query"] for case in cases if case["bucket"] == "B_model"]
    assert any("xh-3000a" in query for query in queries), "缺小写型号变体"
    assert any("XH 3000B" in query for query in queries), "缺带空格型号变体"


# ---------------------------------------------------------------------------
# 锚串能解析到真实知识块
# ---------------------------------------------------------------------------


def _text_of(chunk) -> str:
    return render_bm25_text(chunk)


def test_every_anchor_resolves_to_between_one_and_three_chunks(cases, chunks):
    texts = [_text_of(chunk) for chunk in chunks]

    for case in cases:
        for anchor in case["gold_anchors"]:
            hits = [index for index, text in enumerate(texts) if anchor in text]
            assert hits, f"{case['id']} 的锚串 {anchor!r} 在语料里一块都没命中（标注或语料写错了）"
            assert len(hits) <= 3, (
                f"{case['id']} 的锚串 {anchor!r} 命中 {len(hits)} 块，太泛了，换个更长的串"
            )


def test_must_contain_terms_exist_in_gold_chunks(cases, chunks):
    """必现词必须真的在金标块里 —— 否则「答案里该出现 X」这条判定永远不可能满足。"""
    texts = [_text_of(chunk) for chunk in chunks]

    for case in cases:
        if case["behavior"] != "answer":
            continue
        gold_text = "\n".join(
            text
            for text in texts
            if any(anchor in text for anchor in case["gold_anchors"])
        )
        for term in case["must_contain"]:
            assert term in gold_text, (
                f"{case['id']} 的 must_contain {term!r} 不在金标块里："
                "要么标注写错，要么语料里没有这个事实"
            )


def test_refuse_cases_do_not_look_answerable(cases, chunks):
    """拒答题的每个关键词都不该在语料里成块出现 —— 否则它就是「能答」的题。"""
    texts = [_text_of(chunk) for chunk in chunks]
    keywords = {
        "A11": "货到付款",
        "A12": "积分",
        "B11": "XH-9999",
        "B12": "防水等级",
        "C11": "便宜",
        "C12": "线下门店",
        "E11": "花呗",
        "E12": "WiFi",
    }
    for case in cases:
        if case["behavior"] != "refuse":
            continue
        keyword = keywords[case["id"]]
        assert not any(keyword in text for text in texts), (
            f"{case['id']} 的关键词 {keyword!r} 在语料里出现了 —— 那它是能答的题，"
            "不该放在 refuse 组"
        )
