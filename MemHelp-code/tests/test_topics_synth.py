"""ch10 合成语料与预标：prompt 构造 + 输出解析 + 标签冲突检测。

prompt 是数据质量的源头，但它没法被单测「跑出好数据」；能钉住的是
**结构性约束** —— 类目表必须整份进 prompt、解析器必须只吐出 17 类里的
标签、模型的坏输出不能让整批任务崩掉。
"""

import pytest

from app.topics.synth import (
    Conflict,
    find_conflicts,
    generation_messages,
    labeling_messages,
    parse_generated,
    parse_labeled,
)
from app.topics.taxonomy import CATEGORIES, TaxonomyError

FENCED = """```json
{"text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]}
{"text": "快递到哪了", "labels": ["物流"]}
```"""


def test_generation_prompt_carries_the_whole_taxonomy():
    msgs = generation_messages(CATEGORIES[0], 12, second_label="尺码")
    joined = "\n".join(m["content"] for m in msgs)
    assert "退换货" in joined and "其他" in joined and "保修维修" in joined
    assert "12" in joined and "尺码" in joined
    assert msgs[0]["role"] == "system"


def test_parse_generated_strips_fences():
    out = parse_generated(FENCED)
    assert [p.text for p in out] == ["买大了想退，运费谁出", "快递到哪了"]
    assert out[0].labels == ["尺码", "退换货", "运费"]


def test_parse_generated_rejects_unknown_label():
    with pytest.raises(TaxonomyError):
        parse_generated('{"text": "想退款", "labels": ["退款"]}')


def test_parse_generated_skips_malformed_lines_but_keeps_rest():
    raw = '干扰行\n{"text": "快递到哪了", "labels": ["物流"]}\n{坏 json'
    assert len(parse_generated(raw)) == 1


def test_labeling_prompt_lists_texts_with_ids():
    msgs = labeling_messages(["买大了想退", "发票能开吗"])
    joined = "\n".join(m["content"] for m in msgs)
    assert "买大了想退" in joined and "发票能开吗" in joined
    assert "rationale" in joined


def test_parse_labeled_requires_rationale():
    raw = '{"text": "买大了想退", "labels": ["尺码"], "rationale": "字面提到买大了"}'
    out = parse_labeled(raw)
    assert out[0].rationale == "字面提到买大了"


def test_find_conflicts_compares_sets_ignoring_order():
    g = parse_generated('{"text": "a", "labels": ["尺码", "退换货"]}')
    p = parse_labeled('{"text": "a", "labels": ["退换货", "尺码"], "rationale": "x"}')
    assert find_conflicts(g, p) == []


def test_find_conflicts_flags_real_difference():
    g = parse_generated('{"text": "a", "labels": ["尺码", "退换货"]}')
    p = parse_labeled('{"text": "a", "labels": ["尺码"], "rationale": "x"}')
    assert find_conflicts(g, p) == [
        Conflict(text="a", generated=["尺码", "退换货"], prelabeled=["尺码"])
    ]


def test_force_multi_prompt_demands_every_row():
    """multi 分块原来只说「约三成」，模型就真只给三成 —— 多标签比例掉到 10%。"""

    from app.topics.synth import generation_messages
    from app.topics.taxonomy import CATEGORIES

    category = next(c for c in CATEGORIES if c.name == "退换货")
    content = generation_messages(category, 8, second_label="运费", force_multi=True)[-1]["content"]
    assert "每一条" in content
    assert "「退换货」" in content and "「运费」" in content
    assert "约三成" not in content
