"""ch10 类目表：17 类、固定顺序、边界说明、多标签校验。

这些断言就是「全系统统一一份权威类目表」的可执行定义 —— 任何改动
（加一类、调顺序、删边界说明）都必须在这里先红一次。
"""

import pytest

from app.topics.taxonomy import (
    CATEGORIES,
    CATEGORY_NAMES,
    LABEL_TO_INDEX,
    OTHER_LABEL,
    TaxonomyError,
    prompt_fragment,
    taxonomy_hash,
    taxonomy_payload,
    validate_labels,
)

EXPECTED = (
    "退换货",
    "物流",
    "尺码",
    "发票",
    "质量问题",
    "运费",
    "优惠活动",
    "价保",
    "支付",
    "订单修改",
    "库存补货",
    "商品信息",
    "保修维修",
    "账号",
    "会员积分",
    "评价",
    "其他",
)


def test_seventeen_categories_in_fixed_order():
    assert CATEGORY_NAMES == EXPECTED
    assert [c.index for c in CATEGORIES] == list(range(1, 18))
    assert LABEL_TO_INDEX == {name: i for i, name in enumerate(EXPECTED, start=1)}


def test_every_category_has_boundary_and_triggers():
    for c in CATEGORIES:
        assert len(c.boundary) >= 10, c.name  # 一句边界说明，不许留空
        assert len(c.triggers) >= 2, c.name


def test_required_neighbour_boundaries_are_written_down():
    """需求点名的三组近邻必须两边都写明，否则标注时必然互相抢样本。"""
    by_name = {c.name: c for c in CATEGORIES}
    # 修归保修维修、退归退换货
    assert "保修维修" in by_name["退换货"].boundary
    assert "质量问题" in by_name["保修维修"].boundary
    # 运费管钱、物流管货
    assert "运费" in by_name["物流"].boundary
    assert "物流" in by_name["运费"].boundary
    # 价保是补差价、优惠活动是券和满减
    assert "优惠活动" in by_name["价保"].boundary
    assert "价保" in by_name["优惠活动"].boundary


def test_validate_dedupes_and_sorts_by_index():
    assert validate_labels(["运费", "退换货", "退换货"]) == ["退换货", "运费"]


def test_other_must_not_cooccur():
    with pytest.raises(TaxonomyError, match="其他"):
        validate_labels(["尺码", OTHER_LABEL])


def test_unknown_label_rejected():
    with pytest.raises(TaxonomyError, match="退款"):
        validate_labels(["退款"])


def test_empty_rejected():
    with pytest.raises(TaxonomyError):
        validate_labels([])


def test_hash_is_stable_across_calls():
    assert taxonomy_hash() == taxonomy_hash(taxonomy_payload())
    assert len(taxonomy_hash()) == 64


def test_hash_changes_when_a_category_disappears():
    payload = taxonomy_payload()
    truncated = {**payload, "categories": payload["categories"][:16]}
    assert taxonomy_hash(payload) != taxonomy_hash(truncated)


def test_prompt_fragment_covers_all_seventeen():
    text = prompt_fragment()
    for name in EXPECTED:
        assert name in text
    assert "独占" in text  # 「其他」的口径必须进 prompt
