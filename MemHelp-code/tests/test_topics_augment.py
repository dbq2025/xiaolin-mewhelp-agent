"""ch10 数据增强：同义词替换 + 句式微调，**只扩训练集**。

「只扩训练集」不是靠调用方自觉：``augment_rows`` 只收「行」，没有任何
split 参数，val / test 根本递不进来。这是接口形状上的防污染。
"""

import random

from app.topics.augment import augment_rows, reshape, synonym_replace

ROW = {"id": "t1", "text": "买大了想退", "labels": ["尺码", "退换货"], "source": "synthetic"}


def test_originals_are_preserved_untouched():
    out = augment_rows([ROW], copies=1)
    originals = [r for r in out if "aug_from" not in r]
    assert originals == [ROW]


def test_augmented_keep_same_labels_and_point_back():
    out = augment_rows([ROW], copies=2)
    aug = [r for r in out if "aug_from" in r]
    assert len(aug) == 2
    assert all(r["labels"] == ROW["labels"] for r in aug)
    assert all(r["aug_from"] == "t1" for r in aug)
    assert all(r["id"] != "t1" for r in aug)


def test_augmented_text_actually_differs():
    out = augment_rows([ROW], copies=1, seed=3)
    aug = [r for r in out if "aug_from" in r][0]
    assert aug["text"] != ROW["text"]


def test_placeholders_are_never_replaced():
    text = "我的订单 [ORDER] 到哪了"
    rng = random.Random(0)
    for _ in range(50):
        assert "[ORDER]" in (synonym_replace(text, rng) or text)


def test_synonym_replace_returns_none_when_nothing_to_do():
    assert synonym_replace("嗯", random.Random(0)) is None


def test_reshape_is_idempotent_on_prefix():
    once = reshape("买大了想退", random.Random(1))
    twice = reshape(once, random.Random(1))
    assert not twice.startswith("请问请问")


def test_determinism_by_seed():
    a = augment_rows([ROW], copies=2, seed=11)
    b = augment_rows([ROW], copies=2, seed=11)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_output_ids_are_unique():
    out = augment_rows([ROW], copies=3)
    assert len({r["id"] for r in out}) == len(out)
