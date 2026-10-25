"""推理层的阈值化与兜底 —— 不加载模型也能测的部分。

这里刻意不碰 torch：``TopicClassifier`` 的判定逻辑是纯 Python，scorer 是可替换
的「文本 → 17 个分数」函数。模型只有 ``load()`` 才碰，于是「阈值怎么比、其他
什么时候让位、形状不对怎么办」这些最容易写错的规则都能在毫秒级跑完。
"""

import pytest

from app.topics.infer import TopicClassifier, TopicModelError
from app.topics.taxonomy import CATEGORY_NAMES, taxonomy_hash


def _scorer(fixed):
    return lambda texts: [fixed for _ in texts]


def _thresholds(value=0.5):
    return {name: value for name in CATEGORY_NAMES}


def _scores(**over):
    base = {name: 0.01 for name in CATEGORY_NAMES}
    base.update(over)
    return [base[name] for name in CATEGORY_NAMES]


def test_threshold_selects_multiple_labels_sorted():
    clf = TopicClassifier.from_parts(
        _scorer(_scores(尺码=0.9, 退换货=0.8, 运费=0.7)), _thresholds()
    )
    p = clf.predict(["买大了想退，运费谁出"])[0]
    assert p.labels == ["退换货", "尺码", "运费"]  # 按类目编号升序，不是分数序
    assert p.fallback is False
    assert p.scores["尺码"] == pytest.approx(0.9)


def test_all_below_threshold_falls_back_to_other():
    clf = TopicClassifier.from_parts(_scorer(_scores()), _thresholds())
    p = clf.predict(["嗯"])[0]
    assert p.labels == ["其他"] and p.fallback is True


def test_per_label_threshold_is_respected():
    th = _thresholds()
    th["物流"] = 0.9
    clf = TopicClassifier.from_parts(_scorer(_scores(物流=0.6)), th)
    assert clf.predict(["到哪了"])[0].labels == ["其他"]


def test_other_is_dropped_when_another_label_fires():
    # 即使模型给了「其他」高分，只要还有别的类过阈值，「其他」必须让位（D10 独占）
    clf = TopicClassifier.from_parts(_scorer(_scores(其他=0.9, 物流=0.8)), _thresholds())
    assert clf.predict(["快递到哪了"])[0].labels == ["物流"]


def test_wrong_threshold_keys_raise():
    bad = {"物流": 0.5}
    with pytest.raises(TopicModelError, match="17"):
        TopicClassifier.from_parts(_scorer(_scores()), bad)


def test_taxonomy_hash_mismatch_raises():
    with pytest.raises(TopicModelError, match="taxonomy"):
        TopicClassifier.from_parts(
            _scorer(_scores()),
            _thresholds(),
            model_taxonomy_hash="deadbeef",
            current_hash=taxonomy_hash(),
        )


def test_scorer_output_shape_is_validated():
    with pytest.raises(TopicModelError):
        TopicClassifier.from_parts(lambda t: [[0.1, 0.2]], _thresholds()).predict(["x"])


def test_empty_text_still_returns_one_prediction():
    clf = TopicClassifier.from_parts(_scorer(_scores()), _thresholds())
    assert clf.predict([""])[0].labels == ["其他"]
