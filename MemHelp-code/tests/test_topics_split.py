"""ch10 分层抽样：多标签迭代分层 80/10/10。

分层单位是**完整标签组合**而不是单个标签：按单标签分层会把
「尺码+退换货」这类多诉求样本切碎，训练集与测试集的多标签比例
就对不上，评测数字会系统性偏好单标签。
"""

import random

from app.topics.split import iterative_stratify


def _rows(n: int = 200) -> list[dict]:
    rng = random.Random(7)
    pool = [["退换货"], ["物流"], ["尺码"], ["运费"], ["尺码", "退换货"], ["物流", "运费"]]
    return [
        {"id": f"q{i}", "text": f"问题{i}", "labels": rng.choice(pool), "source": "synthetic"}
        for i in range(n)
    ]


def test_sizes_follow_ratios():
    r = iterative_stratify(_rows())
    total = len(r.train) + len(r.val) + len(r.test)
    assert total == 200
    assert 0.76 <= len(r.train) / total <= 0.84
    assert 0.07 <= len(r.val) / total <= 0.13
    assert 0.07 <= len(r.test) / total <= 0.13


def test_no_row_appears_in_two_splits():
    r = iterative_stratify(_rows())
    ids = [x["id"] for x in r.train + r.val + r.test]
    assert len(ids) == len(set(ids)) == 200


def test_same_seed_is_deterministic():
    a, b = iterative_stratify(_rows()), iterative_stratify(_rows())
    assert [x["id"] for x in a.test] == [x["id"] for x in b.test]
    assert [x["id"] for x in a.val] == [x["id"] for x in b.val]


def test_report_counts_match_actual():
    r = iterative_stratify(_rows())
    assert r.report["sizes"] == {
        "train": len(r.train),
        "val": len(r.val),
        "test": len(r.test),
    }
    for label, counts in r.report["per_label"].items():
        for name in ("train", "val", "test"):
            assert counts[name] == sum(1 for x in getattr(r, name) if label in x["labels"])


def test_rare_combo_all_kept_in_train():
    """只出现 1 次的标签组合不足以切片，必须整组落 train —— 不能为了凑比例
    把它拆到 val/test 去，那等于凭空制造分布外的样本。"""

    rows = _rows(60) + [
        {"id": "rare", "text": "少见", "labels": ["评价", "会员积分"], "source": "synthetic"}
    ]
    r = iterative_stratify(rows)
    assert "rare" in {x["id"] for x in r.train}


def test_real_rows_forced_into_test():
    real = [{"id": "r1", "text": "真实问题", "labels": ["物流"], "source": "real"}]
    r = iterative_stratify(_rows(), real_rows=real)
    assert "r1" in {x["id"] for x in r.test}
    assert r.report["real_in_test"] == 1


def test_zero_in_split_is_reported():
    r = iterative_stratify(_rows(40))
    assert isinstance(r.report["zero_in_split"], list)
    # 每一条都要能定位到「哪一类、缺在哪一份」，否则报告没法据以补数
    for item in r.report["zero_in_split"]:
        assert set(item) == {"label", "split"}
