"""T8 测试集人工标注的工具链：出空白工作表、回填 gold、生成抽审表。

标注本身是人的判断，这里只保证「不漏行、不写错标签、不悄悄改掉预标」三件事。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.topics.taxonomy import TaxonomyError
from scripts.label_test_set import (
    apply_gold,
    dump_worksheet,
    render_review_csv,
    write_jsonl,
)

TEST_ROWS = [
    {
        "id": "r1",
        "text": "买大了想退，运费谁出",
        "labels": [],
        "source": "real",
        "needs_human_label": True,
    },
    {
        "id": "s01",
        "text": "快递三天了还没到",
        "labels": ["物流"],
        "source": "synthetic",
    },
    {
        "id": "s02",
        "text": "发票能开公司抬头吗",
        "labels": ["发票"],
        "source": "synthetic",
    },
]


def test_dump_worksheet_leaves_labels_and_rationale_empty() -> None:
    worksheet = dump_worksheet(TEST_ROWS)

    assert [row["id"] for row in worksheet] == ["r1", "s01", "s02"]
    assert all(row["labels"] == [] for row in worksheet)
    assert all(row["rationale"] == "" for row in worksheet)
    # 预标留作参考列，标注时看得见、但不冒充答案
    assert worksheet[1]["labels_prelabeled"] == ["物流"]
    assert worksheet[0]["labels_prelabeled"] == []


def test_apply_gold_writes_labels_and_keeps_prelabel_copy() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]

    result = apply_gold(TEST_ROWS, worksheet)

    merged = {row["id"]: row for row in result["rows"]}
    # 标签按权威类目编号排序（退换货=1、尺码=3、运费=6），不是按标注时的手写顺序
    assert merged["r1"]["labels"] == ["退换货", "尺码", "运费"]
    assert merged["r1"]["labels_prelabeled"] == []
    assert "needs_human_label" not in merged["r1"]
    assert merged["s01"]["labels_prelabeled"] == ["物流"]
    # 顺序必须和原 test.jsonl 一致，否则预测和标签会对不上
    assert [row["id"] for row in result["rows"]] == ["r1", "s01", "s02"]
    assert [row["id"] for row in result["gold"]] == ["r1", "s01", "s02"]


def test_apply_gold_rejects_unknown_label() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退", "labels": ["退货"]},  # 不是权威类目名
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]

    with pytest.raises(TaxonomyError):
        apply_gold(TEST_ROWS, worksheet)


def test_apply_gold_rejects_other_plus_real_label() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退", "labels": ["其他", "退换货"]},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]

    with pytest.raises(TaxonomyError):
        apply_gold(TEST_ROWS, worksheet)


def test_apply_gold_requires_every_row_and_nonempty_labels() -> None:
    missing = [
        {"id": "r1", "text": "买大了想退", "labels": ["退换货"]},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
    ]
    with pytest.raises(ValueError, match="s02"):
        apply_gold(TEST_ROWS, missing)

    empty = [
        {"id": "r1", "text": "买大了想退", "labels": ["退换货"]},
        {"id": "s01", "text": "快递三天了还没到", "labels": []},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]
    with pytest.raises(ValueError, match="s01"):
        apply_gold(TEST_ROWS, empty)


def test_apply_gold_requires_rationale_when_configured() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退", "labels": ["退换货"], "rationale": ""},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]

    with pytest.raises(ValueError, match="rationale"):
        apply_gold(TEST_ROWS, worksheet, require_rationale=True)

    # 不强制时，缺 rationale 不算错（测试里的小样例可以不写）
    result = apply_gold(TEST_ROWS, worksheet)
    assert result["gold"][0]["rationale"] == ""


def test_apply_gold_counts_disagreements_with_prelabels() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"]},
        {"id": "s02", "text": "发票能开公司抬头吗", "labels": ["发票"]},
    ]

    result = apply_gold(TEST_ROWS, worksheet)
    # r1 本来就没有预标（真实问题），算「无参考」，不记成分歧
    assert result["disagreements"] == []

    worksheet[1]["labels"] = ["物流", "运费"]
    result = apply_gold(TEST_ROWS, worksheet)
    assert [row["id"] for row in result["disagreements"]] == ["s01"]
    assert result["disagreements"][0]["prelabeled"] == ["物流"]
    assert result["disagreements"][0]["gold"] == ["物流", "运费"]


def test_render_review_csv_has_header_and_rows() -> None:
    worksheet = [
        {"id": "r1", "text": "买大了想退", "labels": ["尺码", "退换货"], "rationale": "买大=尺码，想退=退换货"},
        {"id": "s01", "text": "快递三天了还没到", "labels": ["物流"], "rationale": "只问到达进度"},
    ]
    gold = [{"id": row["id"], "labels": row["labels"]} for row in worksheet]

    text = render_review_csv(gold, worksheet)

    lines = text.strip().splitlines()
    assert lines[0] == "id,text,my_labels,my_rationale,user_verdict,user_note"
    assert lines[1].startswith("r1,买大了想退,尺码|退换货,买大=尺码")
    assert len(lines) == 3


def test_write_jsonl_roundtrip() -> None:
    # 不用 tmp_path：沙箱会在用例退出后收回新建目录的权限
    target = Path(__file__).parent / "_tmp_gold.jsonl"
    try:
        write_jsonl(target, [{"id": "r1", "labels": ["退换货"]}])

        text = target.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text.strip())["id"] == "r1"
    finally:
        target.unlink(missing_ok=True)
