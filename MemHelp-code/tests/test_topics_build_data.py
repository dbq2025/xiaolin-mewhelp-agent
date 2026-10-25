"""ch10 数据流水线 CLI：清洗 → 分层 → 增强 → 导出术语表。

这里测的不是「模型好不好」，而是**流水线的形状**：噪声真的被丢掉、三份文件
真的写出来、增强真的只落在 train。数据类任务没法 TDD 出质量，但可以钉住
这四条不变量，跑挂任意一条就说明语料被污染了。

真实数据的目录是 ``data/topics/``，测试一律写到 ``out_dir``，绝不碰它 ——
一个用 12 行 fixture 覆盖掉真语料的测试，比没测试更危险。
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_topic_data import build_dataset, export_taxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures_topics"
OUT = FIXTURES / "_tmp_out"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build(*steps: str) -> dict:
    return build_dataset(FIXTURES, steps=steps, llm=None, seed=1, out_dir=OUT)


def setup_function() -> None:
    import shutil

    shutil.rmtree(OUT, ignore_errors=True)


def test_clean_filters_noise_and_dedupes():
    out = _build("clean", "split")
    assert out["clean"]["kept"] > 0
    assert out["clean"]["noise_dropped"] >= 1
    cleaned = _read_jsonl(Path(out["dirs"]["clean"], "synthetic.jsonl"))
    texts = [row["text"] for row in cleaned]
    assert len(texts) == len(set(texts)), "清洗后必须去重"
    assert all(row["text"] == row["text"].strip() for row in cleaned)


def test_clean_masks_phone_and_keeps_raw():
    out = _build("clean", "split")
    cleaned = _read_jsonl(Path(out["dirs"]["clean"], "synthetic.jsonl"))
    masked = [row for row in cleaned if "[PHONE]" in row["text"]]
    assert masked, "手机号必须被脱敏"
    assert "13800138000" not in masked[0]["text"]
    assert "13800138000" in masked[0]["text_raw"]
    assert any(change["rule"] == "mask_phone" for change in masked[0]["changes"])


def test_split_writes_three_files_and_report():
    out = _build("clean", "split")
    for name in ("train", "val", "test"):
        assert Path(out["dirs"]["splits"], f"{name}.jsonl").is_file()
    report = json.loads(
        Path(out["dirs"]["splits"], "strata_report.json").read_text(encoding="utf-8")
    )
    assert report["sizes"]["train"] > 0
    assert report["real_in_test"] == 2


def test_real_rows_land_in_test():
    out = _build("clean", "split")
    test_rows = _read_jsonl(Path(out["dirs"]["splits"], "test.jsonl"))
    assert {row["id"] for row in test_rows} == {"r1", "r2"}


def test_augment_only_touches_train():
    out = _build("clean", "split", "augment")
    train_aug = _read_jsonl(Path(out["dirs"]["splits"], "train_aug.jsonl"))
    val_rows = _read_jsonl(Path(out["dirs"]["splits"], "val.jsonl"))
    assert any("aug_from" in row for row in train_aug)
    assert all("aug_from" not in row for row in val_rows)


def test_export_taxonomy_writes_json_and_markdown():
    out = export_taxonomy(OUT)
    assert Path(out, "taxonomy.json").is_file()
    md = Path(out, "taxonomy.md").read_text(encoding="utf-8")
    assert md.count("|") > 17 * 3
    data = json.loads(Path(out, "taxonomy.json").read_text(encoding="utf-8"))
    assert len(data["categories"]) == 17
    assert data["categories"][0]["name"] == "退换货"
    assert data["hash"]


def test_llm_steps_need_llm():
    import pytest

    with pytest.raises(RuntimeError):
        build_dataset(FIXTURES, steps=("generate",), llm=None, out_dir=OUT)
