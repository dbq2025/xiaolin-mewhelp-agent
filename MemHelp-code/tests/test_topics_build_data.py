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


class _FakeLLM:
    """只回一条问题，用来数「到底调了几次」。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001, ANN201 - 测试桩
        self.calls += 1
        text = f"模拟问题{self.calls}"
        line = json.dumps({"text": text, "labels": ["其他"]}, ensure_ascii=False)
        return type("Msg", (), {"content": line})()


def test_generate_resumes_from_partial_file():
    """生成 68 次调用太贵：已落盘的类目不能重造，否则一崩就全白跑。"""

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seeded = {
        "id": "s0001",
        "text": "已有问题",
        "labels": ["退换货"],
        "source": "synthetic",
        "category": "退换货",
    }
    (raw / "synthetic.partial.jsonl").write_text(
        json.dumps(seeded, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    llm = _FakeLLM()
    out = build_dataset(FIXTURES, steps=("generate",), llm=llm, seed=1, out_dir=OUT, count=17)
    rows = _read_jsonl(Path(out["dirs"]["raw"], "synthetic.jsonl"))
    assert len(rows) == 17
    assert llm.calls == 16, "已生成的类目不该再调大模型"
    assert any(row["text"] == "已有问题" for row in rows), "断点数据必须并进最终语料"
    assert not (raw / "synthetic.partial.jsonl").is_file(), "成功后清掉断点，避免下次误续"


def test_llm_steps_need_llm():
    import pytest

    with pytest.raises(RuntimeError):
        build_dataset(FIXTURES, steps=("generate",), llm=None, out_dir=OUT)


class _ConcurrentLLM:
    """记录同时在飞的调用数：串行跑永远到不了 2。"""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001, ANN201 - 测试桩
        import time

        with self._lock:
            self.calls += 1
            serial = self.calls
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
        finally:
            with self._lock:
                self._active -= 1
        line = json.dumps({"text": f"并发问题{serial}", "labels": ["其他"]}, ensure_ascii=False)
        return type("Msg", (), {"content": line})()


def test_generate_runs_chunks_concurrently():
    """68 次调用串行要一小时：分块并发跑，落盘和去重只由主线程做。"""
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _ConcurrentLLM()

    out = build_dataset(
        FIXTURES, steps=("generate",), llm=llm, seed=1, out_dir=OUT, count=68, workers=4
    )

    rows = _read_jsonl(Path(out["dirs"]["raw"], "synthetic.jsonl"))
    # 桩每次只回一条，所以收上来的条数 = 调用次数（真实模型会按 take 条返回）
    assert len(rows) == llm.calls > 17
    assert len({row["text"] for row in rows}) == len(rows), "并发块之间不能写出重复问题"
    assert len({row["id"] for row in rows}) == len(rows)
    assert llm.max_active >= 2, "开了 workers 就该真有并发"
    assert not (Path(out["dirs"]["raw"]) / "synthetic.partial.jsonl").is_file()

def test_generate_resume_counts_multi_group():
    """multi 组的断点必须被认出来：键用 (类目, single|multi)，不是 (类目, 搭配类目)。

    之前用 (类目, second or "") 当键、落盘却写 gen_group，两边永远对不上，
    于是每次重跑都把全部类目重造一遍 —— 语料悄悄涨到 1986 条。
    """
    import shutil

    OUT.mkdir(parents=True, exist_ok=True)
    llm_fresh = _FakeLLM()
    build_dataset(FIXTURES, steps=("generate",), llm=llm_fresh, seed=1, out_dir=OUT, count=102)
    fresh_calls = llm_fresh.calls
    assert fresh_calls > 17

    shutil.rmtree(OUT, ignore_errors=True)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seeded = [
        {
            "id": f"sx{index}",
            "text": f"已有的多诉求问题{index}",
            "labels": ["物流"],
            "source": "synthetic",
            "category": "物流",
            "gen_group": "multi",
        }
        for index in (1, 2)
    ]
    (raw / "synthetic.partial.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in seeded) + "\n",
        encoding="utf-8",
    )

    llm_resume = _FakeLLM()
    out = build_dataset(FIXTURES, steps=("generate",), llm=llm_resume, seed=1, out_dir=OUT, count=102)

    # count=102 → 每类 6 条，multi 组 2 条：这两条已经在断点里，物流的 multi 块不该再调
    assert out["generate"]["resumed"] == 2
    assert llm_resume.calls == fresh_calls - 1

