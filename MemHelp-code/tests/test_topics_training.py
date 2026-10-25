"""ch10 训练：纯函数先钉住算式，torch 部分只验「循环能跑完、产物能装回」。

训练质量没法单测 —— 数据集只有真跑一次才知道 val-F1 是多少。所以这里分两层：
纯 Python 的 multi-hot / F1 / 阈值扫描必须精确到小数点（它们是报告的可信来源）；
``train_model`` 只用一个 1 层 tiny BERT 跑通 2 个 epoch，断言六件产物齐全、且能被
``TopicClassifier.load`` 原样装回去预测。真模型的分数在 T11/真训练时看。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.topics.training import macro_f1, multilabel_targets, per_label_f1, threshold_sweep

LABELS = ["退换货", "物流", "尺码"]
FIXTURES = Path(__file__).resolve().parent / "fixtures_topics"


def test_targets_are_multi_hot():
    assert multilabel_targets([["物流", "退换货"]], LABELS) == [[1.0, 1.0, 0.0]]


def test_targets_reject_unknown_label():
    with pytest.raises(ValueError):
        multilabel_targets([["发票"]], LABELS)


def test_macro_f1_hand_computed():
    # 类别 0：tp=1 fn=1 → P=1 R=0.5 F1=2/3；类别 1：tp=1 fp=1 → 2/3；类别 2：tp=0 → 0
    y_true = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert per_label_f1(y_true, y_pred, LABELS) == pytest.approx(
        {"退换货": 0.666667, "物流": 0.666667, "尺码": 0.0}, abs=1e-4
    )
    assert macro_f1(y_true, y_pred, LABELS) == pytest.approx(0.444444, abs=1e-4)


def test_macro_f1_ignores_labels_with_no_support_and_no_prediction():
    # 全 0 的类目：P/R/F1 无定义，按 0 计，但不能让 0/0 抛异常
    assert per_label_f1([[1, 0, 0]], [[1, 0, 0]], LABELS)["尺码"] == 0.0


def test_threshold_sweep_picks_best_point_per_label():
    # 类别 1 的分数刚好跨在 0.45 两侧：0.45 阈值能捞到 tp，0.5 阈值捞不到
    y_true = [[0, 1, 0], [0, 0, 0]]
    y_prob = [[0.1, 0.45, 0.1], [0.1, 0.1, 0.1]]
    out = threshold_sweep(y_true, y_prob, LABELS)
    assert out["物流"]["threshold"] <= 0.45
    assert set(out) == set(LABELS)
    assert all(0.2 <= v["threshold"] <= 0.7 for v in out.values())


def test_threshold_sweep_is_deterministic():
    a = threshold_sweep([[1, 0, 0]], [[0.9, 0.1, 0.1]], LABELS)
    assert a == threshold_sweep([[1, 0, 0]], [[0.9, 0.1, 0.1]], LABELS)


# -- torch 部分：没装 [ml] 就整段跳过 ----------------------------------------

torch = pytest.importorskip("torch")


def _tiny_tokenizer(tmp_name: str):
    from transformers import BertTokenizer

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "买",
        "大",
        "了",
        "想",
        "退",
        "货",
        "运",
        "费",
        "快",
        "递",
        "到",
        "发",
        "票",
        "尺",
        "码",
    ]
    path = FIXTURES / tmp_name
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    try:
        return BertTokenizer(vocab_file=str(path), do_lower_case=True)
    finally:
        path.unlink(missing_ok=True)


def test_train_model_smoke_writes_artifacts_and_reloads():
    from transformers import BertConfig, BertForSequenceClassification

    from app.topics.infer import TopicClassifier
    from app.topics.taxonomy import CATEGORY_NAMES, taxonomy_hash
    from app.topics.training import TrainConfig, train_model

    out_dir = FIXTURES / "_tmp_train"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    tokenizer = _tiny_tokenizer("_tmp_vocab.txt")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(CATEGORY_NAMES),
        problem_type="multi_label_classification",
    )
    model = BertForSequenceClassification(config)
    rows = [
        {"text": "买大了想退货", "labels": ["尺码", "退换货"]},
        {"text": "快递还没到", "labels": ["物流"]},
        {"text": "运费谁出", "labels": ["运费"]},
        {"text": "发票开错了", "labels": ["发票"]},
    ]
    try:
        metrics = train_model(
            TrainConfig(epochs=2, batch_size=2, max_length=16, patience=1, lr=1e-3),
            rows,
            rows[:2],
            device="cpu",
            out_dir=out_dir,
            taxonomy_hash=taxonomy_hash(),
            model=model,
            tokenizer=tokenizer,
        )
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "thresholds.json",
            "metrics.json",
            "taxonomy.json",
        ):
            assert (out_dir / name).is_file(), f"缺产物 {name}"
        # transformers v5 把词表收进 tokenizer.json，v4 是 vocab.txt，两种都算齐全
        assert (out_dir / "tokenizer.json").is_file() or (out_dir / "vocab.txt").is_file()
        thresholds = json.loads((out_dir / "thresholds.json").read_text(encoding="utf-8"))
        thresholds = thresholds.get("thresholds", thresholds)
        assert set(thresholds) == set(CATEGORY_NAMES)
        assert metrics["val_macro_f1"] >= 0.0
        assert metrics["epochs_run"] >= 1
        assert (out_dir / "metrics.json").is_file()

        classifier = TopicClassifier.load(out_dir)
        prediction = classifier.predict(["买大了想退货"])[0]
        assert prediction.labels, "至少要命中一个类目"
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
