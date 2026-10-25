"""ch10 训练：RoBERTa-wwm-ext 全参微调 + 每类阈值扫描。

三个刻意的取舍：

1. **不用 HF ``Trainer``**。Trainer 把早停、阈值、产物落盘都藏在回调里，本章
   要的恰恰是「每个 epoch 在 val 上看 macro-F1、按 patience 停、最后拿 val 扫
   每类阈值」这三件事写在自己眼皮底下 —— 报告里的数字才解释得清。
2. **纯函数与 torch 分离**。``multilabel_targets`` / ``per_label_f1`` /
   ``macro_f1`` / ``threshold_sweep`` 只用标准库，没装 ``[ml]`` 也能跑、也能
   单测；torch 全部函数级 import，import 本模块不会拖 2.7 GiB 的轮子。
3. **17 个独立 sigmoid + BCE**，不是 softmax。「买大了想退」要同时命中尺码与
   退换货，softmax 归一化后只有一个赢家，结构性做不到多标签。

早停盯的是 **0.5 阈值**下的 val macro-F1：阈值扫描是训练之后的一次性动作，
拿扫完的分数去早停等于用 val 调了两遍，报告会虚高。
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.topics.infer import judge_scores
from app.topics.taxonomy import (
    CATEGORY_NAMES,
    taxonomy_hash as compute_taxonomy_hash,
    taxonomy_payload,
)

#: 早停与「最优轮」都看这个阈值，避免用扫过的阈值再调一遍。
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 64
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 8
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 42
    label_smoothing: float = 0.0


# -- 纯函数：可单测的部分 ----------------------------------------------------


def multilabel_targets(
    labels: Sequence[Sequence[str]], label_names: Sequence[str]
) -> list[list[float]]:
    """标签名 → multi-hot。未知标签直接抛，绝不静默丢。"""

    index = {name: i for i, name in enumerate(label_names)}
    out: list[list[float]] = []
    for row in labels:
        vector = [0.0] * len(label_names)
        for name in row:
            if name not in index:
                raise ValueError(f"未知类目：{name}")
            vector[index[name]] = 1.0
        out.append(vector)
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def per_label_f1(
    y_true: Sequence[Sequence[float]],
    y_pred: Sequence[Sequence[float]],
    label_names: Sequence[str],
) -> dict[str, float]:
    """每类 F1。support 与预测都为 0 的类目记 0.0（无定义，但不许炸）。"""

    scores: dict[str, float] = {}
    for index, name in enumerate(label_names):
        tp = fp = fn = 0
        for truth, pred in zip(y_true, y_pred):
            t = bool(truth[index])
            p = bool(pred[index])
            tp += int(t and p)
            fp += int(p and not t)
            fn += int(t and not p)
        scores[name] = _f1(tp, fp, fn)
    return scores


def macro_f1(
    y_true: Sequence[Sequence[float]],
    y_pred: Sequence[Sequence[float]],
    label_names: Sequence[str],
) -> float:
    scores = per_label_f1(y_true, y_pred, label_names)
    return sum(scores.values()) / len(scores) if scores else 0.0


def threshold_sweep(
    y_true: Sequence[Sequence[float]],
    y_prob: Sequence[Sequence[float]],
    label_names: Sequence[str],
    *,
    lo: float = 0.2,
    hi: float = 0.7,
    step: float = 0.05,
) -> dict[str, dict[str, float]]:
    """每类在自己的 val 分数上扫阈值，取 F1 最高的点（并列取较小阈值）。

    并列时取较小阈值是刻意的：多标签场景漏标比多标更难补，宁可多召回。
    """

    count = int(round((hi - lo) / step)) + 1
    thresholds = [round(lo + i * step, 4) for i in range(count)]
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(label_names):
        best_threshold = thresholds[0]
        best_f1 = -1.0
        for threshold in thresholds:
            tp = fp = fn = 0
            for truth, prob in zip(y_true, y_prob):
                t = bool(truth[index])
                p = prob[index] >= threshold
                tp += int(t and p)
                fp += int(p and not t)
                fn += int(t and not p)
            score = _f1(tp, fp, fn)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        out[name] = {"threshold": best_threshold, "f1": round(best_f1, 6)}
    return out


# -- torch 部分 --------------------------------------------------------------


def encode_batch(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, Any]:
    return dict(
        tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    )


def _batches(count: int, batch_size: int, *, shuffle: bool, rng: random.Random) -> list[list[int]]:
    order = list(range(count))
    if shuffle:
        rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def _autocast(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _forward_logits(model: Any, tokenizer: Any, texts: Sequence[str], config: TrainConfig, device: str):
    import torch

    batch = encode_batch(tokenizer, texts, config.max_length)
    batch = {key: value.to(device) for key, value in batch.items()}
    with _autocast(device):
        logits = model(**batch).logits
    return logits.float()


def _val_probabilities(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: TrainConfig,
    device: str,
) -> list[list[float]]:
    import torch

    model.eval()
    probs: list[list[float]] = []
    with torch.inference_mode():
        for chunk in _batches(len(rows), config.batch_size, shuffle=False, rng=random.Random(0)):
            texts = [rows[i]["text"] for i in chunk]
            logits = _forward_logits(model, tokenizer, texts, config, device)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
    return probs


def _binary(probs: Sequence[Sequence[float]], threshold: float = DEFAULT_THRESHOLD) -> list[list[int]]:
    return [[int(value >= threshold) for value in row] for row in probs]


def _soft_targets(targets: Sequence[Sequence[float]], smoothing: float) -> list[list[float]]:
    if smoothing <= 0:
        return [list(row) for row in targets]
    return [[value * (1 - smoothing) + smoothing / 2 for value in row] for row in targets]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def train_model(
    config: TrainConfig,
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    *,
    device: str = "cuda",
    out_dir: Path,
    taxonomy_hash: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """全参微调 + 早停 + val 阈值扫描 + 落盘六件产物。返回 ``metrics.json`` 内容。"""

    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    if not train_rows:
        raise ValueError("训练集为空：先跑 scripts/build_topic_data.py")
    if not val_rows:
        raise ValueError("验证集为空：没法早停，也没法扫阈值")

    started = time.perf_counter()
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(CATEGORY_NAMES),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
    model.to(device)

    train_targets = _soft_targets(
        multilabel_targets([row["labels"] for row in train_rows], CATEGORY_NAMES),
        config.label_smoothing,
    )
    val_targets = multilabel_targets([row["labels"] for row in val_rows], CATEGORY_NAMES)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / (config.batch_size * config.grad_accum))
    total_steps = max(1, steps_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state: dict[str, Any] | None = None
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = _batches(len(train_rows), config.batch_size, shuffle=True, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        for step, chunk in enumerate(batches, 1):
            texts = [train_rows[i]["text"] for i in chunk]
            labels = torch.tensor([train_targets[i] for i in chunk], dtype=torch.float32)
            labels = labels.to(device)
            batch = {
                key: value.to(device)
                for key, value in encode_batch(tokenizer, texts, config.max_length).items()
            }
            with _autocast(device):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels) / config.grad_accum
            loss.backward()
            epoch_loss += float(loss.detach()) * config.grad_accum
            if step % config.grad_accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        probs = _val_probabilities(model, tokenizer, val_rows, config, device)
        val_f1 = macro_f1(val_targets, _binary(probs), CATEGORY_NAMES)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(epoch_loss / max(1, len(batches)), 6),
                "val_macro_f1": round(val_f1, 6),
            }
        )
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    probs = _val_probabilities(model, tokenizer, val_rows, config, device)
    swept = threshold_sweep(val_targets, probs, CATEGORY_NAMES)
    swept_thresholds = {name: value["threshold"] for name, value in swept.items()}
    # 与 sidecar 同一套判定（含「其他」独占 / 兜底），否则报告里的 tuned 指标
    # 描述的是一个线上并不存在的分类器。
    tuned_pred = []
    for row in probs:
        decision = judge_scores(dict(zip(CATEGORY_NAMES, row)), swept_thresholds)
        tuned_pred.append([int(name in decision.labels) for name in CATEGORY_NAMES])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    snapshot = {**taxonomy_payload(), "hash": taxonomy_hash or compute_taxonomy_hash()}
    _write_json(out_dir / "taxonomy.json", snapshot)
    _write_json(
        out_dir / "thresholds.json",
        {
            "thresholds": swept_thresholds,
            "source": "val",
            "macro_f1_at_0.5": round(best_f1, 6),
            "max_length": int(config.max_length),
        },
    )
    metrics: dict[str, Any] = {
        "model_name": config.model_name,
        "device": device,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "val_macro_f1": round(best_f1, 6),
        "val_macro_f1_tuned": round(macro_f1(val_targets, tuned_pred, CATEGORY_NAMES), 6),
        "per_label_f1_at_0.5": per_label_f1(val_targets, _binary(probs), CATEGORY_NAMES),
        "per_label_f1_tuned": per_label_f1(val_targets, tuned_pred, CATEGORY_NAMES),
        "thresholds": {name: value["threshold"] for name, value in swept.items()},
        "history": history,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "seed": config.seed,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "label_smoothing": config.label_smoothing,
        "taxonomy_hash": taxonomy_hash or compute_taxonomy_hash(),
    }
    _write_json(out_dir / "metrics.json", metrics)
    return metrics
