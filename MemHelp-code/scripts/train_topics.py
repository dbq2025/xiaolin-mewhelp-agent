"""ch10 训练入口：读切好的 train/val，全参微调 RoBERTa-wwm-ext，落盘到 ``models/topics/<run_id>/``。

    python scripts/train_topics.py --epochs 8 --batch-size 16

产物目录（六件）::

    config.json  model.safetensors  tokenizer.json(v5)/vocab.txt(v4)
    thresholds.json  metrics.json  taxonomy.json

``models/topics/latest`` 是给 sidecar 与批跑用的「当前模型」。Windows 建符号链接
要管理员权限，所以用 ``shutil.copytree`` 复制一份 —— 一个 run 几十 MB，值得。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from app.topics.training import TrainConfig, train_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED = ROOT / "models" / "pretrained" / "chinese-roberta-wwm-ext"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"找不到数据文件：{path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_id(now: float | None = None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now))


def _publish_latest(out_root: Path, run_dir: Path) -> Path:
    latest = out_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    return latest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch10 主题分类模型训练")
    parser.add_argument("--train", default="data/topics/splits/train_aug.jsonl")
    parser.add_argument("--val", default="data/topics/splits/val.jsonl")
    parser.add_argument("--out", default="models/topics")
    parser.add_argument("--model", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(
            f"找不到本地预训练快照：{model_path}\n"
            "先跑一次 snapshot_download 落到 models/pretrained/（见 dev-notes 阶段 3）。"
        )

    train_rows = _read_jsonl(Path(args.train))
    val_rows = _read_jsonl(Path(args.val))
    run_dir = Path(args.out) / (args.run_id or _run_id())

    config = TrainConfig(
        model_name=str(model_path),
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    print(
        f"[train] train={len(train_rows)} val={len(val_rows)} "
        f"batch={config.batch_size} max_len={config.max_length} -> {run_dir}",
        flush=True,
    )
    try:
        metrics = train_model(
            config, train_rows, val_rows, device=args.device, out_dir=run_dir
        )
    except RuntimeError as exc:  # 显存不够：降 batch 再试一次，日志里留痕
        if "out of memory" not in str(exc).lower():
            raise
        print(f"[train] OOM，降 batch 到 {config.batch_size // 2} + 累积 2 重试", flush=True)
        config = TrainConfig(
            **{
                **config.__dict__,
                "batch_size": max(1, config.batch_size // 2),
                "grad_accum": config.grad_accum * 2,
            }
        )
        metrics = train_model(
            config, train_rows, val_rows, device=args.device, out_dir=run_dir
        )

    latest = _publish_latest(Path(args.out), run_dir)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "latest": str(latest),
                "best_epoch": metrics["best_epoch"],
                "epochs_run": metrics["epochs_run"],
                "val_macro_f1": metrics["val_macro_f1"],
                "val_macro_f1_tuned": metrics["val_macro_f1_tuned"],
                "elapsed_s": metrics["elapsed_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
