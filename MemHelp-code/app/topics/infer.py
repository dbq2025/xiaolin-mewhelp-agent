"""ch10：多标签推理 —— 分数怎么变成标签。

三件事必须分开，否则「线上结果和评测报告对不上」这类问题查不动：

1. **打分**（``scorer``：一批文本 → 每行 17 个 sigmoid 分数）—— 只有这里碰模型；
2. **判定**（阈值、 「其他」独占、空集兜底）—— 纯 Python，毫秒级可单测；
3. **产物校验**（类目表哈希、阈值键集合、分数形状）—— 不成立就拒绝服务。

第 3 条是这份代码存在的核心理由：模型权重和类目表是**两份**会各自漂移的产物，
训练用旧表、上线换新表，模型会把「尺码」的分打到「发票」那一列上，而准确率
只会「莫名其妙掉一点」，没人查得出来。哈希对不上就直接崩，比默默错位好。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.topics.taxonomy import (
    CATEGORY_NAMES,
    OTHER_LABEL,
    names_at,
    taxonomy_hash,
    validate_labels,
)

#: 「一批文本 → 每行 17 个分数」的可替换打分函数。测试注入假实现，线上是 torch。
Scorer = Callable[[Sequence[str]], list[list[float]]]

#: 与训练时一致的最大长度。改这里必须同时重训，否则分数分布会漂。
MAX_LENGTH = 64


class TopicModelError(RuntimeError):
    """模型产物与代码不一致，或打分结果形状不对 —— 一律拒绝服务，不猜。"""


@dataclass(frozen=True, slots=True)
class TopicPrediction:
    labels: list[str]
    scores: dict[str, float]
    fallback: bool


def _normalize_thresholds(thresholds: Mapping[str, float]) -> dict[str, float]:
    keys = set(thresholds)
    expected = set(CATEGORY_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise TopicModelError(
            "thresholds 必须正好覆盖 17 个类目："
            f"缺 {missing or '无'}，多 {extra or '无'}"
        )
    return {name: float(thresholds[name]) for name in CATEGORY_NAMES}


def judge_scores(
    scores: Mapping[str, float], thresholds: Mapping[str, float]
) -> TopicPrediction:
    """把一行 17 个概率判成标签。训练侧算「调阈值后」的指标也走这里，
    保证报告里的 F1 与 sidecar 实际行为是同一套规则。"""

    fired = [
        index
        for index, name in enumerate(CATEGORY_NAMES, start=1)
        if name != OTHER_LABEL and scores[name] >= thresholds[name]
    ]
    if fired:
        # 「其他」即使过阈值也让位：它是兜底类，不与其他类共现（D10）。
        return TopicPrediction(
            labels=names_at(fired), scores=dict(scores), fallback=False
        )
    # 一个都没过阈值 → 兜底「其他」。这条路径的比例要报出来：
    # 兜底率异常升高通常意味着模型退化了，而不是「问题真的都归其他」。
    return TopicPrediction(
        labels=validate_labels([OTHER_LABEL]), scores=dict(scores), fallback=True
    )


class TopicClassifier:
    """阈值化 + 兜底的判定器。构造它不需要模型，``load()`` 才需要。"""

    def __init__(
        self,
        scorer: Scorer,
        thresholds: Mapping[str, float],
        model_taxonomy_hash: str,
        *,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self._scorer = scorer
        self._thresholds = _normalize_thresholds(thresholds)
        self._model_taxonomy_hash = model_taxonomy_hash
        self._max_length = int(max_length)

    @property
    def model_taxonomy_hash(self) -> str:
        return self._model_taxonomy_hash

    @property
    def max_length(self) -> int:
        return self._max_length

    @classmethod
    def from_parts(
        cls,
        scorer: Scorer,
        thresholds: Mapping[str, float],
        *,
        model_taxonomy_hash: str = "",
        current_hash: str | None = None,
        max_length: int = MAX_LENGTH,
    ) -> "TopicClassifier":
        if current_hash is not None and model_taxonomy_hash != current_hash:
            raise TopicModelError(
                "模型产物的 taxonomy 哈希与当前类目表不一致："
                f"模型 {model_taxonomy_hash!r} vs 当前 {current_hash!r}。"
                "改过类目表就必须重训，否则标签会整体错位。"
            )
        return cls(scorer, thresholds, model_taxonomy_hash, max_length=max_length)

    @classmethod
    def load(cls, model_dir: Path) -> "TopicClassifier":
        """从 ``models/topics/<run_id>/`` 装一个真模型。"""

        model_dir = Path(model_dir)
        taxonomy_path = model_dir / "taxonomy.json"
        thresholds_path = model_dir / "thresholds.json"
        for path in (taxonomy_path, thresholds_path):
            if not path.is_file():
                raise TopicModelError(f"模型产物缺文件：{path}")

        snapshot: dict[str, Any] = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        snapshot_hash = str(snapshot.get("hash") or taxonomy_hash(snapshot))
        snapshot_names = tuple(c.get("name", "") for c in snapshot.get("categories", []))
        if snapshot_names != CATEGORY_NAMES:
            raise TopicModelError(
                "taxonomy.json 的类目名/顺序与当前类目表不一致：快照 "
                f"{len(snapshot_names)} 类 vs 当前 {len(CATEGORY_NAMES)} 类。"
            )

        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        thresholds = payload.get("thresholds", payload) if isinstance(payload, dict) else payload
        max_length = int(
            payload.get("max_length", MAX_LENGTH) if isinstance(payload, dict) else MAX_LENGTH
        )

        return cls.from_parts(
            _torch_scorer(model_dir, max_length),
            thresholds,
            model_taxonomy_hash=snapshot_hash,
            current_hash=taxonomy_hash(),
            max_length=max_length,
        )

    def predict(self, texts: Sequence[str]) -> list[TopicPrediction]:
        texts = list(texts)
        if not texts:
            return []
        rows = self._scorer(texts)
        if len(rows) != len(texts):
            raise TopicModelError(
                f"打分函数返回 {len(rows)} 行，但输入 {len(texts)} 条 —— 行与文本必须一一对应"
            )
        return [self._judge(row) for row in rows]

    def _judge(self, row: Sequence[float]) -> TopicPrediction:
        if len(row) != len(CATEGORY_NAMES):
            raise TopicModelError(
                f"每行必须是 {len(CATEGORY_NAMES)} 个类目分数，实际 {len(row)} 个"
            )
        scores = {name: float(score) for name, score in zip(CATEGORY_NAMES, row)}
        return judge_scores(scores, self._thresholds)


def _torch_scorer(model_dir: Path, max_length: int = MAX_LENGTH) -> Scorer:
    """把 HF 权重包成 ``scorer``。torch/transformers 只在这里 import。"""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def scorer(texts: Sequence[str]) -> list[list[float]]:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**encoded).logits
            else:
                logits = model(**encoded).logits
            probabilities = torch.sigmoid(logits.float())
        return probabilities.cpu().tolist()

    return scorer
