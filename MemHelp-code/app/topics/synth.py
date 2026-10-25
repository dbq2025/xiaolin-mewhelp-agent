"""ch10 合成语料 + 大模型预标：prompt 构造、输出解析、标签冲突检测。

真实低置信度问题只有 12 条去重，训练不起来。这里照 17 类权威表造模拟问题，
每条都要求模型**只打句子字面提到的诉求**，并在 prompt 里带上完整的类目 +
边界说明 —— 少放一条边界说明，模型就会把「修」打到退换货、「运费」打到物流。

解析器必须比模型更严格：模型偶尔会把 ``退款`` 这种不在表里的词当标签、
把整段 JSON 包进代码围栏、或在中途插一句解释。语料生成要跑 60 次调用，
一次坏输出不能让整批白跑，但**未知标签必须当场炸**——放过去就是一条脏标签
进了训练集，而且是最难查的那种。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.topics.taxonomy import CATEGORIES, Category, validate_labels

_SYSTEM = (
    "你是电商客服质检的标注专家。你只输出机器可解析的 JSON，不写解释、不加代码围栏。"
)

_PARSE_ERRORS = (json.JSONDecodeError, TypeError, ValueError)


@dataclass(frozen=True, slots=True)
class ParsedQuestion:
    text: str
    labels: list[str]


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    text: str
    labels: list[str]
    rationale: str


@dataclass(frozen=True, slots=True)
class Conflict:
    text: str
    generated: list[str]
    prelabeled: list[str]


@dataclass(frozen=True, slots=True)
class SynthResult:
    questions: list[ParsedQuestion] = field(default_factory=list)
    skipped: int = 0


def _taxonomy_block() -> str:
    lines = []
    for category in CATEGORIES:
        lines.append(f"{category.index}. {category.name}：{category.boundary}")
    return "\n".join(lines)


def generation_messages(
    category: Category,
    count: int,
    *,
    second_label: str | None = None,
    force_multi: bool = False,
) -> list[dict[str, str]]:
    """造模拟问题的 prompt。``second_label`` 用来定向造多诉求句子。

    ``force_multi`` 与「约三成」的区别很关键：只写「约三成」时模型实测只兑现
    了约 10%，训练集多标签比例从设计的 25% 掉到 10%，模型于是学会了「只报一个
    标签」—— 补造多诉求语料时必须用 ``force_multi`` 把话说死。
    """

    if force_multi and second_label:
        multi = (
            f"\n这 {count} 条**每一条**都必须同时提到「{category.name}」和「{second_label}」"
            "两个诉求，labels 必须同时给出这两个类目名，不许只给一个。"
        )
    elif second_label:
        multi = (
            f"\n其中约三成句子要**同时**提到「{category.name}」和「{second_label}」两个诉求，"
            "并给两个标签。"
        )
    else:
        multi = ""
    user = f"""照下面的权威类目表，写 {count} 条真实用户会说的电商客服问题，主题是「{category.name}」。

类目表（编号. 名称：边界说明）：
{_taxonomy_block()}

规则：
1. 每条一句话，20 到 40 个字，像真人打字，可以有错别字和口语。
2. 一条只写一个诉求，除非下面另有要求{multi}
3. labels 只填句子字面提到的诉求，用类目表里的名称，一个不多一个不少。
4. 句子只有寒暄、没有诉求的不要。
5. 每行输出一个 JSON 对象：{{"text": "...", "labels": ["..."]}}，不要输出数组、不要代码围栏。"""
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def labeling_messages(texts: Sequence[str]) -> list[dict[str, str]]:
    """预标 prompt：给一批句子，让模型逐条给标签 + 一句理由。"""

    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
    user = f"""照下面的权威类目表，给每条问题打多标签。

类目表（编号. 名称：边界说明）：
{_taxonomy_block()}

待标问题：
{numbered}

规则：
1. 只打句子字面提到的诉求。一句话提到几个就打几个，一个不多一个不少。
2. 标签必须用类目表里的名称，不许自造。
3. 拿不准主题、或只是寒暄的，打「其他」。
4. 每行输出一个 JSON 对象：{{"text": "...", "labels": ["..."], "rationale": "一句话说明为什么是这些标签"}}。"""
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def _iter_json_lines(raw: str) -> Iterable[dict[str, Any]]:
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        try:
            obj = json.loads(stripped)
        except _PARSE_ERRORS:
            continue
        if isinstance(obj, dict):
            yield obj


def _clean_label_list(obj: Any) -> list[str] | None:
    if not isinstance(obj, list) or not obj:
        return None
    labels = [str(x).strip() for x in obj if str(x).strip()]
    if not labels:
        return None
    validate_labels(labels)  # 未知 / 重复 / 「其他」共现 → TaxonomyError
    return labels


def parse_generated(raw: str) -> list[ParsedQuestion]:
    """解析生成结果。坏行跳过，未知标签直接抛 ``TaxonomyError``。"""

    out: list[ParsedQuestion] = []
    for obj in _iter_json_lines(raw):
        text = str(obj.get("text", "")).strip()
        labels = _clean_label_list(obj.get("labels"))
        if not text or labels is None:
            continue
        out.append(ParsedQuestion(text=text, labels=labels))
    return out


def parse_labeled(raw: str) -> list[ParsedLabel]:
    """解析预标结果。缺 rationale 的行丢掉（理由就是人工抽审的抓手）。"""

    out: list[ParsedLabel] = []
    for obj in _iter_json_lines(raw):
        text = str(obj.get("text", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        labels = _clean_label_list(obj.get("labels"))
        if not text or not rationale or labels is None:
            continue
        out.append(ParsedLabel(text=text, labels=labels, rationale=rationale))
    return out


def find_conflicts(
    generated: Sequence[ParsedQuestion], prelabeled: Sequence[ParsedLabel]
) -> list[Conflict]:
    """两次标注不一致的样本，给人工抽审优先看。"""

    by_text = {item.text: item.labels for item in prelabeled}
    conflicts: list[Conflict] = []
    for item in generated:
        other = by_text.get(item.text)
        if other is None:
            continue
        if set(other) != set(item.labels):
            conflicts.append(
                Conflict(text=item.text, generated=list(item.labels), prelabeled=list(other))
            )
    return conflicts
