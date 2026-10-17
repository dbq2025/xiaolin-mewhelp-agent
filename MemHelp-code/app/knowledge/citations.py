"""证据编号、首尾编排与引用解析 —— 纯函数，无 IO。

## 编号按相关度固定，不随呈现顺序变

用户在页面上点开 ``[3]``，看到的那条必须和正文里 ``[3]`` 指的同一条。如果编号跟着
「塞进 prompt 的顺序」变，同一段回答在两次渲染里 ``[3]`` 指的不是同一块 ——
点开就是错的证据。

## 首尾编排（lost in the middle）

模型对上下文**中段**的注意力最弱。所以把最相关的放在首尾、最不相关的压中间：

```
k=10 → [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
k=5  → [1, 3, 5, 4, 2]
```

通式：``[n | n 为奇数且升序] + [n | n 为偶数且降序]``。编号本身不变。

## 解析只认半角 ``[n]``

全角 ``【1】``、``[1,2]``、``[0]`` 都不算合法引用：前两者不是我们约定的角标格式，
后者不存在。越界编号**不删正文**，只丢弃映射 —— 模型多写一个角标，
不该让整段回答消失。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "CITATION_PATTERN",
    "CitationParse",
    "Evidence",
    "build_evidence",
    "evidence_from_payload",
    "evidence_payload",
    "parse_citations",
    "prompt_order",
]

#: 正文角标：半角方括号 + 1~2 位数字。
CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Evidence:
    """一条证据。

    :param n: 相关度序号（1 起），**用户看到的角标就是它**。
    :param question: 该块的问法/标题（ch02 出参里的 ``items[].question`` 就是它），
        只在工具出参里用，不进落库快照。
    :param quote: 证据原文快照（chunk 的 answer）。落库后即使 chunk 被重切，
        审计时仍能看到当时依据。
    :param score: 重排分或融合分，只进日志/报告。
    :param source: ``vector`` / ``bm25`` / ``hybrid`` / ``rerank``。
        几个内部字段有默认值，是为了让「从工具出参还原证据」不必硬塞它们。
    """

    n: int
    chunk_id: int
    section_path: str
    category: str
    quote: str
    question: str = ""
    score: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class CitationParse:
    """一次引用解析的结果。

    :param valid: 能映射到证据的编号（去重、保序）。
    :param dropped: 映射不到的编号（越界 / 0 / 已无对应证据）。
    :param coverage: 被引用证据数 / 证据总数。低覆盖意味着有证据没用上。
    """

    valid: tuple[int, ...]
    dropped: tuple[int, ...]
    coverage: float


def build_evidence(chunks: Sequence[object]) -> tuple[Evidence, ...]:
    """按**传入顺序**（= 相关度）编号 1..N。

    传入的应该是已经排好序的 ``RetrievedChunk`` 序列；这里不再排序 ——
    排序是检索层的职责，重复排序会让「谁决定顺序」变得含糊。
    """
    return tuple(
        Evidence(
            n=index,
            chunk_id=int(getattr(chunk, "chunk_id")),
            section_path=getattr(chunk, "section_path", None) or "",
            category=getattr(chunk, "category", "") or "",
            quote=getattr(chunk, "answer", "") or "",
            question=getattr(chunk, "question", "") or "",
            score=float(getattr(chunk, "score", 0.0) or 0.0),
            source=getattr(chunk, "source", "") or "",
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def prompt_order(evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
    """首尾编排：奇数位升序 + 偶数位降序（见模块 docstring）。"""
    odds = [item for item in evidence if item.n % 2 == 1]
    evens = [item for item in evidence if item.n % 2 == 0]
    return tuple(odds) + tuple(reversed(evens))


def parse_citations(answer: str, evidence: Sequence[Evidence]) -> CitationParse:
    """解析正文里的 ``[n]``，分成「能映射」与「映射不到」两组。

    **不修改 ``answer``**：正文原样返回给用户，丢弃只发生在映射层。
    """
    known = {item.n for item in evidence}
    valid: list[int] = []
    dropped: list[int] = []

    for match in CITATION_PATTERN.finditer(answer or ""):
        number = int(match.group(1))
        if number in known:
            if number not in valid:
                valid.append(number)
        elif number not in dropped:
            dropped.append(number)

    coverage = len(valid) / len(evidence) if evidence else 0.0
    return CitationParse(valid=tuple(valid), dropped=tuple(dropped), coverage=coverage)


def evidence_payload(evidence: Sequence[Evidence]) -> list[dict]:
    """落库/传输用的快照。**不含** ``score`` / ``source``。

    为什么砍掉这两个：它们是「这一轮怎么排出来的」调试信息，用户点开引用时不需要，
    而 ``messages.citations`` 会跟着每条回答长期留存 —— 存少一点更好。
    """
    return [
        {
            "n": item.n,
            "chunk_id": item.chunk_id,
            "section_path": item.section_path,
            "category": item.category,
            "quote": item.quote,
        }
        for item in evidence
    ]


def evidence_from_payload(items: Sequence[dict]) -> tuple[Evidence, ...]:
    """把工具出参的 ``items``（或库里的快照）还原成 :class:`Evidence`。

    编排层只需要编号与展示字段；缺 ``score`` / ``source`` 时用默认值补齐 ——
    不为了补两个内部字段去改工具契约（``query_faq`` 的出参是跨章节契约）。
    """
    return tuple(
        Evidence(
            n=int(item["n"]),
            chunk_id=int(item["chunk_id"]),
            section_path=item.get("section_path") or "",
            category=item.get("category") or "",
            quote=item.get("quote") or "",
            question=item.get("question") or "",
            score=float(item.get("score") or 0.0),
            source=item.get("source") or "",
        )
        for item in items
    )
