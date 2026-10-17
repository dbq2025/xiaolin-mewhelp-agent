"""``query_faq`` —— 混合检索（BM25 + 稠密）+ 重排后的「带编号证据」。

## 契约只增不改

ch02 的出参是 ``{ok, keyword, count, items:[{question, answer, category}]}``；
ch04 在这一层**只增字段**：每个 item 多出 ``n`` / ``chunk_id`` / ``section_path`` /
``quote``，顶层多出 ``refused`` / ``refusal_source`` / ``refusal_reason``。
前端徽章、历史评估集、工具 Schema 全都还认这个形状。

## ``items`` 的顺序是「首尾编排」，不是相关度序

最相关的两条落在首尾、最弱的压中间（``citations.prompt_order``）——
模型对上下文中段的注意力最弱。但每个 item 的 ``n`` 仍是**相关度序号**，
不随呈现顺序变化：用户在页面上点 ``[3]`` 必须看到同一条。

## 拒答时 ``items`` 为空

``refused=true`` 时不给证据：给了模型就可能「手上有一堆材料舍不得浪费」而硬答。
拒答话术由编排层直接吐，这里只负责把事实（为什么拒）带上去。

## 为什么不降级回关键词

检索层异常原样冒给 ``executor.py``：偷着回退会让「向量库/重排故障」伪装成
「检索功能正常但结果一般」，而故障被藏起来比故障本身更贵。
"""

from __future__ import annotations

import json
from typing import Protocol, Sequence

from app.knowledge.citations import (
    Evidence,
    evidence_payload,
    prompt_order,
)
from app.knowledge.pipeline import RetrievalOutcome

__all__ = ["AnswerContextLike", "query_faq"]

#: 兜底取数（与 ``Settings.rag_top_k`` 的默认值一致，仅用于文档说明）。
MAX_FAQ_ITEMS = 10

#: 单条证据喂给模型的正文上限。
#:
#: 为什么必须截断：``executor`` 对工具结果有 ``MAX_TOOL_RESULT_CHARS = 4000`` 的硬上限，
#: 10 条 × 完整答案（最长 600 字）再加元数据会直接超 —— 超了之后模型收到的是被切断的
#: Python repr，而不是 JSON：它看不懂，引用编号也就不可能标对。
#: 截断只影响**给模型看的**那一份；落库的引用快照由编排层按 chunk 回库取全文补全。
MAX_TOOL_ANSWER_CHARS = 220


class AnswerContextLike(Protocol):
    """工具层对「在线检索管线」的最小依赖。"""

    async def answer_context(
        self, query: str, *, category: str | None = None
    ) -> RetrievalOutcome: ...


def _item(evidence: Evidence) -> dict:
    answer = evidence.quote or ""
    truncated = len(answer) > MAX_TOOL_ANSWER_CHARS
    item = {
        "n": evidence.n,
        "chunk_id": evidence.chunk_id,
        "section_path": evidence.section_path,
        "category": evidence.category,
        # question/answer 是 ch02 以来的「问答对语义」。
        # 不再额外发一份 ``quote``（它与 answer 同源）—— 那是白白多一倍字节，
        # 直接把 10 条证据顶过 executor 的 4000 字上限。
        "question": evidence.question or (
            evidence.section_path.split(" / ")[-1] if evidence.section_path else ""
        ),
        "answer": answer[:MAX_TOOL_ANSWER_CHARS],
    }
    if truncated:
        # 让编排层知道「这条给模型的只是节选、落库前要回库取全文」。
        item["truncated"] = True
    return item


async def query_faq(
    keyword: str,
    *,
    retriever: AnswerContextLike,
    category: str | None = None,
) -> str:
    """检索平台规则/商品参数，返回带编号的证据。

    :param keyword: **用户问题的完整原话**。不要截成名词 —— Query 理解与向量检索
        都靠整句语义；截词反而丢信息。空白串直接返回空结果（不调检索）。
    :param category: 用户明确点名的品类（如「售后政策」「商品参数」）。
        传了就**先过滤再检索**；``None`` 表示不过滤。
    :return: JSON 字符串，见模块 docstring。

    :raises Exception: 检索层的异常**原样冒出去**，由 ``executor.py`` 分类成
        ``ok=false``。这里不能吞成空结果 —— 否则「基础设施挂了」与「真的没查到」
        会变成同一个信号。
    """
    query = (keyword or "").strip()
    if not query:
        return _payload(query, outcome=None)

    outcome = await retriever.answer_context(query, category=category)
    return _payload(query, outcome=outcome)


def _payload(query: str, *, outcome: RetrievalOutcome | None) -> str:
    if outcome is None:
        return json.dumps(
            {
                "ok": True,
                "keyword": query,
                "count": 0,
                "items": [],
                "refused": False,
                "refusal_source": None,
                "refusal_reason": None,
            },
            ensure_ascii=False,
        )

    items: Sequence[dict] = (
        []
        if outcome.refused
        else [_item(evidence) for evidence in prompt_order(outcome.evidence)]
    )
    return json.dumps(
        {
            "ok": True,
            "keyword": query,
            "count": len(items),
            "items": list(items),
            "refused": outcome.refused,
            "refusal_source": outcome.refusal_source,
            "refusal_reason": outcome.refusal_reason,
        },
        ensure_ascii=False,
    )
