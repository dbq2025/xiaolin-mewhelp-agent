"""``query_faq`` —— 向量语义检索（BGE-M3 + Milvus）。

## 契约没变，内核换了

| 项 | ch02（关键词） | ch03（向量） |
|---|---|---|
| 入参 | ``keyword: str`` | **不变** |
| 出参 | ``{ok, keyword, count, items:[{question, answer, category}]}`` | **不变** |
| 匹配方式 | ``LIKE %kw%`` | 语义相似度 Top-K |

所以这个文件只做三件事：调检索器、把命中映射成原来的 JSON 形状、把异常原样放行。
**任何改写出参键名的冲动都要停下来** —— 前端徽章、评估集、ch02 的用例都钉着它。

## 为什么不降级回关键词

本章明确不做混合检索/关键词兜底。Milvus 挂了就让异常冒出去，由 ``executor.py``
分类成 ``ok=false``：模型会如实告诉用户「暂时查不到规则」。
偷偷回退到 ``LIKE`` 会让「向量库故障」伪装成「检索功能正常但结果一般」——
故障被藏起来，比故障本身更贵。

## 空白关键词

模型偶尔会传纯空白。空串直接返回 ``count=0`` 且**不调检索器**（省一次前向）；
纯标点（如「？」）走正常检索路径，交给嵌入模型处理 —— 它不会因此崩溃，
而且对「？」这种输入返回空结果本来就是对的。
"""

import json

from app.knowledge.retrieval import DEFAULT_TOP_K, KnowledgeHit, RetrieverLike

__all__ = ["MAX_FAQ_ITEMS", "RetrieverLike", "query_faq"]

#: 兜底取数（与 ``Settings.rag_top_k`` 的默认值一致）。
#: 真正生效的值来自配置：``KnowledgeRetriever`` 构造时用 ``RAG_TOP_K``，
#: ``query_faq`` 不传 ``top_k`` 时由检索器决定。
MAX_FAQ_ITEMS = DEFAULT_TOP_K


async def query_faq(
    keyword: str, *, retriever: RetrieverLike, top_k: int | None = None
) -> str:
    """按语义检索平台规则库，返回最相关的问答对。

    :param keyword: **用户问题的完整原话**（如「邮费是多少」）。不要截成名词 ——
        向量检索靠整句语义，截词反而丢信息。空白串直接返回空结果。
    :param retriever: 检索器。生产用 ``KnowledgeRetriever``，测试用假实现。
    :param top_k: 最多返回几条；``None`` 表示用检索器配置的默认值（``RAG_TOP_K``）。
    :return: JSON 字符串，含 ``ok`` / ``keyword`` / ``count`` / ``items``
        （每项含 ``question`` / ``answer`` / ``category``）。

        ``count == 0`` 是**正常的业务结果**，不是错误。

    :raises Exception: 检索层的异常**原样冒出去**，由 ``executor.py`` 分类成
        ``ok=false``。这里不能吞成空结果 —— 否则「向量库挂了」与「真的没查到」
        会变成同一个信号。
    """
    query = keyword.strip()
    hits: list[KnowledgeHit] = [] if not query else await retriever.search(query, top_k)

    return json.dumps(
        {
            "ok": True,
            "keyword": query,
            "count": len(hits),
            "items": [
                {"question": hit.question, "answer": hit.answer, "category": hit.category}
                for hit in hits
            ],
        },
        ensure_ascii=False,
    )
