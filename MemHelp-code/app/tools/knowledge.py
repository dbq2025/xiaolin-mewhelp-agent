"""``query_faq`` —— 按关键词查 ``faq`` 表。

## 与 Task 5 那三个工具的关键区别

这个工具**真的连数据**（``faq`` 表）。为此它接受一个 ``faq_repo`` 参数，
类型是 ``app.db.repositories.FaqRepository`` 的**结构**（duck typing）——
测试可以塞内存假仓库进来，于是本文件的用例不需要 MySQL。

## 验收标准 3 的落点

「邮费是多少」查不到，是**预期结果**，必须表现为 ``ok=true, count=0``：
模型收到的信号是「查了，库里确实没有这条」，于是它如实回答；
若做成 ``ok=false``，模型收到的信号是「工具坏了」，它会道歉或编造 ——
那样漏召回就观察不到了，也就失去了留给 ch03 做改造前基线的作用。

## 转义由仓库层负责

本工具把 ``keyword`` **原样**传给 ``faq_repo.search()``。
``LIKE`` 的通配符转义属于「怎么拼 SQL」，在 ``FaqRepository`` 里做。
这里再转一次会变成**双重转义**（``100%`` → ``100\\%`` → ``100\\\\%``），
结果是本来能命中的词反而查不到 —— 而且看起来像「库里没数据」。
"""

import json
from typing import Protocol

from app.db.repositories import FaqHit

#: 单次返回的最大条数。命中太多会把上下文预算一次吃光，
#: 而且模型也不需要 20 条 FAQ 来回答一个问题。
MAX_FAQ_ITEMS = 5


class FaqSearchRepo(Protocol):
    """本工具对 FAQ 仓库的最小依赖。只要求这一个方法。"""

    async def search(self, keyword: str, limit: int = 5) -> list[FaqHit]: ...


async def query_faq(keyword: str, *, faq_repo: FaqSearchRepo) -> str:
    """按关键词检索常见问题。

    :param keyword: 检索关键词。取用户问题里最核心的名词，不要整句照抄
        （``LIKE`` 是子串匹配，整句几乎必然零命中）。
    :param faq_repo: FAQ 仓库。生产用 ``FaqRepository``，测试用内存假仓库。
    :return: JSON 字符串，含 ``ok`` / ``keyword`` / ``count`` / ``items``
        （每项含 ``question`` / ``answer`` / ``category``）。

        ``count == 0`` 是**正常的业务结果**，不是错误。

    :raises Exception: 仓库层的异常**原样冒出去**，由 ``executor.py`` 分类成
        ``ok=false``。这里不能吞成空结果 —— 否则「数据库挂了」与「真的没查到」
        会变成同一个信号。
    """
    keyword = keyword.strip()
    hits = await faq_repo.search(keyword, MAX_FAQ_ITEMS)

    return json.dumps(
        {
            "ok": True,
            "keyword": keyword,
            "count": len(hits),
            "items": [
                {"question": hit.question, "answer": hit.answer, "category": hit.category}
                for hit in hits
            ],
        },
        ensure_ascii=False,
    )
