"""Query 理解 —— 把口语问法归一到标准问法，并做**只在检索侧**的同义词扩展。

## 为什么分两级

1. **静态归一表**：常见口语问法（「钱啥时候能回来」）直接查表，零延迟、纯函数、可单测。
2. **模型改写**：表里没有、且看起来像口语的问法才调一次小模型。

每一问都过一次模型会多付一个 RTT，而绝大多数用户问法是标准问法 ——
「运费怎么算」含疑问词，但它本来就标准，改写它只是白花钱。

## 同义词只扩 BM25

BGE-M3 这类向量模型对同义表达本来就不敏感；把同义词拼进 dense 查询会稀释语义中心
（一句话变成三个意思）。BM25 是词面匹配，扩同义词才有收益。所以这里区分
``standard``（给嵌入用）与 ``bm25_text``（给 BM25 用）。

## 型号归一（真机探针逼出来的）

实测：语料里写 `XH-3000A` 时，BM25 对 `xh-3000a` / `XH 3000A` / `XH3000A`
**全部零命中** —— Milvus 的内置 chinese analyzer 是 jieba + cnalphanumonly，
不做大小写归一，连字符/空格的切分也不同。

分析器是用户定死的选型，所以问题在**查询侧**解决：把形如「字母+数字+可选字母」的
型号 token 统一成「大写 + 连字符」再进 BM25。代价是语料里的型号必须按同一规范书写
（``knowledge/商品型号手册.md`` 里写死了这条）。

## 缓存

按 ``(enabled, 原话)`` 缓存最终结果，FIFO 上限 512。**键里必须带 enabled** ——
关掉改写是排障手段，被缓存绕过就失去意义了。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_QUERY_CHARS",
    "LlmRewriter",
    "QueryUnderstander",
    "Rewriter",
    "STATIC_NORMALISE",
    "SYNONYMS",
    "UnderstoodQuery",
    "normalise_model_tokens",
]

#: 超长查询截断。上游嵌入/改写对这种输入只会更慢或直接拒，长度不是有效信息。
MAX_QUERY_CHARS = 512

#: 静态归一表：口语原话 → 标准问法。键是**去尾标点后**的整句。
STATIC_NORMALISE: Mapping[str, str] = {
    "钱啥时候能回来": "退款到账时间",
    "退钱多久到账": "退款到账时间",
    "退款多久能到": "退款到账时间",
    "包不包邮": "包邮门槛",
    "满多少免运费": "包邮门槛",
    "寄回去要我出钱吗": "退货运费承担",
    "退货邮费谁出": "退货运费承担",
    "拍下多久能发": "发货时效",
    "多久能发货": "发货时效",
    "碎了找谁赔": "质量问题赔付标准",
    "东西坏了怎么赔": "质量问题赔付标准",
    "尺码不合适能换吗": "换货条件",
    "开票要等多久": "发票开具时间",
}

#: 同义词表：领域词 → 用户可能说的其它说法。**只进 BM25 查询串**。
SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "包邮": ("免邮", "运费", "邮费"),
    "运费": ("邮费", "快递费"),
    "退款": ("退钱", "返款"),
    "退货": ("退回", "退掉"),
    "换货": ("换一个", "调换"),
    "发票": ("开票", "票据"),
    "赔付": ("赔偿", "补偿"),
    "发货": ("出库", "寄出"),
    "续航": ("电池", "使用时间"),
    "到账": ("入账", "到帐"),
}

#: 判定「像口语」的标记词。
_COLLOQUIAL_MARKERS: tuple[str, ...] = (
    "啥", "咋", "呢", "吧", "啊", "呀", "嘛",
    "多久", "多少", "怎么", "能不能", "可以不可以", "要不要",
)

#: 型号 token：两字母以上 + 可选连字符/空格 + 两位以上数字 + 可选尾字母。
#:
#: 两侧用**ASCII 边界断言**而不是 ``\b``：中文也是 ``\w``，``\b`` 在「型号紧贴中文」
#: 时（``XH3000A续航多久``）根本不成立 —— 那是用户最常写的一种形状，
#: 用 ``\b`` 会让归一静默失效（真机评审实测）。
_MODEL_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,})\s*[-－—]?\s*(\d{2,})([A-Za-z]?)(?![A-Za-z0-9])"
)

_TAIL_PUNCT = "？?。.！!~～…、,，:：;；\"'“”‘’ "


@dataclass(frozen=True)
class UnderstoodQuery:
    """一次 Query 理解的产物。

    :param original: 截断后的用户原话（保留原始说法，供入池与日志）。
    :param standard: 归一后的标准问法（**给嵌入用**）。
    :param bm25_text: 标准问法 + 同义词扩展（**给 BM25 用**）。
    :param expanded: 实际命中的同义词，进日志与报告。
    :param degraded: 改写失败/被跳过时为 ``True``，用于日志区分「没改写」与「改写失败」。
    """

    original: str
    standard: str
    bm25_text: str
    expanded: tuple[str, ...] = ()
    degraded: bool = False


class Rewriter(Protocol):
    """改写器的最小依赖。测试注入假实现即可完全离线。"""

    async def rewrite(self, query: str) -> str | None: ...


def normalise_model_tokens(text: str) -> str:
    """把型号 token 统一成「大写 + 连字符」。

    只认「字母 + 至少两位数字」的形状：`XH-3000A`、`XH 3000A`、`xh3000a` 都归一成
    `XH-3000A`；`满 99 元包邮`、`运费怎么算` 不受影响（没有字母前缀）。

    误伤面：`iPhone 15` 会变成 `IPHONE-15`。这类词在知识库里不存在，归一后同样查不到，
    所以最坏情况是「没命中」，不会变成「命中错的」。
    """

    def replace(match: re.Match[str]) -> str:
        prefix, digits, suffix = match.group(1), match.group(2), match.group(3)
        return f"{prefix.upper()}-{digits}{suffix.upper()}"

    return _MODEL_TOKEN.sub(replace, text)


def _strip_tail_punct(text: str) -> str:
    return text.rstrip(_TAIL_PUNCT)


def _looks_colloquial(text: str) -> bool:
    """看起来像口语吗？

    已经命中领域词（``SYNONYMS`` 的键）的问法一律不算口语 —— 它已经是标准问法，
    再改写一遍只是多一次调用。

    带型号 token 的问法也一律不改写：改写 prompt 没有「原样保留型号」的约束，
    模型完全可能把 ``XH-3000A`` 改写成「这款巡检仪」—— 型号一丢，BM25 直接全漏。
    """
    if any(term in text for term in SYNONYMS):
        return False
    if _MODEL_TOKEN.search(text):
        return False
    if any(marker in text for marker in _COLLOQUIAL_MARKERS):
        return True
    return len(text) <= 12


class LlmRewriter:
    """用聊天模型把口语问法改写成标准问法。

    严格 JSON 出参；解析失败/超时/上游异常一律返回 ``None``（由调用方回落原话）。
    **不抛异常**：改写是提质手段，不该让检索失败。
    """

    def __init__(self, model: object, *, timeout: float = 3.0) -> None:
        self._model = model
        self._timeout = timeout

    async def rewrite(self, query: str) -> str | None:
        messages = [
            SystemMessage(
                content=(
                    "你是客服知识库的检索助手。把用户的口语问题改写成一句"
                    "标准、完整、可检索的客服问法（不超过 30 字）。\n"
                    "只输出 JSON，形如 {\"rewritten\": \"退款到账时间\"}，不要解释、不要 markdown。\n"
                    "不要回答用户的问题，只做改写。"
                )
            ),
            HumanMessage(content=query),
        ]
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke(messages), timeout=self._timeout
            )
            reply = getattr(response, "content", response)
            if not isinstance(reply, str):
                return None
            return _parse_rewritten(reply)
        except Exception:  # noqa: BLE001 - 改写失败回落原话，不打断检索
            logger.warning("Query 改写失败，回落原话（query=%r）", query, exc_info=True)
            return None


def _parse_rewritten(reply: str) -> str | None:
    """从模型回复里取出 ``rewritten``。容忍 ```json 代码块包裹。"""
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    rewritten = parsed.get("rewritten")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return None
    return rewritten.strip()


class QueryUnderstander:
    """把用户原话变成 :class:`UnderstoodQuery`。

    :param rewriter: 口语改写器；``None`` 表示只用静态表。
    :param enabled: ``False`` 时完全跳过模型改写（排障用）。
    :param cache_size: 进程内缓存条数上限（FIFO 淘汰）。
    """

    def __init__(
        self,
        rewriter: Rewriter | None = None,
        *,
        enabled: bool = True,
        cache_size: int = 512,
    ) -> None:
        self._rewriter = rewriter
        self._enabled = enabled
        self._cache_size = cache_size
        self._cache: dict[tuple[bool, str], UnderstoodQuery] = {}

    async def understand(self, query: str) -> UnderstoodQuery:
        original = (query or "").strip()[:MAX_QUERY_CHARS]
        key = (self._enabled, original)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = await self._understand_uncached(original)
        self._remember(key, result)
        return result

    async def _understand_uncached(self, original: str) -> UnderstoodQuery:
        standard, degraded = await self._normalise(original)
        standard = normalise_model_tokens(standard)
        expanded = self._expand(standard)
        bm25_text = " ".join((standard, *expanded)).strip() if expanded else standard
        return UnderstoodQuery(
            original=original,
            standard=standard,
            bm25_text=bm25_text,
            expanded=expanded,
            degraded=degraded,
        )

    async def _normalise(self, original: str) -> tuple[str, bool]:
        static = STATIC_NORMALISE.get(_strip_tail_punct(original))
        if static is not None:
            return static, False
        if not self._enabled or self._rewriter is None:
            return original, False
        if not _looks_colloquial(original):
            return original, False

        rewritten = await self._rewriter.rewrite(original)
        if not rewritten:
            return original, True
        return rewritten, False

    @staticmethod
    def _expand(standard: str) -> tuple[str, ...]:
        expanded: list[str] = []
        for term, synonyms in SYNONYMS.items():
            if term not in standard:
                continue
            for synonym in synonyms:
                if synonym not in standard and synonym not in expanded:
                    expanded.append(synonym)
        return tuple(expanded)

    def _remember(self, key: tuple[bool, str], value: UnderstoodQuery) -> None:
        if self._cache_size <= 0:
            return
        if len(self._cache) >= self._cache_size:
            # FIFO：dict 保持插入序，丢掉最早的一条。
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[key] = value
