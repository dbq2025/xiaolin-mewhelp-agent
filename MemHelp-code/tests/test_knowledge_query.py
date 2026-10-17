"""Query 理解：口语归一、同义词扩展、型号归一、缓存。

## 为什么改写要分两级

静态表命中就**不调模型** —— 一个纯函数能解决的问题不该付一次 RTT。
只有「静态表没收录、且看起来像口语」的问法才交给模型。

## 为什么同义词只扩 BM25

向量检索对同义表达本来就不敏感；把同义词拼进 dense 查询会稀释语义中心。
BM25 是词面匹配，扩同义词才有收益。所以 ``standard`` 与 ``bm25_text`` 是两个字段。
"""

from __future__ import annotations

import pytest

from app.knowledge.query import (
    MAX_QUERY_CHARS,
    LlmRewriter,
    QueryUnderstander,
    normalise_model_tokens,
)


class _FakeRewriter:
    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def rewrite(self, query: str) -> str | None:
        self.calls.append(query)
        return self.answer


# ---------------------------------------------------------------------------
# 静态表
# ---------------------------------------------------------------------------


async def test_static_table_hit_does_not_call_the_rewriter():
    rewriter = _FakeRewriter("不该被调用")
    out = await QueryUnderstander(rewriter=rewriter).understand("钱啥时候能回来")

    assert out.standard == "退款到账时间"
    assert rewriter.calls == []
    assert out.degraded is False


async def test_standard_question_with_domain_term_is_not_rewritten():
    """「运费怎么算」含疑问词，但已经命中领域词 —— 再改写就是白花一次调用。"""
    rewriter = _FakeRewriter("不该被调用")
    out = await QueryUnderstander(rewriter=rewriter).understand("运费怎么算")

    assert out.standard == "运费怎么算"
    assert rewriter.calls == []


# ---------------------------------------------------------------------------
# LLM 改写
# ---------------------------------------------------------------------------


async def test_colloquial_miss_uses_the_llm():
    rewriter = _FakeRewriter("发票开具流程")
    out = await QueryUnderstander(rewriter=rewriter).understand("那个票咋整啊")

    assert out.standard == "发票开具流程"
    assert rewriter.calls == ["那个票咋整啊"]
    assert out.degraded is False


async def test_rewriter_failure_falls_back_to_the_original():
    """改写失败不该让用户查不到东西。"""
    out = await QueryUnderstander(rewriter=_FakeRewriter(None)).understand("那个票咋整啊")

    assert out.standard == "那个票咋整啊"
    assert out.degraded is True


async def test_rewriting_can_be_disabled():
    rewriter = _FakeRewriter("发票开具流程")
    out = await QueryUnderstander(rewriter=rewriter, enabled=False).understand("那个票咋整啊")

    assert out.standard == "那个票咋整啊"
    assert rewriter.calls == []


async def test_model_queries_are_never_sent_to_the_rewriter():
    """改写 prompt 没要求保留型号 —— 模型可能把 `XH-3000A` 改写成「这款巡检仪」。

    型号一丢，BM25 直接全漏，所以带型号的问法一律不改写。
    """
    rewriter = _FakeRewriter("巡检仪续航时间")
    out = await QueryUnderstander(rewriter=rewriter).understand("xh3000a续航咋样啊")

    assert rewriter.calls == []
    assert out.standard == "XH-3000A续航咋样啊"


async def test_cache_avoids_a_second_llm_call():
    rewriter = _FakeRewriter("发票开具流程")
    understander = QueryUnderstander(rewriter=rewriter)

    first = await understander.understand("那个票咋整啊")
    second = await understander.understand("那个票咋整啊")

    assert len(rewriter.calls) == 1
    assert first == second


def test_the_api_reuses_one_understander_per_process():
    """缓存挂在实例上 —— 每请求新建实例等于缓存永不命中（真机评审实测）。

    spec §4 要的是「进程内 512 条缓存」，所以 API 层必须复用同一个理解器。
    """
    from app.api import _UNDERSTANDERS, _understander_for
    from app.config import Settings

    _UNDERSTANDERS.clear()
    settings = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m"
    )
    model = object()

    first = _understander_for(settings, model)
    second = _understander_for(settings, model)

    assert first is second
    _UNDERSTANDERS.clear()


async def test_cache_key_includes_the_enabled_flag():
    """关掉改写是排障手段；被缓存绕过就失去了意义。"""
    rewriter = _FakeRewriter("发票开具流程")
    on = QueryUnderstander(rewriter=rewriter, enabled=True)
    off = QueryUnderstander(rewriter=rewriter, enabled=False)

    assert (await on.understand("那个票咋整啊")).standard == "发票开具流程"
    assert (await off.understand("那个票咋整啊")).standard == "那个票咋整啊"


# ---------------------------------------------------------------------------
# 同义词与型号归一
# ---------------------------------------------------------------------------


async def test_synonyms_expand_bm25_text_only():
    out = await QueryUnderstander(enabled=False).understand("包邮")

    assert out.standard == "包邮"          # 不扩进 standard（dense 用这个）
    assert "免邮" in out.bm25_text        # 只扩进 bm25_text
    assert out.expanded  # 命中的同义词要能进日志/报告


async def test_no_synonym_hit_leaves_bm25_text_equal_to_standard():
    """没有命中同义词表时，bm25_text 必须与 standard 完全一致（不要加空格/前缀）。"""
    out = await QueryUnderstander(enabled=False).understand("XH-3000A 价格")
    assert out.bm25_text == out.standard
    assert out.expanded == ()


def test_model_tokens_are_uppercased_and_hyphenated():
    """探针实测：BM25 对写法变体零命中（chinese analyzer 不做大小写归一）。

    语料里的型号规范是「大写 + 连字符」，所以检索侧把用户写法归一到那个形状。
    """
    assert normalise_model_tokens("xh-3000a 续航") == "XH-3000A 续航"
    assert normalise_model_tokens("XH 3000A 续航") == "XH-3000A 续航"
    assert normalise_model_tokens("XH3000A 续航") == "XH-3000A 续航"


def test_model_tokens_adjacent_to_chinese_are_normalised():
    """型号**紧贴中文**是最常见的写法，也是 `\\b` 版本的盲区。

    中文也是 `\\w`，所以 `XH3000A续航` 里 `\\b` 根本不成立 —— 归一静默不发生，
    而「未归一的变体 BM25 全漏」已被 Task 0 探针证实。这三条是那条洞的回归测试。
    """
    assert normalise_model_tokens("XH3000A续航多久") == "XH-3000A续航多久"
    assert normalise_model_tokens("xh3000a的续航") == "XH-3000A的续航"
    assert normalise_model_tokens("请问XH3000A续航多久") == "请问XH-3000A续航多久"


def test_model_normalisation_leaves_plain_words_alone():
    assert normalise_model_tokens("运费怎么算") == "运费怎么算"
    assert normalise_model_tokens("满 99 元包邮") == "满 99 元包邮"


async def test_model_variant_is_normalised_before_bm25():
    out = await QueryUnderstander(enabled=False).understand("xh-3000a 续航多久")
    assert out.standard == "XH-3000A 续航多久"
    assert "XH-3000A" in out.bm25_text


# ---------------------------------------------------------------------------
# 长度
# ---------------------------------------------------------------------------


async def test_overlong_query_is_truncated():
    out = await QueryUnderstander(enabled=False).understand("退" * 2000)
    assert len(out.original) == MAX_QUERY_CHARS


async def test_punctuation_only_query_survives():
    out = await QueryUnderstander(enabled=False).understand("？？？")
    assert out.standard == "？？？"


# ---------------------------------------------------------------------------
# LlmRewriter 本身
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[object] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return type("Msg", (), {"content": self.reply})()


async def test_llm_rewriter_parses_strict_json():
    rewriter = LlmRewriter(_FakeModel('{"rewritten": "退款到账时间"}'), timeout=1.0)
    assert await rewriter.rewrite("钱啥时候回来") == "退款到账时间"


async def test_llm_rewriter_returns_none_on_garbage():
    rewriter = LlmRewriter(_FakeModel("我觉得是退款吧"), timeout=1.0)
    assert await rewriter.rewrite("钱啥时候回来") is None


async def test_llm_rewriter_returns_none_on_empty_answer():
    rewriter = LlmRewriter(_FakeModel('{"rewritten": "  "}'), timeout=1.0)
    assert await rewriter.rewrite("q") is None


async def test_llm_rewriter_returns_none_when_the_model_raises():
    class _Broken:
        async def ainvoke(self, messages):
            raise RuntimeError("上游挂了")

    assert await LlmRewriter(_Broken(), timeout=1.0).rewrite("q") is None


@pytest.mark.parametrize("bad", ['```json\n{"rewritten": "x"}\n```'])
async def test_llm_rewriter_tolerates_code_fences(bad):
    """模型偶尔会把 JSON 包进 markdown 代码块 —— 直接判失败太浪费一次调用。"""
    assert await LlmRewriter(_FakeModel(bad), timeout=1.0).rewrite("q") == "x"
