"""证据编号、首尾编排、引用解析。

## 编号按相关度固定，不随呈现顺序变

用户在页面上点开 `[3]`，看到的那条必须和正文里 `[3]` 指的同一条。
如果编号跟着「塞进 prompt 的顺序」变，同一段回答在两次渲染里 `[3]` 指的不是同一块 ——
点开就是错的证据。

## 首尾编排

模型对上下文**中段**的注意力最弱。所以把最相关的两条放在首尾，
最不相关的压中间：``[1,3,5,7,9,10,8,6,4,2]``。编号本身不变。
"""

from __future__ import annotations

from app.knowledge.citations import (
    Evidence,
    build_evidence,
    evidence_from_payload,
    evidence_payload,
    parse_citations,
    prompt_order,
)
from app.knowledge.hybrid import RetrievedChunk


def chunk(chunk_id: int, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        question=f"问法{chunk_id}",
        answer=f"答案{chunk_id}",
        category="售后政策",
        section_path=f"售后手册 / 第{chunk_id}节",
        content_type="manual",
        score=score,
        source="rerank",
    )


def evidence_of(*numbers: int) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(n=number, chunk_id=number, section_path=f"路径{number}",
                 category="售后政策", quote=f"原文{number}")
        for number in numbers
    )


# ---------------------------------------------------------------------------
# 编号
# ---------------------------------------------------------------------------


def test_numbering_follows_relevance_order():
    built = build_evidence([chunk(3), chunk(1)])
    assert [(item.n, item.chunk_id) for item in built] == [(1, 3), (2, 1)]


def test_evidence_keeps_quote_and_section_path():
    built = build_evidence([chunk(7, score=0.42)])
    assert built[0].quote == "答案7"
    assert built[0].section_path == "售后手册 / 第7节"
    assert built[0].score == 0.42
    assert built[0].source == "rerank"


def test_empty_chunks_produce_no_evidence():
    assert build_evidence([]) == ()


# ---------------------------------------------------------------------------
# 首尾编排
# ---------------------------------------------------------------------------


def test_head_tail_order_for_ten():
    assert [item.n for item in prompt_order(evidence_of(*range(1, 11)))] == [
        1, 3, 5, 7, 9, 10, 8, 6, 4, 2,
    ]


def test_head_tail_order_for_odd_count():
    assert [item.n for item in prompt_order(evidence_of(*range(1, 6)))] == [1, 3, 5, 4, 2]


def test_head_tail_order_for_single_item():
    assert [item.n for item in prompt_order(evidence_of(1))] == [1]


def test_head_tail_order_keeps_every_item_exactly_once():
    order = [item.n for item in prompt_order(evidence_of(*range(1, 8)))]
    assert sorted(order) == list(range(1, 8))


# ---------------------------------------------------------------------------
# 引用解析
# ---------------------------------------------------------------------------


def test_parse_keeps_valid_and_drops_out_of_range():
    result = parse_citations("满 99 包邮[1]，破损赔 500[3]。", evidence_of(1, 2))
    assert result.valid == (1,)
    assert result.dropped == (3,)
    assert result.coverage == 0.5


def test_parse_ignores_full_width_brackets():
    result = parse_citations("参考【2】和[1]", evidence_of(1, 2))
    assert result.valid == (1,)
    assert result.dropped == ()


def test_parse_drops_index_zero():
    result = parse_citations("参考[0]", evidence_of(1))
    assert result.valid == ()
    assert result.dropped == (0,)


def test_parse_deduplicates_repeated_citations():
    result = parse_citations("包邮[1]，真的包邮[1]", evidence_of(1, 2))
    assert result.valid == (1,)
    assert result.coverage == 0.5


def test_parse_without_citations():
    result = parse_citations("没有任何角标的回答", evidence_of(1))
    assert result.valid == ()
    assert result.dropped == ()
    assert result.coverage == 0.0


def test_parse_with_empty_evidence_does_not_divide_by_zero():
    result = parse_citations("满 99 包邮[1]", ())
    assert result.valid == ()
    assert result.dropped == (1,)
    assert result.coverage == 0.0


def test_parse_does_not_modify_the_answer():
    """丢弃只发生在映射层；正文一个字都不改（模型多写一个角标不该让回答消失）。"""
    answer = "满 99 包邮[1]，破损赔 500[9]。"
    parse_citations(answer, evidence_of(1))
    assert answer == "满 99 包邮[1]，破损赔 500[9]。"


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------


def test_payload_has_exactly_the_snapshot_fields():
    payload = evidence_payload(evidence_of(1))
    assert set(payload[0]) == {"n", "chunk_id", "section_path", "category", "quote"}


def test_payload_roundtrips_to_evidence():
    original = build_evidence([chunk(1, score=0.8)])
    restored = evidence_from_payload(evidence_payload(original))
    assert [(item.n, item.chunk_id, item.quote) for item in restored] == [
        (item.n, item.chunk_id, item.quote) for item in original
    ]
    # 快照里没有的字段用默认值补齐，不抛
    assert restored[0].score == 0.0
    assert restored[0].source == ""
