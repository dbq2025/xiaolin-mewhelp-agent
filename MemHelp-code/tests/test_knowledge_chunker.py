"""Markdown 结构感知切分器。

这一层是纯函数（无 IO、无数据库），所以整章最该被 TDD 钉死的就是它：
切错了不会报错，只会让**检索命中一段缺头少尾的话**，而那看起来像「模型答得不好」。

三个容易翻车的点，用例逐一钉住：
1. 大表格按行切时**每块都要复制表头** —— 丢了表头的后半截表，语义直接不可用；
2. 重叠必须**回退到句末** —— 从半句开始的重叠比不重叠更糟（模型会顺着半句编）；
3. 只有标题没有正文的节**不产出块** —— 空 answer 的块会污染检索结果。
"""

from __future__ import annotations

from app.knowledge.chunker import Chunk, render_embed_text, split_markdown

DOC = """# 退货政策

## 退货条件

### 七天无理由

签收后 7 天内可申请无理由退货。商品需保持完好，不影响二次销售。

### 运费承担

退回运费由买家承担，质量问题除外。
"""


# ---------------------------------------------------------------------------
# 结构映射
# ---------------------------------------------------------------------------


def test_section_path_and_category_build_from_heading_stack():
    chunks = split_markdown(DOC, content_type="policy")
    by_q = {c.questions: c for c in chunks}

    assert by_q["七天无理由"].section_path == "退货政策 / 退货条件 / 七天无理由"
    assert by_q["七天无理由"].category == "退货政策 / 退货条件"
    assert by_q["七天无理由"].answer.startswith("签收后 7 天内")


def test_faq_heading_becomes_question_and_ordinals_are_sequential():
    """FAQ 文档把标题写成真实问法 —— 于是它与政策手册共用同一条切分规则。"""
    doc = "# 商品 FAQ\n\n## 能退货吗？\n\n可以，7 天无理由。\n\n## 几天发货？\n\n48 小时内。\n"
    chunks = split_markdown(doc, content_type="faq")

    assert [c.questions for c in chunks] == ["能退货吗？", "几天发货？"]
    assert [c.ordinal for c in chunks] == [0, 1]
    assert all(c.content_type == "faq" for c in chunks)


def test_top_level_section_uses_its_own_title_as_category():
    doc = "# 售后手册\n\n退换货总则说明。\n"
    chunks = split_markdown(doc, content_type="manual")
    assert chunks[0].section_path == "售后手册"
    assert chunks[0].category == "售后手册"


def test_section_without_body_is_skipped():
    """只有标题没有正文的节（常见于目录式文档）不产出空块。"""
    doc = "# 政策\n\n## 空节\n\n## 有正文\n\n内容。\n"
    chunks = split_markdown(doc, content_type="policy")
    assert [c.questions for c in chunks] == ["有正文"]


# ---------------------------------------------------------------------------
# 超长递归与重叠
# ---------------------------------------------------------------------------


def test_long_section_splits_recursively_under_max_chars():
    body = "。".join(["这是一句足够长的售后说明文字"] * 60) + "。"
    doc = f"# 手册\n\n## 长篇\n\n{body}"
    chunks = split_markdown(doc, content_type="manual", max_chars=200, overlap=40)

    assert len(chunks) > 3
    assert all(len(c.answer) <= 200 for c in chunks)


def test_overlap_repeats_whole_sentences_from_previous_chunk():
    body = "。".join(["第%d句话说明售后规则" % i for i in range(1, 80)]) + "。"
    doc = f"# 手册\n\n## 长篇\n\n{body}"
    chunks = split_markdown(doc, content_type="manual", max_chars=120, overlap=30)

    prev, cur = chunks[0].answer, chunks[1].answer
    lead = cur.split("。")[0]  # 第二块的第一句
    assert lead + "。" in prev  # 确实是上一块的尾部内容（不是新内容）
    assert len(lead) <= 30  # 复述长度不超过 overlap
    assert cur == cur.lstrip()


def test_overlap_is_not_applied_across_sections():
    """章节之间不加重叠：跨节复述会让「退货」块里出现「发票」的内容。"""
    doc = "# 手册\n\n## 甲\n\n甲的内容。\n\n## 乙\n\n乙的内容。\n"
    chunks = split_markdown(doc, content_type="manual", max_chars=600, overlap=80)
    assert chunks[1].answer == "乙的内容。"


def test_no_punctuation_long_text_terminates_without_empty_chunks():
    doc = "# 手册\n\n## 无标点\n\n" + "无标点很长的正文" * 100
    chunks = split_markdown(doc, content_type="manual", max_chars=100, overlap=20)

    assert chunks
    assert all(c.answer.strip() for c in chunks)
    assert all(len(c.answer) <= 100 for c in chunks)


# ---------------------------------------------------------------------------
# 表格
# ---------------------------------------------------------------------------


def _big_table_doc(rows: int = 25) -> str:
    body = "\n".join(f"| P{i} | {i} 元 |" for i in range(1, rows + 1))
    return f"# 价格表\n\n## 运费\n\n| 商品 | 价格 |\n| --- | --- |\n{body}\n"


def test_large_table_repeats_header_in_every_chunk():
    chunks = split_markdown(_big_table_doc(), content_type="manual", max_chars=300, overlap=0)

    table_chunks = [c for c in chunks if "| 商品 | 价格 |" in c.answer]
    assert len(table_chunks) >= 2
    assert all("| --- | --- |" in c.answer for c in table_chunks)


def test_table_rows_never_split_across_chunks():
    chunks = split_markdown(_big_table_doc(), content_type="manual", max_chars=300, overlap=0)
    joined = "\n".join(c.answer for c in chunks)

    for i in range(1, 26):
        assert joined.count(f"| P{i} | {i} 元 |") == 1


def test_single_overlong_table_row_stays_intact_even_if_chunk_exceeds_max_chars():
    """「一行不跨块」优先于「不超过 max_chars」：宁可块超限，也不把一行掰两半。"""
    long_row = "| 超长说明 | " + "这一行特别长且没有标点" * 20 + " |"
    doc = f"# 价格表\n\n## 运费\n\n| 商品 | 说明 |\n| --- | --- |\n{long_row}\n"
    chunks = split_markdown(doc, content_type="manual", max_chars=50, overlap=0)

    assert any(long_row.strip() in c.answer for c in chunks)
    assert all(c.answer.strip() for c in chunks)


def test_small_table_stays_in_one_block():
    doc = "# 价格表\n\n## 运费\n\n| 商品 | 价格 |\n| --- | --- |\n| P1 | 9 元 |\n| P2 | 19 元 |\n"
    chunks = split_markdown(doc, content_type="manual")
    assert len(chunks) == 1
    assert "| P1 | 9 元 |" in chunks[0].answer and "| P2 | 19 元 |" in chunks[0].answer


# ---------------------------------------------------------------------------
# 关键条款与编码
# ---------------------------------------------------------------------------


def test_key_clause_marked_by_comment_and_by_heading():
    doc = "# 政策\n\n## 赔付 <!-- key-clause -->\n\n最高赔 500 元。\n\n## 时效（关键条款）\n\n48 小时内处理。\n"
    chunks = split_markdown(doc, content_type="policy")

    assert all(c.is_key_clause for c in chunks)
    # 标记本身不能留在标题里，否则它会进向量化文本
    assert [c.questions for c in chunks] == ["赔付", "时效"]


def test_key_clause_marker_in_body_is_stripped_from_answer():
    doc = "# 政策\n\n## 赔付\n\n<!-- key-clause -->\n\n最高赔 500 元。\n"
    chunks = split_markdown(doc, content_type="policy")
    assert chunks[0].is_key_clause is True
    assert "key-clause" not in chunks[0].answer


def test_bom_and_crlf_are_stripped():
    doc = "\ufeff# 政策\r\n\r\n## 退货\r\n\r\n七天内可退。\r\n"
    chunks = split_markdown(doc, content_type="policy")

    assert chunks[0].answer == "七天内可退。"
    assert "\r" not in chunks[0].answer
    assert "\ufeff" not in chunks[0].questions


# ---------------------------------------------------------------------------
# 向量化文本
# ---------------------------------------------------------------------------


def test_render_embed_text_has_three_labelled_slots():
    chunk = Chunk(
        category="售后 / 退货",
        questions="能退吗",
        answer="7 天无理由",
        section_path="售后 / 退货 / 能退吗",
        content_type="faq",
        is_key_clause=False,
        ordinal=0,
    )
    assert render_embed_text(chunk) == "分类：售后 / 退货\n问法：能退吗\n答案：7 天无理由"


def test_render_embed_text_works_for_repository_rows():
    """建库时拿到的可能是仓库层的 ChunkRow（duck typing），不是 Chunk。"""
    from types import SimpleNamespace

    row = SimpleNamespace(category="物流", questions="运费怎么算", answer="满 99 包邮")
    assert render_embed_text(row) == "分类：物流\n问法：运费怎么算\n答案：满 99 包邮"
