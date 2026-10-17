"""演示语料的结构验证。

语料是**数据**，不走 TDD；这里用「结构不变量」代替单测，钉住几件切分器与演示
都依赖的事实：

1. 三份文档存在且是 UTF-8（Windows 上很容易被存成 GBK）；
2. 每个切出来的块都非空且有章节路径（空块会污染检索）；
3. 商品 FAQ 的标题是**真实问法**（这是「questions 填真实问法」的落地方式）；
4. 退货政策里至少有一条**关键条款**（否则那条元数据永远测不到）；
5. 售后手册的大表格会被按行切开，且每块都复制表头；
6. **所有语料都不含「邮费」二字** —— 这是评估集里「换说法」用例能公平对照
   关键词基线的前提：关键词基线必须真的查不到，向量检索的命中才有意义。
"""

from __future__ import annotations

from pathlib import Path

from app.knowledge.chunker import split_markdown

DOCS = Path(__file__).resolve().parents[1] / "knowledge"


def _read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_three_docs_exist_and_are_utf8():
    names = {"退货政策.md", "商品FAQ.md", "售后手册.md"}
    assert {p.name for p in DOCS.glob("*.md")} >= names
    for path in DOCS.glob("*.md"):
        _read(path.name)  # 解码失败会抛，等价于「必须是 UTF-8」


def test_every_chunk_is_non_empty_and_has_a_section_path():
    for path in DOCS.glob("*.md"):
        chunks = split_markdown(_read(path.name), content_type="policy", source=path.name)
        assert chunks, path.name
        for chunk in chunks:
            assert chunk.answer.strip(), path.name
            assert chunk.section_path.strip(), path.name
            assert chunk.questions.strip(), path.name


def test_faq_sections_are_real_questions():
    """商品 FAQ 的标题写成问句 —— 于是它与政策手册共用同一条切分规则。"""
    chunks = split_markdown(_read("商品FAQ.md"), content_type="faq", source="商品FAQ.md")
    questions = [c.questions for c in chunks if c.content_type == "faq"]
    assert sum(q.endswith(("？", "?")) for q in questions) >= 5


def test_policy_doc_has_at_least_one_key_clause():
    chunks = split_markdown(_read("退货政策.md"), content_type="policy", source="退货政策.md")
    assert any(c.is_key_clause for c in chunks)


def test_manual_large_table_is_split_with_header_repeated():
    chunks = split_markdown(_read("售后手册.md"), content_type="manual", source="售后手册.md")
    table_chunks = [c for c in chunks if "|" in c.answer]
    assert len(table_chunks) >= 2, "大表格应按行切开"
    assert all("| --- |" in c.answer for c in table_chunks), "每块都要复制表头"


def test_corpus_never_contains_the_paraphrase_word():
    """「邮费」是评估集里用来测「换说法」的词。

    语料里一旦出现这两个字，关键词基线就能命中，「向量检索解决了漏召回」
    这条结论立刻失去说服力。所以把它钉成结构约束。
    """
    for path in DOCS.glob("*.md"):
        assert "邮费" not in _read(path.name), f"{path.name} 出现了「邮费」，会污染关键词基线"
