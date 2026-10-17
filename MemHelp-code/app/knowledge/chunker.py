"""Markdown → 知识块：结构感知切分。

## 为什么不能按固定字数切

按 500 字硬切会把「### 七天无理由」的标题与它的正文切开，于是检索命中的块
读起来像一句没头没尾的话；表格被拦腰截断后，后半截没有表头，列名全靠猜。
这两类错误都不会报错，只会让模型答得「差不多但不对」。

所以这里按**标题层级**切分（``section_path`` / ``category`` / ``questions`` 都由
标题栈推出），只在**单节超长**时才递归降级到段落、句子、硬切。

## 三个刻意的约束

1. **表格只按行切，且每块复制表头。** 表格不参与「按字符递归切」—— 否则后半截
   必然丢表头。单行超长时宁可让块超过 ``max_chars``，也不把一行掰两半。
2. **重叠回退到句末。** 相邻块的重叠取上一块尾部 ≤ ``overlap`` 字，并从**第一个
   句末标点之后**开始复述：从半句开始的重叠比不重叠更糟。
3. **关键条款只认显式标记**（标题里的「（关键条款）」或正文里的
   ``<!-- key-clause -->``）。用「赔付/期限」这类词猜会污染元数据，而元数据是给人看的。

## 向量化文本

``render_embed_text`` 是**唯一**的拼接实现，建库与检索都调它。两处各写一份，
迟早有一处改了另一处没改 —— 表现为「建库时嵌入的文本」与「检索时假想的文本」
悄悄分叉，召回率无声下滑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "KEY_CLAUSE_MARKER",
    "KEY_CLAUSE_TITLE_SUFFIX",
    "SENTENCE_ENDINGS",
    "Chunk",
    "render_embed_text",
    "split_markdown",
]

#: 正文里的关键条款标记。整行写，切分时会被删掉，不进 ``answer``。
KEY_CLAUSE_MARKER = "<!-- key-clause -->"

#: 标题后缀形式的关键条款标记。
KEY_CLAUSE_TITLE_SUFFIX = "（关键条款）"

#: 句末标点。重叠回退与递归切句都用这一组。
SENTENCE_ENDINGS = "。！？；.!?;"

#: 超过这个数据行数的表格按行切（切出来的每块都复制表头）。
TABLE_MAX_ROWS = 20

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.M)
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_SENTENCE_RE = re.compile(r"(?<=[。！？；.!?;])")


@dataclass(frozen=True)
class Chunk:
    """一个知识块。

    进向量化文本的只有 ``category`` / ``questions`` / ``answer``；
    ``section_path`` / ``content_type`` / ``is_key_clause`` 是元数据（只存不进向量）。
    ``ordinal`` 是文档内序号，仓库层用来回填前后块指针，不落库。
    """

    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool
    ordinal: int


@dataclass
class _Section:
    """解析中间态：一个标题及其正文。"""

    title: str
    path: list[str]
    body: str
    is_key_clause: bool


# ---------------------------------------------------------------------------
# 文本预处理
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """去掉 BOM、统一换行。

    不做的话 ``answer`` 里会混进 ``\\r`` 和 ``\\ufeff`` —— 它们进向量化文本与
    MySQL 都是脏数据，而且在界面上看不出来。
    """
    return (
        text.lstrip("\ufeff")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _clean_title(raw: str) -> tuple[str, bool]:
    """标题 → (干净标题, 是否关键条款)。"""
    title = raw.strip()
    key_clause = KEY_CLAUSE_TITLE_SUFFIX in title
    title = title.replace(KEY_CLAUSE_TITLE_SUFFIX, "").strip()
    if KEY_CLAUSE_MARKER in title:
        key_clause = True
        title = title.replace(KEY_CLAUSE_MARKER, "").strip()
    return title, key_clause


# ---------------------------------------------------------------------------
# 解析成「节」
# ---------------------------------------------------------------------------


def _parse_sections(text: str, source: str) -> list[_Section]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        body = text.strip()
        if not body:
            return []
        # 没有标题的文档：用文件名当节标题，至少让 questions 不为空。
        title = Path(source).stem if source else "正文"
        return [_Section(title=title, path=[title], body=body, is_key_clause=False)]

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []

    for index, match in enumerate(matches):
        level = len(match.group(1))
        title, title_is_key = _clean_title(match.group(2))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]

        # 文档最前面的导语并进第一个节，而不是丢掉。
        if index == 0:
            preamble = text[: match.start()].strip()
            if preamble:
                body = f"{preamble}\n\n{body}"

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        sections.append(
            _Section(
                title=title,
                path=[t for _, t in stack],
                body=body,
                is_key_clause=title_is_key or KEY_CLAUSE_MARKER in body,
            )
        )
    return sections


# ---------------------------------------------------------------------------
# 正文 → 原子单元
# ---------------------------------------------------------------------------


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """超长文本三级降级：整段 → 按句 → 硬切。"""
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        # 单句仍超长（例如整段没有标点）：只能硬切。硬切的块可能从半句开始，
        # 但这是唯一选择 —— 无限大的块进不了嵌入模型的上下文窗口。
        units.extend(
            sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)
        )
    return units


def _table_units(lines: list[str], max_chars: int, title: str) -> list[tuple[str, bool]]:
    """表格 → 若干**原子**单元（不再被后续按字符切分）。

    ``| a | b |`` 第一行是表头、第二行是分隔行（如果存在），其余是数据行。
    数据行超过 :data:`TABLE_MAX_ROWS` 或整表超长时按行切，**每块都复制表头**。
    """
    header = lines[0].rstrip()
    has_separator = len(lines) > 1 and bool(_SEPARATOR_RE.match(lines[1]))
    separator = lines[1].rstrip() if has_separator else ""
    data = lines[2:] if has_separator else lines[1:]
    prefix_lines = [header] + ([separator] if separator else [])

    whole = "\n".join(lines)
    if len(data) <= TABLE_MAX_ROWS and len(whole) <= max_chars:
        return [(whole, True)]

    units: list[tuple[str, bool]] = []
    buffer: list[str] = []
    first = True

    def flush() -> None:
        nonlocal buffer, first
        if not buffer:
            return
        lead = "" if first else f"（接上表：{title}）\n"
        units.append((lead + "\n".join(prefix_lines + buffer), True))
        buffer = []
        first = False

    for row in data:
        buffer.append(row.rstrip())
        if len("\n".join(prefix_lines + buffer)) >= max_chars:
            flush()
    flush()
    return units


def _body_units(body: str, max_chars: int, title: str) -> list[tuple[str, bool]]:
    """正文 → 单元序列；表格是原子单元，其余可以继续被切开。"""
    units: list[tuple[str, bool]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                units.extend((piece, False) for piece in _split_long_text(text, max_chars))
            paragraph.clear()

    lines = body.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append(lines[index])
                index += 1
            units.extend(_table_units(table, max_chars, title))
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        index += 1
    flush_paragraph()
    return units


def _sentence_overlap(previous: str, overlap: int) -> str:
    """取上一块尾部 ≤ ``overlap`` 字，并从**第一个句末标点之后**开始。

    找不到句末标点就返回空串 —— 不复述半句。跨节不调用本函数。
    """
    if overlap <= 0 or not previous:
        return ""
    tail = previous[-overlap:]
    for position, char in enumerate(tail):
        if char in SENTENCE_ENDINGS:
            candidate = tail[position + 1 :].strip()
            if candidate:
                return candidate
            return ""
    return ""


def _pack(
    units: list[tuple[str, bool]], max_chars: int, overlap: int
) -> list[str]:
    """把单元贪心合并成块，并在同节内的相邻块之间加句末对齐的重叠。"""
    pieces: list[str] = []
    current = ""

    for text, atomic in units:
        if atomic:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(text)
            continue

        if not current:
            current = text
        elif len(current) + 2 + len(text) <= max_chars:
            current = f"{current}\n\n{text}"
        else:
            pieces.append(current)
            prefix = _sentence_overlap(current, overlap)
            candidate = prefix + text if prefix else text
            current = candidate if len(candidate) <= max_chars else text

    if current:
        pieces.append(current)
    return pieces


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def split_markdown(
    text: str,
    *,
    content_type: str,
    source: str = "",
    max_chars: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """把一份 Markdown 切成知识块。

    :param content_type: ``faq`` / ``policy`` / ``manual``（落库的元数据）。
    :param source: 来源标识（文件名）。仅在没有标题的文档里用来兜底当节标题。
    :param max_chars: 单块最大字符数。表格按行切时不受此限。
    :param overlap: 同节内相邻块的重叠上限；按句回退。
    :return: 按文档顺序排列的块；只有标题没有正文的节被跳过。
    """
    normalised = _normalise(text)
    chunks: list[Chunk] = []
    ordinal = 0

    for section in _parse_sections(normalised, source):
        body = section.body.replace(KEY_CLAUSE_MARKER, "").strip()
        if not body:
            continue

        units = _body_units(body, max_chars, section.title)
        base_path = " / ".join(section.path)
        emitted = 0
        for piece in _pack(units, max_chars, overlap):
            answer = piece.strip()
            if not answer:
                continue
            emitted += 1
            # 同一节切出的第 2、3…块，在 section_path 后加序号。
            # **这不只是给人看的**：section_path 是仓库层自然键的一部分
            # （content_type + section_path + questions）。同节多块共用同一个路径时，
            # 写入 MySQL 会被判成同一行 —— 后一块覆盖前一块，大表格后半截静默消失，
            # 建库报告只会显示 created 比 total 少，不报错。
            chunks.append(
                Chunk(
                    category=" / ".join(section.path[:-1]) or section.path[-1],
                    questions=section.title,
                    answer=answer,
                    section_path=base_path if emitted == 1 else f"{base_path} #{emitted}",
                    content_type=content_type,
                    is_key_clause=section.is_key_clause,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    return chunks


def render_embed_text(chunk: object) -> str:
    """把一块拼成**向量化文本**：分类 + 问法 + 答案三格。

    接受 duck-typed 行（``Chunk`` 或仓库层的 ``ChunkRow``）—— 建库与检索共用
    同一个拼接，避免两处各写一份后悄悄分叉。

    BGE-M3 不需要 instruction 前缀，但给三格加中文标签能让「分类」「问法」「答案」
    在语义上可区分：否则一段纯正文和它的标题会挤在同一个向量空间角落里。
    """
    category = getattr(chunk, "category", "")
    questions = getattr(chunk, "questions", "")
    answer = getattr(chunk, "answer", "")
    return f"分类：{category}\n问法：{questions}\n答案：{answer}"
