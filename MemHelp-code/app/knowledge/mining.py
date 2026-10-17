"""从历史客服对话里挖知识：分批抽取 → 暂存 → 整体去重 → 入库。

## 两段式的原因

```
分批喂 LLM（防串味）→ qa_extraction_staging
全部批次跑完 → 全量去重（跨批同义才算得出来）→ knowledge_chunks
```

一次喂太多会话，模型会把 A 会话的答案接到 B 会话的问题上；而逐批去重会让
「第 2 批里与第 1 批同义的问法」各存一份。所以抽取按批写、去重对全量做。

## 幂等

``qa_extraction_staging.source_ref`` 记会话 id；已经挖过的会话默认跳过，
``force=True`` 才重挖。批次号 ``batch_no`` 让「哪一批失败了」可追溯。

## 失败语义

**每批一个 try/except**：失败的那批一行都不写（不做半批），记下批号继续下一批。
整条任务不会因为一次上游 500 丢掉前面已经抽好的东西。

## 去重口径（用户裁定）

先做字面归一化（NFKC + 去空白标点 + casefold）——完全相同的直接丢；
剩下的与**已保留**的问题算余弦，≥ 阈值判同义，保留先入的、把问法并进它的
``questions``（最多 5 条，防止无限膨胀）。
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from app.db.repositories import Repositories, StagingRow
from app.knowledge.embedder import Embedder

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationDigest",
    "DedupeReport",
    "ExtractedQa",
    "ExtractionBatch",
    "MiningReport",
    "build_extraction_prompt",
    "dedupe_and_ingest",
    "extract_batch",
    "mine_conversations",
    "normalise_question",
]

#: 挖矿产物的 ``content_type``。与文档类 FAQ 区分开，排查召回时能看出「这条是挖的」。
MINED_CONTENT_TYPE = "mined"

#: 挖矿产物的 ``category``。
MINED_CATEGORY = "对话挖掘"

#: 单条消息进 prompt 前的截断长度。
MAX_TURN_CHARS = 500

#: 每个会话最多取几轮（user+assistant 为一轮）。
MAX_TURNS = 8

#: 合并同义问法时的上限。
MAX_QUESTIONS_PER_ROW = 5

_WHITESPACE_OR_PUNCT = re.compile(
    r"[\s\u3000!-/:-@\[-`{-~，。！？；：、“”‘’（）【】《》…—·]+"
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class ExtractedQa(BaseModel):
    """LLM 抽出的一条问答对。

    ``session_id`` **必须**带：暂存表的 ``source_ref`` 要记来源，
    否则「这条知识是从哪个对话挖出来的」永远查不回来。
    """

    session_id: str = Field(description="这条问答来自哪个会话，必须从给定列表里原样选")
    question: str = Field(description="用户真实问过的问题，改写为通用表述")
    answer: str = Field(description="客服给出的可复用答案，去掉订单号等一次性信息")


class ExtractionBatch(BaseModel):
    """一次抽取调用的结构化输出。"""

    pairs: list[ExtractedQa] = Field(default_factory=list)


@dataclass(frozen=True)
class ConversationDigest:
    """喂给 LLM 的一段会话摘要。"""

    session_id: str
    turns: list[tuple[str, str]]


@dataclass
class MiningReport:
    batches: int = 0
    pairs_written: int = 0
    skipped_sessions: int = 0
    failed_batches: list[str] = field(default_factory=list)


@dataclass
class DedupeReport:
    kept: int = 0
    discarded: int = 0
    ingested: int = 0


# ---------------------------------------------------------------------------
# 归一化与相似度
# ---------------------------------------------------------------------------


def normalise_question(text: str) -> str:
    """字面归一化：NFKC（全角→半角）→ 去空白与标点 → casefold。

    只用来判断「字面是否完全相同」，不做语义判断。
    """
    normalised = unicodedata.normalize("NFKC", text or "")
    return _WHITESPACE_OR_PUNCT.sub("", normalised).casefold()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------


_PROMPT_HEADER = """你是电商客服的知识库编辑。下面是若干段历史客服对话。

请从中抽取可以沉淀进知识库的问答对，规则：
1. 只抽**用户真实问过的问题** + **客服给出的可复用答案**；
2. 寒暄、道谢、订单号/运单号这类一次性信息丢弃；
3. 答案改写成与具体用户无关的通用表述（去掉「您的订单」「刚才」这类指代）；
4. 不要编造对话里没有的规则；
5. 每条问答必须标注它来自哪个会话，``session_id`` 从下面出现过的会话 id 里原样选。
"""


def build_extraction_prompt(digests: Sequence[ConversationDigest]) -> str:
    """把若干会话渲染成一段 prompt。

    每个会话以 ``### 会话 <session_id>`` 开头 —— 这个标记是模型回填
    ``session_id`` 的锚点，也是测试断言 prompt 格式的地方。
    """
    blocks = [_PROMPT_HEADER]
    for digest in digests:
        lines = [f"### 会话 {digest.session_id}"]
        for role, content in digest.turns:
            speaker = "用户" if role == "user" else "客服"
            lines.append(f"{speaker}：{content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _digest_of(session_id: str, rows: Iterable[Any]) -> ConversationDigest:
    turns: list[tuple[str, str]] = []
    for row in rows:
        content = (row.content or "").strip()
        if row.role not in ("user", "assistant") or not content:
            continue
        turns.append((row.role, content[:MAX_TURN_CHARS]))
        if len(turns) >= MAX_TURNS * 2:
            break
    return ConversationDigest(session_id=session_id, turns=turns)


async def _build_digests(
    repositories: Repositories, session_ids: Sequence[str]
) -> list[ConversationDigest]:
    digests: list[ConversationDigest] = []
    for session_id in session_ids:
        conversation_id = await repositories.conversations.find_id(session_id)
        rows = (
            await repositories.messages.list_for_conversation(conversation_id)
            if conversation_id is not None
            else []
        )
        digest = _digest_of(session_id, rows)
        if digest.turns:
            digests.append(digest)
    return digests


def _filter_pairs(
    pairs: Sequence[ExtractedQa], allowed_session_ids: set[str]
) -> list[ExtractedQa]:
    """丢掉：编造来源、空字段、同批重复。"""
    kept: list[ExtractedQa] = []
    seen: set[str] = set()
    for pair in pairs:
        question = pair.question.strip()
        answer = pair.answer.strip()
        if not question or not answer:
            continue
        if pair.session_id not in allowed_session_ids:
            logger.warning("丢弃来源不明的问答对（session_id=%r）", pair.session_id)
            continue
        key = normalise_question(question)
        if key in seen:
            continue
        seen.add(key)
        kept.append(
            ExtractedQa(session_id=pair.session_id, question=question, answer=answer)
        )
    return kept


async def extract_batch(llm: Any, digests: Sequence[ConversationDigest]) -> list[ExtractedQa]:
    """对一批会话做一次结构化抽取。

    :raises Exception: 上游故障原样抛出，由 ``mine_conversations`` 按批处理。
    """
    prompt = build_extraction_prompt(digests)
    structured = llm.with_structured_output(ExtractionBatch)
    result = await structured.ainvoke([SystemMessage(content=prompt)])
    pairs = getattr(result, "pairs", None) or []
    return _filter_pairs(pairs, {digest.session_id for digest in digests})


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def mine_conversations(
    *,
    repositories: Repositories,
    llm: Any,
    embedder: Embedder,
    batch_size: int = 8,
    limit: int | None = None,
    force: bool = False,
    batch_prefix: str | None = None,
) -> MiningReport:
    """把历史会话分批喂给 LLM，抽出的问答对写进暂存表。

    :param force: 已挖过的会话是否重挖（默认跳过，靠 ``source_ref`` 判重）。
    :param batch_prefix: 批次号前缀；默认按当前时间生成。
    """
    report = MiningReport()
    sessions = await repositories.conversations.list_sessions(limit=limit)
    known = set() if force else await repositories.qa_staging.known_sources()

    pending_sessions = [s.session_id for s in sessions if s.session_id not in known]
    report.skipped_sessions = len(sessions) - len(pending_sessions)

    digests = await _build_digests(repositories, pending_sessions)
    if not digests:
        return report

    prefix = batch_prefix or f"ch03-{datetime.now():%Y%m%d%H%M%S}"
    batches = [
        digests[start : start + batch_size] for start in range(0, len(digests), batch_size)
    ]
    report.batches = len(batches)

    for index, batch in enumerate(batches, start=1):
        batch_no = f"{prefix}-{index}"
        try:
            pairs = await extract_batch(llm, batch)
        except Exception as exc:  # noqa: BLE001 - 单批失败不该拖垮整条任务
            logger.warning("第 %d 批抽取失败（%s）：%s", index, batch_no, exc)
            report.failed_batches.append(batch_no)
            continue

        if not pairs:
            logger.info("第 %d 批没有抽到可入库的问答对（%s）", index, batch_no)
            continue

        await repositories.qa_staging.add_rows(
            [
                {
                    "batch_no": batch_no,
                    "source_ref": pair.session_id,
                    "question": pair.question,
                    "answer": pair.answer,
                }
                for pair in pairs
            ]
        )
        report.pairs_written += len(pairs)

    return report


# ---------------------------------------------------------------------------
# 去重入库
# ---------------------------------------------------------------------------


class _MinedChunk:
    """给 ``upsert_chunk`` 用的最小行形状。"""

    __slots__ = (
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "ordinal",
    )

    def __init__(
        self, *, questions: str, answer: str, section_path: str, ordinal: int
    ) -> None:
        self.category = MINED_CATEGORY
        self.questions = questions
        self.answer = answer
        self.section_path = section_path
        self.content_type = MINED_CONTENT_TYPE
        self.is_key_clause = False
        self.ordinal = ordinal


async def dedupe_and_ingest(
    *,
    repositories: Repositories,
    embedder: Embedder,
    threshold: float = 0.92,
    max_questions: int = MAX_QUESTIONS_PER_ROW,
) -> DedupeReport:
    """对暂存表里全部 ``extracted`` 行做整体去重，保留项入 ``knowledge_chunks``。

    **不向量化** —— 向量由 CLI 在它之后调 ``vectorize_pending`` 补齐：
    两件事分开，各自可以单独重跑。
    """
    rows = await repositories.qa_staging.list_by_status("extracted")
    if not rows:
        return DedupeReport()

    vectors = await embedder.embed([row.question for row in rows])

    kept_rows: list[StagingRow] = []
    kept_vectors: list[Sequence[float]] = []
    kept_questions: list[list[str]] = []
    literal_keys: set[str] = set()
    discarded_ids: list[int] = []

    for row, vector in zip(rows, vectors):
        key = normalise_question(row.question)
        if not key:
            discarded_ids.append(row.id)
            continue

        if key in literal_keys:
            discarded_ids.append(row.id)
            continue

        duplicate_index = next(
            (
                index
                for index, other in enumerate(kept_vectors)
                if _cosine(vector, other) >= threshold
            ),
            None,
        )
        if duplicate_index is not None:
            # 同义问法：并入保留行的 questions（有上限，防止无限膨胀）。
            if len(kept_questions[duplicate_index]) < max_questions:
                kept_questions[duplicate_index].append(row.question.strip())
            discarded_ids.append(row.id)
            continue

        literal_keys.add(key)
        kept_rows.append(row)
        kept_vectors.append(vector)
        kept_questions.append([row.question.strip()])

    if discarded_ids:
        await repositories.qa_staging.mark(discarded_ids, "discarded")
    if kept_rows:
        await repositories.qa_staging.mark([row.id for row in kept_rows], "kept")

    ingested = 0
    for row, questions in zip(kept_rows, kept_questions):
        merged = "\n".join(questions)
        await repositories.qa_staging.update_questions(row.id, merged)
        await repositories.knowledge.upsert_chunk(
            _MinedChunk(
                questions=merged,
                answer=row.answer,
                section_path=f"{MINED_CATEGORY} / {row.source_ref or 'unknown'}",
                ordinal=ingested,
            )
        )
        ingested += 1

    return DedupeReport(
        kept=len(kept_rows), discarded=len(discarded_ids), ingested=ingested
    )
