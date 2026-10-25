"""问题标准化：把口语原话改写成标准 FAQ 式问题，并配一条示例答案。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.parsing import extract_json, message_text

logger = logging.getLogger(__name__)

__all__ = ["NormalizedQuestion", "normalize_question", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = """你是客服知识库的编辑。把用户口语化的提问改写成一条标准 FAQ 式问题，并给出一条示例答案备查。

要求：
1. 问题要完整、书面、可被检索；保留原问题里的关键条件（商品、时间、场景），不要臆造。
2. 示例答案是「如果知识库补上这一条，应该怎么写」，只写通用规则，不要编造具体订单信息。
3. 只输出 JSON 对象，不要 markdown，不要解释：
{"question": "标准问题", "answer": "示例答案"}"""


@dataclass(frozen=True)
class NormalizedQuestion:
    normalized_question: str
    suggested_answer: str
    degraded: bool = False


def _evidence_lines(evidence: Sequence[Any], *, limit: int = 5) -> str:
    lines: list[str] = []
    for item in evidence[:limit]:
        if isinstance(item, dict):
            quote = item.get("quote") or item.get("answer") or ""
            section = item.get("section_path") or ""
        else:
            quote = getattr(item, "quote", "") or getattr(item, "answer", "")
            section = getattr(item, "section_path", "")
        text = str(quote).strip().replace("\n", " ")
        if text:
            lines.append(f"- [{section}] {text[:300]}")
    return "\n".join(lines) or "（这一轮没有召回任何片段）"


async def normalize_question(
    model: Any,
    *,
    raw_question: str,
    evidence: Sequence[Any],
    timeout: float = 20.0,
) -> NormalizedQuestion:
    """模型失败/坏 JSON 时降级为原话，**绝不丢掉这条缺口**。"""
    raw = (raw_question or "").strip()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"用户原话：{raw}\n\n"
                f"当轮召回片段（供你理解上下文，不要把片段里没有的规则写进答案）：\n"
                f"{_evidence_lines(evidence)}"
            )
        ),
    ]
    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
        data = extract_json(message_text(response)) or {}
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        if not question:
            raise ValueError("模型没有返回 question 字段")
        return NormalizedQuestion(question, answer, degraded=False)
    except Exception:  # noqa: BLE001 - 标准化是增强步骤，失败也要把原话送进队列
        logger.warning("问题标准化失败，降级为原话（raw=%r）", raw, exc_info=True)
        return NormalizedQuestion(raw, "", degraded=True)
