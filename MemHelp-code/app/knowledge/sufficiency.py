"""生成前自评 —— 让模型判断「召回的证据够不够回答这个问题」。

## 为什么 fail-open

JSON 解析失败 / 超时是**我方 prompt 或模型格式**的问题，不是证据不足。
fail-closed 会把每一次格式抖动变成「客服什么都不知道」，还会污染低置信度池；
而证据已经过了检索层的分数闸（确定性检查）。所以失败时按「证据可用」继续，
并记 ``degraded=True`` 让日志与指标能看见。

## 为什么空证据不调模型

没有证据时答案必然是「答不了」，再花一次调用问模型只是浪费；直接判不足。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.knowledge.citations import Evidence

logger = logging.getLogger(__name__)

__all__ = ["SufficiencyJudge", "SufficiencyVerdict"]

_SYSTEM_PROMPT = """你是客服知识库的质检员。判断「给出的证据是否足以回答用户的问题」。

规则：
- 只看看证据。**不要**用你自己的常识补全，也不要在意证据之外的领域知识。
- 证据能直接支撑答案（数字、条件、流程对得上）→ sufficient = true。
- 证据缺失关键信息（比如问赔付上限，证据只讲流程）→ sufficient = false，
  并在 missing 里写清缺什么。
- 只输出 JSON，不要解释、不要 markdown：
  {"sufficient": true 或 false, "missing": "缺什么", "reason": "一句话理由"}"""


@dataclass(frozen=True)
class SufficiencyVerdict:
    """自评结果。

    :param missing: 缺什么（足以让 ``sufficient=false`` 的那部分信息）。
    :param reason: 一句话理由，进低置信度池的 ``reason`` 字段。
    :param degraded: 自评本身降级了（解析失败/超时），结论按「够」处理。
    """

    sufficient: bool
    missing: str = ""
    reason: str = ""
    degraded: bool = False


class SufficiencyJudge:
    """调一次模型做自评，严格 JSON 出参。

    :param timeout: 单次自评超时（秒）。超时按 fail-open 处理。
    """

    def __init__(self, model: object, *, timeout: float = 5.0) -> None:
        self._model = model
        self._timeout = timeout

    async def judge(
        self, question: str, evidence: Sequence[Evidence]
    ) -> SufficiencyVerdict:
        if not evidence:
            return SufficiencyVerdict(
                sufficient=False, missing="没有检索到任何证据", reason="证据为空"
            )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{question}\n\n证据：\n{_render(evidence)}"),
        ]
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke(messages), timeout=self._timeout
            )
        except Exception:  # noqa: BLE001 - 超时/上游异常都 fail-open
            logger.warning(
                "自评失败（超时或上游异常），按证据可用继续（self_check_degraded）",
                exc_info=True,
            )
            return SufficiencyVerdict(
                sufficient=True, reason="自评不可用，按证据可用继续", degraded=True
            )

        reply = getattr(response, "content", response)
        if not isinstance(reply, str):
            return _degraded("自评返回了非文本内容")
        return _parse(reply)


def _render(evidence: Sequence[Evidence]) -> str:
    """证据按相关度序渲染；``[n]`` 与正文角标同名同号。"""
    return "\n\n".join(
        f"[{item.n}] {item.section_path}\n{item.quote}" for item in evidence
    )


def _degraded(reason: str) -> SufficiencyVerdict:
    logger.warning("自评结果无法解析，按证据可用继续（self_check_degraded）：%s", reason)
    return SufficiencyVerdict(sufficient=True, reason=reason, degraded=True)


def _parse(reply: str) -> SufficiencyVerdict:
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return _degraded("不是合法 JSON")

    if not isinstance(parsed, dict) or not isinstance(parsed.get("sufficient"), bool):
        return _degraded("缺少布尔字段 sufficient")

    return SufficiencyVerdict(
        sufficient=bool(parsed["sufficient"]),
        missing=str(parsed.get("missing") or ""),
        reason=str(parsed.get("reason") or ""),
    )
