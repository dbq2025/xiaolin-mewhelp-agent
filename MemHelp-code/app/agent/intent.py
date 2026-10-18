"""意图识别 —— 八类（七类业务 + other 兜底）+ 质量档 / 成本档。

## 为什么有质量档和成本档

用户要求「默认配够格的大模型先把准确率做上去，成本吃紧时再走降级路」。
所以默认只调主模型；`INTENT_COST_MODE=true` 且有 `INTENT_CHEAP_MODEL` 时，
才先用小模型判一次，置信度低于阈值再让主模型重判。

## 三层兜底

1. **快速通道**：整句命中 `TRIVIAL_CHITCHAT` → chitchat，零模型调用。
   只做整句匹配，不做包含匹配：「在吗，退货政策是什么」不能被吞掉。
2. **模型**：强制 JSON；坏 JSON/超时/异常都不抛。
3. **启发式**：关键词优先级表，默认 `other`（交给主力 Agent 澄清），
   不再默认 `after_sales` —— 拿不准的问题不该被硬塞进业务意图。

## `other` 不是「模型失败」

`other` 是模型的合法输出，`source` 仍是 `llm`；只有模型调用失败才走
`heuristic`。两者在日志里必须能区分。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import INTENT_PROMPT, TRIVIAL_CHITCHAT
from app.agent.state import ALL_INTENTS, Intent
from app.chat import _message_text

logger = logging.getLogger(__name__)

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "heuristic_intent",
    "is_trivial_chitchat",
    "parse_intent_reply",
]

#: 模型偶尔会输出中文类名，容忍它，别为了一个翻译让整轮掉到启发式。
_ALIASES: dict[str, Intent] = {
    "物流": "logistics",
    "订单": "order",
    "商品咨询": "product_consult",
    "退款退货": "refund_return",
    "售后": "after_sales",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "其他": "other",
}

IntentSource = Literal["llm", "llm_cheap", "llm_escalated", "heuristic", "rule"]


@dataclass(frozen=True)
class IntentResult:
    """一次意图判断的结果。"""

    intent: Intent
    confidence: float
    raw: str
    source: IntentSource
    llm_calls: int = 0


def _normalise(message: str) -> str:
    """去空白与标点，小写化。中文标点也算分隔符。"""
    return re.sub(r"[\s\W_]+", "", (message or "").lower(), flags=re.UNICODE)


def is_trivial_chitchat(message: str) -> bool:
    """整句是否只是寒暄。

    **不做包含匹配**：`"在吗，退货政策是什么"` 里虽然有「在吗」，但它是一个
    真实的知识类问题；前缀匹配会把它送进固定话术，用户再也拿不到政策答案。
    """
    normalised = _normalise(message)
    if not normalised or len(normalised) > 8:
        return False
    return normalised in TRIVIAL_CHITCHAT


#: 关键词优先级：越靠前越优先。投诉必须压过退款退货 ——
#: 「我要投诉并退款」的主诉求是投诉，先安抚+给人工选项，而不是直接答退款政策。
_HEURISTIC_RULES: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        "complaint",
        ("投诉", "举报", "曝光", "消协", "12315", "道歉", "赔偿", "气死", "太失望"),
    ),
    (
        "refund_return",
        ("退款", "退货", "退换", "退钱", "七天无理由", "无理由", "退款流程"),
    ),
    (
        "logistics",
        ("物流", "快递", "到哪", "发货", "派送", "签收", "揽收", "运单", "到货", "没收到", "还没到", "运输"),
    ),
    ("order", ("订单", "下单", "订单号", "订单状态", "付款", "待付款", "已付款", "金额")),
    (
        "after_sales",
        ("售后", "换货", "维修", "保修", "质保", "发票", "补发", "质量问题", "坏了", "碎了", "破损"),
    ),
    (
        "product_consult",
        ("价格", "多少钱", "库存", "有货", "参数", "型号", "续航", "尺寸", "颜色", "功能", "商品"),
    ),
)


def heuristic_intent(message: str) -> Intent:
    """关键词优先级兜底。默认 `other`（business 出口，由 Agent 澄清）。"""
    text = message or ""
    for intent, keywords in _HEURISTIC_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return "other"


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def _loads_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        # 模型偶尔在 JSON 前后加一句解释；只取第一个 { 到最后一个 }。
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None


def parse_intent_reply(reply: str) -> IntentResult | None:
    """严格解析模型输出；解析不出来返回 None，由调用方兜底。"""
    text = _strip_code_fence((reply or "").strip())
    parsed = _loads_json(text)
    if not isinstance(parsed, dict):
        return None

    raw_intent = str(parsed.get("intent") or "").strip()
    intent = _ALIASES.get(raw_intent, raw_intent.lower())  # type: ignore[arg-type]
    if intent not in ALL_INTENTS:
        return None

    confidence = parsed.get("confidence", 0.5)
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.5
    value = min(1.0, max(0.0, value))

    return IntentResult(
        intent=intent,  # type: ignore[arg-type]
        confidence=value,
        raw=text,
        source="llm",
    )


class IntentClassifier:
    """一次意图判断；**永不抛**（异常降级为启发式）。"""

    def __init__(
        self,
        primary_model: object,
        *,
        cheap_model: object | None = None,
        timeout: float = 5.0,
        confidence_threshold: float = 0.6,
        cost_mode: bool = False,
    ) -> None:
        self._primary = primary_model
        self._cheap = cheap_model
        self._timeout = timeout
        self._threshold = confidence_threshold
        self._cost_mode = cost_mode

    async def classify(self, message: str) -> IntentResult:
        if is_trivial_chitchat(message):
            return IntentResult(
                intent="chitchat", confidence=1.0, raw="", source="rule"
            )

        calls = 0
        last_raw = ""

        if self._cost_mode and self._cheap is not None:
            cheap, last_raw = await self._call(self._cheap, message)
            calls += 1
            if cheap is not None and cheap.confidence >= self._threshold:
                return replace(cheap, source="llm_cheap", llm_calls=1)

        primary, raw = await self._call(self._primary, message)
        calls += 1
        if raw:
            last_raw = raw
        if primary is not None:
            source: IntentSource = "llm_escalated" if calls > 1 else "llm"
            return replace(primary, source=source, llm_calls=calls)

        return IntentResult(
            intent=heuristic_intent(message),
            confidence=0.0,
            raw=last_raw,
            source="heuristic",
            llm_calls=calls,
        )

    async def _call(self, model: object, message: str) -> tuple[IntentResult | None, str]:
        raw = ""
        try:
            response = await asyncio.wait_for(
                model.ainvoke(  # type: ignore[attr-defined]
                    [
                        SystemMessage(content=INTENT_PROMPT),
                        HumanMessage(content=message),
                    ]
                ),
                timeout=self._timeout,
            )
            raw = _message_text(response)
            parsed = parse_intent_reply(raw)
            if parsed is None:
                logger.warning(
                    "意图分类返回无法解析的 JSON，尝试下一档：%r", raw[:120]
                )
            return parsed, raw
        except Exception as exc:  # noqa: BLE001 - 分类失败不该把聊天变成 500
            raw = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("意图分类模型调用失败，尝试下一档", exc_info=True)
            return None, raw
