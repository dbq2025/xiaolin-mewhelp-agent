"""意图识别 —— 模型分类 + 确定性兜底 + 逐字寒暄快速通道。

## 三层，按成本递增

1. **快速通道**：整句命中 `TRIVIAL_CHITCHAT` → 直接 chitchat，零模型调用。
   需求要求「闲聊不花模型调用」；但「在吗，退货政策是什么」不能被这条吞掉，
   所以只做**整句**匹配，不做前缀/包含匹配。
2. **模型分类**：一个简单 Prompt 输出七类 JSON；超时/异常/坏 JSON 都不抛。
3. **启发式兜底**：关键词优先级表，默认 `after_sales`（business 出口）。
   把真实问题误判成闲聊的代价，比多走一次 Agent 大。

## 为什么 raw 要留着

意图错路由时，第一手证据是「模型原话是什么」。`intent_raw` 只进日志，不回给
用户（可能含用户原句，但日志本来就有；不要进 SSE）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
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
}


@dataclass(frozen=True)
class IntentResult:
    """一次意图判断的结果。"""

    intent: Intent
    confidence: float
    raw: str
    source: Literal["llm", "heuristic", "rule"]


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
    """关键词优先级兜底。默认 `after_sales`（business 出口），不是 chitchat。"""
    text = message or ""
    for intent, keywords in _HEURISTIC_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return "after_sales"


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

    def __init__(self, model: object, *, timeout: float = 5.0) -> None:
        self._model = model
        self._timeout = timeout

    async def classify(self, message: str) -> IntentResult:
        if is_trivial_chitchat(message):
            return IntentResult(intent="chitchat", confidence=1.0, raw="", source="rule")

        raw = ""
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke(  # type: ignore[attr-defined]
                    [
                        SystemMessage(content=INTENT_PROMPT),
                        HumanMessage(content=message),
                    ]
                ),
                timeout=self._timeout,
            )
            raw = _message_text(response)
            parsed = parse_intent_reply(raw)
            if parsed is not None:
                return parsed
            logger.warning("意图分类返回无法解析的 JSON，走启发式兜底：%r", raw[:120])
        except Exception as exc:  # noqa: BLE001 - 分类失败不该把聊天变成 500
            raw = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("意图分类模型调用失败，走启发式兜底", exc_info=True)

        return IntentResult(
            intent=heuristic_intent(message),
            confidence=0.0,
            raw=raw,
            source="heuristic",
        )
