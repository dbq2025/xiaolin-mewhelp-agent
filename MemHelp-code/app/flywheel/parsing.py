"""从模型输出里取 JSON —— 飞轮只接受 JSON，但不该被一句客套话搞崩。"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_json", "message_text"]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def extract_json(text: str) -> dict[str, Any] | None:
    """宽容地取第一个 JSON 对象：允许 ``` 围栏、前后解释文字。"""
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = _FENCE.search(raw)
    candidates = [fenced.group(1) if fenced else raw]
    object_match = _OBJECT.search(raw)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None
