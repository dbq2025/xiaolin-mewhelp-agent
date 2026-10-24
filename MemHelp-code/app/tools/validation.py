"""ch08：按 JSON Schema 统一校验工具参数。

内置工具的 Schema 由 Pydantic 的 ``model_json_schema()`` 导出，MCP 工具直接用
Server 的 ``inputSchema``，所以这里只认**普通 dict**，不依赖任何 Pydantic 类型。

校验失败返回一句中文说明（``字段: 原因``，多条用「；」连接），由执行引擎包成
工具结果回灌给模型；校验器自身出错也降级成一句说明，绝不向上抛。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

_QUOTED = re.compile(r"'([^']+)'")


def _field_name(error: Any) -> str:
    """取错误对应的字段名；顶层错误（如 additionalProperties）从文案里抠。"""
    if error.validator == "required":
        found = _QUOTED.search(error.message)
        return found.group(1) if found else ""
    if error.validator == "additionalProperties":
        found = _QUOTED.findall(error.message)
        return "、".join(found)
    path = list(error.absolute_path)
    return str(path[-1]) if path else ""


def _reason(error: Any) -> str:
    """把 jsonschema 的英文断言翻成给模型看的中文。"""
    if error.validator == "required":
        return "是必填项"
    if error.validator == "enum":
        allowed = "、".join(str(item) for item in error.validator_value)
        return f"取值必须是：{allowed}"
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            expected = "、".join(str(item) for item in expected)
        return f"类型应为 {expected}"
    if error.validator == "minimum":
        return f"不能小于 {error.validator_value}"
    if error.validator == "maximum":
        return f"不能大于 {error.validator_value}"
    if error.validator == "minLength":
        return f"长度不能少于 {error.validator_value}"
    if error.validator == "maxLength":
        return f"长度不能超过 {error.validator_value}"
    if error.validator == "additionalProperties":
        return "不允许该字段"
    if error.validator == "pattern":
        return "格式不符合要求"
    return str(error.message)


def validate_against_schema(schema: dict[str, Any], args: Mapping[str, Any]) -> str | None:
    """通过返回 ``None``，否则返回可直接回灌给模型的中文错误说明。"""
    try:
        # 先验 Schema 本身：``required`` 写成字符串这类坏定义，jsonschema 会把它
        # 当字符序列逐字符报「必填」，那比报错更难排查。
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(dict(args)),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    except Exception as exc:  # 坏 Schema 也不能把工具调用打断
        return f"参数不合法 —— {type(exc).__name__}: {exc}"

    if not errors:
        return None

    parts: list[str] = []
    for error in errors:
        field = _field_name(error)
        reason = _reason(error)
        text = f"{field}: {reason}" if field else reason
        if text not in parts:
            parts.append(text)
    return "；".join(parts)
