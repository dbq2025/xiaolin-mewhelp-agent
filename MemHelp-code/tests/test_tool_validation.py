"""ch08 Task 2：JSON Schema 参数校验（中文错误说明）。"""

from __future__ import annotations

from app.tools.validation import validate_against_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "ticket_type": {"type": "string", "enum": ["售后", "投诉", "咨询"]},
        "count": {"type": "integer", "minimum": 1, "maximum": 9},
    },
    "required": ["description", "ticket_type"],
    "additionalProperties": False,
}


def test_valid_args_pass() -> None:
    assert validate_against_schema(SCHEMA, {"description": "x", "ticket_type": "售后"}) is None


def test_missing_required_is_chinese_and_named() -> None:
    err = validate_against_schema(SCHEMA, {"ticket_type": "售后"})
    assert err is not None
    assert "description" in err and "必填" in err


def test_enum_and_range_errors() -> None:
    err = validate_against_schema(
        SCHEMA, {"description": "x", "ticket_type": "退货", "count": 0}
    )
    assert err is not None
    assert "ticket_type" in err and "售后" in err
    assert "count" in err


def test_additional_property_rejected() -> None:
    err = validate_against_schema(
        SCHEMA, {"description": "x", "ticket_type": "售后", "oops": 1}
    )
    assert err is not None
    assert "oops" in err


def test_wrong_type_is_reported_not_raised() -> None:
    err = validate_against_schema(SCHEMA, {"description": 5, "ticket_type": "售后"})
    assert err is not None and "description" in err


def test_never_raises_on_broken_schema() -> None:
    err = validate_against_schema({"type": "object", "required": "oops"}, {})
    assert err is None or "参数不合法" in err
