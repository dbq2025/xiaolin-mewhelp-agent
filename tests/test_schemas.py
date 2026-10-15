import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesRequest, ChatRequest, ExpectedSolution, RequestType


def test_after_sales_request_defaults_when_missing_info():
    req = AfterSalesRequest(
        order_no=None,
        request_type=RequestType.LOGISTICS,
        detail="订单昨天该到今天没到",
    )
    assert req.expected_solution is ExpectedSolution.UNKNOWN


def test_after_sales_request_enum_values_are_lowercase_strings():
    assert RequestType.REFUND.value == "refund"
    assert ExpectedSolution.RESHIP.value == "reshipping"


def test_after_sales_request_rejects_unknown_request_type():
    with pytest.raises(ValidationError):
        AfterSalesRequest(order_no=None, request_type="complaint", detail="x")


def test_chat_request_stream_defaults_true():
    assert ChatRequest(message="你好").stream is True
    assert ChatRequest(message="你好").session_id is None


def test_after_sales_request_field_descriptions_exist_for_schema():
    schema = AfterSalesRequest.model_json_schema()
    for field in ("order_no", "request_type", "expected_solution", "detail"):
        assert schema["properties"][field].get("description")
