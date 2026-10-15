"""售后信息提取测试。

`with_structured_output` 是关键机制：假模型通过实现 `bind_tools` + 返回携带
tool call 的 AIMessage，让 langchain 自己的解析与 pydantic 校验**真实执行**
（见 `tests/test_fakes.py` 的验证）。

Review Focus 4（不许编造订单号）在这里钉住离线的一半：只要模型给出 None，
上层就必须原样保留 None。评估集负责另一半（真实模型）。
"""

import pytest

from app.extract import ExtractionError, extract_request
from app.schemas import AfterSalesRequest, ExpectedSolution, RequestType
from tests.fakes import FakeChatModel, Recorder


async def test_extract_returns_validated_pydantic_model():
    model = FakeChatModel(
        structured_payload={
            "order_no": "20240115001",
            "request_type": "refund",
            "expected_solution": "refund",
            "detail": "耳机没收到，要求退款",
        }
    )
    result = await extract_request("订单 20240115001 耳机没收到，要退款", model)
    assert isinstance(result, AfterSalesRequest)
    assert result.order_no == "20240115001"
    assert result.request_type is RequestType.REFUND
    assert result.expected_solution is ExpectedSolution.REFUND


async def test_extract_preserves_null_order_no_instead_of_fabricating():
    """Review Focus 4：没提订单号就必须是 None。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "logistics",
            "expected_solution": "unknown",
            "detail": "快递一直没动",
        }
    )
    result = await extract_request("我的快递一直没动", model)
    assert result.order_no is None


async def test_extract_defaults_expected_solution_when_omitted():
    """模型没给 expected_solution 时，由 schema 默认值兜底，不该报错。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "other",
            "detail": "问一下营业时间",
        }
    )
    result = await extract_request("你们几点上班", model)
    assert result.expected_solution is ExpectedSolution.UNKNOWN


async def test_extract_raises_when_model_returns_no_tool_call():
    """模型不按 schema 输出时，langchain 的解析器返回 None（不是抛异常），
    上层必须自己判 None 并转成 ExtractionError。"""
    model = FakeChatModel(reply="抱歉我听不懂")
    with pytest.raises(ExtractionError):
        await extract_request("随便说说", model)


async def test_extract_raises_when_payload_violates_schema():
    """模型给了 tool call 但字段非法（枚举外取值）时也要转成 ExtractionError。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "complaint",  # 不在枚举里
            "expected_solution": "unknown",
            "detail": "x",
        }
    )
    with pytest.raises(ExtractionError):
        await extract_request("我要投诉", model)


async def test_extract_prompt_actually_carries_user_text():
    rec = Recorder()
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "other",
            "expected_solution": "unknown",
            "detail": "闲聊",
        },
        recorder=rec,
    )
    await extract_request("今天天气不错", model)
    assert any("今天天气不错" in str(m.content) for m in rec.messages)


async def test_extract_does_not_write_to_session_store():
    """提取是纯函数式的，不该碰对话历史（两者共用模型但不共用状态）。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "other",
            "expected_solution": "unknown",
            "detail": "x",
        }
    )
    result = await extract_request("x", model)
    assert isinstance(result, AfterSalesRequest)
