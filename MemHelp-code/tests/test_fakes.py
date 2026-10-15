"""假模型自身的验证。

这些测试证明假模型确实驱动了被测代码的**真实路径**：

- ``astream`` 逐块产出（尾块为空串，与真实上游一致）；
- ``with_structured_output`` 走的是 langchain 自己的 tool-call 解析 +
  Pydantic 校验，而不是让假模型直接返回对象 —— 否则 Task 8 的测试就是自证；
- ``Recorder`` 能把模型实际收到的消息回传给测试，用于验证多轮上下文。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.schemas import AfterSalesRequest, RequestType
from tests.fakes import FakeChatModel, Recorder


async def test_astream_yields_content_chunks_and_a_trailing_empty_one():
    """尾块为空串，与真实上游一致 —— 上层必须过滤掉空串。"""
    model = FakeChatModel(chunks=["你", "好"])
    received = [chunk.content async for chunk in model.astream([HumanMessage("hi")])]
    assert received[:2] == ["你", "好"]
    assert received[-1] == ""


def test_recorder_observes_messages_the_model_received():
    rec = Recorder()
    model = FakeChatModel(chunks=["a"], recorder=rec)
    model.invoke([HumanMessage("问到你了")])
    assert [m.content for m in rec.messages] == ["问到你了"]


def test_with_structured_output_parses_tool_call_into_pydantic():
    """关键：langchain 用 schema 名匹配 tool call 名，假模型必须对得上。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": "20240115001",
            "request_type": "refund",
            "expected_solution": "refund",
            "detail": "没收到货，要退款",
        }
    )
    result = model.with_structured_output(AfterSalesRequest).invoke(
        [HumanMessage("订单 20240115001 没收到，要退款")]
    )
    assert isinstance(result, AfterSalesRequest)
    assert result.order_no == "20240115001"
    assert result.request_type is RequestType.REFUND


def test_with_structured_output_raises_on_payload_violating_schema():
    """payload 非法时应该被 langchain/pydantic 拦下，而不是悄悄放过。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "complaint",  # 不在枚举里
            "expected_solution": "unknown",
            "detail": "x",
        }
    )
    with pytest.raises(Exception):
        model.with_structured_output(AfterSalesRequest).invoke([HumanMessage("x")])


def test_missing_structured_payload_yields_none_not_an_exception():
    """重要发现：没有 tool call 时 PydanticToolsParser 返回 None，**不抛异常**。

    所以上层必须显式检查返回值是不是 None（Task 8 的 extract_request 靠这个
    分支把「模型没按要求输出」转成 ExtractionError），否则会把 None 当成成功。
    """
    model = FakeChatModel(reply="我不太明白")
    result = model.with_structured_output(AfterSalesRequest).invoke([HumanMessage("x")])
    assert result is None


async def test_fail_after_raises_mid_stream():
    model = FakeChatModel(chunks=["a", "b", "c"], fail_after=2)
    received = []
    with pytest.raises(RuntimeError, match="fake upstream failure"):
        async for chunk in model.astream([HumanMessage("hi")]):
            received.append(chunk.content)
    assert received == ["a", "b"]
