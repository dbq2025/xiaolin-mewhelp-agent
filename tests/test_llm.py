"""模型工厂测试。

除了「配置有没有接对」，这里还钉住一个**上游适配**：某些 OpenAI 兼容网关
只接受 ``tool_choice`` 为 ``none`` / ``auto`` / ``required``，而 langchain 的
``with_structured_output(method="function_calling")`` 会发 ``tool_choice=<函数名>``
（实测 DeepSeek 网关返回 422）。适配层在模型工厂里完成，业务代码不用管。
"""

from app.config import Settings
from app.llm import get_chat_model


def _settings() -> Settings:
    return Settings(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="deepseek-chat",
    )


def test_get_chat_model_wires_model_name():
    model = get_chat_model(_settings())
    assert model.model_name == "deepseek-chat"


def test_get_chat_model_wires_custom_base_url():
    model = get_chat_model(_settings())
    assert "gw.example.com" in str(model.openai_api_base)


def test_get_chat_model_uses_settings_api_key():
    model = get_chat_model(_settings())
    assert model.openai_api_key.get_secret_value() == "sk-test"


def test_only_llm_module_imports_langchain_openai():
    """全局约束：只有 app/llm.py 可以 import langchain_openai。"""
    from pathlib import Path

    offenders = []
    for path in Path("app").rglob("*.py"):
        if path.as_posix() == "app/llm.py":
            continue
        if "langchain_openai" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert offenders == []


# ---------------------------------------------------------------------------
# tool_choice 适配
# ---------------------------------------------------------------------------


def test_tool_choice_tool_name_is_rewritten_to_auto():
    """langchain 会把函数名放进 tool_choice，很多网关不认 —— 改写成 auto。

    ``with_structured_output(method="function_calling")`` 正是发
    ``tool_choice=<schema 的类名>``，所以这里用真实的 schema 复现那个路径。
    """
    from app.schemas import AfterSalesRequest

    model = get_chat_model(_settings())
    bound = model.bind_tools([AfterSalesRequest], tool_choice="AfterSalesRequest")
    assert bound.kwargs.get("tool_choice") == "auto"


def test_tool_choice_auto_is_left_alone():
    from app.schemas import AfterSalesRequest

    model = get_chat_model(_settings())
    bound = model.bind_tools([AfterSalesRequest], tool_choice="auto")
    assert bound.kwargs.get("tool_choice") == "auto"


def test_tool_choice_required_is_left_alone():
    from app.schemas import AfterSalesRequest

    model = get_chat_model(_settings())
    bound = model.bind_tools([AfterSalesRequest], tool_choice="required")
    assert bound.kwargs.get("tool_choice") == "required"


def test_tool_choice_none_is_left_alone():
    from app.schemas import AfterSalesRequest

    model = get_chat_model(_settings())
    bound = model.bind_tools([AfterSalesRequest], tool_choice="none")
    assert bound.kwargs.get("tool_choice") == "none"


def test_tool_choice_dict_form_is_preserved():
    """显式指定某个函数的 dict 形态不做改写（调用方是故意的就尊重）。"""
    from app.schemas import AfterSalesRequest

    model = get_chat_model(_settings())
    explicit = {"type": "function", "function": {"name": "AfterSalesRequest"}}
    bound = model.bind_tools([AfterSalesRequest], tool_choice=explicit)
    assert bound.kwargs.get("tool_choice") == explicit


def test_with_structured_output_still_parses_after_adaptation():
    """适配之后 with_structured_output 的契约不变：仍然解析成 Pydantic 实例。"""
    from langchain_core.messages import HumanMessage

    from app.schemas import AfterSalesRequest, RequestType
    from tests.fakes import FakeChatModel

    # 用假模型验证「with_structured_output 这条路仍然走得通」的机制
    fake = FakeChatModel(
        structured_payload={
            "order_no": "20240115001",
            "request_type": "refund",
            "expected_solution": "refund",
            "detail": "x",
        }
    )
    result = fake.with_structured_output(
        AfterSalesRequest, method="function_calling"
    ).invoke([HumanMessage("x")])
    assert isinstance(result, AfterSalesRequest)
    assert result.request_type is RequestType.REFUND
