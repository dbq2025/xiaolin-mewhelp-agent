"""售后信息提取用例 —— 用 ``with_structured_output`` 把自由文本抽成固定字段。

用真实的 ``with_structured_output``（而不是让假模型直接返回对象），这样
langchain 的 tool-call 解析与 pydantic 校验都被真实执行。

## 两个必须显式处理的失败路径

1. **模型不吐 tool call**：langchain 的 ``PydanticToolsParser`` 这时返回 ``None``
   而**不抛异常**（已读源码确认，见 ``tests/test_fakes.py``）。若不检查，
   就会把 ``None`` 当成成功结果返回给调用方。
2. **字段非法**（例如枚举外的取值）：pydantic 校验抛异常。

两者都统一转成 ``ExtractionError``，由 ``api.py`` 映射成 HTTP 422 并带上原始
失败原因 —— 这类错误几乎都是 Prompt 或模型能力问题，排查时需要看到原文。
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.prompts import EXTRACT_PROMPT
from app.schemas import AfterSalesRequest


class ExtractionError(RuntimeError):
    """模型输出无法解析成 ``AfterSalesRequest``。"""


async def extract_request(text: str, model: BaseChatModel) -> AfterSalesRequest:
    """把一段售后描述抽成结构化字段。

    :param text: 用户原始描述。
    :param model: 任何说 OpenAI 协议的 chat model（测试注入假模型）。
    :raises ExtractionError: 模型没按 schema 输出，或输出无法通过校验。
    """
    structured = model.with_structured_output(AfterSalesRequest)
    messages = EXTRACT_PROMPT.format_messages(text=text)

    try:
        result = await structured.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 - 上游各种异常统一转成领域错误
        raise ExtractionError(f"模型输出无法解析为结构化结果：{exc}") from exc

    # 关键：解析器在「没有 tool call」时返回 None 而不抛异常，必须显式拦住。
    if not isinstance(result, AfterSalesRequest):
        raise ExtractionError(f"模型未按 schema 返回：{result!r}")

    return result
