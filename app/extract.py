"""售后信息提取用例 —— 用 ``with_structured_output`` 把自由文本抽成固定字段。

用真实的 ``with_structured_output``（而不是让假模型直接返回对象），这样
langchain 的 tool-call 解析与 pydantic 校验都被真实执行。

## 三类失败，必须区分开

1. **上游基础设施错误**（连不上 / 认证失败 / 超时）：原样抛出，由 ``api.py``
   映射成 502 ``upstream_*``。**不能**在这里包成「提取失败」—— 那样调用方会
   以为是这段文本抽不出东西，实际是上游不可达，排查方向完全错。
2. **模型不吐 tool call**：langchain 的 ``PydanticToolsParser`` 这时返回
   ``None`` 而**不抛异常**（已读源码确认，见 ``tests/test_fakes.py``）。
   若不检查，就会把 ``None`` 当成成功结果返回。
3. **字段非法**（例如枚举外的取值）：pydantic 校验抛异常。

2 和 3 都是「模型给了东西但不符合 schema」，统一转成 ``ExtractionError``，
由 ``api.py`` 映射成 HTTP 422 —— 这类错误几乎都是 Prompt 或模型能力问题，
排查时需要看到原文。
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.prompts import EXTRACT_PROMPT
from app.schemas import AfterSalesRequest


class ExtractionError(RuntimeError):
    """模型输出无法解析成 ``AfterSalesRequest``（**不含**上游基础设施错误）。"""


def is_infrastructure_error(exc: BaseException) -> bool:
    """判断异常是不是「上游连不上/认证失败」这类基础设施问题。

    多特征判断，任一命中即为真：``openai`` 自己的异常类型、（链式）原因里的
    类型、以及消息里的关键词。比只匹配消息文本稳 —— ``openai`` 的异常消息会
    随版本变（例如 ``APIConnectionError`` 的 str 只有 "Connection error."）。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        if any(
            token in name
            for token in ("apiconnectionerror", "apitimeouterror", "authenticationerror", "permissiondeniederror")
        ):
            return True
        if any(
            token in message
            for token in (
                "connection error",
                "connection refused",
                "name or service not known",
                "temporary failure in name resolution",
                "getaddrinfo failed",
                "timed out",
                "timeout",
                "incorrect api key",
                "invalid api key",
                "unauthorized",
                "401",
                "403",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


async def extract_request(text: str, model: BaseChatModel) -> AfterSalesRequest:
    """把一段售后描述抽成结构化字段。

    :param text: 用户原始描述。
    :param model: 任何说 OpenAI 协议的 chat model（测试注入假模型）。
    :raises ExtractionError: 模型没按 schema 输出，或输出无法通过校验。
    :raises Exception: 上游基础设施错误**原样抛出**，交由 HTTP 层分类。
    """
    structured = model.with_structured_output(AfterSalesRequest)
    messages = EXTRACT_PROMPT.format_messages(text=text)

    try:
        result = await structured.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001
        if is_infrastructure_error(exc):
            raise  # 交给 api.py 映射成 upstream_* + 502
        raise ExtractionError(f"模型输出无法解析为结构化结果：{exc}") from exc

    # 关键：解析器在「没有 tool call」时返回 None 而不抛异常，必须显式拦住。
    if not isinstance(result, AfterSalesRequest):
        raise ExtractionError(f"模型未按 schema 返回：{result!r}")

    return result
