"""模型工厂 —— 全项目**唯一**知道上游地址 / 密钥 / 模型名的地方。

除了本模块，任何文件都不得 ``import langchain_openai``：换 GPT / Claude /
DeepSeek / Ollama 只需要改 ``.env``（必要时改 ``OPENAI_BASE_URL``），业务代码
零改动。``tests/test_llm.py`` 有一条测试专门钉住这个约束。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

#: 网关普遍接受的 ``tool_choice`` 字面量。
_WELL_KNOWN_TOOL_CHOICES = frozenset({"none", "auto", "required", "any"})


class GatewayChatOpenAI(ChatOpenAI):
    """针对 OpenAI 兼容网关做的一处适配。

    ``with_structured_output(method="function_calling")`` 会调
    ``bind_tools([schema], tool_choice=<schema 类名>)`` —— 也就是发
    ``tool_choice: {"type": "function", "function": {"name": "AfterSalesRequest"}}``。

    不少兼容网关**只接受** ``none`` / ``auto`` / ``required`` 这三种字面量，
    收到函数名形态直接 422（实测 DeepSeek：``tool_choice: expected one of
    ``none``, ...``）。而 langchain 里这个值是硬编码的，没有配置项能改。

    所以在这里把「函数名字符串」改写成 ``"auto"``：语义上仍然允许模型调用工具，
    解析依旧由 langchain 的 ``PydanticToolsParser`` 负责，业务代码完全不用变。

    只改写「裸字符串且不是已知字面量」的情况 —— 显式传 dict 或 ``none`` /
    ``required`` 时按调用方本意原样保留。
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        tool_choice = kwargs.get("tool_choice")
        if isinstance(tool_choice, str) and tool_choice not in _WELL_KNOWN_TOOL_CHOICES:
            kwargs["tool_choice"] = "auto"
        return super().bind_tools(tools, **kwargs)


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """按配置产出一个说 OpenAI 协议的上游客户端。

    应用侧统一按 OpenAI 协议说话，所以这里只认 ``base_url`` / ``api_key`` /
    ``model`` 三件事 —— 具体的上游是谁对上层完全透明。

    ``stream_usage=True`` 让流式响应附带 token 用量，为后续章节接 Langfuse
    埋点留好数据（当前不消费）。
    """
    s = settings or get_settings()
    return GatewayChatOpenAI(
        model=s.openai_model,
        base_url=s.openai_base_url,
        api_key=s.openai_api_key,
        timeout=s.request_timeout,
        stream_usage=True,
    )
