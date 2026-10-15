"""模型工厂 —— 全项目**唯一**知道上游地址 / 密钥 / 模型名的地方。

除了本模块，任何文件都不得 ``import langchain_openai``：换 GPT / Claude /
DeepSeek / Ollama 只需要改 ``.env``（必要时改 ``OPENAI_BASE_URL``），业务代码
零改动。``tests/test_llm.py`` 有一条测试专门钉住这个约束。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """按配置产出一个说 OpenAI 协议的上游客户端。

    应用侧统一按 OpenAI 协议说话，所以这里只认 ``base_url`` / ``api_key`` /
    ``model`` 三件事 —— 具体的上游是谁对上层完全透明。

    ``stream_usage=True`` 让流式响应附带 token 用量，为后续章节接 Langfuse
    埋点留好数据（当前不消费）。
    """
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.openai_model,
        base_url=s.openai_base_url,
        api_key=s.openai_api_key,
        timeout=s.request_timeout,
        stream_usage=True,
    )
