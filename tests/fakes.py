"""离线测试用的假模型与工具。

设计目标：让上层用例的**真实代码路径**被跑到，而不是把整条链打桩。

- 走 ``model.astream(...)`` 时按 ``chunks`` 逐块吐出 ``AIMessageChunk``，
  和上游的逐 token 流式同形（尾块会是空串，与真实上游一致）；
  ``fail_after`` 用来模拟「流到一半上游挂了」。
- 走 ``model.with_structured_output(Schema)`` 时，langchain 的实现是
  ``self.bind_tools([schema], tool_choice="any") | PydanticToolsParser``。
  所以这里实现 ``bind_tools`` 记录下被绑定的 schema，再由 ``_generate``
  返回一条携带 **OpenAI 风格 tool call** 的 ``AIMessage`` —— 解析与 Pydantic
  校验由 langchain 真实执行，而不是让假模型直接返回对象。
- 不给 ``structured_payload`` 时就返回一条**不带 tool call** 的普通消息，
  于是 ``PydanticToolsParser`` 会真实地解析失败 —— 用来验证上层是否把
  「模型不按 schema 输出」转成了明确的错误。
- ``captured`` 传入一个 list 时，每次调用都会把收到的消息追加进去，
  用来断言「第二轮请求里到底带了哪些历史」。
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


@dataclass
class Recorder:
    """记录模型每次调用收到的消息。

    必须用 dataclass 而不是 `list` 字段：pydantic 在构造模型时会**复制** list
    字段，外部拿到的就不是同一个对象了。dataclass 属于任意类型，按引用传递。
    """

    messages: list[BaseMessage] = field(default_factory=list)


class FakeChatModel(BaseChatModel):
    """一个够用就好的假 chat model。"""

    chunks: list[str] = Field(default_factory=list)
    raw_chunks: list[Any] = Field(default_factory=list)
    """直接作为 ``content`` 使用的原始块。

    用来模拟「content 是 content-block 列表」的 provider（``[{"type": "text",
    "text": "..."}]``），以及推理模型那种只有 reasoning 块、没有文本块的情况。
    """
    reply: str = "ok"
    structured_payload: dict[str, Any] | None = None
    fail_after: int | None = None
    recorder: Recorder | None = None
    # 由 bind_tools 写入：被绑定的 schema（with_structured_output 会绑一个）
    bound_schemas: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"chunks": len(self.chunks), "reply": self.reply}

    # -- 工具绑定（with_structured_output 的前置条件）----------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        """记录被绑定的 schema，返回一个带该记录的副本。

        langchain 的 ``with_structured_output`` 先调本方法，再用
        ``PydanticToolsParser`` 解析产出的 tool call。这里只记录 schema 名，
        真正的解析交给 langchain。
        """
        return self.model_copy(update={"bound_schemas": list(tools)})

    # -- 内部工具 ---------------------------------------------------------

    def _record(self, messages: Sequence[BaseMessage]) -> None:
        if self.recorder is not None:
            self.recorder.messages.extend(messages)

    def _structured_message(self) -> AIMessage:
        """把 structured_payload 包成一条带 tool call 的 AIMessage。

        tool call 的 name 必须是绑定 schema 的类名 —— langchain 的
        ``PydanticToolsParser`` 按名字匹配。
        """
        assert self.structured_payload is not None
        name = getattr(self.bound_schemas[0], "__name__", "schema") if self.bound_schemas else "schema"
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": dict(self.structured_payload),
                    "id": "call_fake_1",
                    "type": "tool_call",
                }
            ],
        )

    def _reply_message(self) -> AIMessage:
        wants_structured = bool(self.bound_schemas)
        if wants_structured and self.structured_payload is not None:
            return self._structured_message()
        # 没有 payload（或没绑 schema）时给一条普通消息：
        # 若调用方期望结构化输出，PydanticToolsParser 会真实地解析失败。
        return AIMessage(content=self.reply)

    # -- 同步路径 ---------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._record(messages)
        return ChatResult(generations=[ChatGeneration(message=self._reply_message())])

    def _stream_items(self) -> list[Any]:
        """要逐个作为 ``content`` 吐出去的块（``raw_chunks`` 优先）。"""
        return list(self.raw_chunks) if self.raw_chunks else list(self.chunks)

    # -- 流式路径 ---------------------------------------------------------

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._record(messages)
        for index, content in enumerate(self._stream_items()):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("fake upstream failure mid-stream")
            yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._record(messages)
        for index, content in enumerate(self._stream_items()):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("fake upstream failure mid-stream")
            yield ChatGenerationChunk(message=AIMessageChunk(content=content))


def fake_structured_message(payload: dict[str, Any], name: str = "AfterSalesRequest") -> AIMessage:
    """独立造一条携带 tool call 的 AIMessage，供需要直接构造的场景使用。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": dict(payload),
                "id": "call_fake_1",
                "type": "tool_call",
            }
        ],
    )


class FailingChatModel(FakeChatModel):
    """每次都抛指定异常的假模型。

    用来离线测「上游不可达 / 认证失败」这类基础设施错误在 HTTP 层被映射成什么 ——
    不需要真的联网，只要异常类型与消息和真实上游一致即可
    （例如 ``openai.APIConnectionError`` 的 str 就是 ``"Connection error."``）。
    """

    error: Exception = Field(default_factory=lambda: RuntimeError("boom"))

    def _raise(self) -> None:
        raise self.error

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:  # type: ignore[override]
        self._raise()

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:  # type: ignore[override]
        self._raise()
        yield  # pragma: no cover - 仅为让本方法是异步生成器

    def with_structured_output(self, schema: Any = None, **kwargs: Any):  # type: ignore[override]
        error = self.error

        class _AlwaysFails:
            async def ainvoke(self, *args: Any, **kw: Any) -> Any:
                raise error

            def invoke(self, *args: Any, **kw: Any) -> Any:
                raise error

        return _AlwaysFails()
