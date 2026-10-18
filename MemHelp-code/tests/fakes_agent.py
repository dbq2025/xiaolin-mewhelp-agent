"""ch05 的脚本化假模型：按调用顺序返回「文本 / 工具调用 / 多工具 / 空回复」。

## 为什么不用现有 `FakeChatModel`

`FakeChatModel` 是为 ch01/ch02 的「一次工具调用」写的：它只会在第一次
`_generate` 时把 `tool_calls_then_stream` 吐出来。ReAct 需要的是**每一步都由
测试指定**：第 1 步查订单、第 2 步查物流、第 3 步收敛。脚本模型让每一步的
tool_calls、内容、usage 都可控，同时仍然走真实的 `bind_tools` / `astream` 路径。

## `state` 为什么是 dataclass

`bind_tools` 会返回 `model_copy`；若把「脚本走到第几条」放在 pydantic 字段里，
副本与原对象会各走各的。用 dataclass 共享可变状态，副本与本体看到同一个 index。
"""

from __future__ import annotations

import json
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

Usage = tuple[int, int, int]  # input, output, total


@dataclass
class ScriptState:
    """跨 `bind_tools` 副本共享的脚本游标与调用记录。"""

    index: int = 0
    calls: list[list[BaseMessage]] = field(default_factory=list)
    bound_tool_names: list[str] = field(default_factory=list)


@dataclass
class ScriptTurn:
    """一步模型响应：以 chunk 形式定义，非流式路径由它们合并。"""

    chunks: list[AIMessageChunk]

    def as_message(self) -> AIMessage:
        full = self.chunks[0]
        for chunk in self.chunks[1:]:
            full = full + chunk
        return AIMessage(
            content=full.content or "",
            tool_calls=list(full.tool_calls or []),
            additional_kwargs=dict(full.additional_kwargs),
            usage_metadata=full.usage_metadata,
            id=full.id,
        )


def _with_usage(chunk: AIMessageChunk, usage: Usage) -> AIMessageChunk:
    input_tokens, output_tokens, total_tokens = usage
    return chunk.model_copy(
        update={
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        }
    )


def text_turn(*pieces: str, usage: Usage = (10, 5, 15)) -> ScriptTurn:
    chunks = [AIMessageChunk(content=piece) for piece in pieces] or [AIMessageChunk(content="")]
    chunks[-1] = _with_usage(chunks[-1], usage)
    return ScriptTurn(chunks=chunks)


def empty_turn(*, usage: Usage = (10, 0, 10)) -> ScriptTurn:
    return ScriptTurn(chunks=[_with_usage(AIMessageChunk(content=""), usage)])


def tool_turn(
    name: str,
    args: dict[str, Any],
    *,
    call_id: str = "call_1",
    preamble: str = "",
    usage: Usage = (10, 5, 15),
    extra_kwargs: dict[str, Any] | None = None,
) -> ScriptTurn:
    chunk = AIMessageChunk(
        content=preamble,
        tool_call_chunks=[
            {
                "name": name,
                "args": json.dumps(args, ensure_ascii=False),
                "id": call_id,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        additional_kwargs=dict(extra_kwargs or {}),
    )
    return ScriptTurn(chunks=[_with_usage(chunk, usage)])


def multi_tool_turn(
    specs: Sequence[tuple[str, dict[str, Any], str]],
    *,
    usage: Usage = (10, 5, 15),
) -> ScriptTurn:
    chunks = [
        {
            "name": name,
            "args": json.dumps(args, ensure_ascii=False),
            "id": call_id,
            "index": index,
            "type": "tool_call_chunk",
        }
        for index, (name, args, call_id) in enumerate(specs)
    ]
    chunk = AIMessageChunk(content="", tool_call_chunks=chunks)
    return ScriptTurn(chunks=[_with_usage(chunk, usage)])


class ScriptedChatModel(BaseChatModel):
    """按 `script` 顺序逐次返回的假模型。"""

    script: list[ScriptTurn] = Field(default_factory=list)
    state: ScriptState = Field(default_factory=ScriptState)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"turns": len(self.script), "cursor": self.state.index}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.state.bound_tool_names.extend(getattr(tool, "name", str(tool)) for tool in tools)
        return self

    def _next(self, messages: Sequence[BaseMessage]) -> ScriptTurn:
        self.state.calls.append(list(messages))
        if self.state.index >= len(self.script):
            raise RuntimeError(
                f"脚本模型没有第 {self.state.index + 1} 步响应（共 {len(self.script)} 步）；"
                "说明被测代码多调用了一次模型"
            )
        turn = self.script[self.state.index]
        self.state.index += 1
        return turn

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=turn.as_message())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._next(messages).chunks:
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._next(messages).chunks:
            yield ChatGenerationChunk(message=chunk)
