"""SSE 帧编码 —— 纯字符串处理，无 IO、无框架依赖。

## 为什么不用 ``fastapi.sse``

FastAPI 0.141.1 提供了原生 ``EventSourceResponse`` / ``ServerSentEvent``，但它
要求路由函数**本身**是异步生成器，并且只认 ``ServerSentEvent``：yield 普通
``Response`` 会被当成数据 JSON 编码，yield 字符串也会被编码成
``data: "..."``。所以「一个路径，``stream`` 开关切换 SSE 与普通 JSON」这件事
它做不到 —— 除非拆成两条路由，而 Starlette 匹配到第一条路径+方法相同的路由
就结束，guard 里抛 ``HTTPException`` 只会直接返回 422，**不会回落到第二条**
（四种组合都实测过）。

于是退回普通 ``StreamingResponse`` + 自己编码 SSE 帧：一个异步生成器既能吐
SSE 帧，也能在 ``stream=false`` 时吐一个 JSON 对象。帧格式按 SSE 规范：
``event: <name>`` / ``data: <json>`` / 空行结束。

手工编码的代价是格式正确性要自己保证，所以 ``tests/test_sse.py`` 专门钉住它。
"""

import json
from typing import Any

#: SSE 的 MIME 类型。
SSE_MEDIA_TYPE = "text/event-stream"


def sse_frame(event: str, data: Any) -> str:
    """把一条事件编码成 SSE 帧。

    ``json.dumps(..., ensure_ascii=False)`` 让中文保持可读（curl 里直接看到汉字，
    也少传字节）。JSON 会把消息内部的换行转义成 ``\\n``，所以帧结构不会被撑破。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
