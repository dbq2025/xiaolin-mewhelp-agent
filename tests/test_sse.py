"""SSE 编码器测试。

不用 FastAPI 原生 `fastapi.sse` 的原因（实测结论）：

- 原生 SSE 要求路由函数**本身**是异步生成器，``ServerSentEvent`` 之外的东西
  一律被 JSON 编码成 ``data: "..."``，所以没法在同一个生成器里返回普通 JSON；
- 想用两条同路径路由分流也不行 —— Starlette 匹配到第一条路径+方法相同的路由
  就结束（guard 里抛 ``HTTPException`` 只会直接返回 422，**不会**回落到第二条）。

所以这里自己编码 SSE 帧，用普通 ``StreamingResponse`` 承载。纯字符串处理，
可以完全离线单测，不用管 FastAPI 的行为。
"""

from app.sse import sse_frame


def test_sse_frame_shape():
    assert sse_frame("token", {"delta": "您"}) == 'event: token\ndata: {"delta": "您"}\n\n'


def test_sse_frame_ends_with_blank_line_for_every_event():
    for event in ("meta", "token", "done", "error"):
        assert sse_frame(event, {"a": 1}).endswith("\n\n")


def test_sse_frame_keeps_non_ascii_readable():
    """中文不转义成 \\uXXXX —— curl 直接看到可读内容，也少传字节。"""
    frame = sse_frame("token", {"delta": "你好，我是客服小美"})
    assert "你好，我是客服小美" in frame
    assert "\\u" not in frame


def test_sse_frame_escapes_newlines_inside_json():
    """data 里出现换行时不能把帧结构撑破（JSON 会转义成 \\n）。"""
    frame = sse_frame("token", {"delta": "第一行\n第二行"})
    assert frame.count("\n\n") == 1
    assert "\\n" in frame
    assert frame.startswith("event: token\ndata: {")


def test_sse_frame_data_is_valid_json():
    import json

    frame = sse_frame("done", {"finish_reason": "stop", "kept": 3})
    line = frame.split("data: ", 1)[1].strip()
    assert json.loads(line) == {"finish_reason": "stop", "kept": 3}
