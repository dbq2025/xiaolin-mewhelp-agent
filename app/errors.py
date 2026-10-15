"""上游错误的分类 —— 全项目**唯一**判断「这是哪种上游故障」的地方。

之所以抽成独立模块：早先 ``app/extract.py`` 和 ``app/api.py`` 各自判断了一次，
结果同一个「连不上上游」的错误在 ``/api/extract`` 被报成 422
``extraction_failed``、在 ``/api/chat/completions`` 被报成 502
``upstream_unavailable`` —— 同一件事两种说法，调用方没法判断该重试还是查密钥。

分类只看**异常类型名**和**消息**（含 ``__cause__`` 链），不依赖具体 SDK：

- ``upstream_auth``：认证/权限问题 → 使用者该去查 ``.env`` 里的密钥
- ``upstream_invalid_request``：上游 400 —— 模型名写错、参数不被支持等
  → 使用者该去查 ``.env`` 里的模型名，**不是** Prompt 或文本的问题
- ``upstream_unavailable``：连不上 / 域名解析失败 → 上游或网络问题，可重试
- ``upstream_timeout``：明确的超时（可重试，与不可用区分开便于告警）
- ``upstream_error``：其它上游异常，兜底
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------

#: 常见密钥形态。宁可多脱一点，也不要漏。
#: - ``sk-...`` / ``sk-proj-...``（OpenAI 及多数兼容网关）
#: - ``Bearer <token>`` 授权头
#: - 常见的长 token（32 位以上连续的字母数字）
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(sk-[A-Za-z0-9_\-]{4,})"), "sk-***"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}"), r"\1 ***"),
    (
        re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|secret|password)\b\s*[=:]\s*[\"']?[^\s,\"'}\)]{6,}"),
        r"\1=***",
    ),
)


def redact(text: object) -> str:
    """把文本里可能的密钥替换掉。

    **所有回给客户端的错误信息都必须先过这里**。原因：上游异常的原文常常包含
    请求内容（有些网关会把整个请求体回显在错误里，其中就有 Authorization 头），
    而本地 pydantic 的 ``Settings`` 校验错误会带出 ``input_value`` —— 里面就有
    密钥的前几位。这些文本一旦进了响应体，就等于把凭据交给了调用方。

    服务端日志记原文，响应体记脱敏后的版本。
    """
    result = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class UpstreamFault(RuntimeError):
    """上游/配置故障，带一个可供程序判断的 ``type``。

    定义在这里（而不是 ``app/api.py``）是为了让 ``app/chat.py`` 也能抛它 ——
    否则用例层要反向 import HTTP 层。HTTP 层只负责把 ``type`` 映射成状态码。
    """

    def __init__(self, message: str, type_: str = "upstream_error") -> None:
        super().__init__(message)
        self.type = type_


class UpstreamEmptyResponse(UpstreamFault):
    """上游返回了响应，但一个内容块都没有。

    真实上游会走到这里：推理模型的 reasoning 分块、``content`` 是 content-block
    列表而文本在嵌套结构里的 provider。这时候**不能**把空字符串当成正常回答
    落库 —— 否则下一轮模型会以为客服刚说了句空话。
    """

    def __init__(self, message: str = "上游返回了空内容（没有可用文本块）。") -> None:
        super().__init__(message, type_="upstream_empty_response")


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------

#: 名称里出现这些片段 → 认证/权限问题。
_AUTH_TYPE_MARKERS = (
    "authenticationerror",
    "permissiondeniederror",
    "unauthorizederror",
    "forbidden",
)
_AUTH_MESSAGE_MARKERS = (
    "incorrect api key",
    "invalid api key",
    "unauthorized",
    "authentication",
    "401",
    "403",
)

#: 名称/消息里出现这些片段 → 上游拒绝了请求本身（模型名、参数等）。
#: 典型：DeepSeek 返回 400 "The supported API model names are ..."。
_INVALID_REQUEST_TYPE_MARKERS = ("badrequesterror", "invalidrequesterror", "unprocessableentityerror")
_INVALID_REQUEST_MESSAGE_MARKERS = (
    "invalid_request_error",
    "supported api model names",
    "does not exist",
    "unknown model",
    "model not found",
    "invalid model",
    "unsupported model",
    "400",
)

#: 名称里出现这些片段 → 连不上 / 解析失败。
_CONNECTION_TYPE_MARKERS = ("apiconnectionerror", "connecterror", "connectionerror")
_CONNECTION_MESSAGE_MARKERS = (
    "connection error",
    "connection refused",
    "connection reset",
    "name or service not known",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "nodename nor servname",
    "no such host",
    "502",
    "503",
    "504",
)

#: 名称/消息里出现这些片段 → 超时。
_TIMEOUT_TYPE_MARKERS = ("apitimeouterror", "timeouterror", "readtimeout")
_TIMEOUT_MESSAGE_MARKERS = ("timed out", "timeout")


def _walk(exc: BaseException):
    """遍历异常及其 ``__cause__`` / ``__context__`` 链（去重防环）。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def classify_upstream_error_type(exc: BaseException) -> str:
    """返回错误类型字符串。见模块文档的类型表。

    注意判定顺序：认证 → 无效请求 → 超时 → 连接。把「无效请求」放在「连接」之前，
    是因为上游 400 的消息里可能同时出现 "400" 之类命中连接族关键词的内容。
    """
    for current in _walk(exc):
        name = type(current).__name__.lower()
        message = str(current).lower()

        if any(marker in name for marker in _AUTH_TYPE_MARKERS) or any(
            marker in message for marker in _AUTH_MESSAGE_MARKERS
        ):
            return "upstream_auth"

        if any(marker in name for marker in _INVALID_REQUEST_TYPE_MARKERS) or any(
            marker in message for marker in _INVALID_REQUEST_MESSAGE_MARKERS
        ):
            return "upstream_invalid_request"

        if any(marker in name for marker in _TIMEOUT_TYPE_MARKERS) or any(
            marker in message for marker in _TIMEOUT_MESSAGE_MARKERS
        ):
            return "upstream_timeout"

        if any(marker in name for marker in _CONNECTION_TYPE_MARKERS) or any(
            marker in message for marker in _CONNECTION_MESSAGE_MARKERS
        ):
            return "upstream_unavailable"

    return "upstream_error"


def is_infrastructure_error(exc: BaseException) -> bool:
    """这个异常是不是「上游/网络/配置」这类问题（而非业务解析失败）。

    判定标准：能被归成某个具体的 upstream_* 类型。只有兜底的
    ``upstream_error`` 才交给 ``ExtractionError`` 去当解析失败处理。
    """
    return classify_upstream_error_type(exc) != "upstream_error"


def upstream_error_message(exc: BaseException, type_: str) -> str:
    """按类型给出给使用者看的说明，附上原始错误（**已脱敏**）便于排查。"""
    text = redact(f"{type(exc).__name__}: {exc}")
    if type_ == "upstream_auth":
        return (
            "上游拒绝了本次请求，请检查 .env 里的 OPENAI_API_KEY / OPENAI_BASE_URL。"
            f"原始错误：{text}"
        )
    if type_ == "upstream_invalid_request":
        return (
            "上游认为请求本身不合法（通常是 .env 里的 OPENAI_MODEL 名字不对，"
            f"或该模型不支持当前调用方式）。原始错误：{text}"
        )
    if type_ == "upstream_timeout":
        return f"上游响应超时，请稍后重试。原始错误：{text}"
    if type_ == "upstream_unavailable":
        return f"上游不可用（连不上或域名解析失败），请检查 OPENAI_BASE_URL 与网络。原始错误：{text}"
    return text
