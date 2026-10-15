"""上游错误的分类 —— 全项目**唯一**判断「这是哪种上游故障」的地方。

之所以抽成独立模块：早先 ``app/extract.py`` 和 ``app/api.py`` 各自判断了一次，
结果同一个「连不上上游」的错误在 ``/api/extract`` 被报成 422
``extraction_failed``、在 ``/api/chat/completions`` 被报成 502
``upstream_unavailable`` —— 同一件事两种说法，调用方没法判断该重试还是查密钥。

分类只看**异常类型名**和**消息**（含 ``__cause__`` 链），不依赖具体 SDK：

- ``upstream_auth``：认证/权限问题 → 使用者该去查 ``.env`` 里的密钥
- ``upstream_unavailable``：连不上 / 域名解析失败 / 超时 → 上游或网络问题，可重试
- ``upstream_timeout``：明确的超时（可重试，与不可用区分开便于告警）
- ``upstream_error``：其它上游异常，兜底
"""

from __future__ import annotations

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
    """返回错误类型字符串。见模块文档的类型表。"""
    for current in _walk(exc):
        name = type(current).__name__.lower()
        message = str(current).lower()

        if any(marker in name for marker in _AUTH_TYPE_MARKERS) or any(
            marker in message for marker in _AUTH_MESSAGE_MARKERS
        ):
            return "upstream_auth"

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
    """这个异常是不是「上游/网络不可用」这类问题（而非业务解析失败）。"""
    return classify_upstream_error_type(exc) != "upstream_error"


def upstream_error_message(exc: BaseException, type_: str) -> str:
    """按类型给出给使用者看的说明，附上原始错误便于排查。"""
    text = f"{type(exc).__name__}: {exc}"
    if type_ == "upstream_auth":
        return (
            "上游拒绝了本次请求，请检查 .env 里的 OPENAI_API_KEY / OPENAI_BASE_URL。"
            f"原始错误：{text}"
        )
    if type_ == "upstream_timeout":
        return f"上游响应超时，请稍后重试。原始错误：{text}"
    if type_ == "upstream_unavailable":
        return f"上游不可用（连不上或域名解析失败），请检查 OPENAI_BASE_URL 与网络。原始错误：{text}"
    return text
