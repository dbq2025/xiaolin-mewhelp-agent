"""五个工具的**参数 Schema**。

这些 Pydantic 模型有双重身份，两个身份都很重要：

1. **发给模型**：``@tool(args_schema=...)`` 会把它们转成 JSON Schema 塞进
   ``tools`` 字段。所以每个字段的 ``description`` 是**写给模型看的提示词**，
   不是写给同事看的注释 —— 描述不清会让模型漏参数或给错类型。
2. **执行前校验**：``executor.py`` 在真正调用工具前再 ``model_validate`` 一次。
   模型偶尔会漏掉必填参数或把数字给成字符串，在它自己的调用上失败并拿到
   可读的错误，比让 ``KeyError`` 冒到 SSE 连接上要好得多。

字段名刻意用英文（``order_id``、``keyword``）：模型对英文参数名的工具调用
稳定性更好，而且中文参数名在部分网关的 JSON Schema 处理里会出问题。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CreateTicketArgs",
    "QueryFaqArgs",
    "QueryOrderArgs",
    "QueryProductArgs",
]


def _reject_blank(value: str) -> str:
    """去掉首尾空白后拒绝空串。

    空 id 若放过去，工具会返回一份「订单 的物流到哪了」这种荒谬数据，
    模型再一本正经地转述 —— 用户看到的是幻觉，而不是一句「请提供订单号」。
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("不能是空白字符串，请提供具体的编号或关键词")
    return stripped


class _Args(BaseModel):
    """参数模型的共同配置。"""

    # 拒绝多余字段：模型偶尔会自作主张加参数（比如 ``{"order_id": "1001", "user": "x"}``），
    # 静默忽略会让它以为参数生效了。报错能让它在下一轮修正，也让日志里看得见原因。
    model_config = ConfigDict(extra="forbid")


class QueryOrderArgs(_Args):
    """``query_order`` 的参数。"""

    order_id: str = Field(
        ..., description="订单号，纯数字或带字母前缀均可，例如 1001。"
    )

    _normalise = field_validator("order_id")(_reject_blank)


class QueryProductArgs(_Args):
    """``query_product`` 的参数。"""

    product_id: str = Field(..., description="商品编号，例如 P2001。")

    _normalise = field_validator("product_id")(_reject_blank)


class QueryFaqArgs(_Args):
    """``query_faq`` 的参数。

    **刻意不暴露 ``category``**：实测模型会自作主张猜一个品类（把型号题猜成
    「商品参数」），而语料里的品类是「商品型号手册 / 巡检仪」这种真实路径 ——
    猜错就会把检索结果过滤成空，用户得到的是一句「知识库里没有」。
    品类过滤是**程序可达能力**（``query_faq(category=...)`` 与评估脚本都能用），
    但不该交给模型自由填字符串。
    """

    keyword: str = Field(
        ...,
        description=(
            "用户问题的**完整原话**，例如「邮费是多少」「买了七天还能退吗」"
            "「发票怎么开」。向量检索靠整句语义匹配，"
            "**不要**截成名词，也不要带引号。"
        ),
    )

    _normalise = field_validator("keyword")(_reject_blank)


class CreateTicketArgs(_Args):
    """``create_ticket`` 的参数。"""

    description: str = Field(
        ...,
        description="工单要解决的问题描述。写清用户诉求与已知信息，便于人工接手。",
    )
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(
        ...,
        description=(
            "工单类型：商品质量/退换货/退款/物流问题选「售后」；"
            "对服务或商品不满要追责选「投诉」；一般性询问选「咨询」。"
        ),
    )
    conversation_id: str = Field(
        ...,
        description=(
            "当前会话标识。**必须**原样使用系统提示里给你的会话 ID，"
            "不要自己编造，否则人工客服无法倒查聊天记录。"
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "可选的一次性键。用户在前端点「建工单」时由前端生成；"
            "同一个键重复提交应返回同一张工单，避免网络重试建出两张。"
        ),
    )

    _normalise_description = field_validator("description")(_reject_blank)
    _normalise_conversation_id = field_validator("conversation_id")(_reject_blank)
