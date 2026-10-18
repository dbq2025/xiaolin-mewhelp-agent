"""请求 / 响应 / 结构化输出的数据结构。

本模块只有数据定义，不含任何业务逻辑；字段的 ``description`` 会进入
JSON Schema 交给模型，因此它们是提取准确率的一部分，不能省略。
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# 售后信息提取（结构化输出）
# --------------------------------------------------------------------------


class RequestType(str, Enum):
    """用户诉求的类型。"""

    REFUND = "refund"  # 退款
    RETURN = "return"  # 退货
    EXCHANGE = "exchange"  # 换货
    LOGISTICS = "logistics"  # 物流（未发货 / 未送达 / 停滞）
    INVOICE = "invoice"  # 发票
    OTHER = "other"  # 其他或无法归类


class ExpectedSolution(str, Enum):
    """用户期望的解决方案。"""

    REFUND = "refund"  # 退款
    EXCHANGE = "exchange"  # 换货
    RESHIP = "reshipping"  # 补发
    REPAIR = "repair"  # 维修
    COMPENSATION = "compensation"  # 赔偿
    UNKNOWN = "unknown"  # 用户未表达或无法判断


class AfterSalesRequest(BaseModel):
    """从用户的售后描述里提取出来的固定字段。"""

    order_no: str | None = Field(
        default=None,
        description=(
            "订单号。只填用户原文中明确出现的订单号，原样保留（不要改写成别的格式）。"
            "用户没有提到订单号时填 null，禁止推测或编造。"
        ),
    )
    request_type: RequestType = Field(
        description=(
            "诉求类型。可选值："
            "refund=退款，return=退货，exchange=换货，"
            "logistics=物流问题（未发货/未送达/停滞），invoice=发票，"
            "other=其他或无法归类。"
        ),
    )
    expected_solution: ExpectedSolution = Field(
        default=ExpectedSolution.UNKNOWN,
        description=(
            "用户期望的解决方案。可选值："
            "refund=退款，exchange=换货，reshipping=补发，repair=维修，"
            "compensation=赔偿，unknown=用户未表达或无法判断。"
        ),
    )
    detail: str = Field(
        description="用户问题的一句话摘要，保留关键事实（时间、商品、现象），不要加入用户没说过的信息。",
    )


# --------------------------------------------------------------------------
# 对话接口
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """对话请求。"""

    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "会话 ID。只允许字母、数字、下划线和连字符，最长 64 字符；"
            "缺省时由服务端生成并在响应中回传。"
        ),
    )
    message: str = Field(min_length=1, description="用户本轮消息。")
    stream: bool = Field(default=True, description="true 走 SSE 流式，false 返回完整 JSON。")


class SessionId(BaseModel):
    """路径参数里的会话 ID，校验规则与 ``ChatRequest.session_id`` 保持一致。"""

    session_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID。",
    )


class ErrorBody(BaseModel):
    """统一错误响应体。"""

    type: str = Field(description="错误类型，便于程序判断。")
    message: str = Field(description="给人看的错误说明。")
    details: list[dict] | None = Field(default=None, description="校验类错误的明细。")


class ErrorResponse(BaseModel):
    """``{"error": {...}}`` 外壳，所有非 2xx 响应都长这样。"""

    error: ErrorBody = Field(description="错误详情。")


class ContextStats(BaseModel):
    """本轮对话的上下文裁剪统计。"""

    kept_messages: int = Field(default=0, description="保留在上下文里的历史消息条数。")
    trimmed_messages: int = Field(default=0, description="因超出 token 预算被裁掉的历史消息条数。")
    budget_tokens: int = Field(default=0, description="历史消息的 token 预算。")
    used_tokens: int = Field(default=0, description="保留的历史消息实际占用的近似 token 数。")
    system_prompt_tokens: int = Field(default=0, description="System Prompt 的近似 token 数（不计入预算）。")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    session_id: str = Field(description="会话 ID。")
    content: str = Field(description="模型完整回复。")
    context: ContextStats = Field(default_factory=ContextStats, description="本轮上下文裁剪统计。")
    #: ch05：与 SSE `done` 同形的编排摘要。默认值让旧调用方（ch01/ch02 测试）
    #: 不需要改构造参数。
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "本轮工具轨迹。每项含 name / args / ok / duration_ms / attempts / "
            "started_at / result / error，供界面展示工具调用记录。"
        ),
    )
    citations: list[dict[str, Any]] = Field(default_factory=list, description="本轮引用快照。")
    actions: list[dict[str, Any]] = Field(
        default_factory=list, description="本轮建议动作（转人工/建工单）。"
    )
    intent: str | None = Field(default=None, description="本轮意图。")
    route: str | None = Field(default=None, description="本轮出口。")
    resolved_message: str | None = Field(
        default=None, description="补全指代后的完整问题；未改写时等于原话。"
    )
    coref_changed: bool = Field(default=False, description="本轮是否补全了指代/改写。")
    awaiting_input: dict[str, Any] | None = Field(
        default=None,
        description="非流式路径遇到订单选择时的中断载荷；非空表示本轮未完成。",
    )
    agent_steps: int = Field(default=0, description="ReAct 执行了几轮工具。")
    stop_reason: str | None = Field(default=None, description="本轮停止原因。")


class SessionMessagesResponse(BaseModel):
    """某会话已保存的历史消息。

    ``messages`` 每项形如
    ``{"role": "human"|"ai"|"tool", "content": "...", "tools": [...],
      "citations": [...], "actions": [...]}``。

    ``tools`` / ``citations`` / ``actions`` **只在那一轮确实有内容时才出现**：
    省略而不是填 ``null``，是为了让前端只需一处 truthiness 判断。
    """

    session_id: str = Field(description="会话 ID。")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "历史消息。``tools`` 字段（可缺省）是本轮的工具轨迹，"
            "每项含 name / args / ok / duration_ms / attempts / started_at / "
            "result / error（后三项在旧历史里可能缺省）；"
            "``actions`` 是本轮建议动作（转人工 / 建工单）。"
        ),
    )


# --------------------------------------------------------------------------
# ch05：建议动作（建工单端点）
# --------------------------------------------------------------------------


class CreateTicketRequest(BaseModel):
    """用户点「建工单」时的请求。

    ``description`` 可缺省：后端回退到该会话最后一条用户消息。前端只传
    ``session_id`` 也能建单，避免把整段聊天历史回传。
    """

    session_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID，与聊天接口同一个值。",
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="问题描述；缺省时用该会话最后一条用户消息。",
    )
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(
        default="售后",
        description="工单类型，取自 DDL 的三个 ENUM 取值。",
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "前端生成的一次性键；同一个键重复提交返回同一张工单，"
            "避免网络重试建出两张。"
        ),
    )


class CreateTicketResponse(BaseModel):
    """建单结果。"""

    ok: bool = True
    ticket_no: str = Field(description="工单号，形如 T20261018001。")
    ticket_type: str = Field(description="工单类型。")
    status: str = Field(description="工单状态。")
    message: str = Field(description="给用户看的一句话。")


# --------------------------------------------------------------------------
# ch06：订单选择续跑 / 退款单
# --------------------------------------------------------------------------

RefundReason = Literal[
    "商品质量问题",
    "尺寸/颜色/型号不合适",
    "与描述不符",
    "不想要了（七天无理由）",
    "物流破损/少件",
    "其他",
]


class RefundSelectRequest(BaseModel):
    """用户点选订单卡片后的续跑请求。"""

    session_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID，与聊天接口同一个值。",
    )
    order_id: str = Field(
        min_length=4,
        max_length=32,
        pattern=r"^[A-Za-z0-9-]+$",
        description="用户点选的订单号；必须在本次候选列表里。",
    )


class CreateRefundRequest(BaseModel):
    """用户提交退款单。"""

    session_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID。",
    )
    order_id: str = Field(
        min_length=4,
        max_length=32,
        pattern=r"^[A-Za-z0-9-]+$",
        description="退款订单号。",
    )
    reason: RefundReason = Field(description="退款原因，固定六类之一。")
    idempotency_key: str | None = Field(
        default=None,
        max_length=64,
        description="前端生成的一次性键；重复提交返回同一张工单。",
    )


class CreateRefundResponse(BaseModel):
    """退款单提交结果（复用 tickets，以售后工单形式落库）。"""

    ok: bool = True
    refund_no: str = Field(description="退款单号（即工单号）。")
    ticket_no: str = Field(description="工单号。")
    order_id: str = Field(description="订单号。")
    reason: str = Field(description="退款原因。")
    status: str = Field(description="工单状态。")
    message: str = Field(description="给用户看的一句话。")


# --------------------------------------------------------------------------
# 结构化提取接口
# --------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    """售后描述提取请求。"""

    text: str = Field(min_length=1, description="用户的售后描述原文。")


# --------------------------------------------------------------------------
# ch04：编造个案台账
# --------------------------------------------------------------------------


class FaithCaseUpdateRequest(BaseModel):
    """台账处置请求。

    ``status`` 用 ``Literal`` 收窄到 DDL 的三个 ENUM 取值：非法状态在**参数校验**阶段
    就返回 422，而不是等到 flush 时抛 ``LookupError``。
    """

    status: Literal["未解决", "已解决", "无需解决"] = Field(
        description="处置状态。标「已解决/无需解决」必须同时写 resolution；退回「未解决」会清空它。"
    )
    resolution: str | None = Field(
        default=None,
        description="处置说明：标已解决要写清怎么解决的，标无需解决要写清为什么不用改。",
    )
