"""内置工具的注册规格：闭包捕获当轮依赖，导出统一 ``ToolSpec``。

## 为什么依赖用闭包

``query_faq`` 与 ``create_ticket`` 需要注入仓库，而仓库是**每请求**一个。
但工具的 JSON Schema 就是**发给模型**的工具定义，里面不能出现 ``faq_repo``
这种参数 —— 模型不知道那是什么，会给它编一个值。

所以依赖在构造时闭包绑定，模型只看见 ``keyword``、``order_id`` 这些业务参数。

## ``description`` 是写给模型看的提示词

它直接决定模型「该不该调我、调哪一个」。改这些文案要重跑
``evals/run_tool_evals.py``。

## 物流为什么不在内置清单里

ch08 起物流查询由独立的 MCP Server（``mcp_servers/logistics_server.py``，
工具名 ``track_shipment``）接管。内置的 ``query_logistics`` 已下线，
避免同一能力出现两个名字、模型二选一。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.db.repositories import Repositories
from app.tools.business import query_order as _query_order
from app.tools.business import query_product as _query_product
from app.tools.contract import ToolSpec
from app.tools.knowledge import AnswerContextLike, query_faq as _query_faq
from app.tools.schemas import (
    CreateTicketArgs,
    QueryFaqArgs,
    QueryOrderArgs,
    QueryProductArgs,
)
from app.tools.ticketing import create_ticket as _create_ticket

#: 内置工具名（不含已移交 MCP 的物流）。
BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "query_order",
    "query_product",
    "query_faq",
    "create_ticket",
)


def build_builtin_tools(
    repositories: Repositories, retriever: AnswerContextLike | None = None
) -> list[ToolSpec]:
    """构造绑定到当轮仓库（与检索器）的内置工具规格。"""

    @tool("query_order", args_schema=QueryOrderArgs)
    def query_order(order_id: str) -> str:
        """按订单号查询订单状态、金额与下单时间。

        当用户提到**具体的订单号**、问某笔订单怎么样时用这个。
        只问物流轨迹请改用 track_shipment（物流由 MCP Server 提供）。
        """
        return _query_order(order_id)

    @tool("query_product", args_schema=QueryProductArgs)
    def query_product(product_id: str) -> str:
        """按商品编号查询商品名称、价格、库存与分类。

        当用户提到**具体的商品编号**、问某个商品多少钱或还有没有货时用这个。
        """
        return _query_product(product_id)

    @tool("query_faq", args_schema=QueryFaqArgs)
    async def query_faq(keyword: str) -> str:
        """检索平台的常见问题、规则库与商品型号参数，返回**带编号的依据**。

        用于**平台规则类**问题（退货政策、退款流程、运费与包邮门槛、发票开具、
        质保与赔付标准）与**具体商品型号**问题（某个型号的参数、配件、价格）。
        参数请传**用户问题的完整原话**（如「邮费是多少」），不要截成名词 ——
        检索靠整句语义匹配，换一种说法也能召回。

        返回里每个 item 带 ``n`` 与 ``section_path``：**正文引用时用 [n] 标注**。
        若 ``refused`` 为 true，说明知识库里没有能回答这个问题的依据，请如实拒答。
        """
        # 闭包捕获当次请求的检索器 —— 它不是给模型的参数，所以不出现在签名里。
        # 必须 async + await：底层是协程函数，同步返回会把 coroutine 对象回灌给模型。
        if retriever is None:
            raise RuntimeError(
                "query_faq 未装配检索器（retriever）。"
                "请检查 create_app 的装配或测试注入。"
            )
        return await _query_faq(keyword, retriever=retriever)

    @tool("create_ticket", args_schema=CreateTicketArgs)
    async def create_ticket(
        description: str,
        ticket_type: str,
        conversation_id: str,
        idempotency_key: str | None = None,
    ) -> str:
        """创建人工工单，转交人工客服跟进。

        **只有用户明确要求「建工单 / 提交投诉 / 转人工 / 申请售后」时才调本工具。**
        调用后系统会把工单预览交给用户确认，用户点「确认提交」之前**不会真的建单**；
        所以缺问题描述时要先追问，不要自己编。

        已有订单号就写进 description，让接手的人不用回头问。

        天气、闲聊、能直接答的规则问题**不要**建工单。

        conversation_id 必须原样使用系统提示中给出的会话 ID。
        """
        # 同上：底层 ``_create_ticket`` 是协程函数，必须 await。
        return await _create_ticket(
            description,
            ticket_type,
            conversation_id,
            idempotency_key=idempotency_key,
            ticket_repo=repositories.tickets,
            conversation_repo=repositories.conversations,
        )

    specs = [
        ToolSpec(
            name="query_order",
            description=query_order.description,
            args_schema=QueryOrderArgs.model_json_schema(),
            source="builtin",
            mcp_server=None,
            read_only=True,
            tool=query_order,
        ),
        ToolSpec(
            name="query_product",
            description=query_product.description,
            args_schema=QueryProductArgs.model_json_schema(),
            source="builtin",
            mcp_server=None,
            read_only=True,
            tool=query_product,
        ),
        ToolSpec(
            name="query_faq",
            description=query_faq.description,
            args_schema=QueryFaqArgs.model_json_schema(),
            source="builtin",
            mcp_server=None,
            read_only=True,
            tool=query_faq,
        ),
        ToolSpec(
            name="create_ticket",
            description=create_ticket.description,
            args_schema=CreateTicketArgs.model_json_schema(),
            source="builtin",
            mcp_server=None,
            read_only=False,
            tool=create_ticket,
        ),
    ]
    assert tuple(spec.name for spec in specs) == BUILTIN_TOOL_NAMES
    return specs
