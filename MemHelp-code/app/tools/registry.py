"""五个工具的注册表 —— 名字（字符串）与工具对象的唯一对应表。

## 为什么是工厂而不是模块级常量

``query_faq`` 与 ``create_ticket`` 需要注入仓库，而仓库是**每请求**一个
（每个请求一个 DB Session）。但工具的 JSON Schema 就是**发给模型**的工具定义，
里面不能出现 ``faq_repo`` 这种参数 —— 模型不知道那是什么，会给它编一个值，
于是真实依赖被编造的值覆盖，表现为「工具总是查不到数据」。

所以依赖用**闭包**在构造时绑定，注册表暴露工厂 ``get_tools(repositories)``：

- 模型只看见 ``keyword``、``order_id``、``ticket_type`` 这些业务参数；
- 每次请求拿到绑定了当次仓库的工具实例，不跨请求串数据。

## ``description`` 是写给模型看的提示词

每个工具的 ``description`` 直接决定验收标准 1 / 2 的成败：模型靠它判断
「用户这句话该不该调我、调哪一个」。所以它必须写清楚**什么时候用**，
而不只是复述工具名。改这些文案要重跑 ``evals/run_tool_evals.py``。

## 原始纯函数保持可直接调用

``@tool`` 包装的是 ``app/tools/business.py`` 里那些函数的**副本** ——
``business.py`` 本身不 import langchain。这让「不构造 LangChain 信封也能单测
这三个工具」成立，也避免工具层被框架渗进来。
"""

from langchain_core.tools import BaseTool, tool

from app.db.repositories import Repositories
from app.tools.business import (
    query_logistics as _query_logistics,
    query_order as _query_order,
    query_product as _query_product,
)
from app.tools.knowledge import AnswerContextLike
from app.tools.knowledge import query_faq as _query_faq
from app.tools.schemas import (
    CreateTicketArgs,
    QueryFaqArgs,
    QueryLogisticsArgs,
    QueryOrderArgs,
    QueryProductArgs,
)
from app.tools.ticketing import create_ticket as _create_ticket

#: SSE 的 ``status`` 帧用的中文短语。
#:
#: 它在工具**执行之前**就发给前端，让用户立刻看到「在查数据」而不是干等。
#: 以省略号结尾表示进行中。键必须与工具名逐一对应。
TOOL_LABELS: dict[str, str] = {
    "query_order": "正在查询订单…",
    "query_product": "正在查询商品…",
    "query_logistics": "正在查询物流…",
    "query_faq": "正在检索常见问题…",
    "create_ticket": "正在创建工单…",
}


def _make_tools(repositories: Repositories, retriever: AnswerContextLike | None) -> list[BaseTool]:
    """按当次请求的仓库构造五个绑好依赖的工具。

    两个需要仓库的工具用**闭包**捕获 ``repositories``，``query_faq`` 额外捕获
    ``retriever``（向量检索器），并把真正的业务参数显式列在签名里 —— 签名即给模型的 Schema。
    """

    @tool("query_order", args_schema=QueryOrderArgs)
    def query_order(order_id: str) -> str:
        """按订单号查询订单状态、金额与下单时间。

        当用户提到**具体的订单号**、问某笔订单怎么样时用这个。
        只问物流轨迹请改用 query_logistics。
        """
        return _query_order(order_id)

    @tool("query_product", args_schema=QueryProductArgs)
    def query_product(product_id: str) -> str:
        """按商品编号查询商品名称、价格、库存与分类。

        当用户提到**具体的商品编号**、问某个商品多少钱或还有没有货时用这个。
        """
        return _query_product(product_id)

    @tool("query_logistics", args_schema=QueryLogisticsArgs)
    def query_logistics(order_id: str) -> str:
        """按订单号查询物流轨迹：承运商、运单号、当前状态、途经节点、预计到达。

        当用户问「到哪了」「什么时候到」「发货了没」「物流不动了」时用这个。
        用户只给订单号而问物流，就调这个，不要先调 query_order。
        """
        return _query_logistics(order_id)

    @tool("query_faq", args_schema=QueryFaqArgs)
    async def query_faq(keyword: str) -> str:
        """检索平台的常见问题、规则库与商品型号参数，返回**带编号的依据**。

        用于**平台规则类**问题（退货政策、退款流程、运费与包邮门槛、发票开具、
        质保与赔付标准）与**具体商品型号**问题（某个型号的参数、配件、价格）。
        参数请传**用户问题的完整原话**（如「邮费是多少」），不要截成名词 ——
        检索靠整句语义匹配，换一种说法也能召回。

        返回里每个 item 带 ``n`` 与 ``section_path``：**正文引用时用 [n] 标注**。
        若 ``refused`` 为 true，说明知识库里没有能回答这个问题的依据，请如实拒答，
        不要编造规则，也不要引用不存在的编号。
        """
        # 闭包捕获当次请求的检索器 —— 它不是给模型的参数，所以不出现在签名里。
        #
        # **必须 async + await**：底层 ``_query_faq`` 是协程函数（它要 await 检索）。
        # 写成同步函数再 ``return _query_faq(...)`` 会返回一个 **coroutine 对象**，
        # LangChain 把它当成工具结果原样回灌给模型 —— 表现为模型复述
        # 「我帮你正在检索常见问题了：<coroutine object query_faq at 0x...>」。
        # 这个 bug 单测测不出来（假执行器不经过真工具），只有端到端能发现。
        if retriever is None:
            # 宁可失败得明明白白，也不要静默返回空结果 —— 那会把「装配漏了」
            # 伪装成「库里没有这条规则」。
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

        **用户明确要求「建工单 / 提交投诉 / 转人工 / 申请售后」时，直接调本工具。**
        那是要**执行的动作**，不要先去查订单状态 —— 用户没有问订单状态，
        先查订单会导致动作没被执行，用户以为你已经受理了。

        已有订单号就写进 description，让接手的人不用回头问。

        也用于：问题需要人工介入（赔付、质量索赔、反复沟通未解决）。

        天气、闲聊、能直接答的规则问题**不要**建工单。

        conversation_id 必须原样使用系统提示中给出的会话 ID。
        idempotency_key 由前端「建工单」按钮生成；同一个键重复提交返回同一张工单。
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

    return [query_order, query_product, query_logistics, query_faq, create_ticket]


#: 工具名的权威顺序（也是它们提供给模型的顺序）。
TOOL_NAMES: tuple[str, ...] = (
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
)


def get_tools(
    repositories: Repositories, retriever: AnswerContextLike | None = None
) -> list[BaseTool]:
    """构造绑定到当次请求仓库（与检索器）的五个工具，供 ``model.bind_tools(...)``。

    :param retriever: 向量检索器。``query_faq`` 必需；为 ``None`` 时该工具会在
        **被调用时**抛一条可读的错误（由 executor 分类成 ``ok=false``），
        而不是静默返回空结果。
    """
    tools = _make_tools(repositories, retriever)
    # 自检：注册表键、工具自报名、标签三者必须一致。三者错位是**静默**故障
    # （SSE 徽章显示不出、评估集全红），所以在构造时就撞一下，而不是等线上。
    names = [t.name for t in tools]
    if set(names) != set(TOOL_NAMES):
        raise RuntimeError(f"工具注册表与实现不一致：注册 {TOOL_NAMES}，实际 {tuple(names)}")
    if set(TOOL_LABELS) != set(TOOL_NAMES):
        raise RuntimeError(
            f"TOOL_LABELS 与注册表不一致：标签 {tuple(TOOL_LABELS)}，注册 {TOOL_NAMES}"
        )
    return tools


def get_tool(
    name: str, repositories: Repositories, retriever: AnswerContextLike | None = None
) -> BaseTool | None:
    """按名字取一个工具；未知名字返回 ``None``。

    返回 ``None`` 而不是抛异常：未知工具名是**模型的输入**（它可能编一个名字），
    属于要转成结构化失败并回灌的正常情况，不是程序缺陷。
    """
    for candidate in get_tools(repositories, retriever):
        if candidate.name == name:
            return candidate
    return None
