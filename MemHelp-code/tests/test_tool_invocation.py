"""**真正调用**注册表里的工具（而不是假执行器）。

## 为什么需要这一层

``tests/test_chat_tools.py`` 用注入的假执行器验证编排逻辑，所以它**从不经过真工具**；
``tests/test_tool_registry.py`` 只检查工具的名字/描述/Schema，也不调用它们。

这中间的空档藏着一个真实缺陷（端到端跑验收时才发现）：

``query_faq`` 与 ``create_ticket`` 底层是**协程函数**（它们要 await 仓库/检索器），
而注册表里最初写成同步函数直接 ``return _query_faq(...)`` —— 于是工具返回的是
一个 **coroutine 对象**。LangChain 把它当结果原样回灌，模型复述出来是：

    我帮你正在检索常见问题了：<coroutine object query_faq at 0x...>

单元测试全绿，因为没人真的执行过这些工具。

本文件专治这一类「包装层忘记 await」的问题：**调用真工具，断言结果是一条字符串**。
"""

import json

import pytest

from app.db.repositories import Repositories
from app.tools.registry import get_tool


@pytest.fixture
def repos(sqlite_session):
    return Repositories.from_session(sqlite_session)


async def test_query_faq_returns_a_string_not_a_coroutine(repos, empty_retriever):
    """工具 ``ainvoke`` 的结果必须是一条可解析的 JSON 字符串。

    返回 coroutine 时本用例会红在 ``json.loads`` 上 —— 而 LangChain 的真实行为
    是把它当工具结果回灌，模型于是复述出 ``<coroutine object ...>``。
    """
    tool = get_tool("query_faq", repos, empty_retriever)
    assert tool is not None

    raw = await tool.ainvoke({"keyword": "退货"})

    assert isinstance(raw, str), f"工具返回了 {type(raw).__name__}，应为 str"
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert "coroutine" not in raw


async def test_query_faq_passes_the_question_to_the_retriever(repos):
    """ch03 起检索走向量库：工具必须把**整句**交给检索器，并把命中映射进出参。"""
    from app.knowledge.retrieval import KnowledgeHit

    from tests.fakes import SearchAsAnswerContext

    class _Retriever(SearchAsAnswerContext):
        def __init__(self):
            self.queries: list[str] = []

        async def search(self, query, top_k=5):
            self.queries.append(query)
            return [KnowledgeHit("运费怎么算", "满 99 包邮", "物流", 0.9)]

    retriever = _Retriever()
    tool = get_tool("query_faq", repos, retriever)
    payload = json.loads(await tool.ainvoke({"keyword": "邮费是多少"}))

    assert retriever.queries == ["邮费是多少"]
    assert payload["count"] == 1
    assert payload["items"][0]["answer"] == "满 99 包邮"


async def test_query_faq_without_a_retriever_fails_loudly(repos):
    """装配漏了检索器时必须**明确报错**，不能静默返回空结果。

    静默失败会把「装配漏了」伪装成「库里没有这条规则」，排查方向完全不同。
    """
    tool = get_tool("query_faq", repos)
    assert tool is not None

    with pytest.raises(Exception, match="retriever"):
        await tool.ainvoke({"keyword": "退货"})


async def test_create_ticket_returns_a_string_and_writes_a_row(repos, sqlite_session):
    """建工单工具也必须真的落到 ``tickets`` 表里。"""
    from app.db.models import Ticket

    tool = get_tool("create_ticket", repos)
    assert tool is not None

    raw = await tool.ainvoke(
        {"description": "杯子碎了", "ticket_type": "售后", "conversation_id": "sess-tool-1"}
    )

    assert isinstance(raw, str), f"工具返回了 {type(raw).__name__}，应为 str"
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["ticket_no"].startswith("T")

    rows = sqlite_session.query(Ticket).all()
    assert len(rows) == 1
    assert rows[0].ticket_no == payload["ticket_no"]
    assert rows[0].ticket_type == "售后"


async def test_create_ticket_creates_the_conversation_row(repos, sqlite_session):
    """工单的外键指向会话 —— 会话不存在时必须被建出来，否则真库上外键失败。"""
    from app.db.models import Conversation

    tool = get_tool("create_ticket", repos)
    await tool.ainvoke(
        {"description": "咨询一下", "ticket_type": "咨询", "conversation_id": "brand-new-sess"}
    )

    rows = sqlite_session.query(Conversation).all()
    assert [r.session_id for r in rows] == ["brand-new-sess"]


async def test_query_order_returns_parsable_json(repos):
    """三个演示工具不连库，但同样要确认包装层没问题。"""
    tool = get_tool("query_order", repos)
    payload = json.loads(await tool.ainvoke({"order_id": "1001"}))

    assert payload["ok"] is True
    assert payload["order_id"] == "1001"


async def test_query_logistics_returns_parsable_json(repos):
    tool = get_tool("query_logistics", repos)
    payload = json.loads(await tool.ainvoke({"order_id": "1001"}))

    assert payload["carrier"]
    assert payload["traces"]


async def test_query_product_returns_parsable_json(repos):
    tool = get_tool("query_product", repos)
    payload = json.loads(await tool.ainvoke({"product_id": "P2001"}))

    assert payload["ok"] is True
    assert payload["name"]


@pytest.mark.parametrize(
    "name,args",
    [
        ("query_order", {"order_id": ""}),
        ("query_faq", {"keyword": "   "}),
        ("create_ticket", {"description": "  ", "ticket_type": "售后", "conversation_id": "s"}),
    ],
)
async def test_tool_arguments_are_validated_by_the_real_tool(repos, name, args):
    """真工具的 Schema 校验也要真的生效（空白参数必须被拒）。"""
    from pydantic import ValidationError

    tool = get_tool(name, repos)
    with pytest.raises(ValidationError):
        await tool.ainvoke(args)


async def test_every_registered_tool_returns_a_string(repos, empty_retriever):
    """把五个工具都真调一遍，逐个断言返回类型。

    这条是**兜底**：以后有人再加工具、又在包装层忘了 await，这里会红。
    """
    cases = [
        ("query_order", {"order_id": "1001"}),
        ("query_product", {"product_id": "P1"}),
        ("query_logistics", {"order_id": "1001"}),
        ("query_faq", {"keyword": "测试"}),
        ("create_ticket", {"description": "描述", "ticket_type": "咨询", "conversation_id": "s1"}),
    ]

    for name, args in cases:
        tool = get_tool(name, repos, empty_retriever)
        assert tool is not None, name
        raw = await tool.ainvoke(args)
        assert isinstance(raw, str), f"{name} 返回了 {type(raw).__name__}，应为 str"
        json.loads(raw)  # 不可解析会抛
