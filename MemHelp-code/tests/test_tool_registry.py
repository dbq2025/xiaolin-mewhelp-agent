"""工具注册表的测试。

## 为什么注册表是「按请求构造」而不是模块级常量

``query_faq`` 与 ``create_ticket`` 需要注入仓库，而仓库是**每请求**一个
（每个请求一个 DB Session）。但工具的 JSON Schema 里**不能**出现 ``faq_repo``
这种参数 —— 模型会试图给它编一个值。

所以依赖用**闭包**绑定，注册表暴露 ``get_tools(repositories)`` 这个工厂。
这样：

- 模型只看见 ``keyword``、``ticket_type`` 这些真正的业务参数；
- 每次请求拿到的是绑定了当次仓库的工具实例，不会跨请求串数据。

## 这里钉住什么

名字是字符串，而字符串是**静默失联**的温床：SSE 帧、落库、评估集都用工具名。
改个名不会报错，只会让徽章显示不出、评估集全红。所以注册表的键必须与
``.name`` 逐一对应，且五个工具一个不多一个不少。
"""

import pytest

EXPECTED_TOOL_NAMES = {
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
}


@pytest.fixture
def repos(sqlite_session):
    """一份接到 SQLite 内存库上的真仓库（不是假仓库 —— 注册表要真的能绑）。"""
    from app.db.repositories import Repositories

    return Repositories.from_session(sqlite_session)


def test_registry_covers_exactly_the_five_business_tools(repos):
    from app.tools.registry import TOOL_NAMES

    assert set(TOOL_NAMES) == EXPECTED_TOOL_NAMES


def test_registry_keys_match_tool_names(repos):
    """注册表的键必须与每个工具的 ``.name`` 相同。

    不同的话会出现「按名字取到了工具、但工具自报另一个名字」，
    于是落库的 ``tool_calls.name`` 与徽章显示的、与评估集断言的三者错位。
    """
    from app.tools.registry import get_tools

    for tool in get_tools(repos):
        assert tool.name in EXPECTED_TOOL_NAMES


def test_every_tool_has_a_pydantic_args_schema(repos):
    """参数 Schema 是发给模型的工具定义，缺了它模型不知道要传什么。"""
    from pydantic import BaseModel

    from app.tools.registry import get_tools

    for tool in get_tools(repos):
        assert isinstance(tool.args_schema, type)
        assert issubclass(tool.args_schema, BaseModel)


def test_every_tool_has_a_description_that_mentions_its_purpose(repos):
    """``description`` 是给模型看的「什么时候该调我」。

    它直接决定验收标准 1 与 2 的成败（模型选不选对工具），
    所以不能是空串，也不能只有工具名。
    """
    from app.tools.registry import get_tools

    for tool in get_tools(repos):
        assert tool.description.strip()
        assert len(tool.description) > 20, f"{tool.name} 的描述太短，模型无从判断"


def test_get_tool_returns_none_for_unknown_name(repos):
    """未知工具名要能表达「没有」，由执行器转成结构化失败。"""
    from app.tools.registry import get_tool

    assert get_tool("query_nothing", repos) is None


def test_get_tool_finds_each_registered_tool(repos):
    from app.tools.registry import TOOL_NAMES, get_tool

    for name in TOOL_NAMES:
        found = get_tool(name, repos)
        assert found is not None
        assert found.name == name


def test_every_tool_has_a_chinese_label_for_the_status_frame(repos):
    """SSE 的 ``status`` 帧要在工具执行**前**就告诉用户「正在查什么」。

    所以标签必须是给人看的中文短语，且以省略号结尾（表示进行中）。
    """
    from app.tools.registry import TOOL_LABELS

    assert set(TOOL_LABELS) == EXPECTED_TOOL_NAMES
    for name, label in TOOL_LABELS.items():
        assert label.endswith("…"), name
        assert any("\u4e00" <= ch <= "\u9fff" for ch in label), f"{name} 的标签不是中文"


def test_tool_schemas_do_not_leak_infrastructure_parameters(repos):
    """**关键约束**：模型的工具定义里不能出现仓库、session 之类的参数。

    泄漏的话模型会给它们编值（它不知道那是什么），而执行器会拿这个编造的值去
    覆盖真实依赖 —— 表现为「工具总是查不到数据」，且很难归因。
    """
    from app.tools.registry import get_tools

    forbidden = {"faq_repo", "ticket_repo", "conversation_repo", "session", "repositories"}
    for tool in get_tools(repos):
        params = set(tool.args_schema.model_fields)
        assert not (params & forbidden), f"{tool.name} 泄漏了基础设施参数 {params & forbidden}"


def test_create_ticket_schema_asks_the_model_for_the_session_id(repos):
    """会话 id 是**业务参数**（模型必须给），但它得来自系统提示，不是仓库注入。

    这条和上一条是一对：仓库不能出现，会话标识必须出现。
    """
    from app.tools.registry import get_tool

    tool = get_tool("create_ticket", repos)
    assert tool is not None
    assert "conversation_id" in tool.args_schema.model_fields


def test_two_requests_get_independent_tool_instances(sqlite_session):
    """两个请求的工具必须绑各自的仓库，不能共享。

    共享的话，请求 A 的消息会写进请求 B 的会话 —— 表现为「聊天记录串台」，
    而且是间歇性的（取决于并发时序）。
    """
    from app.db.repositories import Repositories
    from app.tools.registry import get_tools

    repos_a = Repositories.from_session(sqlite_session)
    repos_b = Repositories.from_session(sqlite_session)

    tool_a = next(t for t in get_tools(repos_a) if t.name == "query_faq")
    tool_b = next(t for t in get_tools(repos_b) if t.name == "query_faq")

    assert tool_a is not tool_b


def test_business_tool_functions_remain_directly_callable():
    """``@tool`` 包装的是**副本**，原始纯函数必须还能直接调用。

    理由：``tests/test_tool_business.py`` 与将来可能的批处理脚本都直接调它们，
    不必构造 LangChain 的调用信封。所以包装要发生在注册表里，
    而不是用装饰器改写 ``business.py`` 里的函数本身。
    """
    import json

    from app.tools.business import query_order

    assert json.loads(query_order("1001"))["order_id"] == "1001"
