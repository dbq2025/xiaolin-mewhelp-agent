"""工具注册中心的测试。

## 为什么注册中心是「按轮构造」而不是模块级常量

``query_faq`` 与 ``create_ticket`` 需要注入仓库，而仓库是**每次请求**一个
（每个请求一个 DB Session）。但工具的 JSON Schema 里**不能**出现 ``faq_repo``
这种参数 —— 模型会试图给它编一个值。

所以依赖用**闭包**绑定，注册中心每轮构造一次；MCP 工具由进程级 provider 发现、
每轮合并进来。这样：

- 模型只看见 ``keyword``、``ticket_type`` 这些真正的业务参数；
- 每次请求拿到的是绑定了当次仓库的工具实例，不会跨请求串数据。

## 这里钉住什么

名字是字符串，而字符串是**静默失联**的温床：SSE 帧、落库、评估集都用工具名。
改个名不会报错，只会让徽章显示不出、评估集全红。所以注册表的键必须与
``.name`` 逐一对应，且物流已下线 —— 它由 MCP Server 接管，不能留重名。
"""

from __future__ import annotations

import pytest

EXPECTED_TOOL_NAMES = {
    "query_order",
    "query_product",
    "query_faq",
    "create_ticket",
}


@pytest.fixture
def repos(sqlite_session):
    """一份接到 SQLite 内存库上的真仓库（不是假仓库 —— 注册表要真的能绑）。"""
    from app.db.repositories import Repositories

    return Repositories.from_session(sqlite_session)


def test_registry_has_no_logistics_and_keeps_four(repos):
    from app.tools.registry import TOOL_NAMES, build_registry

    reg = build_registry(repos, None)
    names = [spec.name for spec in reg.snapshot()]
    assert "query_logistics" not in names
    assert set(names) == EXPECTED_TOOL_NAMES
    assert "query_logistics" not in TOOL_NAMES


def test_duplicate_name_rejected(repos):
    from app.tools.registry import build_registry

    reg = build_registry(repos, None)
    spec = reg.get("query_order")
    assert spec is not None
    with pytest.raises(ValueError):
        reg.register(spec)


def test_snapshot_is_stable_sorted(repos):
    from app.tools.registry import build_registry

    names = [spec.name for spec in build_registry(repos, None).snapshot()]
    assert names == sorted(names)


def test_get_returns_none_for_unknown_name(repos):
    from app.tools.registry import build_registry

    assert build_registry(repos, None).get("query_nothing") is None


def test_every_spec_carries_the_three_registration_fields(repos):
    """名字、用途描述、JSON Schema —— 三样缺一，模型就选不对或填不对。"""
    from app.tools.registry import build_registry

    for spec in build_registry(repos, None).snapshot():
        assert spec.name
        assert len(spec.description) > 20, f"{spec.name} 的描述太短，模型无从判断"
        assert isinstance(spec.args_schema, dict)
        assert spec.args_schema.get("type") == "object"
        assert "properties" in spec.args_schema
        assert (spec.source == "builtin") is True
        assert spec.mcp_server is None


def test_read_only_flag_marks_create_ticket_as_write(repos):
    from app.tools.registry import build_registry

    reg = build_registry(repos, None)
    assert reg.get("create_ticket").read_only is False  # type: ignore[union-attr]
    assert reg.get("query_order").read_only is True  # type: ignore[union-attr]


def test_tool_schemas_do_not_leak_infrastructure_parameters(repos):
    """**关键约束**：模型的工具定义里不能出现仓库、session 之类的参数。"""
    from app.tools.registry import build_registry

    forbidden = {"faq_repo", "ticket_repo", "conversation_repo", "session", "repositories"}
    for spec in build_registry(repos, None).snapshot():
        params = set(spec.args_schema.get("properties", {}))
        assert not (params & forbidden), f"{spec.name} 泄漏了基础设施参数 {params & forbidden}"


def test_create_ticket_schema_asks_the_model_for_the_session_id(repos):
    from app.tools.registry import build_registry

    spec = build_registry(repos, None).get("create_ticket")
    assert spec is not None
    assert "conversation_id" in spec.args_schema["properties"]


def test_every_tool_has_a_chinese_label_for_the_status_frame(repos):
    from app.tools.registry import TOOL_LABELS, TOOL_NAMES

    assert set(TOOL_LABELS) == set(TOOL_NAMES) == EXPECTED_TOOL_NAMES
    for name, label in TOOL_LABELS.items():
        assert label.endswith("…"), name
        assert any("\u4e00" <= ch <= "\u9fff" for ch in label), f"{name} 的标签不是中文"


def test_agent_tools_keeps_create_ticket_for_gated_flow(repos):
    """create_ticket 留在工具集里，但由 ReAct 拦截、图节点确认后才执行。"""
    from app.agent.tools import agent_tools
    from app.tools.registry import build_registry

    names = {
        tool.name
        for tool in agent_tools(
            build_registry(repos, None),
            include_faq=True,
            record_actions=[],
        )
    }
    assert "create_ticket" in names
    assert "query_logistics" not in names


def test_agent_tools_can_hide_faq_and_order_queries(repos):
    from app.agent.tools import agent_tools
    from app.tools.registry import build_registry

    reg = build_registry(repos, None)
    no_faq = {tool.name for tool in agent_tools(reg, include_faq=False, record_actions=[])}
    assert "query_faq" not in no_faq
    refund = {
        tool.name
        for tool in agent_tools(reg, include_faq=True, include_order=False, record_actions=[])
    }
    assert "query_order" not in refund
    assert "query_product" in refund


def test_two_registries_get_independent_tool_instances(sqlite_session):
    """两个请求的工具必须绑各自的仓库，不能共享。"""
    from app.db.repositories import Repositories
    from app.tools.registry import build_registry

    repos_a = Repositories.from_session(sqlite_session)
    repos_b = Repositories.from_session(sqlite_session)

    spec_a = build_registry(repos_a, None).get("query_faq")
    spec_b = build_registry(repos_b, None).get("query_faq")
    assert spec_a is not None and spec_b is not None
    assert spec_a.tool is not spec_b.tool


def test_business_tool_functions_remain_directly_callable():
    import json

    from app.tools.business import query_order

    assert json.loads(query_order("1001"))["order_id"] == "1001"
