"""ch06 Task 2：State 契约与八类意图 → 五出口的写死映射。

文件名保留 ch05，避免无谓移动；断言已按 ch06 契约更新。
"""

from app.agent.state import ALL_INTENTS, INTENT_TO_ROUTE, AgentState, route_for


def test_eight_intents_map_to_five_exits():
    assert len(ALL_INTENTS) == 8
    assert INTENT_TO_ROUTE == {
        "logistics": "business",
        "order": "business",
        "product_consult": "knowledge",
        "refund_return": "refund",
        "after_sales": "refund",
        "complaint": "complaint",
        "chitchat": "chitchat",
        "other": "business",
    }


def test_route_for_unknown_falls_back_to_business():
    """认不出来的问题宁可多走一次 Agent，也不要被当成闲聊糊弄掉。"""
    assert route_for("unknown") == "business"
    assert route_for("") == "business"
    assert route_for("other") == "business"


def test_state_has_the_contract_fields_other_nodes_depend_on():
    fields = set(AgentState.__annotations__)
    for name in (
        "session_id",
        "turn_id",
        "user_message",
        "messages",
        "resolved_message",
        "coref_changed",
        "coref_source",
        "coref_llm_calls",
        "intent",
        "intent_llm_calls",
        "route",
        "refund_order_id",
        "refund_order_candidates",
        "refund_order",
        "refund_order_error",
        "expanded_queries",
        "query_expansion_degraded",
        "refund_form",
        "retrieval_attempted",
        "retrieval_score",
        "evidence",
        "gate_passed",
        "answer",
        "tool_traces",
        "tool_records",
        "pending_messages",
        "citations",
        "suggested_actions",
        "ticket_type",
        "agent_steps",
        "token_usage",
        "stop_reason",
        "trace",
        "turn_log",
    ):
        assert name in fields, name
