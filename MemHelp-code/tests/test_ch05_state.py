"""ch05 Task 3：State 契约与七类意图 → 四出口的写死映射。"""

from app.agent.state import ALL_INTENTS, INTENT_TO_ROUTE, AgentState, route_for


def test_seven_intents_map_to_four_exits():
    assert len(ALL_INTENTS) == 7
    assert INTENT_TO_ROUTE == {
        "logistics": "business",
        "order": "business",
        "product_consult": "knowledge",
        "refund_return": "knowledge",
        "after_sales": "business",
        "complaint": "complaint",
        "chitchat": "chitchat",
    }


def test_route_for_unknown_falls_back_to_business():
    """认不出来的问题宁可多走一次 Agent，也不要被当成闲聊糊弄掉。"""
    assert route_for("unknown") == "business"
    assert route_for("") == "business"


def test_state_has_the_contract_fields_other_nodes_depend_on():
    fields = set(AgentState.__annotations__)
    for name in (
        "session_id",
        "turn_id",
        "user_message",
        "messages",
        "resolved_message",
        "intent",
        "route",
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
