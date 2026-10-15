from langchain_core.messages import AIMessage, HumanMessage

from app.memory import InMemorySessionStore, get_session_store, new_session_id


def test_append_then_get_preserves_order():
    store = InMemorySessionStore()
    store.append("s1", [HumanMessage("一"), AIMessage("答一")])
    store.append("s1", [HumanMessage("二")])
    assert [m.content for m in store.get("s1")] == ["一", "答一", "二"]


def test_sessions_are_isolated():
    """Review Focus：客服绝不能把甲的订单号说给乙。"""
    store = InMemorySessionStore()
    store.append("a", [HumanMessage("甲的问题")])
    store.append("b", [HumanMessage("乙的问题")])
    assert [m.content for m in store.get("a")] == ["甲的问题"]
    assert [m.content for m in store.get("b")] == ["乙的问题"]


def test_get_returns_copy_so_callers_cannot_corrupt_state():
    store = InMemorySessionStore()
    store.append("s1", [HumanMessage("一")])
    got = store.get("s1")
    got.append(HumanMessage("外部偷加的"))
    assert len(store.get("s1")) == 1


def test_get_unknown_session_returns_empty_list():
    assert InMemorySessionStore().get("never") == []


def test_clear_removes_only_that_session():
    store = InMemorySessionStore()
    store.append("a", [HumanMessage("甲")])
    store.append("b", [HumanMessage("乙")])
    store.clear("a")
    assert store.get("a") == []
    assert [m.content for m in store.get("b")] == ["乙"]


def test_new_session_id_is_unique_and_short():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 12 for i in ids)


def test_get_session_store_is_singleton():
    assert get_session_store() is get_session_store()
