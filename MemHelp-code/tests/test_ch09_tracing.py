"""ch09 观测层：编译期回调、一轮一根 span、意图 tag、降级为 no-op。"""

from __future__ import annotations

from contextlib import contextmanager

from app.observability import tracing


@contextmanager
def _null_attributes(**_kwargs):
    yield


class _FakeObservation:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.observation = _FakeObservation()
        self.calls: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        yield self.observation


def test_tracing_disabled_is_noop(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_handler", None)
    with tracing.turn_span("s1", "你好") as span:
        span.finish({"stop_reason": "chitchat"})
    with tracing.observation("retrieval", as_type="retriever"):
        pass
    assert tracing.callback_handler() is None


def test_graph_attaches_callback(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_handler", "HANDLER")
    from app.agent.graph import build_agent_graph

    graph = build_agent_graph(checkpointer=None)
    assert graph.config["callbacks"] == ["HANDLER"]
    assert hasattr(graph, "aget_state")
    assert graph.checkpointer is not None


def test_turn_span_finishes_with_payload(monkeypatch):
    tracing.reset_for_tests()
    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_client", fake)
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(tracing, "_attributes", _null_attributes)

    with tracing.turn_span("sess-9", "退款多久到账？") as span:
        span.finish(
            {
                "intent": "refund",
                "route": "refund",
                "gate": {"passed": True, "score": 0.8},
                "token_usage": {"total_tokens": 12},
                "stop_reason": "answered",
                "message_id": 5,
            }
        )

    assert fake.calls[0]["name"] == "chat-turn"
    update = fake.observation.updates[-1]
    assert update["metadata"]["intent"] == "refund"
    assert update["metadata"]["token_usage"] == {"total_tokens": 12}
    assert update["output"]["message_id"] == 5


def test_tag_intent_is_noop_when_disabled(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_enabled", False)
    with tracing.tag_intent("refund", 0.9):
        pass


class _BrokenClient:
    def start_as_current_observation(self, **_kwargs):
        raise RuntimeError("langfuse 内部错误")


def test_observation_failure_does_not_break_chat(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.setattr(tracing, "_client", _BrokenClient())
    monkeypatch.setattr(tracing, "_enabled", True)

    with tracing.observation("knowledge_retrieve", as_type="retriever") as span:
        assert span is None
