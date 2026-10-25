"""ch10 Task 14：主应用侧两个接口的契约。

不测模型、也不起 sidecar：把 ``httpx.AsyncClient.post`` 换掉，验的是
「HTTP 形状、错误映射、以及批跑真的落了库」。
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from app.db.models import LowConfidenceQuestion, TopicClassification
from tests.fakes import FakeChatModel


def _client(settings: Settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def _seed_pending(session, n: int = 2) -> None:
    session.add_all(
        [
            LowConfidenceQuestion(raw_question=f"问题{i}", source="retrieval_low_conf")
            for i in range(n)
        ]
    )
    session.commit()


def _ok_results(texts, labels=("物流",)):
    return {
        "results": [
            {"labels": list(labels), "scores": {}, "fallback": False} for _ in texts
        ]
    }


def test_distribution_shape_on_empty_db(test_settings, sqlite_session):
    with _client(test_settings) as client:
        r = client.get("/api/topics/distribution")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "total",
        "by_label",
        "all_labels",
        "multi_label_share",
        "pending",
        "min_pending",
        "last_classified_at",
    }
    assert len(body["all_labels"]) == 17 and body["all_labels"][0] == "退换货"
    assert body["total"] == 0 and body["by_label"] == [] and body["pending"] == 0
    assert body["multi_label_share"] == 0.0
    assert body["min_pending"] == test_settings.topic_batch_min_pending


def test_distribution_reports_labels_and_pending(test_settings, sqlite_session):
    _seed_pending(sqlite_session, 3)
    sqlite_session.add_all(
        [
            TopicClassification(question_id=1, labels=["物流"]),
            TopicClassification(question_id=2, labels=["退换货", "尺码"]),
        ]
    )
    sqlite_session.commit()
    with _client(test_settings) as client:
        body = client.get("/api/topics/distribution").json()
    assert body["total"] == 2 and body["pending"] == 1
    assert {row["label"] for row in body["by_label"]} == {"物流", "退换货", "尺码"}
    assert body["multi_label_share"] == 0.5
    assert body["last_classified_at"] is not None


def test_classify_batch_writes_rows(test_settings, sqlite_session, monkeypatch):
    _seed_pending(sqlite_session, 2)

    async def fake_post(self, url, **kwargs):  # noqa: ANN001, ARG001
        texts = kwargs["json"]["texts"]
        return httpx.Response(200, json=_ok_results(texts))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with _client(test_settings) as client:
        r = client.post("/api/topics/classify-batch", json={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["processed"] == 2 and body["pending_after"] == 0
    labels = [tuple(row.labels) for row in sqlite_session.query(TopicClassification).all()]
    assert labels == [("物流",), ("物流",)]


def test_classify_batch_returns_503_with_command_when_sidecar_down(
    test_settings, sqlite_session, monkeypatch
):
    _seed_pending(sqlite_session, 1)

    async def refuse(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", refuse)
    with _client(test_settings) as client:
        r = client.post("/api/topics/classify-batch", json={"limit": 10})
    assert r.status_code == 503
    assert "uvicorn app.topics.service:app --port 8100" in r.json()["error"]["message"]
    # 失败不该写进任何一行
    assert sqlite_session.query(TopicClassification).count() == 0


def test_batch_rejects_negative_limit(test_settings, sqlite_session):
    with _client(test_settings) as client:
        assert client.post("/api/topics/classify-batch", json={"limit": -1}).status_code == 422


def test_batch_uses_chunks_of_at_most_sixty_four(test_settings, sqlite_session, monkeypatch):
    """超过 sidecar 单请求上限时必须切块，而不是发一个 100 条的请求。"""
    _seed_pending(sqlite_session, 70)
    sizes: list[int] = []

    async def fake_post(self, url, **kwargs):  # noqa: ANN001, ARG001
        texts = kwargs["json"]["texts"]
        sizes.append(len(texts))
        return httpx.Response(200, json=_ok_results(texts))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with _client(test_settings) as client:
        r = client.post("/api/topics/classify-batch", json={"limit": 70})
    assert r.status_code == 200 and r.json()["processed"] == 70
    assert sizes == [64, 6]
