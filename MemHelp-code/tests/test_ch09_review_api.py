"""ch09 Task 8：审核队列 API、详情与通过后知识库回写。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from tests.fakes import FakeChatModel


class _Embedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _Store:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[Any] = []

    async def upsert(self, records: list[Any]) -> None:
        if self.fail:
            raise RuntimeError("milvus down")
        self.records.extend(records)


def _client(settings: Settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def _patch_vector(monkeypatch, store: _Store, embedder: _Embedder) -> None:
    monkeypatch.setattr("app.api._vector_store_for", lambda *args, **kwargs: store)
    monkeypatch.setattr("app.api.get_embedder", lambda *args, **kwargs: embedder)


async def _seed(repositories: Any, *, snapshot: list[dict[str, Any]] | None = None):
    review_id = await repositories.reviews.add("退款多久到账？", "1-3 个工作日")
    question_id = await repositories.low_confidence.add(
        None,
        "钱啥时候回来",
        "user_feedback",
        retrieved_chunks=snapshot
        if snapshot is not None
        else [
            {
                "n": 1,
                "chunk_id": "7",
                "section_path": "售后政策 / 退款",
                "category": "售后政策",
                "quote": "退款将在 1-3 个工作日内原路退回。",
                "score": 0.42,
            }
        ],
    )
    await repositories.low_confidence.attach_review(question_id, review_id)
    return review_id, question_id


async def test_list_and_detail_include_questions_and_snapshot(
    test_settings, fake_repositories, monkeypatch
):
    _patch_vector(monkeypatch, _Store(), _Embedder())
    review_id, _ = await _seed(fake_repositories)
    client = _client(test_settings)

    listed = client.get("/api/review-queue", params={"status": "待审"})
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert [item["id"] for item in items] == [review_id]

    detail = client.get(f"/api/review-queue/{review_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["review"]["normalized_question"] == "退款多久到账？"
    assert payload["questions"][0]["raw_question"] == "钱啥时候回来"
    assert payload["snapshot"][0]["quote"].startswith("退款将在")
    assert payload["snapshot"][0]["score"] == 0.42


async def test_approve_writes_knowledge_and_marks(
    test_settings, fake_repositories, monkeypatch
):
    store, embedder = _Store(), _Embedder()
    _patch_vector(monkeypatch, store, embedder)
    review_id, _ = await _seed(fake_repositories)
    client = _client(test_settings)

    resp = client.post(
        f"/api/review-queue/{review_id}/approve",
        json={"approved_answer": "1-3 个工作日原路退回"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["vectorize_status"] == "ok"
    assert payload["chunk_id"] > 0
    assert store.records and store.records[0].text
    assert embedder.batches

    row = await fake_repositories.reviews.get(review_id)
    assert row.review_status == "通过"
    assert row.approved_answer == "1-3 个工作日原路退回"
    chunks = await fake_repositories.knowledge.list_all()
    assert any("退款多久到账" in chunk.questions for chunk in chunks)


async def test_reject_marks_without_touching_knowledge(
    test_settings, fake_repositories, monkeypatch
):
    _patch_vector(monkeypatch, _Store(), _Embedder())
    review_id, _ = await _seed(fake_repositories)
    client = _client(test_settings)

    resp = client.post(f"/api/review-queue/{review_id}/reject", json={})
    assert resp.status_code == 200
    assert (await fake_repositories.reviews.get(review_id)).review_status == "驳回"
    assert await fake_repositories.knowledge.list_all() == []


async def test_second_approve_conflicts(test_settings, fake_repositories, monkeypatch):
    _patch_vector(monkeypatch, _Store(), _Embedder())
    review_id, _ = await _seed(fake_repositories)
    client = _client(test_settings)

    first = client.post(
        f"/api/review-queue/{review_id}/approve", json={"approved_answer": "可以"}
    )
    second = client.post(
        f"/api/review-queue/{review_id}/approve", json={"approved_answer": "再改"}
    )
    assert first.status_code == 200
    assert second.status_code == 409


async def test_vectorize_failure_keeps_approval_pending(
    test_settings, fake_repositories, monkeypatch
):
    _patch_vector(monkeypatch, _Store(fail=True), _Embedder())
    review_id, _ = await _seed(fake_repositories)
    client = _client(test_settings)

    resp = client.post(
        f"/api/review-queue/{review_id}/approve", json={"approved_answer": "可以"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["vectorize_status"] == "pending"
    assert (await fake_repositories.reviews.get(review_id)).review_status == "通过"


async def test_unknown_review_is_404(test_settings, fake_repositories, monkeypatch):
    _patch_vector(monkeypatch, _Store(), _Embedder())
    client = _client(test_settings)

    assert client.get("/api/review-queue/999").status_code == 404
    assert (
        client.post(
            "/api/review-queue/999/approve", json={"approved_answer": "x"}
        ).status_code
        == 404
    )
    assert client.post("/api/review-queue/999/reject", json={}).status_code == 404
