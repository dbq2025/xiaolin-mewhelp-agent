"""sidecar 的 HTTP 契约。

只测契约（状态码、字段、边界），不测模型 —— 模型用假 classifier 注入。
真正「权重能不能加载」的检查在 ``lifespan`` 里，测不到就真起一次（见计划 Step 5）。
"""

import pytest
from fastapi.testclient import TestClient

from app.topics.service import create_app
from app.topics.taxonomy import CATEGORY_NAMES, taxonomy_hash


def _clf(**kw):
    from app.topics.infer import TopicClassifier

    base = {n: 0.01 for n in CATEGORY_NAMES}
    base.update(kw)
    return TopicClassifier.from_parts(
        lambda t: [[base[n] for n in CATEGORY_NAMES] for _ in t],
        {n: 0.5 for n in CATEGORY_NAMES},
        model_taxonomy_hash=taxonomy_hash(),
    )


def test_health_reports_model_state():
    with TestClient(create_app(classifier=_clf())) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["taxonomy_hash"] == taxonomy_hash()


def test_classify_returns_all_seventeen_scores():
    with TestClient(create_app(classifier=_clf(物流=0.9))) as c:
        r = c.post("/classify", json={"texts": ["快递到哪了"]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["labels"] == ["物流"]
    assert len(res["scores"]) == 17 and res["scores"]["物流"] == pytest.approx(0.9)


def test_empty_texts_is_ok_not_error():
    with TestClient(create_app(classifier=_clf())) as c:
        assert c.post("/classify", json={"texts": []}).json() == {"results": []}


def test_more_than_sixty_four_texts_is_422():
    with TestClient(create_app(classifier=_clf())) as c:
        assert c.post("/classify", json={"texts": ["x"] * 65}).status_code == 422


def test_scores_summary_hides_internal_paths(monkeypatch):
    def boom(texts):
        raise RuntimeError(r"加载失败 C:\Users\dbq\secret\w.bin")

    from app.topics.infer import TopicClassifier

    clf = TopicClassifier.from_parts(
        boom, {n: 0.5 for n in CATEGORY_NAMES}, model_taxonomy_hash=taxonomy_hash()
    )
    with TestClient(create_app(classifier=clf)) as c:
        r = c.post("/classify", json={"texts": ["x"]})
    assert r.status_code == 500 and "secret" not in r.text
