"""编造个案台账的查询 / 处置端点。

只做最小面：列表（按状态筛）+ 改状态（必须写处置说明）。
没有管理页面 —— 页面不在本章前端范围内，但「人工点按钮改」这件事需要接口先存在。

同步测试 + ``anyio.run`` 造数据：``TestClient`` 是同步的，而仓库层是 async。
SQLite 用的是 ``StaticPool``（见 conftest），跨线程共享同一个连接，所以这样混用是安全的。
"""

from __future__ import annotations

import anyio
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(fake_repositories, test_settings):
    return TestClient(main.create_app(settings=test_settings))


def _seed(repositories, eval_id="A01"):
    async def _run():
        await repositories.faith_cases.upsert_from_judgement(
            eval_id=eval_id,
            bucket="A_policy",
            query="你们支持货到付款吗",
            answer="支持的。",
            reason="证据里没有这条",
            strategy="hybrid_rerank",
            citations=[
                {
                    "n": 1,
                    "chunk_id": 3,
                    "section_path": "售后手册 / 赔付标准",
                    "category": "售后手册",
                    "quote": "满 99 元包邮",
                }
            ],
            judge_model="fake-judge",
        )

    anyio.run(_run)


def _set_status(repositories, eval_id, status, resolution=None):
    anyio.run(repositories.faith_cases.update_status, eval_id, status, resolution)


def _pending(client) -> list[dict]:
    return client.get("/api/faith-cases", params={"status": "未解决"}).json()["items"]


def test_list_returns_rows(client, fake_repositories):
    _seed(fake_repositories)

    response = client.get("/api/faith-cases")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["eval_id"] == "A01"
    assert item["status"] == "未解决"
    assert item["seen_count"] == 1
    assert item["citations"][0]["chunk_id"] == 3


def test_list_filters_by_status(client, fake_repositories):
    _seed(fake_repositories, "A01")
    _seed(fake_repositories, "B02")
    _set_status(fake_repositories, "B02", "已解决", "补了条款")

    assert [item["eval_id"] for item in _pending(client)] == ["A01"]
    resolved = client.get("/api/faith-cases", params={"status": "已解决"}).json()["items"]
    assert [item["eval_id"] for item in resolved] == ["B02"]


def test_patch_requires_a_resolution(client, fake_repositories):
    _seed(fake_repositories)

    response = client.patch("/api/faith-cases/A01", json={"status": "已解决", "resolution": ""})

    assert response.status_code == 422


def test_patch_updates_status_and_resolution(client, fake_repositories):
    _seed(fake_repositories)

    response = client.patch(
        "/api/faith-cases/A01",
        json={"status": "已解决", "resolution": "补了货到付款条款"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "已解决"
    assert response.json()["resolution"] == "补了货到付款条款"
    assert _pending(client) == []


def test_patch_can_move_back_to_unresolved(client, fake_repositories):
    _seed(fake_repositories)
    _set_status(fake_repositories, "A01", "已解决", "补了条款")

    response = client.patch("/api/faith-cases/A01", json={"status": "未解决"})

    assert response.status_code == 200
    assert response.json()["status"] == "未解决"
    assert response.json()["resolution"] is None


def test_patch_rejects_unknown_status(client, fake_repositories):
    _seed(fake_repositories)

    response = client.patch(
        "/api/faith-cases/A01", json={"status": "搞定了", "resolution": "x"}
    )

    assert response.status_code == 422


def test_unknown_eval_id_returns_404(client):
    response = client.patch(
        "/api/faith-cases/ZZZ", json={"status": "无需解决", "resolution": "题号写错了"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"
