"""ch10 Task 15：主题分布页的冒烟断言。

前端按用户要求走 Vibe Coding（不套 TDD / code review），所以这里只钉一条：
**页面能被主应用挂出来**，且引用的静态资源确实存在。视觉效果靠人眼看。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from tests.fakes import FakeChatModel


def _client(settings: Settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def test_topics_page_is_served(test_settings):
    with _client(test_settings) as client:
        assert client.get("/topics/").status_code == 200
        # 不带斜杠时是 307 → /topics/（TestClient 默认跟随，这里关掉才看得到重定向）
        assert client.get("/topics", follow_redirects=False).status_code == 307
        page = client.get("/topics/").text
        assert "/topics/app.js" in page and "/topics/style.css" in page
        assert client.get("/topics/app.js").status_code == 200
        assert client.get("/topics/style.css").status_code == 200


def test_topics_page_renders_bars_from_api(test_settings, sqlite_session):
    """页面本身不判空，但接口得先能给出 17 类的清单（页面靠它画满 17 行）。"""
    from app.db.models import LowConfidenceQuestion, TopicClassification

    sqlite_session.add(LowConfidenceQuestion(raw_question="买大了想退，运费谁出", source="self_check"))
    sqlite_session.commit()
    sqlite_session.add(TopicClassification(question_id=1, labels=["退换货", "尺码"]))
    sqlite_session.commit()

    with _client(test_settings) as client:
        body = client.get("/api/topics/distribution").json()
    assert body["total"] == 1 and body["multi_label_share"] == 1.0
    assert len(body["all_labels"]) == 17
    counts = {row["label"]: row["count"] for row in body["by_label"]}
    assert counts == {"退换货": 1, "尺码": 1}


def test_admin_acceptance_and_question_pages_are_served(test_settings):
    """后台首页、验收四页、类目问题列表都能挂出来，且引用的样式/脚本可达。"""
    pages = {
        "/admin/": "/acceptance/style.css",
        "/acceptance/": "/acceptance/overview.js",
        "/acceptance/eval/": "/acceptance/eval/eval.js",
        "/acceptance/data/": "/acceptance/data/data.js",
        "/acceptance/errors/": "/acceptance/errors/errors.js",
        "/topics/questions/": "/topics/questions/questions.js",
    }
    with _client(test_settings) as client:
        for page, asset in pages.items():
            response = client.get(page)
            assert response.status_code == 200, page
            assert asset in response.text, page
            assert client.get(asset).status_code == 200, asset


def test_acceptance_api_reads_report_files(test_settings):
    """验收接口读的是仓库里的报告快照，字段足够页面渲染。"""
    with _client(test_settings) as client:
        data = client.get("/api/acceptance/data").json()
        assert len(data["taxonomy"]) == 17
        assert data["splits"]["test"] > 0

        errors = client.get("/api/acceptance/errors").json()
        assert set(errors) >= {"run_id", "count", "rows"}

        summary = client.get("/api/acceptance/summary")
        if summary.status_code == 200:
            body = summary.json()
            assert set(body) >= {"run_id", "overall", "dataset", "worst_classes"}
            assert len(body["worst_classes"]) <= 5
