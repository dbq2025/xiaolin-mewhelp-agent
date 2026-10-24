"""ch09 Task 9：待审队列后台页面的静态挂载。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main
from tests.fakes import FakeChatModel


def _client(settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def test_review_page_is_served_with_its_assets(test_settings):
    client = _client(test_settings)

    redirect = client.get("/review", follow_redirects=False)
    assert redirect.status_code in {307, 308}
    assert redirect.headers["location"] == "/review/"

    page = client.get("/review/")
    assert page.status_code == 200
    assert "待审队列" in page.text
    assert "/review/app.js" in page.text

    script = client.get("/review/app.js")
    assert script.status_code == 200
    assert "review-queue" in script.text

    style = client.get("/review/style.css")
    assert style.status_code == 200
    assert "--brand" in style.text
