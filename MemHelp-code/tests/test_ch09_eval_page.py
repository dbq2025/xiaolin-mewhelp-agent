"""ch09 Task 12：评估趋势页的静态挂载。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main
from tests.fakes import FakeChatModel


def _client(settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def test_eval_trend_page_is_served_with_its_assets(test_settings):
    client = _client(test_settings)

    redirect = client.get("/eval-trend", follow_redirects=False)
    assert redirect.status_code in {307, 308}
    assert redirect.headers["location"] == "/eval-trend/"

    page = client.get("/eval-trend/")
    assert page.status_code == 200
    assert "指标趋势" in page.text
    assert "/eval-trend/trend.js" in page.text

    script = client.get("/eval-trend/trend.js")
    assert script.status_code == 200
    assert "api/eval-runs" in script.text
    assert "waitForNewRun" in script.text

    style = client.get("/eval-trend/style.css")
    assert style.status_code == 200
    assert ".sparkline" in style.text
