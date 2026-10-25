"""ch09 观测与成本页的静态挂载。

讲义第九章「验收过程」给的是 ``http://localhost:8000/observability``：一个**页面**，
而不是 Langfuse 平台。本章改用 Langfuse Cloud 之后本地没有 3000 端口，所以这个
页面必须由本服务自己托管，成本与趋势都读本项目已有的接口
（``/api/intent-cost``、``/api/eval-runs``、``/healthz``）。

这两条用例钉的是**挂载契约**，不是页面长相：路径、重定向、资产引用、以及「从
其它页面点得进来」（否则页面就是个只有作者知道地址的孤儿页）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main
from tests.fakes import FakeChatModel


def _client(settings) -> TestClient:
    return TestClient(
        app_main.create_app(model=FakeChatModel(chunks=["好"]), settings=settings)
    )


def test_observability_page_is_served_with_its_assets(test_settings):
    client = _client(test_settings)

    redirect = client.get("/observability", follow_redirects=False)
    assert redirect.status_code in {307, 308}
    assert redirect.headers["location"] == "/observability/"

    page = client.get("/observability/")
    assert page.status_code == 200
    assert "观测与成本" in page.text
    assert "/observability/observability.js" in page.text
    assert "/observability/style.css" in page.text
    # 讲义验收地址是 /observability，页面自述要说明「数据从哪来」，
    # 免得看的人以为这是 Langfuse 的另一个入口。
    assert "Langfuse" in page.text

    script = client.get("/observability/observability.js")
    assert script.status_code == 200
    assert "api/intent-cost" in script.text
    assert "api/eval-runs" in script.text
    assert "healthz" in script.text

    style = client.get("/observability/style.css")
    assert style.status_code == 200
    assert ".bar-fill" in style.text


def test_every_page_links_to_observability(test_settings):
    """四个已有页面都要能点进观测页 —— 否则它只会被讲义地址访问到。"""
    client = _client(test_settings)

    for path in ("/chat/", "/rag-eval/", "/review/", "/eval-trend/"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert 'href="/observability/"' in page.text, path
