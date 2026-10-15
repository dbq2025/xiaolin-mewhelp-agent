"""应用入口。

配置**不在导入期**读取：这样没有 ``.env`` 时也能 ``import app.main``
（测试与 ``--help`` 之类场景都需要），配置问题推迟到请求时以明确的
503 ``configuration_error`` 暴露，而不是启动就崩。

起服务::

    uv run uvicorn app.main:app --reload --port 8000
"""

from app.api import create_app

app = create_app()
