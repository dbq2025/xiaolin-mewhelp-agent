"""DB 引擎装配与健康探测。

``ping()`` 是 ``/healthz`` 的数据来源 —— 它的契约是**永不抛异常**。
一旦它抛了，健康检查就从「503 + 结构化外壳」退化成裸 500，
编排器拿不到能判读的原因。
"""

import pytest


async def test_ping_returns_false_instead_of_raising_on_dead_engine():
    """连不上的库必须返回 ``False``，不能把异常冒出去。

    ``connect_timeout=1`` 让这条用例最多等 1 秒，而不是等 MySQL 驱动默认的 10 秒。
    """
    from sqlalchemy import create_engine

    from app.db.session import ping

    dead = create_engine(
        "mysql+pymysql://u:p@127.0.0.1:1/nope",
        connect_args={"connect_timeout": 1},
    )
    assert await ping(dead) is False


async def test_ping_true_on_live_engine():
    from sqlalchemy import create_engine

    from app.db.session import ping

    assert await ping(create_engine("sqlite+pysqlite:///:memory:")) is True


def test_create_engine_from_uses_settings_url(test_settings):
    from app.db.session import create_engine_from

    engine = create_engine_from(test_settings)
    assert engine.url.drivername == "mysql+pymysql"
    assert engine.url.database == "mewhelp"
    assert engine.url.query.get("charset") == "utf8mb4"
    engine.dispose()


def test_create_engine_from_sets_pre_ping_and_recycle(test_settings):
    """MySQL 的 ``wait_timeout`` 会掐掉闲置连接。

    没有 ``pool_pre_ping`` 时，一个闲置 8 小时后的请求会拿到**已经被对端关掉**的连接，
    报 ``Lost connection to MySQL server``；而 ``pool_recycle`` 提前把它换掉，
    让这种错误根本不发生。两个都要设，缺一个都会在生产里偶发。
    """
    from app.db.session import create_engine_from

    engine = create_engine_from(test_settings)
    assert engine.pool._pre_ping is True
    assert engine.pool._recycle == 1800
    engine.dispose()


def test_session_factory_yields_sessions_bound_to_the_engine(test_settings):
    from app.db.session import create_engine_from, session_factory

    engine = create_engine_from(test_settings)
    factory = session_factory(engine)
    session = factory()
    try:
        assert session.bind is engine
    finally:
        session.close()
        engine.dispose()


def test_repositories_from_session_returns_all_four(sqlite_session):
    """容器装配在真实 Session 上也能工作（不只是 SQLite fixture 里）。"""
    from app.db.repositories import Repositories

    repos = Repositories.from_session(sqlite_session)
    assert repos.conversations is not None
    assert repos.messages is not None
    assert repos.faq is not None
    assert repos.tickets is not None


@pytest.mark.parametrize("bad_port", [0, 70000])
def test_settings_reject_impossible_ports_before_any_connection_is_made(bad_port):
    """非法端口要在配置层就挡住，而不是等驱动去连。

    连之前挡住的好处：错误发生在**装配阶段**，运维看日志一眼知道是配置问题；
    否则报的是驱动的网络错误，看起来像「数据库挂了」。
    """
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            openai_base_url="https://x/v1",
            openai_api_key="k",
            openai_model="m",
            mysql_port=bad_port,
        )
