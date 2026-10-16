"""数据库引擎与 Session 的装配。

职责边界：**本模块只负责「怎么连」和「连接是否活着」**，不写任何查询 ——
查询形状在 ``app/db/repositories.py``，表结构在 ``app/db/models.py``。

## 为什么连接串必须来自 ``Settings.database_url()``

配置单一来源（见 ``agent.md`` 第 5 节）：只有 ``app/config.py`` 读 ``.env``，
所以这里也只接受一个 ``Settings`` 实例，不自己读环境变量。测试因此可以塞一份
写死的配置进来，完全不碰宿主机。

## 为什么 ``ping()`` 永不抛异常

它的唯一调用方是 ``/healthz``。健康检查的契约是「永远能给出一份可判读的答复」：
数据库挂了要报 **503 + ``{"database": "unavailable"}``**，而不是让异常冒到 ASGI 层
变成裸 500（客户端拿到 ``Exception in ASGI application``，完全无法判读）。
"""

import anyio
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
#: 连接池回收周期（秒）。MySQL 服务端 ``wait_timeout`` 默认 28800 秒（8 小时），
#: 但中间还可能有防火墙/NAT 更早掐断空闲连接。1800 秒（30 分钟）是常见取值：
#: 远小于任何一段正常的闲置窗口，又不至于频繁重建连接。
POOL_RECYCLE_SECONDS = 1800


def create_engine_from(settings: Settings) -> Engine:
    """按配置建引擎。

    ``pool_pre_ping=True`` 让每次取连接时先探一下还活着没 —— 服务端已经关掉的
    连接会被丢弃重建，而不是把这个连接交给业务代码去撞 ``Lost connection``。
    它和 ``pool_recycle`` 是互补的：recycle 预防，pre_ping 兜底。
    """
    return create_engine(
        settings.database_url(),
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        # 不 echo：连接串级别的 SQL 日志会把参数打出来，而工具参数里可能有用户原话。
        echo=False,
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """产出一个 Session 工厂。

    ``expire_on_commit=False`` 是刻意的：默认情况下 commit 后所有实例属性失效，
    下一次读取会触发新的 SELECT。仓库层在**工作线程**里用完 session 就把数据
    拆成纯 dataclass 返回（见 repositories.py），所以这里不影响正确性；
    但关掉它能少一批「commit 之后又打一次库」的意外往返。
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


async def ping(engine: Engine) -> bool:
    """探一次数据库可用性。**任何异常都返回 ``False``，绝不抛。**

    同步驱动会阻塞，所以送进工作线程 —— ``/healthz`` 是高频接口，
    不能让它把事件循环按住整个连接超时的时长。
    """

    def _probe() -> bool:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - 健康检查的契约就是「吞掉一切，返回布尔」
            return False

    return await anyio.to_thread.run_sync(_probe)
