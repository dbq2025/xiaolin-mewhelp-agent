"""SQLAlchemy 声明式基类。

单独一个文件是为了解开一个循环：``models.py`` 需要 ``Base``，而
``session.py`` 的 ``create_all`` 与测试要 ``Base.metadata``。把 ``Base`` 放这里，
两边都只依赖本模块，不会互相 import。

**注意**：``Base.metadata`` 只有在 ``app.db.models`` 被 import 过之后才认识那四张表。
所以任何要用 metadata 的地方（建表、对齐检查）都必须先
``import app.db.models``。测试里由 ``conftest.py`` 的 fixture 负责。
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """全部 ORM 模型的基类。

    ``Base.metadata`` 是「ORM 视角的表结构」，而 ``sql/ch02-ddl.sql`` 是
    「数据库视角的真相」。两者必须逐列对齐，由 ``tests/test_db_mysql.py``
    在真实 MySQL 上钉住 —— 本层的单测只能保证模型自身的形状。
    """
