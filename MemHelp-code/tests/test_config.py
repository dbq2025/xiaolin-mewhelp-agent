"""配置校验测试。

**不使用 pytest 的 ``tmp_path``**：DSH 沙箱下测试进程新建的目录在退出后会被
收回权限（见 ``conftest.py``），而 ``.uv-cache/`` 这类目录被 gitignore、干净克隆
上也不存在。所以这里用 ``tempfile.mkdtemp`` 之外的策略 —— 直接写进**自建**目录，
写完即删。

**也不需要 ``_env_file`` 的隐式路径**：``Settings`` 的 ``env_file`` 是绝对路径
（``app.config.ENV_FILE``），与当前工作目录无关。测试通过显式传 ``_env_file=``
覆盖，并清掉宿主环境变量 —— 否则 pydantic-settings 的优先级
（OS 环境变量 > .env 文件）会让断言在意料之外的环境下失败。
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

#: 放临时 .env 的目录，位于工作区内。
#:
#: 为什么不用 ``tempfile`` / ``tmp_path``：沙箱会把**测试进程新建的目录**在退出后
#: 收回权限（见 conftest.py），``tempfile.TemporaryDirectory`` 也一样会撞
#: ``PermissionError``（实测）。所以用工作区里自建的固定目录。
ENV_FIXTURE_DIR = Path(".uv-cache") / "cfg-fixtures"

# 注：「清掉宿主 OPENAI_* 环境变量」与「把 env_file 置空」现在由 conftest.py 的
# 全局 autouse fixture 负责 —— 原因见那里：整个套件的结果不能取决于宿主机有没有
# .env。本文件因此不再需要自己的清理 fixture。


@pytest.fixture
def env_dir():
    """一个自建的可写目录，用于放临时 .env 文件。"""
    ENV_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    return ENV_FIXTURE_DIR


def _write_env(env_dir: Path, name: str, body: str) -> Path:
    path = env_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def test_settings_read_from_env_file(env_dir):
    env = _write_env(
        env_dir,
        "t-read.env",
        "OPENAI_BASE_URL=https://gw.example.com/v1\n"
        "OPENAI_API_KEY=sk-test\n"
        "OPENAI_MODEL=deepseek-chat\n"
        "CONTEXT_TOKEN_BUDGET=1234\n",
    )
    try:
        s = Settings(_env_file=env)
    finally:
        env.unlink(missing_ok=True)

    assert s.openai_base_url == "https://gw.example.com/v1"
    assert s.openai_model == "deepseek-chat"
    assert s.context_token_budget == 1234
    assert s.request_timeout == 60.0


def test_env_file_is_anchored_to_the_code_dir_not_the_cwd():
    """``.env`` 必须相对**代码目录**解析，而不是当前工作目录。

    本仓库布局是「仓库根 + 代码在 MemHelp-code/」（见 ``agent.md``），而
    pydantic-settings 的相对路径按 CWD 解析 —— 实测从仓库根启动报 3 个
    ``Field required``。所以这里断言：``ENV_FILE`` 与 ``app/`` 同级。
    """
    import app

    from app.config import ENV_FILE

    assert ENV_FILE.is_absolute(), "env_file 必须是绝对路径"
    code_dir = Path(app.__file__).resolve().parent.parent
    assert ENV_FILE.parent == code_dir, (
        f".env 应解析到代码目录 {code_dir}，实际是 {ENV_FILE}"
    )
    assert ENV_FILE.name == ".env"


def test_settings_work_regardless_of_cwd(monkeypatch, env_dir):
    """换个工作目录也要能读到同一份 .env。

    ``_env_file`` 本身也是按 CWD 解析的，所以这里显式传绝对路径；要证明的是
    「显式绝对路径 + 任意 CWD」都能读到 —— 生产代码里 ``ENV_FILE`` 正是绝对路径。
    """
    env = _write_env(
        env_dir,
        "t-cwd.env",
        "OPENAI_BASE_URL=https://gw.example.com/v1\n"
        "OPENAI_API_KEY=sk-test\n"
        "OPENAI_MODEL=cwd-model\n",
    ).resolve()
    original = Path.cwd()
    try:
        os.chdir(Path.cwd().anchor)  # 换到一个完全无关的工作目录（盘根）
        s = Settings(_env_file=env)
        assert s.openai_model == "cwd-model"
    finally:
        os.chdir(original)
        env.unlink(missing_ok=True)


def test_settings_strip_surrounding_whitespace():
    s = Settings(
        openai_base_url="  https://gw.example.com/v1  ",
        openai_api_key="  sk-test  ",
        openai_model="  deepseek-chat  ",
    )
    assert s.openai_base_url == "https://gw.example.com/v1"
    assert s.openai_api_key == "sk-test"
    assert s.openai_model == "deepseek-chat"


@pytest.mark.parametrize(
    "overrides",
    [
        {"openai_api_key": "   "},
        {"openai_model": ""},
        {"openai_base_url": "gw.example.com"},
        {"context_token_budget": 0},
        {"context_token_budget": -1},
    ],
    ids=["blank-key", "blank-model", "scheme-missing", "budget-zero", "budget-negative"],
)
def test_settings_reject_invalid_values(overrides):
    base = {
        "openai_base_url": "https://gw.example.com/v1",
        "openai_api_key": "sk-test",
        "openai_model": "m",
    }
    base.update(overrides)
    with pytest.raises(ValidationError):
        Settings(**base)


def test_settings_require_all_three_openai_fields(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------------------------------------------------------------------------
# ch02：MySQL 连接与工具基础设施
# ---------------------------------------------------------------------------


def test_database_url_is_pymysql_with_utf8mb4():
    """连接串形状必须逐字符确定 —— 它是全项目唯一生成 DB URL 的地方。"""
    s = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        mysql_host="db.internal", mysql_port=3307, mysql_user="u",
        mysql_password="p", mysql_database="mewhelp",
    )
    assert s.database_url() == (
        "mysql+pymysql://u:p@db.internal:3307/mewhelp?charset=utf8mb4"
    )


def test_database_url_escapes_special_chars_in_password():
    """密码里有 @ / : / / 时必须 percent-encode。

    不转义的话 ``p@ss`` 会让 URL 解析把 ``@`` 之后当成 host，连接打到错误的机器上 ——
    这种故障的表现是「连不上数据库」，而根因在密码字符，极难查。
    """
    s = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        mysql_password="p@ss:w/rd",
    )
    assert "p%40ss%3Aw%2Frd" in s.database_url()


def test_tool_infra_defaults():
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    assert s.tool_timeout_seconds == 3.0
    assert s.tool_max_retries == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"mysql_port": 0},
        {"mysql_port": 70000},
        {"tool_timeout_seconds": 0},
        {"tool_timeout_seconds": -1},
        {"tool_max_retries": -1},
        {"tool_max_retries": 6},
    ],
    ids=["port-zero", "port-too-big", "timeout-zero", "timeout-negative",
         "retries-negative", "retries-too-many"],
)
def test_ch02_settings_reject_invalid_values(overrides):
    """超时 0 会让每次工具调用立刻超时；重试次数上不封顶会让一次故障拖垮请求。
    这些都必须在配置层挡住，而不是等线上表现成「工具永远失败」。"""
    base = {
        "openai_base_url": "https://gw.example.com/v1",
        "openai_api_key": "sk-test",
        "openai_model": "m",
    }
    base.update(overrides)
    with pytest.raises(ValidationError):
        Settings(**base)


# ---------------------------------------------------------------------------
# ch03：RAG 配置
# ---------------------------------------------------------------------------


def test_rag_settings_have_stable_defaults():
    """默认值必须与 `.env.example` 和 spec §8 一致 —— 三者不同步会让「本机跑得通、
    换台机器跑不通」，而症状通常只是「检索结果莫名其妙」。"""
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    assert s.milvus_uri == "http://127.0.0.1:19530"
    assert s.milvus_collection == "knowledge"
    assert s.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.embedding_timeout == 30.0
    assert s.embedding_batch_size == 32
    assert s.embedding_max_retries == 2
    assert s.rag_top_k == 5
    assert s.rag_warmup is True
    assert s.chunk_max_chars == 600
    assert s.chunk_overlap == 80
    assert s.qa_batch_size == 8
    assert s.qa_dedupe_threshold == 0.92


def test_embedding_key_may_be_absent_until_first_call():
    """没密钥也要能 import 配置、跑离线测试、看 --status。

    只有**真正要发请求**时才允许报错 —— 否则「本机没配 key」会让整个套件
    与健康检查一起崩，而那不是配置错误，是环境差异。
    """
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    assert s.embedding_api_key == ""


def test_require_embedding_key_points_at_env_file():
    s = Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        s.require_embedding_key()


def test_embedding_url_strips_trailing_slash():
    s = Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        embedding_base_url="https://api.siliconflow.cn/v1/",
    )
    assert s.embedding_url() == "https://api.siliconflow.cn/v1/embeddings"


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_dim": 0},
        {"embedding_timeout": 0},
        {"embedding_batch_size": 0},
        {"embedding_batch_size": 999},
        {"embedding_max_retries": -1},
        {"embedding_max_retries": 9},
        {"rag_top_k": 0},
        {"chunk_max_chars": 0},
        {"chunk_overlap": -1},
        {"qa_batch_size": 0},
        {"qa_dedupe_threshold": 1.5},
        {"qa_dedupe_threshold": -0.1},
    ],
    ids=["dim-zero", "timeout-zero", "batch-zero", "batch-too-big",
         "retries-negative", "retries-too-many", "topk-zero", "chunk-zero",
         "overlap-negative", "qa-batch-zero", "threshold-over-one",
         "threshold-negative"],
)
def test_ch03_settings_reject_invalid_values(overrides):
    """去重阈值 >1 会让「永远不判重」，维数 0 会让 Milvus 建集合失败 —
    这些错在配置层报出来，比在检索/建库时表现成「空结果」好查得多。"""
    base = {
        "openai_base_url": "https://gw.example.com/v1",
        "openai_api_key": "sk-test",
        "openai_model": "m",
    }
    base.update(overrides)
    with pytest.raises(ValidationError):
        Settings(**base)
