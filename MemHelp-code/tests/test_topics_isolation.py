"""ch10 Task 14：主应用**必须**保持零 ML 依赖。

这条断言不是洁癖：方案 B 的全部收益都建立在「`app.api` 的 import 图里没有 torch」
之上 —— 一旦有人在 ``app/api.py`` 顶层 import 了 ``app.topics.infer`` 的模型实现
（而不是只 import 轻量的 routes），部署机就又要装 2.7GB 轮子，sidecar 也就白拆了。

用 AST 静态检查而不是 ``sys.modules``：套件里别的测试文件会真的 import torch，
``"torch" not in sys.modules`` 这种断言会随收集顺序时红时绿。
运行时行为另有证据（见 dev-notes 阶段 3）：干净解释器里 ``import app.api`` 后
``'torch' in sys.modules`` 为 False。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEAVY = {"torch", "transformers", "safetensors", "tokenizers"}

MODULES = [
    "app/api.py",
    "app/topics/routes.py",
    "app/topics/batch.py",
    "app/topics/service.py",
    "app/topics/infer.py",
    "scripts/classify_topics.py",
]


def _toplevel_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:  # 只看模块顶层：函数内 import 是延迟加载，允许
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("rel", MODULES)
def test_module_level_imports_stay_light(rel):
    path = ROOT / rel
    assert path.is_file(), f"{rel} 不存在（清单过期了？）"
    assert not (_toplevel_import_roots(path) & HEAVY)


def test_infer_defers_torch_into_functions():
    """推理层只在真要用模型时才 import torch —— 于是纯逻辑测试不用装 GPU 轮子。"""
    tree = ast.parse((ROOT / "app/topics/infer.py").read_text(encoding="utf-8"))
    deferred = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name.split(".")[0] in HEAVY for alias in node.names)
    ]
    assert deferred, "infer.py 里应该有一处函数内 import torch"
    for node in deferred:
        assert node.col_offset > 0, "torch 的 import 必须缩进（在函数体内）"
