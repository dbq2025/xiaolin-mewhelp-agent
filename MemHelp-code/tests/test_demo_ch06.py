"""ch06 Task 13：演示脚本的纯函数/参数校验（不连真实模型）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "scripts" / "demo-ch06.py"
    spec = importlib.util.spec_from_file_location("demo_ch06_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_case_accepts_known_values():
    module = _load_module()
    assert module.parse_case("all") == "all"
    assert module.parse_case("picker") == "picker"


def test_parse_case_rejects_unknown():
    module = _load_module()
    with pytest.raises(ValueError):
        module.parse_case("nope")
