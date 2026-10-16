"""仓库约定的静态检查。

这里放的是「跨文件、无法用行为测试表达」的约束。它们跑得很快，价值在于
把**踩过的坑**变成不会复发的不变量。

## 为什么必须有第一条

``.ps1`` 里带中文时，PowerShell 5.1 会按 **GBK** 读文件（见 ``agent.md`` 第 4 节），
中文会撑破字符串字面量，报出一堆 ``Unexpected token`` —— 而文件在 UTF-8 阅读器里
看起来完全正常。

这个坑在本章踩了**三次**：任何「重写整个文件」的工具（``Set-Content``、
编辑器、甚至一次结构化编辑）都会静默把 UTF-8 BOM 丢掉。丢掉的当下没有任何症状，
直到你运行脚本才炸，而且报错信息指向的是「乱码 token」而不是「编码问题」。

所以这条检查的目的不是「符合某个风格」，而是**让这个静默失效无法再次发生**。
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "MemHelp-code" / "scripts"

UTF8_BOM = b"\xef\xbb\xbf"


def _powershell_scripts() -> list[Path]:
    if not SCRIPTS_DIR.is_dir():  # pragma: no cover - 仓库结构异常
        return []
    return sorted(SCRIPTS_DIR.glob("*.ps1"))


def test_there_are_powershell_scripts_to_check():
    """防呆：glob 写错时下面的用例会一个都不跑，于是「全绿」变成假象。"""
    assert _powershell_scripts(), f"在 {SCRIPTS_DIR} 下没找到任何 .ps1"


@pytest.mark.parametrize("script", _powershell_scripts(), ids=lambda p: p.name)
def test_powershell_scripts_with_chinese_start_with_a_utf8_bom(script: Path):
    """含中文的 ``.ps1`` 必须是 UTF-8 **with BOM**。

    PowerShell 5.1 没有 BOM 时按 GBK 解码：中文注释与字符串会变成乱码，
    字符串字面量被撑破，脚本直接语法错误。加 BOM 是让它正确识别 UTF-8 的唯一办法
    （``# -*- coding: utf-8 -*-`` 那种声明 PowerShell 不认）。
    """
    raw = script.read_bytes()
    has_chinese = any(byte >= 0x80 for byte in raw)

    if not has_chinese:
        pytest.skip(f"{script.name} 是纯 ASCII，不需要 BOM")

    assert raw.startswith(UTF8_BOM), (
        f"{script.name} 含中文但没有 UTF-8 BOM —— PowerShell 5.1 会按 GBK 读它，"
        "中文会撑破字符串字面量。修法："
        f"用 UTF-8 with BOM 重新保存（`[System.IO.File]::WriteAllBytes(path, BOM + bytes)`）"
    )


@pytest.mark.parametrize("script", _powershell_scripts(), ids=lambda p: p.name)
def test_powershell_scripts_are_valid_utf8(script: Path):
    """去掉 BOM 之后必须是合法 UTF-8（防止有人用 GBK 存了整个文件）。"""
    raw = script.read_bytes()
    body = raw[len(UTF8_BOM):] if raw.startswith(UTF8_BOM) else raw
    body.decode("utf-8")  # 解码失败会抛 UnicodeDecodeError


def test_seed_sql_never_contains_the_shipping_fee_keyword():
    """**验收标准 3 的语料基线**：``sql/ch02-seed.sql`` 的 FAQ 数据里不能出现「邮费」。

    这条与 ``tests/test_db_mysql.py`` 里的真库断言是一对：
    那里验「库里没有」，这里验「种子脚本也不会加进去」——
    否则下次有人重建库时基线就被破坏了，而那时已经没人记得这件事。

    注意只看 ``INSERT INTO faq`` 那一段：注释与自检语句里出现「邮费」是**故意**的
    （它们就是在说明这条基线），不能把它们也算进来。
    """
    seed = REPO_ROOT / "sql" / "ch02-seed.sql"
    if not seed.is_file():  # pragma: no cover - 文件被移走时给出明确信息
        pytest.fail(f"找不到种子脚本：{seed}")

    text = seed.read_text(encoding="utf-8")
    start = text.find("INSERT INTO faq")
    assert start != -1, "ch02-seed.sql 里没有 INSERT INTO faq 段"
    end = text.find(";", start)
    faq_block = text[start:end]

    assert "邮费" not in faq_block, (
        "验收标准 3 的基线被破坏：种子语料里出现了「邮费」。"
        "「邮费是多少」将不再漏召回，ch03 的改造前基线随之消失。"
    )
