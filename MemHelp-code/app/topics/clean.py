"""ch10 语料清洗：全半角、脱敏、错别字、空白。

四条规则**顺序固定**，换顺序会出错：
1. 全角→半角：不先转，后面的手机号正则就匹配不到全角数字；
2. 脱敏：不先脱敏，错别字替换可能把手机号里的字符改掉；
3. 错别字：只在已脱敏的文本上跑，避免污染占位符；
4. 空白压缩：最后做，前面每步都可能引入多余空格。

每条改动都记进 ``changes``，为的是两件事：人工抽审时能一眼看出这行被改过
什么，以及回滚时知道该撤回哪一条。**幂等**是硬要求 —— 同一份语料被流水线
重跑两次，第二次必须一条改动都不产生，否则重复跑会不断改写数据。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: 脱敏占位符。**占位符本身是有效特征**（「我的订单 [ORDER] 到哪了」仍然是物流），
#: 所以保留而不是删掉。
PLACEHOLDERS: tuple[str, ...] = ("[IDCARD]", "[PHONE]", "[ORDER]", "[EMAIL]", "[ADDR]")

_DEFAULT_TYPO_PATH = Path(__file__).resolve().parents[2] / "bank" / "typos.json"


@dataclass(frozen=True, slots=True)
class Change:
    """一条改动。``before`` / ``after`` 是违规字符本身，不是整句。"""

    rule: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class CleanResult:
    text: str
    changes: tuple[Change, ...]


def load_typo_map(path: Path | None = None) -> dict[str, str]:
    """读错别字词典。键是错的，值是正确写法。"""

    target = path or _DEFAULT_TYPO_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}


# -- 各条规则 --------------------------------------------------------------

#: 只转**全角数字 / 字母 / 全角空格**，不动中文标点 —— 把「，」转成「,」
#: 会让语料看起来像被机器糊过一遍，而这里没有任何收益。
_FULLWIDTH_ALNUM = {
    **{0xFF10 + i: ord("0") + i for i in range(10)},
    **{0xFF21 + i: ord("A") + i for i in range(26)},
    **{0xFF41 + i: ord("a") + i for i in range(26)},
    0x3000: 0x20,
}

_RE_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
#: 订单号必须带前缀才认：裸数字串太容易误伤年份、价格、型号。
_RE_ORDER = re.compile(r"(?P<label>订单号?|单号)(?P<sep>[:：]?)(?P<num>\d{6,})")
_RE_ADDR = re.compile(
    r"(?:[\u4e00-\u9fa5]{2,}(?:省|市|区|县|镇|乡)){1,3}"
    r"[\u4e00-\u9fa5]{0,10}(?:路|街|道|巷|弄|村)[0-9]{1,4}号?"
)
_RE_SPACE = re.compile(r"\s+")


def _normalize_width(text: str) -> tuple[str, list[Change]]:
    out = text.translate(_FULLWIDTH_ALNUM)
    if out == text:
        return text, []
    return out, [Change("normalize_width", text, out)]


def _mask(text: str) -> tuple[str, list[Change]]:
    changes: list[Change] = []

    def sub(pattern: re.Pattern[str], rule: str, placeholder: str, source: str) -> str:
        def repl(match: re.Match[str]) -> str:
            changes.append(Change(rule, match.group(0), placeholder))
            return placeholder

        return pattern.sub(repl, source)

    text = sub(_RE_IDCARD, "mask_idcard", "[IDCARD]", text)
    text = sub(_RE_PHONE, "mask_phone", "[PHONE]", text)
    text = sub(_RE_EMAIL, "mask_email", "[EMAIL]", text)
    # 订单号只替换数字那一段，保留「订单号」这个词（它本身带语义）。
    def order_repl(match: re.Match[str]) -> str:
        changes.append(Change("mask_order", match.group(0), match.group("label") + "[ORDER]"))
        return match.group("label") + "[ORDER]"

    text = _RE_ORDER.sub(order_repl, text)
    text = sub(_RE_ADDR, "mask_addr", "[ADDR]", text)
    return text, changes


def _fix_typos(text: str, typo_map: Mapping[str, str]) -> tuple[str, list[Change]]:
    changes: list[Change] = []
    # 长键优先：「发错活」必须先于「收活」这一级别被匹配，否则「活」会被截断替换。
    for wrong in sorted(typo_map, key=len, reverse=True):
        if wrong and wrong in text:
            right = typo_map[wrong]
            text = text.replace(wrong, right)
            changes.append(Change("typo", wrong, right))
    return text, changes


def _squeeze_space(text: str) -> tuple[str, list[Change]]:
    out = _RE_SPACE.sub(" ", text).strip()
    if out == text:
        return text, []
    return out, [Change("whitespace", text, out)]


def clean_text(raw: str, *, typo_map: Mapping[str, str] | None = None) -> CleanResult:
    """跑完四条规则，返回清洗后文本与全部改动。"""

    typos = typo_map if typo_map is not None else load_typo_map()
    changes: list[Change] = []

    text, step = _normalize_width(raw)
    changes += step
    text, step = _mask(text)
    changes += step
    text, step = _fix_typos(text, typos)
    changes += step
    text, step = _squeeze_space(text)
    changes += step

    return CleanResult(text=text, changes=tuple(changes))


def is_noise(result: CleanResult) -> bool:
    """清洗后只剩标点 / 占位符 / 表情 → 直接丢，不进语料。

    「在吗」不算噪声：它是个真问题（该归「其他」），只是没有主题词。
    所以判据是「有没有实义字符」，不是「有没有主题词」。
    """

    text = result.text
    for placeholder in PLACEHOLDERS:
        text = text.replace(placeholder, "")
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category[0] in {"P", "S", "Z", "C"}:
            continue
        return False
    return True
