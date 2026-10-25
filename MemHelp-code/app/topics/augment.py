"""ch10 数据增强：同义词替换 + 句式微调，只扩训练集。

1200 条训练样本对 17 类来说偏少，增强是为了让模型别盯着某一种问法记。
两个手段都很朴素 —— 换说法、换口气 —— 因为**标签不能变**：任何会改变
诉求数量的增强（比如把「买大了想退」拆成两句）都会把多标签标注毁掉。

占位符（``[ORDER]`` 等）绝不能被替换：它是脱敏后的有效特征，换掉等于
把「这句话和订单有关」这个信号删了。
"""

from __future__ import annotations

import random
import re
from typing import Any, Sequence

#: 同义说法分组。组内可互换，跨组不换（换了就改语义）。
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("想退", "要退", "能退吗", "申请退款", "想退货"),
    ("换货", "换个", "能换吗", "调换"),
    ("到哪了", "到哪儿了", "几天能到", "什么时候到", "物流到哪"),
    ("包邮", "免运费", "包运费"),
    ("邮费", "运费", "快递费"),
    ("能开票吗", "能开发票吗", "可以开票吗", "能开专票吗"),
    ("偏大", "码数大", "买大了"),
    ("偏小", "码数小", "买小了"),
    ("没货", "缺货", "断货", "卖完了"),
    ("什么时候补货", "啥时候补货", "多久补货", "几时到货"),
    ("保修", "质保", "保修期"),
    ("维修", "修理", "修一下"),
    ("降价", "便宜了", "价格降了"),
    ("补差价", "退差价", "保价"),
    ("优惠券", "券", "代金券"),
    ("满减", "满多少减", "凑单满减"),
    ("登不上", "登录不了", "进不去"),
    ("改地址", "修改地址", "换个地址"),
    ("退款", "退钱", "把钱退我"),
    ("积分", "成长值", "会员分"),
)

#: 句式前缀 / 后缀，用来模拟不同用户的口吻。
_PREFIXES: tuple[str, ...] = ("请问", "麻烦问下", "想问一下", "您好，")
_SUFFIXES: tuple[str, ...] = ("呢", "啊", "呀", "谢谢")

_RE_PLACEHOLDER = re.compile(r"\[(?:IDCARD|PHONE|ORDER|EMAIL|ADDR)\]")


def _split_placeholders(text: str) -> list[str]:
    """把占位符切成独立片段，保证替换只发生在普通文字上。"""

    parts: list[str] = []
    last = 0
    for match in _RE_PLACEHOLDER.finditer(text):
        parts.append(text[last : match.start()])
        parts.append(match.group(0))
        last = match.end()
    parts.append(text[last:])
    return parts


def synonym_replace(text: str, rng: random.Random) -> str | None:
    """随机挑一组同义词换掉其中一处；没有可换的就返回 ``None``。"""

    candidates: list[tuple[int, str, str]] = []
    for group in SYNONYM_GROUPS:
        for variant in group:
            start = text.find(variant)
            while start != -1:
                candidates.append((start, variant, group))
                start = text.find(variant, start + 1)
    if not candidates:
        return None

    start, variant, group = rng.choice(candidates)
    alternatives = [v for v in group if v != variant]
    replacement = rng.choice(alternatives)
    return text[:start] + replacement + text[start + len(variant) :]


def reshape(text: str, rng: random.Random) -> str:
    """加称呼 / 语气词，模拟不同用户的口吻。已有的前缀不重复加。"""

    out = text
    if not any(out.startswith(prefix) for prefix in _PREFIXES) and rng.random() < 0.6:
        out = rng.choice(_PREFIXES) + out
    if not any(out.endswith(suffix) for suffix in _SUFFIXES) and rng.random() < 0.5:
        out = out + rng.choice(_SUFFIXES)
    return out


def _augment_one(text: str, rng: random.Random) -> str:
    parts = _split_placeholders(text)
    # 只对普通片段做同义替换，逐个片段试，第一个成功的就用。
    out_parts = list(parts)
    order = list(range(len(parts)))
    rng.shuffle(order)
    for index in order:
        if _RE_PLACEHOLDER.fullmatch(parts[index]):
            continue
        replaced = synonym_replace(parts[index], rng)
        if replaced is not None:
            out_parts[index] = replaced
            break
    result = "".join(out_parts)
    result = reshape(result, rng)
    if result == text:  # 兜底：增强样本必须真的不同，否则等于复制，白占一条
        result = "请问" + text if not text.startswith("请问") else text + "呢"
    return result


def augment_rows(
    rows: Sequence[dict[str, Any]], *, copies: int = 1, seed: int = 42
) -> list[dict[str, Any]]:
    """返回「原行 + 增强行」。原行原样保留，增强行带 ``aug_from``。"""

    if copies < 0:
        raise ValueError("copies 不能为负")
    rng = random.Random(seed)
    out: list[dict[str, Any]] = [dict(row) for row in rows]
    for row in rows:
        for n in range(copies):
            out.append(
                {
                    "id": f"{row['id']}-a{n + 1}",
                    "text": _augment_one(row["text"], rng),
                    "labels": list(row["labels"]),
                    "source": row.get("source", "synthetic"),
                    "aug_from": row["id"],
                }
            )
    return out
