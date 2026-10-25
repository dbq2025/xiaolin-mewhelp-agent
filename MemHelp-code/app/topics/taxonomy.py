"""ch10：17 类权威类目表 —— 全系统唯一来源。

这张表同时是：合成语料的生成依据、大模型预标的分类依据、训练标签的
index 映射、混淆矩阵的行列顺序、前端条形图的显示顺序。**顺序即编号**，
改顺序等于换模型语义，所以 `taxonomy_hash()` 会把它钉进模型产物里，
sidecar 启动时比对，不一致就拒绝启动（见 `app/topics/infer.py`）。

为什么标签直接用中文类目名、不搞英文 code 中转：需求说的是「全系统统一
一份 17 类的权威类目表」，那说的就是这张中文表。多一层映射只多一处对不上
的地方，而它换不来任何收益 —— labels 存进 JSON 列，前端直接显示。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

OTHER_LABEL = "其他"


class TaxonomyError(ValueError):
    """标签不在表内、空标签集、或违反了「其他」独占的规则。"""


@dataclass(frozen=True, slots=True)
class Category:
    """一个类目。``boundary`` 是给人和给大模型看的分界线，不是给人看的简介。"""

    index: int
    name: str
    boundary: str
    triggers: tuple[str, ...]


#: 17 类，顺序即编号即输出顺序。改这里必须同步重训模型。
CATEGORIES: tuple[Category, ...] = (
    Category(
        1,
        "退换货",
        "退货、换货、退款意图，以及退货条件、流程、时限。"
        "「修」归保修维修，「改地址 / 改规格」归订单修改。",
        ("想退", "能换吗", "退款多久到"),
    ),
    Category(
        2,
        "物流",
        "货的运输状态。「钱」归运费。",
        ("到哪了", "几天到", "哪家快递", "丢件"),
    ),
    Category(
        3,
        "尺码",
        "选码建议、偏大偏小、尺码表。参数材质归商品信息。",
        ("买多大", "偏大吗", "尺码表"),
    ),
    Category(
        4,
        "发票",
        "开票、抬头税号、专普票、重开。付款问题归支付。",
        ("能开票吗", "专票", "发票随货"),
    ),
    Category(
        5,
        "质量问题",
        "到手即坏、破损、少件、发错货、瑕疵异味。用久了坏归保修维修。",
        ("一拆就坏", "发错货", "有味道"),
    ),
    Category(
        6,
        "运费",
        "邮费多少钱、包不包邮、补运费差价、运费险。运输进度归物流。",
        ("包邮吗", "邮费多少", "运费险"),
    ),
    Category(
        7,
        "优惠活动",
        "券、满减、秒杀、赠品、活动规则与叠加。降价补差归价保。",
        ("券能用吗", "能叠加吗", "满减"),
    ),
    Category(
        8,
        "价保",
        "买后降价能否补差价、价保期限与申请。券和满减归优惠活动。",
        ("降价了", "补差价", "价保"),
    ),
    Category(
        9,
        "支付",
        "付款失败、重复扣款、支付方式、分期。开票归发票。",
        ("付不了", "扣两次", "能分期吗"),
    ),
    Category(
        10,
        "订单修改",
        "改地址、改规格颜色、合并拆分单、未发货取消。已发货要退归退换货。",
        ("改地址", "改颜色", "取消订单"),
    ),
    Category(
        11,
        "库存补货",
        "缺货、何时补货、到货通知、预售发货时间。",
        ("没货了", "啥时候补", "到货通知"),
    ),
    Category(
        12,
        "商品信息",
        "材质、参数、型号、功能、适配、真伪、保质期。选码归尺码。",
        ("什么材质", "支持吗", "正品吗"),
    ),
    Category(
        13,
        "保修维修",
        "保修期、上门修、寄修、换配件、维修收费。到手即坏归质量问题。",
        ("保修多久", "怎么修", "上门"),
    ),
    Category(
        14,
        "账号",
        "登录、密码、换绑手机、注销、实名、冻结。",
        ("登不上", "改手机号", "注销"),
    ),
    Category(
        15,
        "会员积分",
        "会员等级、积分获取抵扣过期、权益、成长值。",
        ("积分怎么用", "会员价", "过期"),
    ),
    Category(
        16,
        "评价",
        "晒单、追评、差评申诉、评价奖励、改评价。",
        ("怎么晒单", "能改评价吗"),
    ),
    Category(
        17,
        OTHER_LABEL,
        "以上都不涉及（投诉客服态度、转人工、闲聊）。独占，不与其他类共现。",
        ("转人工", "你们服务太差"),
    ),
)

CATEGORY_NAMES: tuple[str, ...] = tuple(c.name for c in CATEGORIES)
LABEL_TO_INDEX: dict[str, int] = {c.name: c.index for c in CATEGORIES}


def validate_labels(labels: Iterable[str]) -> list[str]:
    """校验并归一：去重 → 查表 → 「其他」独占 → 按编号升序。

    排序不是审美问题：混淆矩阵的行列顺序、多热向量的列顺序都按同一个
    编号走，乱序会让「模型输出第 3 位是尺码」这种对应关系在某个环节断掉。
    """

    ordered = list(dict.fromkeys(labels))  # 去重且保序
    if not ordered:
        raise TaxonomyError("标签集不能为空：每个问题至少要有一个类目")

    unknown = [label for label in ordered if label not in LABEL_TO_INDEX]
    if unknown:
        raise TaxonomyError(f"类目表里没有这些标签：{'、'.join(unknown)}")

    if OTHER_LABEL in ordered and len(ordered) > 1:
        raise TaxonomyError(
            f"「{OTHER_LABEL}」独占，不可与其他类目共现：{'、'.join(ordered)}"
        )

    return sorted(ordered, key=lambda label: LABEL_TO_INDEX[label])


def taxonomy_payload() -> dict[str, Any]:
    """可序列化的快照 —— 落进 ``models/topics/<run_id>/taxonomy.json``。"""

    return {
        "version": 1,
        "categories": [
            {
                "index": c.index,
                "name": c.name,
                "boundary": c.boundary,
                "triggers": list(c.triggers),
            }
            for c in CATEGORIES
        ],
    }


def canonical_json(payload: dict[str, Any]) -> str:
    """哈希用的规范形态：键排序 + 无多余空格 + 中文不转义。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def taxonomy_hash(payload: dict[str, Any] | None = None) -> str:
    """类目表的内容指纹，用于「模型是用哪版类目表训的」这件事。"""

    return hashlib.sha256(canonical_json(payload or taxonomy_payload()).encode("utf-8")).hexdigest()


def prompt_fragment() -> str:
    """喂给合成语料 / 预标 prompt 的那段人话。与 CATEGORIES 永远同源。"""

    lines = ["类目表（编号 / 类目 / 边界 / 典型说法）："]
    for c in CATEGORIES:
        lines.append(
            f"{c.index}. {c.name}｜边界：{c.boundary}｜典型说法：{'、'.join(c.triggers)}"
        )
    lines.append(
        "规则：「其他」独占，只要有一个诉求能落进 1–16 就不打「其他」；"
        "标签按编号升序输出；只打字面出现的诉求，不推断隐含需求。"
    )
    return "\n".join(lines)


def names_at(indices: Sequence[int]) -> list[str]:
    """按编号取类目名（评测报告按类目顺序排版时用）。"""

    by_index = {c.index: c.name for c in CATEGORIES}
    return [by_index[i] for i in indices]
