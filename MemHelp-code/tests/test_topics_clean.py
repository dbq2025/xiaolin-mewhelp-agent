"""ch10 语料清洗：脱敏、错别字、格式，逐条留痕。

清洗是整条流水线的第一道闸门，也是唯一会**改写原始语料**的地方，
所以每条规则都钉一个断言，且要求「清洗两次不再产生改动」（幂等）。
"""

from pathlib import Path

from app.topics.clean import clean_text, is_noise, load_typo_map

TYPOS = {"帐号": "账号", "收活": "收货", "褪回": "退回"}


def test_masks_phone_number():
    r = clean_text("我手机13812345678，麻烦联系我", typo_map=TYPOS)
    assert "[PHONE]" in r.text
    assert "13812345678" not in r.text
    assert any(c.rule == "mask_phone" for c in r.changes)


def test_masks_fullwidth_digits_before_masking():
    """全角数字必须先转半角再脱敏，否则正则匹配不到 —— 最容易漏的一处。"""
    r = clean_text("电话１３８１２３４５６７８", typo_map=TYPOS)
    assert "[PHONE]" in r.text
    assert "１３８" not in r.text


def test_masks_idcard_order_email_address():
    r = clean_text(
        "身份证110101199003071234，订单号202410250001，邮箱a@b.com，"
        "北京市朝阳区建国路88号",
        typo_map=TYPOS,
    )
    for placeholder in ("[IDCARD]", "[ORDER]", "[EMAIL]", "[ADDR]"):
        assert placeholder in r.text


def test_fixes_known_typos():
    r = clean_text("我的帐号登不上，还没收活", typo_map=TYPOS)
    assert "账号" in r.text and "收货" in r.text
    assert any(c.rule == "typo" for c in r.changes)


def test_normalizes_width_and_whitespace():
    r = clean_text("  ＡＢＣ１２３   你好  ", typo_map=TYPOS)
    assert r.text == "ABC123 你好"


def test_clean_is_idempotent():
    """清洗两次必须不再产生 changes，否则批量重跑会不断改写语料。"""
    once = clean_text("我的帐号１３８１２３４５６７８登不上", typo_map=TYPOS)
    twice = clean_text(once.text, typo_map=TYPOS)
    assert twice.text == once.text
    assert twice.changes == ()


def test_noise_detection():
    assert is_noise(clean_text("   ", typo_map=TYPOS))
    assert is_noise(clean_text("？？？", typo_map=TYPOS))
    assert not is_noise(clean_text("在吗", typo_map=TYPOS))


def test_placeholder_alone_is_noise():
    assert is_noise(clean_text("[PHONE]", typo_map=TYPOS))


def test_changes_record_before_and_after():
    r = clean_text("帐号", typo_map=TYPOS)
    assert r.changes[0].before == "帐号" and r.changes[0].after == "账号"


def test_default_typo_map_is_loadable():
    m = load_typo_map(Path("bank/typos.json"))
    assert m["帐号"] == "账号"
