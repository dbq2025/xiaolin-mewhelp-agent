"""生成前自评：模型判断「证据够不够答这个问题」。

## 为什么 fail-open

JSON 解析失败 / 超时是**我方 prompt 或模型格式**的问题，不是证据不足。
fail-closed 会把每一次格式抖动变成「客服什么都不知道」，还会污染低置信度池；
而证据已经过了检索层的分数闸这道确定性检查。

## 为什么空证据不调模型

没有证据时答案必然是「答不了」，再花一次调用问模型只是浪费。
"""

from __future__ import annotations

import asyncio

import pytest

from app.knowledge.citations import Evidence
from app.knowledge.sufficiency import SufficiencyJudge, SufficiencyVerdict


class _FakeModel:
    def __init__(self, reply: str = "", *, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.calls: list[object] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        return type("Msg", (), {"content": self.reply})()


def evidence_of(*numbers: int) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(n=number, chunk_id=number, section_path=f"路径{number}",
                 category="售后政策", quote=f"原文{number}")
        for number in numbers
    )


async def test_sufficient_false_carries_the_reason():
    model = _FakeModel('{"sufficient": false, "missing": "赔付上限", "reason": "证据只讲流程"}')
    verdict = await SufficiencyJudge(model).judge("赔付上限多少", evidence_of(1, 2))

    assert verdict.sufficient is False
    assert verdict.missing == "赔付上限"
    assert verdict.reason == "证据只讲流程"
    assert verdict.degraded is False


async def test_sufficient_true():
    model = _FakeModel('{"sufficient": true, "missing": "", "reason": "够"}')
    verdict = await SufficiencyJudge(model).judge("运费怎么算", evidence_of(1))

    assert verdict.sufficient is True
    assert verdict.degraded is False


async def test_bad_json_fails_open():
    model = _FakeModel("我觉得够了，不用拒答")
    verdict = await SufficiencyJudge(model).judge("q", evidence_of(1))

    assert verdict.sufficient is True
    assert verdict.degraded is True


async def test_timeout_fails_open():
    model = _FakeModel('{"sufficient": false}', delay=1.0)
    verdict = await SufficiencyJudge(model, timeout=0.01).judge("q", evidence_of(1))

    assert verdict.sufficient is True
    assert verdict.degraded is True


async def test_empty_evidence_is_insufficient_without_a_model_call():
    model = _FakeModel('{"sufficient": true}')
    verdict = await SufficiencyJudge(model).judge("q", ())

    assert verdict.sufficient is False
    assert verdict.degraded is False
    assert model.calls == []


async def test_prompt_contains_numbered_evidence():
    model = _FakeModel('{"sufficient": true}')
    await SufficiencyJudge(model).judge("运费怎么算", evidence_of(1, 2))

    rendered = "\n".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for message in model.calls[0]
        for block in (message.content if isinstance(message.content, list) else [message.content])
    )
    assert "[1]" in rendered and "[2]" in rendered
    assert "路径1" in rendered and "原文2" in rendered


async def test_verdict_is_frozen():
    verdict = SufficiencyVerdict(sufficient=True)
    with pytest.raises(Exception):
        verdict.sufficient = False  # type: ignore[misc]
