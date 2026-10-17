"""单轮工具编排的测试 —— 本章的核心。

## 被测的精确语义（spec 4.1）

```
构建消息 → bind_tools(五工具).ainvoke()          ← 第 1 次调用，**非流式**，只为拿 tool_calls
  ├─ 无 tool_calls → 直接用拿到的文本流式吐出（本轮无徽章）
  └─ 有 tool_calls → 只取 [0]
        yield status 帧
        execute_tool(...)                        ← 参数校验 + 超时 + 重试
        messages += [AIMessage(tool_calls), ToolMessage(结果, tool_call_id)]
        astream(messages)                        ← 第 2 次调用，流式
        └─ 一个非空 delta 都没有 → fallback_answer(结果)
落库（四行）→ yield tools 帧
```

**「只调一次就收敛」是结构性保证**：第 2 次调用之后代码里没有任何检查
``tool_calls`` 的分支，也没有循环 —— 所以「模型想连调两次」不可能发生。
下面的用例钉住这一点。

## 为什么第 1 次调用不用流式

工具调用块的增量合并是 LangChain 侧的行为。把它和我们的编排混在一起，
会让「工具到底选了没有」变得难测。用一次非流式调用换确定的 ``tool_calls``，
第 2 次再用流式（用户要的逐 token 就在那一段）。代价是首 token 多一个 RTT。
"""

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.chat_tools import ToolEvent, astream_chat_with_tools, fallback_answer
from app.tools.executor import ToolResult
from tests.fakes import FakeChatModel, Recorder


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


class _RecordingMessageRepo:
    """内存消息仓库：记录 append 调用，并能回放成 LangChain 消息。"""

    def __init__(self, history=None) -> None:
        self.appended: list[dict] = []
        self._history = list(history or [])
        self.fail_on_append = False

    async def append(
        self,
        conversation_id,
        *,
        role,
        content,
        tool_calls=None,
        tool_call_id=None,
        tool_traces=None,
        citations=None,
    ):
        if self.fail_on_append:
            raise RuntimeError("落库失败")
        self.appended.append(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "tool_call_id": tool_call_id,
                "tool_traces": tool_traces,
                "citations": citations,
            }
        )

    async def history_as_messages(self, session_id):
        return list(self._history)

    def roles(self) -> list[str]:
        return [row["role"] for row in self.appended]


class _RecordingConversationRepo:
    def __init__(self, conversation_id=7) -> None:
        self.upserts: list[tuple[str, str]] = []
        self._id = conversation_id

    async def upsert_by_session(self, session_id, user_id):
        self.upserts.append((session_id, user_id))
        return self._id


class _FakeRepos:
    """只带编排用得上的两个仓库。"""

    def __init__(self, messages=None, conversation_id=7) -> None:
        self.messages = messages or _RecordingMessageRepo()
        self.conversations = _RecordingConversationRepo(conversation_id)
        # 本文件的用例都注入假 executor，所以 faq/tickets 永远用不到；
        # 只有「未知工具」那条会走真执行器去注册表查名字 —— 而注册表构造工具时
        # 会去读 repositories.faq / .tickets，所以这里必须给真对象。
        self._faq = None
        self._tickets = None

    @property
    def faq(self):
        if self._faq is None:
            self._faq = _NullRepo()
        return self._faq

    @property
    def tickets(self):
        if self._tickets is None:
            self._tickets = _NullRepo()
        return self._tickets

    @tickets.setter
    def tickets(self, value):
        self._tickets = value


class _NullRepo:
    """占位仓库：注册表只把引用绑进闭包，不会在本文件里调用它。"""

    async def search(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("本文件的用例不应真的查询数据")

    async def create(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("本文件的用例不应真的写数据")

    async def upsert_by_session(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("本文件的用例不应真的 upsert")


@pytest.fixture
def repos():
    return _FakeRepos()


def _tool_call(name="query_logistics", args=None, call_id="call_1"):
    return {
        "name": name,
        "args": args if args is not None else {"order_id": "1001"},
        "id": call_id,
        "type": "tool_call",
    }


def _fake_executor(results):
    """造一个假的 execute_tool：按调用顺序返回给定结果，并记录被请求了什么。"""
    calls: list[tuple[str, dict]] = []

    async def _run(name, args, **_kwargs):
        calls.append((name, args))
        if len(calls) <= len(results):
            return results[len(calls) - 1]
        return ToolResult(
            tool=name, args=args, ok=True, data='{"ok":true}', error=None,
            attempts=1, duration_ms=1,
        )

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


async def _collect(**kwargs):
    return [event async for event in astream_chat_with_tools(**kwargs)]


class _NoHitsRetriever:
    """编排用例不关心检索内容：给一个永远返回空的检索器。"""

    async def search(self, query, top_k=5):
        return []

    async def warmup(self) -> None:
        return None


def _common(repos, model, executor, **overrides):
    kwargs = {
        "session_id": "sess-a",
        "message": "订单 1001 的物流到哪了",
        "model": model,
        "repositories": repos,
        "retriever": _NoHitsRetriever(),
        "budget_tokens": 2000,
        "model_name": "fake-model",
        "timeout": 1.0,
        "max_retries": 0,
        "executor": executor,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 工具路径
# ---------------------------------------------------------------------------


async def test_status_frame_is_emitted_before_tokens(repos):
    """用户要先看到「正在查物流…」，再看到答案。

    顺序反了的话，工具执行那几秒里界面是全白的 —— 用户会以为客服没反应。
    """
    model = FakeChatModel(chunks=["已", "发货"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))

    kinds = [e.kind for e in events]
    assert "status" in kinds
    assert kinds.index("status") < kinds.index("token")


async def test_status_frame_carries_tool_name_args_and_chinese_label(repos):
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))
    status = next(e for e in events if e.kind == "status")

    assert status.payload["tool"] == "query_logistics"
    assert status.payload["args"] == {"order_id": "1001"}
    assert status.payload["label"].endswith("…")


async def test_tokens_are_streamed_after_the_tool_runs(repos):
    model = FakeChatModel(chunks=["你的", "包裹", "已到杭州"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))
    deltas = [e.payload["delta"] for e in events if e.kind == "token"]

    assert deltas == ["你的", "包裹", "已到杭州"]


async def test_tools_event_is_last_and_summarises_the_run(repos):
    """``done.tools`` 是徽章的权威来源，必须在最后给出完整成败信息。"""
    model = FakeChatModel(
        chunks=["好"], tool_calls_then_stream=[_tool_call()], bind_tools_calls=Recorder()
    )
    executor = _fake_executor(
        [ToolResult("query_logistics", {"order_id": "1001"}, True, '{"x":1}', None, 1, 12)]
    )

    events = await _collect(**_common(repos, model, executor))

    assert events[-1].kind == "tools"
    summary = events[-1].payload["tools"][0]
    assert summary["name"] == "query_logistics"
    assert summary["ok"] is True
    assert summary["duration_ms"] == 12
    assert summary["args"] == {"order_id": "1001"}
    assert summary["attempts"] == 1


async def test_tools_are_bound_to_the_model_before_asking(repos):
    """没绑工具的话模型永远不会调 —— 而且它会静默地直接答，看起来「正常」。

    走无工具路径（``tool_calls_then_stream`` 为空）来断言：编排只发**一次**模型调用，
    这一次就绑上了全部五个工具。``bind_tools`` 返回副本，所以用引用类型的
    Recorder 记（见 tests/fakes.py）。
    """
    model = FakeChatModel(chunks=["好"], bind_tools_calls=Recorder())
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    bound = model.bind_tools_calls.messages
    assert len(bound) == 5, f"应绑上全部五个工具，实际 {len(bound)}"
    assert {t.name for t in bound} == {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket",
    }


async def test_tool_result_is_fed_back_with_the_matching_tool_call_id(repos):
    """``ToolMessage.tool_call_id`` 必须与模型给的 id 对上。

    对不上的话上游会直接报错（有些网关报得很含糊），而且工具结果会被丢弃。
    """
    from tests.fakes import Recorder

    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[_tool_call(call_id="call_xyz")])
    model.recorder = Recorder()
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    await _collect(**_common(repos, model, executor))

    tool_messages = [m for m in model.recorder.messages if isinstance(m, ToolMessage)]
    assert tool_messages, "第二次调用必须带上 ToolMessage"
    assert tool_messages[0].tool_call_id == "call_xyz"


# ---------------------------------------------------------------------------
# 单轮收敛（Review Focus 2）
# ---------------------------------------------------------------------------


async def test_only_the_first_tool_call_is_executed_when_the_model_returns_several(repos):
    """Review Focus 2：模型一轮给了多个 ``tool_calls``，只执行第一个。

    这是「只做单轮调用」的硬保证。多执行的后果是同一轮里产生多个工具结果，
    而回灌只需要一个 —— 剩下的会变成孤儿 ``ToolMessage``，上游会拒绝。
    """
    model = FakeChatModel(
        chunks=["好"],
        tool_calls_then_stream=[
            _tool_call("query_logistics", {"order_id": "1001"}, "call_a"),
            _tool_call("query_faq", {"keyword": "运费"}, "call_b"),
            _tool_call("query_order", {"order_id": "1001"}, "call_c"),
        ],
        recorder=Recorder(),
    )
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))

    assert executor.calls == [("query_logistics", {"order_id": "1001"})]
    assert len([e for e in events if e.kind == "status"]) == 1
    # 只回灌一条 ToolMessage
    tool_messages = [m for m in model.recorder.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_a"


async def test_second_call_returning_tool_calls_is_ignored(repos):
    """Review Focus 2 的第二面：就算第 2 次调用还想调工具，也不再执行。

    代码里没有循环结构，所以这一条是**结构性**成立，而不是靠计数刹车。
    这里用一个「永远返回 tool_calls」的模型来验证它真的收敛。
    """

    class AlwaysWantsTools(FakeChatModel):
        def _reply_message(self):  # type: ignore[override]
            from langchain_core.messages import AIMessage

            return AIMessage(content="", tool_calls=[_tool_call()])

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            self._record(messages)
            yield ChatGenerationChunk(message=AIMessageChunk(content="最终还是答了"))

    model = AlwaysWantsTools()
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))

    assert len(executor.calls) == 1, "只执行一次，不进入循环"
    assert [e.payload["delta"] for e in events if e.kind == "token"] == ["最终还是答了"]


# ---------------------------------------------------------------------------
# 无工具路径
# ---------------------------------------------------------------------------


async def test_plain_answer_streams_without_any_tool_events(repos):
    """模型不调工具是**正常路径**，不该报错，也不该有徽章。

    无工具时第 1 次调用（非流式）拿到的文本会**一次**吐出，而不是再发一次
    流式请求 —— 同一次提问打两次上游，钱和延迟都翻倍。
    """
    model = FakeChatModel(chunks=["你好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    events = await _collect(**_common(repos, model, executor))

    assert [e.kind for e in events] == ["token", "tools"]
    assert events[0].payload["delta"] == "你好"
    assert events[-1].payload["tools"] == []
    assert executor.calls == []


async def test_plain_answer_uses_the_non_streaming_reply(repos):
    """无工具路径上，第 1 次（非流式）调用拿到的文本就是最终答案。

    **只设 ``reply``、不设 ``chunks``** —— 这样第 1 次调用必然返回 ``reply``，
    证明编排没有再去发一次流式请求取文本（那会让同一次提问打两次上游）。
    """
    model = FakeChatModel(reply="直接答的")
    executor = _fake_executor([])

    events = await _collect(**_common(repos, model, executor))

    assert [e.payload["delta"] for e in events if e.kind == "token"] == ["直接答的"]


# ---------------------------------------------------------------------------
# 空内容（Review Focus 1）
# ---------------------------------------------------------------------------


async def test_first_call_with_empty_content_and_tool_calls_is_not_an_error(repos):
    """Review Focus 1：第 1 次调用 content 为空**是正常的**。

    它只用来拿 ``tool_calls``。若照搬 ch01 的 ``UpstreamEmptyResponse`` 判定，
    每一次工具调用都会被误判成「上游返回空内容」而整轮失败 ——
    验收标准 1 与 2 会全部红，而错误信息还指向上游。
    """
    model = FakeChatModel(chunks=["查到了"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    events = await _collect(**_common(repos, model, executor))

    assert [e.payload["delta"] for e in events if e.kind == "token"] == ["查到了"]


async def test_second_call_with_empty_content_falls_back_to_tool_result_wording(repos):
    """Review Focus 1 的另一面：第 2 次调用真的没文本时，必须给用户一句人话。

    什么都不发的话，气泡里会是一片空白（前端显示「客服没有返回内容」），
    而工具其实成功了 —— 用户会觉得系统坏了。兜底话术要把工具结果说清楚。
    """
    model = FakeChatModel(
        chunks=[], yield_empty_chunk_when_no_chunks=True, tool_calls_then_stream=[_tool_call()]
    )
    result = ToolResult(
        tool="query_logistics",
        args={"order_id": "1001"},
        ok=True,
        data=json.dumps(
            {"status": "运输中", "carrier": "顺丰速运", "eta": "2026-02-20"},
            ensure_ascii=False,
        ),
        error=None,
        attempts=1,
        duration_ms=5,
    )
    executor = _fake_executor([result])

    events = await _collect(**_common(repos, model, executor))
    deltas = [e.payload["delta"] for e in events if e.kind == "token"]

    assert len(deltas) == 1
    assert "运输中" in deltas[0]


async def test_fallback_answer_for_a_failed_tool_does_not_claim_success(repos):
    """工具失败时兜底话术不能说「查到了」—— 那是编造。"""
    result = ToolResult(
        tool="query_logistics", args={}, ok=False, data=None,
        error="工具执行超时（超过 3.0 秒）", attempts=2, duration_ms=3000, retried=True,
    )

    text = fallback_answer(result)

    assert "查到了" not in text
    assert "运输中" not in text
    # 要说明失败、并给出下一步（建工单 / 人工核实）
    assert "没能取到数据" in text
    assert "核实" in text


async def test_fallback_answer_surfaces_speakable_fields_for_a_success(repos):
    """成功时要把关键字段说清楚，而不是把整坨 JSON 念出来。"""
    result = ToolResult(
        tool="query_logistics",
        args={},
        ok=True,
        data=json.dumps(
            {"status": "运输中", "carrier": "顺丰速运", "eta": "2026-02-20", "internal_id": "x"},
            ensure_ascii=False,
        ),
        error=None,
        attempts=1,
        duration_ms=1,
    )

    text = fallback_answer(result)

    assert "运输中" in text
    assert "顺丰速运" in text
    assert "internal_id" not in text, "不该把原始 JSON 字段名念给用户"


async def test_first_call_with_neither_content_nor_tool_calls_raises(repos):
    """两边都没有 = 上游真的返回了空内容，这时才该报错（沿用 ch01 的判定）。"""
    from app.errors import UpstreamEmptyResponse

    model = FakeChatModel(chunks=[], reply="", tool_calls_then_stream=[])
    executor = _fake_executor([])

    with pytest.raises(UpstreamEmptyResponse):
        await _collect(**_common(repos, model, executor))


# ---------------------------------------------------------------------------
# 落库（Review Focus 3 的另一半）
# ---------------------------------------------------------------------------


async def test_tool_turn_persists_four_rows_in_order(repos):
    """四行：user → assistant(带 tool_calls) → tool(带 tool_call_id) → assistant(答案)。

    顺序错了会让下次重建历史时 ``AIMessage(tool_calls)`` 出现在 ``ToolMessage``
    之后，上游直接拒绝这段历史。
    """
    model = FakeChatModel(chunks=["已", "发货"], tool_calls_then_stream=[_tool_call(call_id="c9")])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    await _collect(**_common(repos, model, executor))

    assert repos.messages.roles() == ["user", "assistant", "tool", "assistant"]
    assistant_call = repos.messages.appended[1]
    assert assistant_call["tool_calls"][0]["id"] == "c9"
    assert assistant_call["content"] == ""
    tool_row = repos.messages.appended[2]
    assert tool_row["tool_call_id"] == "c9"
    assert repos.messages.appended[3]["content"] == "已发货"


async def test_badge_details_are_persisted_separately_from_model_history(repos):
    """徽章信息（含耗时）要落进 ``tool_traces``，**不能**并进 ``tool_calls``。

    为什么必须分开：``tool_calls`` 会被原样塞回 ``AIMessage(tool_calls=...)``
    当模型历史，而 LangChain 对里面的字段是严格的 —— 实测多一个 ``duration_ms``
    会直接抛：``TypeError: tool_call() got an unexpected keyword argument``。

    所以：模型历史只有 ``{name, args, id}``；耗时/成败另存一列。
    这条测试同时钉住**两个方向**，缺一个就会让另一处悄悄坏掉。
    """
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[_tool_call(call_id="c1")])
    executor = _fake_executor(
        [ToolResult("query_logistics", {"order_id": "1001"}, True, "{}", None, 1, 37)]
    )

    await _collect(**_common(repos, model, executor))

    assistant_row = repos.messages.appended[1]

    # 方向一：模型历史必须干净，多一个键都会让上游构造消息失败
    assert set(assistant_row["tool_calls"][0]) == {"name", "args", "id"}

    # 方向二：展示用的耗时/成败必须存下来（否则刷新后徽章丢耗时）
    traces = assistant_row["tool_traces"]
    assert traces and traces[0]["duration_ms"] == 37
    assert traces[0]["ok"] is True
    assert traces[0]["name"] == "query_logistics"


async def test_plain_turn_persists_two_rows(repos):
    model = FakeChatModel(chunks=["你", "好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    assert repos.messages.roles() == ["user", "assistant"]
    assert repos.messages.appended[1]["content"] == "你好"
    assert repos.messages.appended[1]["tool_calls"] is None


async def test_conversation_is_upserted_once_per_turn(repos):
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    assert repos.conversations.upserts == [("sess-a", "anonymous")]


async def test_persisted_rows_carry_the_resolved_conversation_id(repos):
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    assert {row["conversation_id"] for row in repos.messages.appended} == {7}


async def test_failed_tool_still_persists_all_four_rows(repos):
    """工具失败**不是上游故障** —— 那是正常业务流程，四条都要落。

    第 3 行的 content 是 ``{"ok": false, ...}``，这样下次重建历史时模型能看到
    「上次查失败了」，而不是以为没查过。
    """
    model = FakeChatModel(chunks=["抱歉"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor(
        [ToolResult("query_logistics", {}, False, None, "查询超时", 2, 3000)]
    )

    await _collect(**_common(repos, model, executor))

    assert repos.messages.roles() == ["user", "assistant", "tool", "assistant"]
    assert json.loads(repos.messages.appended[2]["content"])["ok"] is False


# ---------------------------------------------------------------------------
# 历史回读与裁剪
# ---------------------------------------------------------------------------


async def test_existing_history_is_replayed_into_the_model_input(repos):
    """多轮对话：第二轮必须带上第一轮。历史来自**库**（不是内存）。"""
    from langchain_core.messages import AIMessage

    from tests.fakes import Recorder

    repos.messages._history = [HumanMessage("上轮问"), AIMessage("上轮答")]
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[])
    model.recorder = Recorder()
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    contents = [getattr(m, "content", "") for m in model.recorder.messages]
    assert "上轮问" in contents
    assert "上轮答" in contents


async def test_session_id_is_present_in_the_system_message(repos):
    """``create_ticket`` 需要 conversation_id，而模型只能从提示里知道它。

    不让它出现在提示里，模型就只能编一个 —— 工单会挂到不存在的会话上
    （真库上直接外键失败）。
    """
    from langchain_core.messages import SystemMessage

    from tests.fakes import Recorder

    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[])
    model.recorder = Recorder()
    executor = _fake_executor([])

    await _collect(**_common(repos, model, executor))

    system = next(m for m in model.recorder.messages if isinstance(m, SystemMessage))
    assert "sess-a" in system.content


async def test_tools_stay_available_on_the_second_call(repos):
    """第 2 次调用也要绑工具吗？

    **要**。上游在「历史里有 tool 消息」时有网关要求 tools 仍然存在，
    否则报错；而且不绑会让部分网关把这段历史判成非法。

    断言总条数是 10 = 两次调用各绑了五个（``bind_tools_calls`` 是引用类型，
    两次的记录都累积在同一个 Recorder 里）。
    """
    model = FakeChatModel(
        chunks=["好"], tool_calls_then_stream=[_tool_call()], bind_tools_calls=Recorder()
    )
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    await _collect(**_common(repos, model, executor))

    assert len(model.bind_tools_calls.messages) == 10, "两次调用各绑五个"


async def test_reasoning_content_from_the_first_call_is_carried_into_the_second(repos):
    """**思考模型的字段必须原样带回第 2 次调用。**

    踩过的坑（端到端跑验收时才暴露）：``deepseek-flash`` 是思考模型，第 1 次响应里
    除了 ``tool_calls`` 还带 ``reasoning_content``（住在 ``additional_kwargs``）。
    第二次请求不带回它就报：

        400 The `reasoning_content` in the thinking mode must be passed back to the API.

    所以编排**不能**用 ``AIMessage(content="", tool_calls=[chosen])`` 重建那条消息 ——
    那会把除 tool_calls 之外的一切丢掉。这个 bug 是**间歇性**的
    （只在模型真的产生思考内容时炸），所以它看起来像随机故障。
    """
    model = FakeChatModel(
        chunks=["好"],
        tool_calls_then_stream=[_tool_call()],
        tool_call_extra_kwargs={"reasoning_content": "用户要查物流，我该调用 query_logistics"},
        recorder=Recorder(),
    )
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    await _collect(**_common(repos, model, executor))

    from langchain_core.messages import AIMessage

    assistant_messages = [m for m in model.recorder.messages if isinstance(m, AIMessage)]
    assert assistant_messages, "第 2 次调用必须带上第 1 次那条 assistant 消息"
    carried = assistant_messages[0].additional_kwargs
    assert carried.get("reasoning_content") == "用户要查物流，我该调用 query_logistics", (
        f"思考内容没被带回第 2 次调用，实际上游会 400。additional_kwargs={carried!r}"
    )


async def test_only_the_first_tool_call_survives_into_the_second_call(repos):
    """多工具调用时，消息里只能留**被执行的那一个**。

    留了未执行的调用，上游会认为我们漏了它们的 ``ToolMessage``。
    这条与「思考字段必须保留」是一对：**裁剪 tool_calls，但保留其它字段**。
    """
    model = FakeChatModel(
        chunks=["好"],
        tool_calls_then_stream=[
            _tool_call("query_logistics", {"order_id": "1001"}, "call_a"),
            _tool_call("query_faq", {"keyword": "运费"}, "call_b"),
        ],
        tool_call_extra_kwargs={"reasoning_content": "先查物流"},
        recorder=Recorder(),
    )
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    await _collect(**_common(repos, model, executor))

    from langchain_core.messages import AIMessage

    assistant = next(m for m in model.recorder.messages if isinstance(m, AIMessage))
    assert len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0]["id"] == "call_a"
    # 裁的是 tool_calls，不是 other fields
    assert assistant.additional_kwargs.get("reasoning_content") == "先查物流"


# ---------------------------------------------------------------------------
# 失败语义（沿用 ch01：不落半截回答）
# ---------------------------------------------------------------------------


async def test_upstream_failure_persists_only_the_user_turn(repos):
    """上游挂了只留用户那一轮。

    落半截回答会污染下一轮上下文 —— 模型会以为它已经答过了。

    注意这里**不能**用 ``fail_after``：它只作用于流式路径，而第 1 次调用是
    非流式的，所以 ``fail_after=0`` 根本不会触发。必须让第 1 次调用就抛。
    """

    class ExplodingFirstCall(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            raise RuntimeError("upstream 500")

    model = ExplodingFirstCall(chunks=["好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    with pytest.raises(RuntimeError):
        await _collect(**_common(repos, model, executor))

    assert repos.messages.roles() == ["user"]


async def test_upstream_failure_after_a_tool_run_still_persists_just_the_user_turn(repos):
    """工具成功但第 2 次调用挂掉：同样只留用户那一轮。

    留一条孤立的 ``tool`` 行没有意义 —— 它没有对应的 assistant 申请单，
    下次重建历史会直接抛「tool 消息缺少 tool_call_id 的对端」。
    """

    class ExplodingSecondCall(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            # 第 1 次调用要给 tool_calls（走父类），流式那一步才炸。
            return super()._generate(messages, stop, run_manager, **kwargs)

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            self._record(messages)
            raise RuntimeError("upstream 500 on second call")
            yield  # pragma: no cover - 让本方法是异步生成器

    model = ExplodingSecondCall(chunks=["好"], tool_calls_then_stream=[_tool_call()])
    executor = _fake_executor([ToolResult("query_logistics", {}, True, "{}", None, 1, 5)])

    with pytest.raises(RuntimeError):
        await _collect(**_common(repos, model, executor))

    assert repos.messages.roles() == ["user"]


async def test_persist_failure_does_not_mask_the_original_upstream_error(repos):
    """**Review I1**：上游挂了、落库也挂了时，对外暴露的必须是**上游**那个异常。

    不改的话：``_persist`` 先抛，用户拿到的是数据库错误，而真因在上游 ——
    排查方向被彻底带偏（去查数据库，而数据库是好的）。

    这条测的正是「两个依赖同时坏」这种最需要准确诊断的场景。
    """

    class ExplodingFirstCall(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            raise RuntimeError("upstream model exploded")

    repos.messages.fail_on_append = True

    model = ExplodingFirstCall(chunks=["好"], tool_calls_then_stream=[])
    executor = _fake_executor([])

    with pytest.raises(RuntimeError) as excinfo:
        await _collect(**_common(repos, model, executor))

    assert "upstream model exploded" in str(excinfo.value), (
        f"原始上游异常被落库异常盖掉了，实际抛出：{excinfo.value}"
    )


async def test_meta_out_receives_the_context_result(repos):
    """HTTP 层要靠它填 ``done`` 帧里的裁剪统计。"""
    model = FakeChatModel(chunks=["好"], tool_calls_then_stream=[])
    executor = _fake_executor([])
    meta_out: dict = {}

    await _collect(**_common(repos, model, executor), meta_out=meta_out)

    assert meta_out["meta"].session_id == "sess-a"
    assert meta_out["meta"].context.budget_tokens == 2000
