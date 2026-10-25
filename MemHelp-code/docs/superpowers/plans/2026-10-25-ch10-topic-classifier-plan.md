# ch10 多标签主题分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统落地一个 17 类多标签主题分类器 —— 用 RoBERTa-wwm-ext 全参微调，把飞轮低置信度问题按主题批量归类，写进 `topic_classifications`，并在后台出一个主题分布页。

**Architecture:** 三条离线流水线 + 一个旁路服务。①数据：清洗 → 归并术语表 → 分层抽样 → 只扩训练集的数据增强，产物落 `data/topics/`。②训练/评测：自写训练循环（17 个独立 sigmoid + BCEWithLogitsLoss + 早停 + val 上每类阈值扫描），产物落 `models/topics/<run_id>/`，评测出 `report.md` + `per_class_prf.csv` + `confusion.html`。③推理：**独立的无状态 sidecar**（`app.topics.service:app`，8100 端口，常驻 torch，**不连数据库**）；主应用零 ML 依赖，只负责统计、代理触发批跑、挂 `/topics/` 页面。

**Tech Stack:** Python 3.12 / PyTorch 2.9.1 (cu128) / transformers / `hfl/chinese-roberta-wwm-ext` / FastAPI 0.143 / SQLAlchemy 2.1 / pytest-asyncio / 纯静态 HTML+CSS+JS。

**Spec:** `MemHelp-code/docs/superpowers/specs/2026-10-25-ch10-topic-classifier-design.md`

---

## 运行方式（本仓库固定写法）

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
$uv = "C:\Users\dbq\.local\bin\uv.exe"
& $uv run --directory MemHelp-code python -m pytest tests/test_topics_taxonomy.py -q --color=no
```

- 工作目录一律 `MemHelp-code/`；**所有代码只写在这下面**。
- 测试**禁止** `tmp_path` / `tempfile`：沙箱在进程退出后收回新建目录的权限（WinError 5）。需要落盘时写进 `tests/` 下已存在的目录，文件名前缀 `_tmp_`，`yield` 后 `unlink(missing_ok=True)`（照 `tests/test_knowledge_ingest.py:88-96`）。
- 含中文的 `.ps1` 必须存成 UTF-8 with BOM（PowerShell 5.1 按 GBK 读）。改中文源文件用编辑工具，不要 `Set-Content`。

## Global Constraints

- **17 类，顺序即编号**：退换货(1)、物流(2)、尺码(3)、发票(4)、质量问题(5)、运费(6)、优惠活动(7)、价保(8)、支付(9)、订单修改(10)、库存补货(11)、商品信息(12)、保修维修(13)、账号(14)、会员积分(15)、评价(16)、其他(17)。
- **近邻边界**：修→保修维修、退→退换货；运费管钱、物流管货；价保补差价、优惠活动管券和满减。
- **多标签口径**：只打字面出现的诉求；同义合并（退/换/退款同属 1 类）；「其他」独占不与其他类共现；标签按编号升序输出；每条带 `rationale`。
- **数据量**：合成 1500 条（25% 双/三诉求）；迭代分层 80/10/10；增强只扩 train；真实池 27 行去重后 12 条**只进 test**。
- **训练超参（定死）**：`max_length=64`、`batch_size=16`（OOM 降 8 + 累积 2）、bf16 autocast、AdamW `lr=2e-5` `weight_decay=0.01`、warmup 10% + 余弦、`epochs<=8`、早停 `patience=2`（盯 val macro-F1@0.5）、梯度裁剪 1.0、dropout 0.1、seed 42。**全参微调，不用 LoRA**。
- **阈值**：只在 val 上按类扫 `0.2–0.7` 步长 `0.05`；测试集只报数、不调参。
- **端口**：主应用 8000、sidecar 8100。sidecar `GET /health`、`POST /classify`（单请求封顶 64 条，返回全部 17 个分数）。
- **配置键（定死）**：`TOPIC_SERVICE_URL=http://127.0.0.1:8100`、`TOPIC_MODEL_DIR=models/topics/latest`、`TOPIC_BATCH_MIN_PENDING=50`、`TOPIC_BATCH_LIMIT=100`、`TOPIC_SERVICE_TIMEOUT=30`。
- **落库**：`labels` JSON 存**中文类目名**数组；同一 `question_id` 重跑是**覆盖**（`UNIQUE uk_question_id`）；空预测兜底 `["其他"]`。
- **依赖边界**：`app/api.py` 及其路由链**永不 import torch/transformers**；只有 `app/llm.py` 能 import `langchain_openai`；只有 `app/config.py` 读配置；错误文本一律过 `redact()`。
- **不做**：LoRA/QLoRA、embedding/意图微调、定时器自动批跑、ONNX、`knowledge_chunks.category` 迁移、前端混淆矩阵。

## Review Focus

1. **空输入与纯噪声**：`""`、`"在吗"`、`"？"`、只含 `[PHONE]` 的串，清洗后可能只剩空串 —— 分类器不能崩，且必须给出一条合法标签（兜底「其他」）。
2. **sidecar 没起或超时**：主应用 `POST /api/topics/classify-batch` 必须返 **503 + 启动命令**，不能 500、不能静默 0 条、不能把异常吞成 200。
3. **同一批重跑**：`UNIQUE uk_question_id` 下第二次跑同一批，`processed` 可能是 0 而不是报错；`pending` 计算必须 `LEFT JOIN ... IS NULL`，不能因重复行把 pending 算多。
4. **类目表被改动**：`taxonomy.py` 顺序/内容变了而模型没重训 → sidecar 必须**拒绝启动**（哈希不符），不能静默把第 3 位当另一个类。
5. **「其他」与其他类共现**：`["尺码","其他"]` 这类输入必须被 `validate_labels` 拒绝，而不是写进库污染分布。
6. **17 类外的标签**：合成语料/大模型预标偶尔会输出「退款」「快递」这类自造词 —— 必须在解析处拦住并计入冲突表，不能落进 labels。

---

## 文件结构（先定边界，再拆任务）

```
app/topics/__init__.py          空包
app/topics/taxonomy.py          17 类单一来源 + 校验 + 哈希 + prompt 片段
app/topics/clean.py             清洗：全半角 / 脱敏 / 错别字 / 空白，逐条记 changes
app/topics/split.py             迭代分层 80/10/10
app/topics/augment.py           同义词替换 + 句式微调（只 train）
app/topics/synth.py             合成语料 prompt 构造 / 解析 / 预标 / 冲突检测
app/topics/training.py          数据集、目标张量、训练循环、早停、阈值扫描
app/topics/evaluate.py          micro/macro/每类 P-R-F1、2×2 混淆矩阵、类对排行、报告渲染
app/topics/infer.py             阈值化 + 兜底（纯 Python 可测），load() 才碰 torch
app/topics/service.py           sidecar FastAPI：/health /classify + 启动自检
app/topics/batch.py             批跑编排 run_batch()（API 与 CLI 共用）
app/topics/routes.py            主应用路由：/api/topics/distribution、/api/topics/classify-batch

scripts/build_topic_data.py     数据流水线 CLI（--from-db/--generate/--clean/--prelabel/--split/--augment/--all）
scripts/train_topics.py         训练 CLI
scripts/eval_topics.py          评测 CLI（出报告 + 多诉求专项 + 基线对比）
scripts/classify_topics.py      批跑 CLI

bank/typos.json                 错别字词典（source，进库）
evals/ch10_multilabel_cases.jsonl   20 条多诉求专项用例
evals/ch10_baseline.json            macro-F1 基线
web/topics/{index.html,style.css,app.js}   主题分布页（纯静态）

data/ 运行产物：taxonomy.json/taxonomy.md、raw/、clean/、labeled/、splits/、review/
models/topics/<run_id>/      六件产物
reports/topics/<run_id>/     report.md / metrics.json / per_class_prf.csv / confusion.html / error_review.csv
```

**为什么 `training.py` / `evaluate.py` 放在 `app/topics/` 而不是全塞进 `scripts/`**：spec §9.2 的接缝清单只列了 `infer/service/batch/routes`，但训练循环与指标计算是本章最需要单测的两块；放脚本里就只能整脚本跑，单测粒度退化成「端到端」。这里是**对 spec 的补充，不是偏离**：多两个模块，别处接口不变。

---

### Task 1: 环境 spike —— torch cu128 / sm_120 / 模型可下载 / 多标签 API 可用

这一步**不写产品代码**，只把三个可能让后面全盘返工的不确定性钉死。按用户工作要求第 1 条，非单测任务用「跑一遍真东西」代替 TDD。

**Files:**
- Modify: `MemHelp-code/pyproject.toml`（加 `[project.optional-dependencies] ml` + `[tool.uv.index]` + `[tool.uv.sources]`）
- Create: `MemHelp-code/app/topics/__init__.py`（空文件，只为了让下面 `python -c "import app.topics"` 成立）
- Modify: `MemHelp-code/dev-notes/ch10.md`（阶段 3 第一条追记）

**Interfaces:**
- Produces: 一个能 `import torch` 且 `torch.cuda.is_available() == True`、`torch.cuda.get_device_capability() == (12, 0)` 的 venv；`transformers` 的实际版本号（后续任务据此决定是否锁 v4）。

- [ ] **Step 1: 按 uv 官方写法把 torch 锁到 cu128 索引**

`Context7 /llmstxt/astral_sh_uv_llms_txt` 确认的写法是 `[[tool.uv.index]] explicit = true` + `[tool.uv.sources] torch = { index = "pytorch-cu128" }`。在 `pyproject.toml` 末尾加：

```toml
[project.optional-dependencies]
# ch10：主题分类器的 ML 依赖。主应用**不装这组**，靠 [ml] 隔离，
# 保证 app/api.py 那条链路永远不背 300MB 的 torch。
ml = [
    "torch==2.9.1",
    "transformers>=5.0",
    "safetensors>=0.8",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
# 只有 torch 走 cu128 索引：RTX 5060 是 sm_120，PyPI 的默认轮子是 cu126 及更早，
# 装上会在第一次前向时报 "no kernel image is available for execution on the device"。
torch = { index = "pytorch-cu128" }
```

- [ ] **Step 2: 装依赖**

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
& $uv sync --directory MemHelp-code --extra ml
```

Expected: 装完无报错；`torch==2.9.1+cu128`（`+cu128` 后缀是索引版本号，缺后缀说明装成了 CPU 版）。

- [ ] **Step 3: 验证 GPU 真的能算**

```powershell
& $uv run --directory MemHelp-code python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0));x=torch.randn(64,64,device='cuda');print((x@x).sum().item())"
```

Expected: `2.9.1+cu128 True (12, 0)` 和一个有限的浮点数。若 `cuda.is_available()` 为 False → 停下来问，不自作主张改 CPU 版（训练会慢一个量级，且违背需求里的「本地微调」隐含期望）。

- [ ] **Step 4: 模型能下载 + 多标签 API 可用**

`Context7` 的 transformers 文档确认多标签靠 `config.problem_type`，但**最新版是 5.20.2（v5 大版本）**，必须实跑验证而不是照记忆写：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
& $uv run --directory MemHelp-code python -c @"
import transformers, torch
from transformers import AutoModelForSequenceClassification, BertTokenizer
print('transformers', transformers.__version__)
tok = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
m = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=17, problem_type='multi_label_classification')
print('problem_type =', m.config.problem_type, '| hidden =', m.config.hidden_size, '| layers =', m.config.num_hidden_layers)
b = tok(['买大了想退，运费谁出'], padding=True, truncation=True, max_length=64, return_tensors='pt')
m.eval()
with torch.no_grad():
    out = m(**b)
print('logits', tuple(out.logits.shape))
"@
```

Expected: `problem_type = multi_label_classification`、`hidden = 768`、`layers = 12`、`logits (1, 17)`。
若 v5 报 `problem_type` 不被接受或 `BertTokenizer` 改名 → 在 `pyproject.toml` 把 `transformers>=5.0` 改成 `transformers>=4.57,<5`，重跑本步，并在 dev-notes 记下「v5 破了什么」。

- [ ] **Step 5: 记追记**

在 `dev-notes/ch10.md` 的「阶段 3 · 任务执行」下加第一条：实际版本号、GPU 型号与算力、模型下载走的是 hf-mirror 还是 huggingface、v5 是否踩坑、装包耗时。

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/pyproject.toml MemHelp-code/uv.lock MemHelp-code/app/topics/__init__.py MemHelp-code/dev-notes/ch10.md
git commit -m "chore(ch10): 锁定 cu128 的 torch 与 [ml] 可选依赖，验证 sm_120 可训练"
```

---

### Task 2: 类目表 `taxonomy.py` —— 全系统唯一来源

**Files:**
- Create: `app/topics/taxonomy.py`
- Test: `tests/test_topics_taxonomy.py`

**Interfaces:**
- Produces:
  - `OTHER_LABEL: str = "其他"`
  - `@dataclass(frozen=True, slots=True) class Category: index: int; name: str; boundary: str; triggers: tuple[str, ...]`
  - `CATEGORIES: tuple[Category, ...]`（长度 17，顺序即编号）
  - `CATEGORY_NAMES: tuple[str, ...]`、`LABEL_TO_INDEX: dict[str, int]`
  - `class TaxonomyError(ValueError)`
  - `validate_labels(labels: Iterable[str]) -> list[str]`（去重 → 校验在表内 → 「其他」独占 → 按编号升序）
  - `taxonomy_payload() -> dict[str, object]`
  - `taxonomy_hash(payload: dict[str, object] | None = None) -> str`（sha256 of `json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`）
  - `prompt_fragment() -> str`（给合成/预标 prompt 用的一段人话，含编号、类目、边界、触发词）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_taxonomy.py
import pytest
from app.topics.taxonomy import (
    CATEGORIES, CATEGORY_NAMES, LABEL_TO_INDEX, OTHER_LABEL, TaxonomyError,
    prompt_fragment, taxonomy_hash, taxonomy_payload, validate_labels,
)

EXPECTED = ("退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
            "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号",
            "会员积分", "评价", "其他")

def test_seventeen_categories_in_fixed_order():
    assert CATEGORY_NAMES == EXPECTED
    assert [c.index for c in CATEGORIES] == list(range(1, 18))
    assert LABEL_TO_INDEX == {name: i for i, name in enumerate(EXPECTED, start=1)}

def test_every_category_has_boundary_and_triggers():
    for c in CATEGORIES:
        assert len(c.boundary) >= 10, c.name          # 一句边界说明，不许留空
        assert len(c.triggers) >= 2, c.name
        assert c.name in c.boundary or c.name == OTHER_LABEL

def test_neighbour_boundaries_are_written_down():
    # 需求点名的四组近邻：修/退、运费/物流、价保/优惠活动
    assert "保修维修" in LABEL_TO_INDEX["保修维修"] and "修" in CATEGORIES[12].boundary
    assert "退" in CATEGORIES[0].boundary and "保修维修" in CATEGORIES[0].boundary
    assert "运费" in CATEGORIES[1].boundary and "钱" in CATEGORIES[1].boundary
    assert "便宜" in CATEGORIES[5].boundary or "钱" in CATEGORIES[5].boundary

def test_validate_dedupes_and_sorts_by_index():
    assert validate_labels(["运费", "退换货", "退换货"]) == ["退换货", "运费"]

def test_other_must_not_cooccur():
    with pytest.raises(TaxonomyError, match="其他"):
        validate_labels(["尺码", OTHER_LABEL])

def test_unknown_label_rejected():
    with pytest.raises(TaxonomyError, match="退款"):
        validate_labels(["退款"])

def test_empty_rejected():
    with pytest.raises(TaxonomyError):
        validate_labels([])

def test_hash_is_stable_across_calls():
    assert taxonomy_hash() == taxonomy_hash(taxonomy_payload())
    assert len(taxonomy_hash()) == 64

def test_hash_changes_when_boundary_changes():
    payload = taxonomy_payload()
    assert taxonomy_hash(payload) != taxonomy_hash({**payload, "categories": payload["categories"][:16]})

def test_prompt_fragment_covers_all_seventeen():
    text = prompt_fragment()
    for name in EXPECTED:
        assert name in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_taxonomy.py -q --color=no`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.topics.taxonomy'`

- [ ] **Step 3: 实现 `app/topics/taxonomy.py`**

17 个 `Category` 按 spec §4.1 的表逐字抄：`boundary` 用表格「边界说明」列 + 近邻归属那句；`triggers` 用「典型触发」列拆词。`validate_labels` 的实现要点（测试没钉但必须这么做）：先 `dict.fromkeys` 去重保序 → 逐个查 `LABEL_TO_INDEX` → 若含「其他」且 `len > 1` 抛 `TaxonomyError("「其他」独占，不可与其他类目共现")` → 最后按 index 排序。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_taxonomy.py -q --color=no`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/app/topics/taxonomy.py MemHelp-code/tests/test_topics_taxonomy.py
git commit -m "feat(ch10): 17 类权威类目表与多标签校验"
```

---

### Task 3: 清洗 `clean.py` —— 脱敏 / 错别字 / 格式，逐条留痕

**Files:**
- Create: `app/topics/clean.py`
- Create: `bank/typos.json`
- Test: `tests/test_topics_clean.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class Change: rule: str; before: str; after: str`
  - `@dataclass(frozen=True, slots=True) class CleanResult: text: str; changes: tuple[Change, ...]`
  - `clean_text(raw: str, *, typo_map: Mapping[str, str] | None = None) -> CleanResult`
  - `load_typo_map(path: Path | None = None) -> dict[str, str]`（默认 `PROJECT_ROOT/bank/typos.json`）
  - `is_noise(result: CleanResult) -> bool`（清洗后为空或只剩标点/占位符 → 丢弃）

  规则顺序（不可换）：①全角→半角（数字/字母/空格）②脱敏 ③错别字 ④空白压缩 ⑤首尾 strip。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_clean.py
from pathlib import Path
from app.topics.clean import clean_text, is_noise, load_typo_map

TYPOS = {"帐号": "账号", "收活": "收货", "褪回": "退回"}

def test_masks_phone_number():
    r = clean_text("我手机13812345678，麻烦联系我", typo_map=TYPOS)
    assert "[PHONE]" in r.text
    assert "13812345678" not in r.text
    assert any(c.rule == "mask_phone" for c in r.changes)

def test_masks_fullwidth_digits_before_masking():
    # 全角数字必须先转半角再脱敏，否则正则匹配不到 —— 这是最容易漏的一处
    r = clean_text("电话１３８１２３４５６７８", typo_map=TYPOS)
    assert "[PHONE]" in r.text and "１３８" not in r.text

def test_masks_idcard_order_email_address():
    r = clean_text("身份证110101199003071234，订单202410250001，邮箱a@b.com，北京市朝阳区建国路88号", typo_map=TYPOS)
    for ph in ("[IDCARD]", "[ORDER]", "[EMAIL]", "[ADDR]"):
        assert ph in r.text

def test_fixes_known_typos():
    r = clean_text("我的帐号登不上，还没收活", typo_map=TYPOS)
    assert "账号" in r.text and "收货" in r.text
    assert any(c.rule == "typo" for c in r.changes)

def test_normalizes_width_and_whitespace():
    r = clean_text("  ＡＢＣ１２３   你好  ", typo_map=TYPOS)
    assert r.text == "ABC123 你好"

def test_clean_is_idempotent():
    # 清洗两次必须不再产生 changes，否则批量重跑会不断改写语料
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_clean.py -q --color=no`
Expected: FAIL，`No module named 'app.topics.clean'`

- [ ] **Step 3: 实现**

`bank/typos.json` 至少含 `帐号→账号`、`收活→收货`、`褪回→退回`、`运单号→运单号` 之外的 20 组客服高频错字（`在吗`类不算）。脱敏正则按「先长后短」顺序：身份证 18 位 → 手机 11 位 → 邮箱 → 带前缀订单号（`订单号?|单号`）→ 地址（`省|市|区|县` + `路|街|道` + 门牌）。`is_noise` 用「去掉 `[\s\p{P}]` 与 5 个占位符后长度 < 1」判定（`str.translate` + `unicodedata.category` 逐字符判 `P*`）。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_clean.py -q --color=no`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/app/topics/clean.py MemHelp-code/bank/typos.json MemHelp-code/tests/test_topics_clean.py
git commit -m "feat(ch10): 语料清洗（脱敏/错字/格式）与逐条留痕"
```

---

### Task 4: 迭代分层抽样 `split.py`

**Files:**
- Create: `app/topics/split.py`
- Test: `tests/test_topics_split.py`

**Interfaces:**
- Consumes: `taxonomy.validate_labels`
- Produces:
  - `@dataclass(frozen=True, slots=True) class SplitResult: train: list[dict]; val: list[dict]; test: list[dict]; report: dict[str, object]`
  - `iterative_stratify(rows: Sequence[dict], *, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42, real_rows: Sequence[dict] = ()) -> SplitResult`

  `row` 形状：`{"id": str, "text": str, "labels": list[str], "source": "synthetic"|"real"}`。
  `report` 形状：`{"total": int, "sizes": {"train": int, "val": int, "test": int}, "per_label": {label: {"train": int, "val": int, "test": int}}, "zero_in_split": [{"label": str, "split": str}], "strata_count": int, "real_in_test": int}`。

  **分层单位是完整标签组合**（`frozenset(labels)` 排序后 join），不是单标签 —— 单标签分层会把「尺码+退换货」这类组合切碎，导致多标签样本在 train/test 分布不一致。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_split.py
import random
from app.topics.split import iterative_stratify

def _rows(n=200):
    rng = random.Random(7)
    pool = [["退换货"], ["物流"], ["尺码"], ["运费"], ["尺码", "退换货"], ["物流", "运费"]]
    return [{"id": f"q{i}", "text": f"问题{i}", "labels": rng.choice(pool), "source": "synthetic"}
            for i in range(n)]

def test_sizes_follow_ratios():
    r = iterative_stratify(_rows())
    total = len(r.train) + len(r.val) + len(r.test)
    assert total == 200
    assert 0.76 <= len(r.train) / total <= 0.84
    assert 0.07 <= len(r.val) / total <= 0.13
    assert 0.07 <= len(r.test) / total <= 0.13

def test_no_row_appears_in_two_splits():
    r = iterative_stratify(_rows())
    ids = [x["id"] for x in r.train + r.val + r.test]
    assert len(ids) == len(set(ids)) == 200

def test_same_seed_is_deterministic():
    a, b = iterative_stratify(_rows()), iterative_stratify(_rows())
    assert [x["id"] for x in a.test] == [x["id"] for x in b.test]

def test_report_counts_match_actual():
    r = iterative_stratify(_rows())
    assert r.report["sizes"] == {"train": len(r.train), "val": len(r.val), "test": len(r.test)}
    for label, counts in r.report["per_label"].items():
        for split in ("train", "val", "test"):
            assert counts[split] == sum(1 for x in getattr(r, split) if label in x["labels"])

def test_rare_combo_all_kept_in_train():
    # 只出现 1 次的标签组合不足以切片，必须整组落 train，不能凭空造出 val/test 样本
    rows = _rows(60) + [{"id": "rare", "text": "少见", "labels": ["评价", "会员积分"], "source": "synthetic"}]
    r = iterative_stratify(rows)
    assert "rare" in {x["id"] for x in r.train}

def test_real_rows_forced_into_test():
    real = [{"id": "r1", "text": "真实问题", "labels": ["物流"], "source": "real"}]
    r = iterative_stratify(_rows(), real_rows=real)
    assert "r1" in {x["id"] for x in r.test}
    assert r.report["real_in_test"] == 1

def test_zero_in_split_is_reported():
    r = iterative_stratify(_rows(40))
    assert isinstance(r.report["zero_in_split"], list)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.split'`

- [ ] **Step 3: 实现**

算法：按 `frozenset(labels)` 分组 → 组内 `rng.shuffle` → 需要 val/test 的份额 `max(1, round(n * ratio))`，但**组样本数 < 3 时整组进 train**（否则 val/test 里出现单条组合，训练分布失真）→ 剩余按「最大余数法」补齐到目标规模 → 每类各 split 计数写进 `report["per_label"]`，其中为 0 的进 `zero_in_split`。

- [ ] **Step 4: 跑测试确认通过** → 7 passed

- [ ] **Step 5: 评测前要人看的数（替代 TDD 的验证步）**

真实跑 `python scripts/build_topic_data.py --split` 之后，用下面这条把 `strata_report.json` 的每类计数打出来，人工确认「每类在 test 里都有 ≥10 条」；不够的类目记进 dev-notes，按 spec §11 决定「补造该类样本」还是「合并进其他」：

```powershell
& $uv run --directory MemHelp-code python -c "import json;d=json.load(open('data/splits/strata_report.json',encoding='utf-8'));[print(k,v) for k,v in sorted(d['per_label'].items())]"
```

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/app/topics/split.py MemHelp-code/tests/test_topics_split.py
git commit -m "feat(ch10): 多标签迭代分层抽样 80/10/10"
```

---

### Task 5: 数据增强 `augment.py` —— 只扩训练集

**Files:**
- Create: `app/topics/augment.py`
- Test: `tests/test_topics_augment.py`

**Interfaces:**
- Produces:
  - `SYNONYM_GROUPS: tuple[tuple[str, ...], ...]`（每组内可互换，例如 `("想退","要退","能退吗","申请退款")`）
  - `augment_rows(rows: Sequence[dict], *, copies: int = 1, seed: int = 42) -> list[dict]`
    - 返回：原行（**不带** `aug_from`）+ 增强行（带 `aug_from=<原 id>`），`labels` 与原行逐字相同
  - `synonym_replace(text: str, rng: random.Random) -> str | None`
  - `reshape(text: str, rng: random.Random) -> str`（句式微调：加「请问/麻烦问下」前缀、句末加「呢/啊」、去掉结尾问号再补）

  **函数签名只接受 train 行** —— 防污染不是靠调用方自觉，是靠接口形状：`augment_rows` 没有 split 参数，val/test 根本进不来。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_augment.py
from app.topics.augment import augment_rows, reshape, synonym_replace
import random

ROW = {"id": "t1", "text": "买大了想退", "labels": ["尺码", "退换货"], "source": "synthetic"}

def test_originals_are_preserved_untouched():
    out = augment_rows([ROW], copies=1)
    originals = [r for r in out if "aug_from" not in r]
    assert originals == [ROW]

def test_augmented_keep_same_labels_and_point_back():
    out = augment_rows([ROW], copies=2)
    aug = [r for r in out if "aug_from" in r]
    assert len(aug) == 2
    assert all(r["labels"] == ROW["labels"] and r["aug_from"] == "t1" for r in aug)
    assert all(r["id"] != "t1" for r in aug)

def test_augmented_text_actually_differs():
    out = augment_rows([ROW], copies=1, seed=3)
    aug = [r for r in out if "aug_from" in r][0]
    assert aug["text"] != ROW["text"]

def test_placeholders_are_never_replaced():
    rng = random.Random(0)
    for _ in range(50):
        s = synonym_replace("我的订单 [ORDER] 到哪了", rng)
        assert "[ORDER]" in s

def test_synonym_replace_returns_none_when_nothing_to_do():
    assert synonym_replace("嗯", random.Random(0)) is None

def test_reshape_is_idempotent_on_prefix():
    once = reshape("买大了想退", random.Random(1))
    twice = reshape(once, random.Random(1))
    assert not twice.startswith("请问请问")

def test_determinism_by_seed():
    a = augment_rows([ROW], copies=2, seed=11)
    b = augment_rows([ROW], copies=2, seed=11)
    assert [r["text"] for r in a] == [r["text"] for r in b]

def test_output_ids_are_unique():
    out = augment_rows([ROW], copies=3)
    assert len({r["id"] for r in out}) == len(out)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.augment'`

- [ ] **Step 3: 实现**

`SYNONYM_GROUPS` 至少 12 组，覆盖 17 类高频说法（退/换/退款、邮费/运费/快递费、到哪了/几天到、能开票/开专票、偏大/偏小/买大了…）。`synonym_replace` 只在**未脱敏片段**上做替换：先把 5 个占位符挖出来（`re.split` 保留分隔符），只对普通片段替换。`reshape` 前缀只加一次（先判 `startswith`）。

- [ ] **Step 4: 跑测试确认通过** → 8 passed

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/app/topics/augment.py MemHelp-code/tests/test_topics_augment.py
git commit -m "feat(ch10): 只扩训练集的数据增强（同义替换 + 句式微调）"
```

---

### Task 6: 合成语料与预标 `synth.py`

**Files:**
- Create: `app/topics/synth.py`
- Test: `tests/test_topics_synth.py`

**Interfaces:**
- Consumes: `taxonomy.prompt_fragment`, `taxonomy.validate_labels`, `taxonomy.TaxonomyError`
- Produces:
  - `@dataclass(frozen=True, slots=True) class ParsedQuestion: text: str; labels: list[str]`
  - `@dataclass(frozen=True, slots=True) class ParsedLabel: text: str; labels: list[str]; rationale: str`
  - `@dataclass(frozen=True, slots=True) class Conflict: text: str; generated: list[str]; prelabeled: list[str]`
  - `generation_messages(category: Category, count: int, *, second_label: str | None) -> list[dict[str, str]]`
  - `parse_generated(raw: str) -> list[ParsedQuestion]`
  - `labeling_messages(texts: Sequence[str]) -> list[dict[str, str]]`
  - `parse_labeled(raw: str) -> list[ParsedLabel]`
  - `find_conflicts(generated, prelabeled) -> list[Conflict]`

  模型输出约定为一行一个 JSON 对象（`{"text": "...", "labels": [...]}`），解析器要能吃下 ```json 围栏。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_synth.py
import pytest
from app.topics.taxonomy import CATEGORIES, TaxonomyError
from app.topics.synth import (
    Conflict, find_conflicts, generation_messages, labeling_messages,
    parse_generated, parse_labeled,
)

FENCED = """```json
{"text": "买大了想退，运费谁出", "labels": ["尺码", "退换货", "运费"]}
{"text": "快递到哪了", "labels": ["物流"]}
```"""

def test_generation_prompt_carries_the_whole_taxonomy():
    msgs = generation_messages(CATEGORIES[0], 12, second_label="尺码")
    joined = "\n".join(m["content"] for m in msgs)
    assert "退换货" in joined and "其他" in joined and "保修维修" in joined
    assert "12" in joined and "尺码" in joined
    assert msgs[0]["role"] == "system"

def test_parse_generated_strips_fences():
    out = parse_generated(FENCED)
    assert [p.text for p in out] == ["买大了想退，运费谁出", "快递到哪了"]
    assert out[0].labels == ["尺码", "退换货", "运费"]

def test_parse_generated_rejects_unknown_label():
    with pytest.raises(TaxonomyError):
        parse_generated('{"text": "想退款", "labels": ["退款"]}')

def test_parse_generated_skips_malformed_lines_but_keeps_rest():
    raw = '干扰行\n{"text": "快递到哪了", "labels": ["物流"]}\n{坏 json'
    assert len(parse_generated(raw)) == 1

def test_labeling_prompt_lists_texts_with_ids():
    msgs = labeling_messages(["买大了想退", "发票能开吗"])
    joined = "\n".join(m["content"] for m in msgs)
    assert "买大了想退" in joined and "发票能开吗" in joined
    assert "rationale" in joined

def test_parse_labeled_requires_rationale():
    raw = '{"text": "买大了想退", "labels": ["尺码"], "rationale": "字面提到买大了"}'
    out = parse_labeled(raw)
    assert out[0].rationale == "字面提到买大了"

def test_find_conflicts_compares_sets_ignoring_order():
    g = parse_generated('{"text": "a", "labels": ["尺码", "退换货"]}')
    p = parse_labeled('{"text": "a", "labels": ["退换货", "尺码"], "rationale": "x"}')
    assert find_conflicts(g, p) == []

def test_find_conflicts_flags_real_difference():
    g = parse_generated('{"text": "a", "labels": ["尺码", "退换货"]}')
    p = parse_labeled('{"text": "a", "labels": ["尺码"], "rationale": "x"}')
    assert find_conflicts(g, p) == [Conflict(text="a", generated=["尺码", "退换货"], prelabeled=["尺码"])]
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.synth'`

- [ ] **Step 3: 实现**

`generation_messages` 的 system 段 = `prompt_fragment()` + spec §4.2 四条口径 + 商品词表（记忆棉靠枕 A57 / 收纳箱 F98 / 巡检仪 / 门禁与探头）+ 口语化约束（8–40 字、允许省略主语、不许用书面语「请问您」）。user 段给「本类目 X、数量 N、25% 叠加 `second_label` 的诉求」。

- [ ] **Step 4: 跑测试确认通过** → 9 passed

- [ ] **Step 5: 用 20 条真样本验证质量（替代 TDD）**

拿真 LLM 各跑一次 `--generate --count 20` 与 `--prelabel`，把结果贴进 dev-notes，逐条看三件事：①有没有 17 类外的自造标签 ②多诉求句的标签是否「一个不多一个不少」③口语化程度像不像真实用户。不合格就改 prompt 重跑，**这一步不通过不许进 Task 7**。

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/app/topics/synth.py MemHelp-code/tests/test_topics_synth.py
git commit -m "feat(ch10): 合成语料与预标 prompt 的构造与解析"
```

---

### Task 7: 数据流水线 CLI `scripts/build_topic_data.py`

**Files:**
- Create: `scripts/build_topic_data.py`
- Test: `tests/test_topics_build_data.py`

**Interfaces:**
- Consumes: `clean.*`、`split.iterative_stratify`、`augment.augment_rows`、`synth.*`、`taxonomy.*`
- Produces（CLI）：
  - `--from-db` → `data/topics/raw/real.jsonl`（读 `low_confidence_questions.raw_question`，精确去重 + 清洗后去重）
  - `--generate --count 1500` → `data/topics/raw/synthetic.jsonl`
  - `--clean` → `data/topics/clean/{real,synthetic}.jsonl`（带 `text_raw` 与 `changes`）
  - `--prelabel` → `data/topics/labeled/synthetic.jsonl`（带 `rationale`）+ `data/topics/review/label_conflicts.csv`
  - `--split` → `data/topics/splits/{train,val,test}.jsonl` + `strata_report.json`
  - `--augment` → `data/topics/splits/train_aug.jsonl`
  - `--export-taxonomy` → `data/topics/taxonomy.json` + `taxonomy.md`
  - `--all` 依次跑全部（`--from-db`/`--generate` 需 `--allow-llm` 才执行，避免误触 API）
- 每个子命令结束打印一行 JSON 摘要（条数、丢弃数、告警数）。

- [ ] **Step 1: 写失败测试**（用 fixture 文件 + 假 llm，不打网络）

```python
# tests/test_topics_build_data.py
import json
from pathlib import Path
from scripts.build_topic_data import build_dataset, export_taxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures_topics"

def test_clean_filters_noise_and_dedupes(tmp_path_factory=None):
    out = build_dataset(FIXTURES, steps=("clean", "split"), llm=None, seed=1)
    assert out["clean"]["kept"] > 0
    assert out["clean"]["noise_dropped"] >= 1

def test_split_writes_three_files_and_report():
    out = build_dataset(FIXTURES, steps=("clean", "split"), llm=None, seed=1)
    for name in ("train", "val", "test"):
        assert Path(out["dirs"]["splits"], f"{name}.jsonl").is_file()
    report = json.loads(Path(out["dirs"]["splits"], "strata_report.json").read_text(encoding="utf-8"))
    assert report["sizes"]["train"] > 0

def test_augment_only_touches_train():
    out = build_dataset(FIXTURES, steps=("clean", "split", "augment"), llm=None, seed=1)
    train_aug = [json.loads(l) for l in Path(out["dirs"]["splits"], "train_aug.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(l) for l in Path(out["dirs"]["splits"], "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("aug_from" in r for r in train_aug)
    assert all("aug_from" not in r for r in val_rows)

def test_export_taxonomy_writes_json_and_markdown():
    out = export_taxonomy(FIXTURES / "out")
    assert Path(out, "taxonomy.json").is_file()
    md = Path(out, "taxonomy.md").read_text(encoding="utf-8")
    assert md.count("|") > 17 * 3
```

`tests/fixtures_topics/` 放 3 个小文件：`raw_synthetic.jsonl`（12 行，含 2 行噪声 `""` / `"？？"`、1 行含手机号、1 行含错别字）、`raw_real.jsonl`（3 行）、`typos.json`。**这个目录里的中文必须 UTF-8 无 BOM。**

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'scripts.build_topic_data'`

- [ ] **Step 3: 实现**（`--from-db` 走 `app.db.session.create_engine_from` + `TopicClassificationRepository` 所属的 session；`--generate` / `--prelabel` 走 `app.llm.get_chat_model`，一次 20 条、失败重试 1 次、进度打 stderr）

- [ ] **Step 4: 跑测试确认通过** → 4 passed

- [ ] **Step 5: 真跑一次（数据类任务的验证步）**

```powershell
& $uv run --directory MemHelp-code python scripts/build_topic_data.py --from-db --export-taxonomy --clean --split --augment
```

Expected 摘要：`real=12 kept=12`、`zeros_in_split=[]` 或告警清单。把摘要贴进 dev-notes。

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/scripts/build_topic_data.py MemHelp-code/tests/test_topics_build_data.py MemHelp-code/tests/fixtures_topics MemHelp-code/data/topics/taxonomy.json MemHelp-code/data/topics/taxonomy.md
git commit -m "feat(ch10): 数据流水线 CLI（捞池/清洗/分层/增强/导出术语表）"
```

---

### Task 8: 测试集人工标注（无代码，产出 gold）

**Files:**
- Create: `data/topics/splits/test_gold.jsonl`（每条含 `labels` + `rationale` + `labeled_by: "agent"`）
- Create: `data/topics/review/test_gold_review.csv`（列：`id,text,my_labels,my_rationale,user_verdict,user_note`）
- Modify: `data/topics/splits/test.jsonl`（用 gold 覆盖 `labels`）

**Interfaces:**
- Produces: 一个 `test.jsonl`，其 `labels` 全部来自人工（我 + 用户抽审），大模型预标**只作为参考列** `labels_prelabeled` 保留。

- [ ] **Step 1: 把 test 的 150 条导出成待标表**

```powershell
& $uv run --directory MemHelp-code python -c @"
import json, csv, pathlib
rows = [json.loads(l) for l in pathlib.Path('data/topics/splits/test.jsonl').read_text(encoding='utf-8').splitlines()]
with pathlib.Path('data/topics/review/test_gold_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['id','text','my_labels','my_rationale','user_verdict','user_note'])
    for r in rows: w.writerow([r['id'], r['text'], '', '', '', ''])
print(len(rows))
"@
```

- [ ] **Step 2: 逐条标注**

对 150 条逐行填 `my_labels`（用 `|` 分隔）与 `my_rationale`（一句，写命中了哪条边界规则）。硬要求：**不许看 `labels_prelabeled` 再抄** —— 顺序上先把 150 条独立标完，再回头看冲突，冲突逐条裁决并记下裁决理由。这条是 spec D11 的全部意义所在。

- [ ] **Step 3: 与预标对比，看分歧集中在哪里**

```powershell
& $uv run --directory MemHelp-code python -c @"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path('data/topics/splits/test.jsonl').read_text(encoding='utf-8').splitlines()]
diff = [r for r in rows if set(r['labels']) != set(r.get('labels_prelabeled', []))]
print('分歧', len(diff), '/', len(rows))
for r in diff[:15]: print(r['id'], r['text'], '| mine:', r['labels'], '| pre:', r.get('labels_prelabeled'))
"@
```

Expected: 分歧率 10–25% 属正常；>40% 说明预标 prompt 或边界规则有问题，**停下来改 §4.1 再重标**。

- [ ] **Step 4: 请用户抽审 ≥30 条**

把 `test_gold_review.csv` 交给用户，请其在 `user_verdict` 填 `同意/不同意`、`user_note` 写理由。用户改过的条目以用户为准写入 `test.jsonl`，报告脚注注明「用户抽审 N 条，修正 M 条」。

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/data/topics/splits/test.jsonl MemHelp-code/data/topics/splits/test_gold.jsonl MemHelp-code/data/topics/review/test_gold_review.csv
git commit -m "data(ch10): 测试集 150 条人工 gold 标注与抽审表"
```

> **这是全章唯一必须用户参与的步骤。** 后续 Task 9 起的评测数字全部建立在它之上，标不完就不要往下跑训练（可以先做 Task 8 之外的 DB/服务任务）。

---

### Task 9: ORM 模型与仓储

**Files:**
- Modify: `app/db/models.py`（末尾加 `TopicClassification`）
- Modify: `app/db/repositories.py`（加 `TopicClassRow` / `TopicCount` / `PendingQuestionRow` / `TopicClassificationRepository` / `Repositories` 门面属性）
- Modify: `tests/test_db_models.py:13`（十二表 → 十三表）
- Test: `tests/test_topics_repository.py`

**Interfaces:**
- Produces:
  - `class TopicClassification(Base)`，`__tablename__ = "topic_classifications"`，字段严格对齐 `sql/ch10-ddl.sql`：`id`（自增 PK）、`question_id`（BigInteger, FK→`low_confidence_questions.id`, `ondelete="CASCADE"`）、`labels: Mapped[list]`（JSON, not null）、`classified_at: Mapped[datetime]`（not null, `server_default=func.now()`），`UniqueConstraint("question_id", name="uk_question_id")`。
  - `@dataclass(frozen=True) class PendingQuestionRow: id: int; raw_question: str; created_at: datetime`
  - `@dataclass(frozen=True) class TopicCount: label: str; count: int`
  - `class TopicClassificationRepository`：
    - `def __init__(self, session: Session) -> None`
    - `async def list_pending(self, limit: int) -> list[PendingQuestionRow]`
    - `async def count_pending(self) -> int`
    - `async def upsert_many(self, rows: Sequence[tuple[int, list[str]]], *, classified_at: datetime) -> int`（返回写入行数）
    - `async def label_counts(self) -> list[TopicCount]`
    - `async def total_classified(self) -> int`
    - `async def multi_label_count(self) -> int`
    - `async def last_classified_at(self) -> datetime | None`
  - `Repositories.topics: TopicClassificationRepository`（`from_session` 一并装配）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_repository.py
from datetime import datetime
from sqlalchemy import select
from app.db.models import LowConfidenceQuestion, TopicClassification
from app.db.repositories import Repositories

NOW = datetime(2026, 10, 25, 12, 0, 0)

async def _seed(session, n=3):
    session.add_all([LowConfidenceQuestion(raw_question=f"问题{i}", source="retrieval_low_conf") for i in range(n)])
    session.commit()
    return list(session.scalars(select(LowConfidenceQuestion.id)).all())

async def test_pending_lists_unclassified_only(sqlite_session):
    ids = await _seed(sqlite_session)
    repo = Repositories.from_session(sqlite_session).topics
    assert [p.id for p in await repo.list_pending(10)] == ids
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=NOW)
    assert [p.id for p in await repo.list_pending(10)] == ids[1:]
    assert await repo.count_pending() == 2

async def test_upsert_twice_overwrites_not_duplicates(sqlite_session):
    ids = await _seed(sqlite_session, 1)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=NOW)
    await repo.upsert_many([(ids[0], ["退换货", "运费"])], classified_at=NOW)
    rows = list(sqlite_session.scalars(select(TopicClassification)).all())
    assert len(rows) == 1 and rows[0].labels == ["退换货", "运费"]

async def test_label_counts_counts_each_label(sqlite_session):
    ids = await _seed(sqlite_session, 3)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many([(ids[0], ["退换货", "运费"]), (ids[1], ["退换货"]), (ids[2], ["其他"])], classified_at=NOW)
    counts = {c.label: c.count for c in await repo.label_counts()}
    assert counts == {"退换货": 2, "运费": 1, "其他": 1}
    assert await repo.total_classified() == 3
    assert await repo.multi_label_count() == 1

async def test_last_classified_at_is_max(sqlite_session):
    ids = await _seed(sqlite_session, 2)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.upsert_many([(ids[0], ["物流"])], classified_at=datetime(2026, 1, 1))
    await repo.upsert_many([(ids[1], ["物流"])], classified_at=datetime(2026, 5, 1))
    assert await repo.last_classified_at() == datetime(2026, 5, 1)

async def test_empty_table_returns_none_and_zero(sqlite_session):
    repo = Repositories.from_session(sqlite_session).topics
    assert await repo.last_classified_at() is None
    assert await repo.total_classified() == 0
    assert await repo.count_pending() == 0

async def test_upsert_many_wraps_in_thread(sqlite_session, monkeypatch):
    # 仓库层约定：同步 DB 调用必须经 anyio.to_thread，否则会阻塞事件循环
    from app.db import repositories as mod
    calls = []
    real = mod.anyio.to_thread.run_sync
    async def spy(fn, *a, **kw):
        calls.append(fn)
        return await real(fn, *a, **kw)
    monkeypatch.setattr(mod.anyio.to_thread, "run_sync", spy)
    repo = Repositories.from_session(sqlite_session).topics
    await repo.count_pending()
    assert calls
```

同时把 `tests/test_db_models.py:13` 的 `test_metadata_has_exactly_twelve_tables` 改名 `test_metadata_has_exactly_thirteen_tables`，断言集合加 `"topic_classifications"`。

- [ ] **Step 2: 跑测试确认失败** → FAIL，`ImportError: cannot import name 'TopicClassification'`

- [ ] **Step 3: 实现**

模型照 `app/db/models.py` 现有风格（`Mapped[...]` + `_AUTOINC_PK` + `JSON`）。仓储照 `EvalRunRepository`（`_xxx` 同步私有方法 + `async def xxx` 包 `_run`）。`list_pending` 用 `select(...).outerjoin(TopicClassification, TopicClassification.question_id == LowConfidenceQuestion.id).where(TopicClassification.id.is_(None)).order_by(LowConfidenceQuestion.id).limit(limit)`。`upsert_many` 单事务：一次 `select` 取已有 `question_id` 集合 → 分 insert / update → 一次 `commit()`。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_repository.py tests/test_db_models.py -q --color=no`
Expected: 7 + 全部 model 测试 passed

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/app/db/models.py MemHelp-code/app/db/repositories.py MemHelp-code/tests/test_topics_repository.py MemHelp-code/tests/test_db_models.py MemHelp-code/sql/ch10-ddl.sql
git commit -m "feat(ch10): topic_classifications 的 ORM 模型与仓储（pending/upsert/计数）"
```

---

### Task 10: 训练 `training.py` + `scripts/train_topics.py`

**Files:**
- Create: `app/topics/training.py`
- Create: `scripts/train_topics.py`
- Test: `tests/test_topics_training.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class TrainConfig: model_name: str = "hfl/chinese-roberta-wwm-ext"; max_length: int = 64; batch_size: int = 16; grad_accum: int = 1; lr: float = 2e-5; weight_decay: float = 0.01; warmup_ratio: float = 0.1; epochs: int = 8; patience: int = 2; grad_clip: float = 1.0; seed: int = 42; label_smoothing: float = 0.0`
  - `def multilabel_targets(labels: Sequence[Sequence[str]], label_names: Sequence[str]) -> list[list[float]]`（**纯 Python**，不依赖 torch，必须可单测）
  - `def threshold_sweep(y_true: Sequence[Sequence[float]], y_prob: Sequence[Sequence[float]], label_names: Sequence[str], *, lo: float = 0.2, hi: float = 0.7, step: float = 0.05) -> dict[str, dict[str, float]]`（每类返回 `{"threshold"[, "f1"]}`）
  - `def macro_f1(y_true, y_pred_binary, label_names) -> float`、`def per_label_f1(...) -> dict[str, float]`
  - `def encode_batch(tokenizer, texts, max_length) -> dict[str, torch.Tensor]`（torch 相关，测试用 `importorskip`）
  - `def train_model(config, train_rows, val_rows, *, device="cuda", out_dir: Path, taxonomy_hash: str) -> dict`（返回 `metrics.json` 内容并落盘六件产物）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_training.py
import pytest
from app.topics.training import macro_f1, multilabel_targets, per_label_f1, threshold_sweep

LABELS = ["退换货", "物流", "尺码"]

def test_targets_are_multi_hot():
    assert multilabel_targets([["物流", "退换货"]], LABELS) == [[1.0, 1.0, 0.0]]

def test_targets_reject_unknown_label():
    with pytest.raises(ValueError):
        multilabel_targets([["发票"]], LABELS)

def test_macro_f1_hand_computed():
    # 类别 0：tp=1 fn=1 → P=1 R=0.5 F1=2/3；类别 1：全对 → 1.0；类别 2：tp=0 → 0
    y_true = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert per_label_f1(y_true, y_pred, LABELS) == pytest.approx(
        {"退换货": 0.666667, "物流": 0.666667, "尺码": 0.0}, abs=1e-4
    )
    assert macro_f1(y_true, y_pred, LABELS) == pytest.approx(0.444444, abs=1e-4)

def test_macro_f1_ignores_labels_with_no_support_and_no_prediction():
    # 全 0 的类目：P/R/F1 无定义，按 0 计，但不能让 0/0 抛异常
    assert per_label_f1([[1, 0, 0]], [[1, 0, 0]], LABELS)["尺码"] == 0.0

def test_threshold_sweep_picks_best_point_per_label():
    # 类别 1 的分数刚好跨在 0.45 两侧：0.4 阈值能捞到 tp，0.5 阈值捞不到
    y_true = [[0, 1, 0], [0, 0, 0]]
    y_prob = [[0.1, 0.45, 0.1], [0.1, 0.1, 0.1]]
    out = threshold_sweep(y_true, y_prob, LABELS)
    assert out["物流"]["threshold"] <= 0.45
    assert set(out) == set(LABELS)
    assert all(0.2 <= v["threshold"] <= 0.7 for v in out.values())

def test_threshold_sweep_is_deterministic():
    a = threshold_sweep([[1, 0, 0]], [[0.9, 0.1, 0.1]], LABELS)
    assert a == threshold_sweep([[1, 0, 0]], [[0.9, 0.1, 0.1]], LABELS)


torch = pytest.importorskip("torch")

def test_train_model_smoke_on_two_rows():
    # 不下载预训练权重：用一个 2 层的小 BertConfig 代替，只验训练循环能跑完、能存盘
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    from app.topics.training import TrainConfig, train_model
    cfg = TrainConfig(model_name="tiny", epochs=1, batch_size=2, max_length=16)
    ...
```

（最后那个 smoke 测试用 `BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=17, problem_type="multi_label_classification")` 本地构造模型与 tokenizer，注入 `train_model(..., model=..., tokenizer=...)`，断言 `out_dir` 下六个产物齐全且 `thresholds.json` 有 17 个键。）

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.training'`

- [ ] **Step 3: 实现 `app/topics/training.py`**

纯函数部分（`multilabel_targets` / `threshold_sweep` / `per_label_f1` / `macro_f1`）用标准库实现，**不 import torch**（`macro_f1` 收 `list[list[int]]`）。torch 只在 `encode_batch` / `train_model` 内部函数级 import，这样纯函数测试在没装 `[ml]` 的环境里也能跑。训练循环：`AdamW` + `get_cosine_schedule_with_warmup`（warmup = `int(total_steps * 0.1)`）+ `torch.autocast("cuda", dtype=torch.bfloat16)`（CUDA 不可用时退 `nullcontext`）+ `clip_grad_norm_(1.0)`；每 epoch 用 **0.5 阈值**算 val macro-F1，`patience=2` 不升则停；保留最优 epoch 的 `state_dict`。

- [ ] **Step 4: 跑测试确认通过** → 6 + 1 passed

- [ ] **Step 5: 实现 `scripts/train_topics.py`**

参数：`--train data/topics/splits/train_aug.jsonl --val data/topics/splits/val.jsonl --out models/topics --seed 42`。做四件事：读数据 → `train_model` → 在 **val** 上 `threshold_sweep` → 写 `models/topics/<run_id>/` 六件产物，并把 `models/topics/latest` 建成指向 `<run_id>` 的目录副本（Windows 无 symlink 权限，用 `shutil.copytree`）。

- [ ] **Step 6: 真训练一次（替代 TDD 的验证步）**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
& $uv run --directory MemHelp-code python scripts/train_topics.py --epochs 8 --batch-size 16
```

Expected: 8GB 显存下 `batch_size=16`、`max_length=64` 不 OOM；早停在第 3–6 轮触发；`metrics.json` 里 val macro-F1@0.5 ≥ 0.75。**若 OOM**：脚本自动降 8 + 累积 2 重试一次（写进日志），仍 OOM 才停下汇报。耗时、显存峰值、最优轮次、各类 val-F1 贴进 dev-notes。

- [ ] **Step 7: Commit**

```bash
git add MemHelp-code/app/topics/training.py MemHelp-code/scripts/train_topics.py MemHelp-code/tests/test_topics_training.py
git commit -m "feat(ch10): RoBERTa-wwm-ext 全参微调训练循环与每类阈值扫描"
```

---

### Task 11: 评测 `evaluate.py` + `scripts/eval_topics.py`

**Files:**
- Create: `app/topics/evaluate.py`
- Create: `scripts/eval_topics.py`
- Create: `evals/ch10_multilabel_cases.jsonl`（20 条）
- Create: `evals/ch10_baseline.json`
- Test: `tests/test_topics_evaluate.py`

**Interfaces:**
- Produces:
  - `def overall_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], label_names: Sequence[str]) -> dict[str, float]`（键：`micro_precision/micro_recall/micro_f1/macro_precision/macro_recall/macro_f1/weighted_f1/hamming_loss/exact_match`）
  - `def per_class_metrics(gold, pred, label_names) -> list[dict[str, object]]`（键：`label/precision/recall/f1/support/tp/fp/fn/tn`）
  - `def label_pair_confusions(gold, pred, label_names) -> list[dict[str, object]]`（`gold_label/pred_label/count`，按 count 降序，排除 gold==pred）
  - `def render_report_md(metrics, per_class, pairs, *, run_id, real_subset=None, attr=None) -> str`
  - `def render_confusion_html(gold, pred, label_names) -> str`（17 个 2×2 色块 + 类对排行表，**纯 HTML+CSS**）
  - `def render_error_review_csv(gold, pred, texts, label_names) -> str`
  - `def baseline_regression(current_macro_f1: float, baseline: float) -> str`（`"ok"` / `"regressed"`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_evaluate.py
import pytest
from app.topics.evaluate import (
    baseline_regression, label_pair_confusions, overall_metrics, per_class_metrics,
    render_confusion_html, render_report_md,
)

NAMES = ["退换货", "物流", "尺码"]
GOLD = [["退换货"], ["物流"], ["尺码", "退换货"]]
PRED = [["退换货"], ["物流"], ["尺码"]]

def test_overall_micro_and_macro():
    m = overall_metrics(GOLD, PRED, NAMES)
    # 4 个正例中命中 3 个 → micro P=R=F1=0.75
    assert m["micro_f1"] == pytest.approx(0.75)
    assert m["exact_match"] == pytest.approx(2 / 3)
    assert m["hamming_loss"] == pytest.approx(1 / 9)

def test_per_class_has_all_fields():
    rows = per_class_metrics(GOLD, PRED, NAMES)
    assert [r["label"] for r in rows] == NAMES
    assert rows[0]["tp"] == 2 and rows[0]["fn"] == 1 and rows[0]["support"] == 3
    assert rows[2]["tn"] == 1

def test_pair_confusions_exclude_correct():
    pairs = label_pair_confusions(GOLD, PRED, NAMES)
    assert pairs == [{"gold_label": "退换货", "pred_label": "尺码", "count": 1}]

def test_report_flags_small_support():
    md = render_report_md(overall_metrics(GOLD, PRED, NAMES), per_class_metrics(GOLD, PRED, NAMES),
                          [], run_id="r1")
    assert "样本不足" in md and "r1" in md

def test_confusion_html_is_self_contained():
    html = render_confusion_html(GOLD, PRED, NAMES)
    assert "<table" in html and "退换货" in html
    assert "http" not in html.split("<style>")[0]   # 不许引外部脚本/样式

def test_baseline_regression_threshold():
    assert baseline_regression(0.80, 0.82) == "ok"
    assert baseline_regression(0.78, 0.82) == "regressed"
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.evaluate'`

- [ ] **Step 3: 实现**

全部纯 Python（无 numpy 依赖，互斥计数手写）。`render_confusion_html` 用内联 `<style>` 的 `background: rgba(220,38,38,alpha)`，alpha 按 `count / max_count`。`render_report_md` 里 `support < 10` 的类目 F1 单元格加「（样本不足，仅供参考）」。

- [ ] **Step 4: 跑测试确认通过** → 6 passed

- [ ] **Step 5: 写 20 条多诉求专项用例**

`evals/ch10_multilabel_cases.jsonl`，每行 `{"id","text","labels"}`，覆盖每组近邻组合：`买大了想退，运费谁出`（尺码+退换货+运费）、`快递到哪了，还包邮吗`（物流+运费）、`刚买就降价了，券还能用吗`（价保+优惠活动）、`屏幕碎了保修吗`（质量问题+保修维修）、`地址写错了能改吗，不然就退了`（订单修改+退换货）、`开发票要税号吗，能开专票吗`（发票）等。**至少 12 条是 2 标签、4 条 3 标签**。

- [ ] **Step 6: 实现 `scripts/eval_topics.py` 并真跑**

```powershell
& $uv run --directory MemHelp-code python scripts/eval_topics.py --run-id latest
```

Expected: `reports/topics/<run_id>/` 下 5 个文件齐全；stdout 打印 `macro-F1`、`多标签完全命中率`、**`至少命中 2 个标签的比例`（目标 20/20）**。

- [ ] **Step 7: 人工抽判错 30 条（替代 TDD 的验证步）**

从 `error_review.csv` 每类最多取 3 条错例、优先假阴性，逐条判 `模型对/人标对/都错/歧义`，把归因分布写进 `report.md` 的「人工复核」段。若「边界争议」占比 >20% → 回改 spec §4.1 边界说明并重标测试集（这是 spec §7.2 的硬规则）。

- [ ] **Step 8: 写基线**

```powershell
& $uv run --directory MemHelp-code python -c "import json;m=json.load(open('reports/topics/latest/metrics.json',encoding='utf-8'));json.dump({'macro_f1':m['macro_f1'],'note':'ch10 首个基线','run_id':'latest'},open('evals/ch10_baseline.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)"
```

- [ ] **Step 9: Commit**

```bash
git add MemHelp-code/app/topics/evaluate.py MemHelp-code/scripts/eval_topics.py MemHelp-code/tests/test_topics_evaluate.py MemHelp-code/evals/ch10_multilabel_cases.jsonl MemHelp-code/evals/ch10_baseline.json
git commit -m "feat(ch10): 分类指标/混淆矩阵/报告渲染与多诉求专项评测"
```

---

### Task 12: 推理 `infer.py` —— 阈值化与兜底（纯 Python 可测）

**Files:**
- Create: `app/topics/infer.py`
- Test: `tests/test_topics_infer.py`

**Interfaces:**
- Produces:
  - `class TopicModelError(RuntimeError)`
  - `@dataclass(frozen=True, slots=True) class TopicPrediction: labels: list[str]; scores: dict[str, float]; fallback: bool`
  - `class TopicClassifier`：
    - `def __init__(self, scorer: Callable[[Sequence[str]], list[list[float]]], thresholds: Mapping[str, float], model_taxonomy_hash: str) -> None`
    - `@classmethod def from_parts(cls, scorer, thresholds, *, model_taxonomy_hash: str, current_hash: str | None = None) -> "TopicClassifier"`
    - `@classmethod def load(cls, model_dir: Path) -> "TopicClassifier"`（读 `taxonomy.json` / `thresholds.json` / 权重，构造真实 scorer）
    - `def predict(self, texts: Sequence[str]) -> list[TopicPrediction]`
  - 校验规则：`thresholds` 键集合必须等于 17 类名（否则 `TopicModelError`）；`model_taxonomy_hash != current_hash` 时（`current_hash` 非空）抛 `TopicModelError`；分数长度必须是 17。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_infer.py
import pytest
from app.topics.taxonomy import CATEGORY_NAMES, taxonomy_hash
from app.topics.infer import TopicClassifier, TopicModelError

def _scorer(fixed):
    return lambda texts: [fixed for _ in texts]

def _thresholds(value=0.5):
    return {name: value for name in CATEGORY_NAMES}

def _scores(**over):
    base = {name: 0.01 for name in CATEGORY_NAMES}
    base.update(over)
    return [base[name] for name in CATEGORY_NAMES]

def test_threshold_selects_multiple_labels_sorted():
    clf = TopicClassifier.from_parts(_scorer(_scores(尺码=0.9, 退换货=0.8, 运费=0.7)), _thresholds())
    p = clf.predict(["买大了想退，运费谁出"])[0]
    assert p.labels == ["退换货", "尺码", "运费"]      # 按类目编号升序，不是分数序
    assert p.fallback is False
    assert p.scores["尺码"] == pytest.approx(0.9)

def test_all_below_threshold_falls_back_to_other():
    clf = TopicClassifier.from_parts(_scorer(_scores()), _thresholds())
    p = clf.predict(["嗯"])[0]
    assert p.labels == ["其他"] and p.fallback is True

def test_per_label_threshold_is_respected():
    th = _thresholds()
    th["物流"] = 0.9
    clf = TopicClassifier.from_parts(_scorer(_scores(物流=0.6)), th)
    assert clf.predict(["到哪了"])[0].labels == ["其他"]

def test_other_is_dropped_when_another_label_fires():
    # 即使模型给了「其他」高分，只要还有别的类过阈值，「其他」必须让位（D10 独占）
    clf = TopicClassifier.from_parts(_scorer(_scores(其他=0.9, 物流=0.8)), _thresholds())
    assert clf.predict(["快递到哪了"])[0].labels == ["物流"]

def test_wrong_threshold_keys_raise():
    bad = {"物流": 0.5}
    with pytest.raises(TopicModelError, match="17"):
        TopicClassifier.from_parts(_scorer(_scores()), bad)

def test_taxonomy_hash_mismatch_raises():
    with pytest.raises(TopicModelError, match="taxonomy"):
        TopicClassifier.from_parts(_scorer(_scores()), _thresholds(),
                                   model_taxonomy_hash="deadbeef", current_hash=taxonomy_hash())

def test_scorer_output_shape_is_validated():
    with pytest.raises(TopicModelError):
        TopicClassifier.from_parts(lambda t: [[0.1, 0.2]], _thresholds()).predict(["x"])

def test_empty_text_still_returns_one_prediction():
    clf = TopicClassifier.from_parts(_scorer(_scores()), _thresholds())
    assert clf.predict([""])[0].labels == ["其他"]
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.infer'`

- [ ] **Step 3: 实现**

`predict` 是纯 Python：调 `scorer` → 校验每行 17 个分数 → 每类与自己的阈值比 → 得到候选集 →「其他」若与别人同时出现则丢弃 → 空集兜底 `["其他"], fallback=True` → `validate_labels` 兜底排序。`load()` 里才 import torch/transformers，并用 `sys.modules` 缓存判断「torch 已加载」以支持隔离测试。

- [ ] **Step 4: 跑测试确认通过** → 9 passed

- [ ] **Step 5: Commit**

```bash
git add MemHelp-code/app/topics/infer.py MemHelp-code/tests/test_topics_infer.py
git commit -m "feat(ch10): 多标签推理的阈值化与「其他」兜底"
```

---

### Task 13: sidecar 服务 `service.py`

**Files:**
- Create: `app/topics/service.py`
- Test: `tests/test_topics_service.py`

**Interfaces:**
- Consumes: `TopicClassifier`
- Produces:
  - `def create_app(*, classifier: TopicClassifier | None = None, model_dir: Path | None = None) -> FastAPI`
  - `GET /health` → `{"status": "ok", "run_id": str, "taxonomy_hash": str, "model_loaded": bool}`
  - `POST /classify` body `{"texts": [str]}` → `{"results": [{"labels": [...], "scores": {...}, "fallback": bool}]}`
  - 校验：`len(texts) == 0` → 200 `{"results": []}`；`len(texts) > 64` → 422；文案含 `redact()` 处理后的异常文本。
  - 启动自检（`lifespan`）：taxonomy 哈希、thresholds 17 键、权重可加载 —— 任一失败 **抛异常阻止启动**。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topics_service.py
import pytest
from fastapi.testclient import TestClient
from app.topics.taxonomy import CATEGORY_NAMES, taxonomy_hash
from app.topics.service import create_app

def _clf(**kw):
    from app.topics.infer import TopicClassifier
    base = {n: 0.01 for n in CATEGORY_NAMES}
    base.update(kw)
    return TopicClassifier.from_parts(lambda t: [[base[n] for n in CATEGORY_NAMES] for _ in t],
                                      {n: 0.5 for n in CATEGORY_NAMES},
                                      model_taxonomy_hash=taxonomy_hash())

def test_health_reports_model_state():
    with TestClient(create_app(classifier=_clf())) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["taxonomy_hash"] == taxonomy_hash()

def test_classify_returns_all_seventeen_scores():
    with TestClient(create_app(classifier=_clf(物流=0.9))) as c:
        r = c.post("/classify", json={"texts": ["快递到哪了"]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["labels"] == ["物流"]
    assert len(res["scores"]) == 17 and res["scores"]["物流"] == pytest.approx(0.9)

def test_empty_texts_is_ok_not_error():
    with TestClient(create_app(classifier=_clf())) as c:
        assert c.post("/classify", json={"texts": []}).json() == {"results": []}

def test_more_than_sixty_four_texts_is_422():
    with TestClient(create_app(classifier=_clf())) as c:
        assert c.post("/classify", json={"texts": ["x"] * 65}).status_code == 422

def test_scores_summary_hides_internal_paths(monkeypatch):
    def boom(texts): raise RuntimeError(r"加载失败 C:\Users\dbq\secret\w.bin")
    from app.topics.infer import TopicClassifier
    clf = TopicClassifier.from_parts(boom, {n: 0.5 for n in CATEGORY_NAMES}, model_taxonomy_hash=taxonomy_hash())
    with TestClient(create_app(classifier=clf)) as c:
        r = c.post("/classify", json={"texts": ["x"]})
    assert r.status_code == 500 and "secret" not in r.text
```

- [ ] **Step 2: 跑测试确认失败** → FAIL，`No module named 'app.topics.service'`

- [ ] **Step 3: 实现**（`Context7 /websites/fastapi_tiangolo` 确认：`@asynccontextmanager` + `FastAPI(lifespan=...)` 是当前推荐写法，`startup`/`shutdown` 事件已弃用；异常处理用 `@app.exception_handler`）

- [ ] **Step 4: 跑测试确认通过** → 5 passed

- [ ] **Step 5: 真启动一次**

```powershell
& $uv run --directory MemHelp-code uvicorn app.topics.service:app --port 8100
# 另一个窗口：
& $uv run --directory MemHelp-code python -c "import httpx;print(httpx.get('http://127.0.0.1:8100/health').json())"
```

Expected: `{'status': 'ok', 'run_id': ..., 'taxonomy_hash': ..., 'model_loaded': True}`。把启动耗时与显存占用贴进 dev-notes。

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/app/topics/service.py MemHelp-code/tests/test_topics_service.py
git commit -m "feat(ch10): 无状态旁路推理 sidecar（/health + /classify + 启动自检）"
```

---

### Task 14: 批跑编排 `batch.py` + 主应用路由 + 配置 + CLI

**Files:**
- Create: `app/topics/batch.py`
- Create: `app/topics/routes.py`
- Create: `scripts/classify_topics.py`
- Modify: `app/config.py`（ch10 分节 5 个字段）
- Modify: `.env.example`（`# ---- ch10：主题分类 ----` 分节）
- Modify: `app/api.py`（`register_topic_routes(app)` + `/topics` 静态挂载）
- Test: `tests/test_topics_batch.py`、`tests/test_topics_routes.py`、`tests/test_topics_isolation.py`

**Interfaces:**
- Produces:
  - `class TopicServiceUnavailable(RuntimeError)`
  - `@dataclass(frozen=True) class BatchOutcome: processed: int; fallback_other: int; elapsed_ms: int; pending_before: int; pending_after: int`
  - `async def run_batch(*, repositories: Repositories, classify: Callable[[Sequence[str]], Awaitable[Sequence[TopicPrediction]]], limit: int, now: datetime | None = None) -> BatchOutcome`
  - `def register_topic_routes(app: FastAPI) -> None`
  - `GET /api/topics/distribution` → `{total, by_label: [{label, count}], multi_label_share, pending, min_pending, last_classified_at}`
  - `POST /api/topics/classify-batch` body `{"limit": int | None}` → `{processed, fallback_other, elapsed_ms}`
  - `scripts/classify_topics.py --limit 27 [--service-url ...]`：走 HTTP 调主应用的 `/api/topics/classify-batch`，打印同样三个数

- [ ] **Step 1: 写失败测试（batch）**

```python
# tests/test_topics_batch.py
from datetime import datetime
import pytest
from sqlalchemy import select
from app.db.models import LowConfidenceQuestion, TopicClassification
from app.db.repositories import Repositories
from app.topics.batch import TopicServiceUnavailable, run_batch
from app.topics.infer import TopicPrediction

NOW = datetime(2026, 10, 25, 12, 0, 0)

def _pred(labels, fallback=False):
    return TopicPrediction(labels=labels, scores={}, fallback=fallback)

async def _seed(session, n=3):
    session.add_all([LowConfidenceQuestion(raw_question=f"问题{i}", source="retrieval_low_conf") for i in range(n)])
    session.commit()

async def test_batch_classifies_pending_and_reports(sqlite_session):
    await _seed(sqlite_session, 2)
    seen = []
    async def classify(texts):
        seen.extend(texts)
        return [_pred(["物流"]), _pred(["其他"], fallback=True)]
    out = await run_batch(repositories=Repositories.from_session(sqlite_session), classify=classify, limit=10, now=NOW)
    assert out.processed == 2 and out.fallback_other == 1
    assert out.pending_before == 2 and out.pending_after == 0
    assert sorted(seen) == ["问题0", "问题1"]
    rows = list(sqlite_session.scalars(select(TopicClassification)).all())
    assert {tuple(r.labels) for r in rows} == {("物流",), ("其他",)}
    assert all(r.classified_at == NOW for r in rows)

async def test_second_run_is_noop(sqlite_session):
    await _seed(sqlite_session, 2)
    async def classify(texts):
        return [_pred(["物流"]) for _ in texts]
    repo = Repositories.from_session(sqlite_session)
    await run_batch(repositories=repo, classify=classify, limit=10, now=NOW)
    out = await run_batch(repositories=repo, classify=classify, limit=10, now=NOW)
    assert out.processed == 0 and out.pending_after == 0

async def test_limit_is_respected(sqlite_session):
    await _seed(sqlite_session, 5)
    async def classify(texts):
        return [_pred(["物流"]) for _ in texts]
    out = await run_batch(repositories=Repositories.from_session(sqlite_session), classify=classify, limit=2, now=NOW)
    assert out.processed == 2 and out.pending_after == 3

async def test_service_error_becomes_domain_error(sqlite_session):
    await _seed(sqlite_session, 1)
    async def classify(texts):
        raise TopicServiceUnavailable("连不上 sidecar")
    with pytest.raises(TopicServiceUnavailable):
        await run_batch(repositories=Repositories.from_session(sqlite_session), classify=classify, limit=10, now=NOW)

async def test_empty_pending_is_not_an_error(sqlite_session):
    async def classify(texts):
        return []
    out = await run_batch(repositories=Repositories.from_session(sqlite_session), classify=classify, limit=10, now=NOW)
    assert out.processed == 0 and out.pending_before == 0
```

- [ ] **Step 2: 写失败测试（路由 + 隔离）**

```python
# tests/test_topics_routes.py
import httpx, pytest
from fastapi.testclient import TestClient

async def test_distribution_shape(sqlite_session, ...):   # 用 conftest 的离线装配 + fake repos
    ...
    body = r.json()
    assert set(body) == {"total", "by_label", "multi_label_share", "pending", "min_pending", "last_classified_at"}

def test_classify_batch_returns_503_with_command_when_sidecar_down(monkeypatch, ...):
    # monkeypatch httpx.AsyncClient.post → raise httpx.ConnectError("refused")
    r = client.post("/api/topics/classify-batch", json={"limit": 10})
    assert r.status_code == 503
    assert "uvicorn app.topics.service:app --port 8100" in r.json()["error"]["message"]

def test_batch_rejects_negative_limit(...):
    assert client.post("/api/topics/classify-batch", json={"limit": -1}).status_code == 422

def test_topics_page_is_mounted(...):
    assert client.get("/topics/").status_code == 200

# tests/test_topics_isolation.py
def test_api_does_not_import_torch():
    import app.api  # noqa: F401
    import sys
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
```

- [ ] **Step 3: 跑测试确认失败** → FAIL，`No module named 'app.topics.batch'`

- [ ] **Step 4: 实现 `batch.py` / `routes.py` / `classify_topics.py` / config / api 接线**

`batch.py`：`pending_before = count_pending()` → `list_pending(limit)` → 空则直接返回 → `texts = [p.raw_question]` → `await classify(texts)` → 校验条数一致（不一致抛 `TopicServiceUnavailable("sidecar 返回条数与请求不一致")`）→ `upsert_many([(p.id, pred.labels) ...], classified_at=now or datetime.now())` → `pending_after = count_pending()`。`fallback_other` = 预测 `fallback` 为真的条数。

`routes.py`：`httpx.AsyncClient(timeout=settings.topic_service_timeout)` 调 `{TOPIC_SERVICE_URL}/classify`；`except (httpx.ConnectError, httpx.TimeoutException)` → `_error(503, "topic_service_unavailable", "先启动 sidecar：uvicorn app.topics.service:app --port 8100")`。`distribution` 从 repository 拿数，`multi_label_share = multi_label_count / total_classified`（`total==0` 时 `0.0`）。

`app/api.py`：在静态页挂载块**之前** `register_topic_routes(app)`；在 observability 挂载后照抄一段 `web/topics/` 的挂载 + `/topics` 重定向。

- [ ] **Step 5: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_topics_batch.py tests/test_topics_routes.py tests/test_topics_isolation.py -q --color=no`
Expected: 全 passed

- [ ] **Step 6: 端到端真跑（验收标准 ② 的核心证据）**

```powershell
# 窗口 1：sidecar
& $uv run --directory MemHelp-code uvicorn app.topics.service:app --port 8100
# 窗口 2：主应用
& $uv run --directory MemHelp-code uvicorn app.api:app --port 8000
# 窗口 3：批跑 27 条真实问题
& $uv run --directory MemHelp-code python scripts/classify_topics.py --limit 27
```

Expected: `processed=27`；`& $uv run --directory MemHelp-code python -c "..."` 查 `SELECT COUNT(*) FROM topic_classifications` = 27。

- [ ] **Step 7: Commit**

```bash
git add MemHelp-code/app/topics/batch.py MemHelp-code/app/topics/routes.py MemHelp-code/scripts/classify_topics.py MemHelp-code/app/config.py MemHelp-code/app/api.py MemHelp-code/.env.example MemHelp-code/tests/test_topics_batch.py MemHelp-code/tests/test_topics_routes.py MemHelp-code/tests/test_topics_isolation.py
git commit -m "feat(ch10): 旁路批跑编排、分布/批跑接口与零 torch 隔离断言"
```

---

### Task 15: 主题分布页（Vibe Coding —— 不套 brainstorm / TDD / code review）

**Files:**
- Create: `web/topics/index.html`、`web/topics/style.css`、`web/topics/app.js`

**Interfaces:**
- Consumes: `GET /api/topics/distribution`、`POST /api/topics/classify-batch`
- Produces: 一个纯静态页，照 `web/review/` 的三件套结构与 `style.css` 变量体系（同一个视觉语言），无构建、无图表库。

- [ ] **Step 1: 先做能看的第一版**

布局照 spec §9.1：标题「主题分布」+ 右上「跑一批」按钮；三张数字卡（已归类 / 待归类 / 多标签占比）；水位提示「待归类 27 条 · 水位 50」；17 行横向条形（CSS `width: x%`，按量降序，「其他」固定最后并淡化）；空状态；sidecar 未起时顶部黄条（带启动命令）。

- [ ] **Step 2: 交给用户**：`http://127.0.0.1:8000/topics/`，按用户口述直接改，改到用户满意为止。

- [ ] **Step 3: 只在最后加一条冒烟断言**（不是 TDD，只是防止页面 404）

```python
def test_topics_page_returns_html(...):
    r = client.get("/topics/")
    assert r.status_code == 200 and "主题分布" in r.text
```

- [ ] **Step 4: Commit**

```bash
git add MemHelp-code/web/topics
git commit -m "feat(ch10): 飞轮后台主题分布页（纯静态，无构建）"
```

---

### Task 16: 全量回归 + 交付收尾

**Files:**
- Modify: `dev-notes/ch10.md`（阶段 3 收尾、阶段 4 code review、阶段 5 finish）
- Modify: `docs/superpowers/specs/2026-10-25-ch10-topic-classifier-design.md`（状态行改「已批准 / 已实现」）
- Modify: `.gitignore`（`data/` → `data/*` + `!data/topics/`；加 `models/` 但放行 `thresholds.json` / `metrics.json`）
- Create: `dev-notes/ch10-delivery.md`（或并入 `DELIVERY.md` 的 ch10 段：交付物表 / 演示命令 / 测试结果 / 接口一览 / 验收对照）

- [ ] **Step 1: `gitignore` 与依赖收尾**

按 spec D20 改 `.gitignore`。确认 `git status` 里没有 `models/*/model.safetensors`（几百 MB）、没有 `data/topics/raw` 之外的大文件。

- [ ] **Step 2: 全量测试**

```powershell
& $uv run --directory MemHelp-code python -m pytest -q --color=no
```

Expected: 全绿（含原有的 13 表断言、零 torch 断言、全章新增 ~70 个用例）。把 `passed` 数贴进 dev-notes。

- [ ] **Step 3: 三条验收自查**

```powershell
# ① 报告能跑出来
& $uv run --directory MemHelp-code python scripts/eval_topics.py --run-id latest
# ② 分布页有统计（需 sidecar + 主应用在跑）
& $uv run --directory MemHelp-code python scripts/classify_topics.py --limit 27
# ③ 多诉求句命中多类（单条演示）
& $uv run --directory MemHelp-code python -c @"
import httpx
r = httpx.post('http://127.0.0.1:8100/classify', json={'texts': ['买大了想退，运费谁出']}, timeout=30)
print(r.json()['results'][0]['labels'])
"@
```

Expected: ③ 打印的列表含 `尺码`、`退换货`、`运费` 中至少 2 个（spec 验收标准 3 的原文证据）。

- [ ] **Step 4: `requesting-code-review` 走一遍全分支评审**

按技能要求把 `git diff main...HEAD` 交给评审，重点看：`app/api.py` 是否真的零 torch、`upsert_many` 的事务边界、`threshold_sweep` 有没有偷看 test、`clean.py` 的正则有没有灾难性回溯。评审结论写进 `dev-notes/ch10.md`「阶段 4」。

- [ ] **Step 5: 追记 `dev-notes/ch10.md` 阶段 3/4/5**

四个小节照 ch09 的格式（`### 用户关键原话` / `### 关键产出` / `### 被拒绝 / 纠偏` / `### 翻车与返工`）。阶段 3 要含：真训练耗时与显存、早停轮次、val/test 的 macro-F1、多标签完全命中率、真实子集 vs 合成子集的 F1 对比；阶段 5 含交付清单。

- [ ] **Step 6: Commit**

```bash
git add MemHelp-code/.gitignore MemHelp-code/dev-notes/ch10.md MemHelp-code/dev-notes/ch10-delivery.md MemHelp-code/docs/superpowers/specs/2026-10-25-ch10-topic-classifier-design.md
git commit -m "docs(ch10): 回归通过、交付说明与开发过程追记"
```

---

## 自审

**1. Spec 覆盖**：§1 目标 1–8 分别落在 T2（类目表）/T3–T7（四步流水线）/T6+8（标注）/T10（训练）/T11（评测）/T13–14（推理落库）/T15（前端页）。§2 的 D1–D20：D1→T10、D2→T1、D3→T13、D4→T14、D5→T14、D6→T9、D7→T14、D8→T2、D9→T12、D10→T2+T12、D11→T8、D12→T10+T11、D13→T4+T8、D14→T6+T7、D15→T10、D16→T10、D17→T11、D18→T9、D19→T1、D20→T16。无遗漏。

**2. 步骤粒度**：每个 Task 都有「写失败测试 → 跑失败 → 实现 → 跑通过 → commit」，非单测任务换成「真跑一次 + 人工看数」（T1/T6/T7/T8/T10/T11/T13/T14/T15），与用户工作要求第 1 条一致。测试代码给了断言，实现只给签名与关键约束，不复述函数体。

**3. 类型一致**：`TopicPrediction`（T12 定义，T13/T14 消费）、`PendingQuestionRow`（T9 定义，T14 消费）、`Repositories.topics`（T9 定义，T14 消费）、`TaxonomyError`（T2 定义，T6 消费）、`BatchOutcome`（T14 定义，CLI 与路由共用）逐一核对过命名与字段。

**4. Review Focus**：6 条各有归属测试 —— ①空输入 T12 `test_empty_text_still_returns_one_prediction`、T3 `test_noise_detection`；②503 T14；③重跑 T14 `test_second_run_is_noop` + T9 `test_upsert_twice_overwrites_not_duplicates`；④哈希 T12 `test_taxonomy_hash_mismatch_raises`；⑤「其他」独占 T2 `test_other_must_not_cooccur` + T12 `test_other_is_dropped_when_another_label_fires`；⑥自造标签 T6 `test_parse_generated_rejects_unknown_label`。

**5. 篇幅比例**：计划约 1100 行 vs spec 392 行，偏长主要来自逐任务的测试断言（数据类任务里这些断言就是验收标准本身）。已删掉所有可以「一句话说清」的实现细节（正则具体写法、prompt 全文、HTML 结构），实现者能一处写完的步骤没有拆成两步。
