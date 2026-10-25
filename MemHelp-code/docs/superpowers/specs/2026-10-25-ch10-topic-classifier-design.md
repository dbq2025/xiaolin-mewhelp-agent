# 第十章设计 · 多标签主题分类器（数据流水线 / RoBERTa-wwm-ext 全参微调 / 旁路批量推理 / 主题分布页）

- 日期：2026-10-25
- 状态：已批准（用户 m00179 批准，六节设计逐节获批）
- 关联：`dev-notes/ch10.md`、`sql/ch10-ddl.sql`、`docs/superpowers/plans/2026-10-25-ch10-topic-classifier-plan.md`
- 前置章节：ch04（低置信度池）、ch09（飞轮问题池 / 审核队列 / 后台静态页）

---

## 1. 目标与非目标

### 1.1 目标

1. **17 类权威类目表**：全系统一份，每类一句边界说明，近邻类目靠边界说明划开；代码里单一来源，合成 prompt / 预标 prompt / 训练映射 / 前端 / 报告全部从它取。
2. **数据处理四步**：清洗（脱敏 + 错别字 + 格式）→ 归并术语表 → 分层抽样 80/10/10 → 数据增强（只扩训练集），每步有产物文件和验证手段。
3. **语料从飞轮低置信度池捞真实问题**，池里只有 27 条（去重 12 条）不够用，按术语表用大模型造 1500 条模拟问题补足，其中 25% 是多诉求多标签。
4. **多标签标注**：一句话字面提到几个诉求就打几个标签，一个不多一个不少；路线是大模型照术语表预标 + 人工抽审。
5. **RoBERTa-wwm-ext 全参微调**（不用 LoRA），训练中盯验证集，weight_decay + dropout + 早停三道防过拟合。
6. **评测**：留出测试集上按类目算 P/R/F1 与每类混淆矩阵，出可运行的报告；自动指标之外人工抽判错样本复核并归因。
7. **推理服务化**：独立旁路进程，按批量跑，结果 upsert 进 `topic_classifications`（关联问题 + 多标签 labels + 归类时间），供飞轮后台看各主题分布；**实时对话主链路不调它**。
8. **主题分布页**：飞轮后台新增一页，各类目问题量做成简单图表，哪类堆得多一眼能看出来。

### 1.2 明确不做

- LoRA / QLoRA（那是大模型手段）。
- embedding 微调、意图识别微调。
- 定时器自动批跑（改由后台按钮 + 脚本手动触发，用户 m00101 决定）。
- 主题分布之外的前端功能；混淆矩阵不上前端页（评测视角与运营视角分离，用户 m00143 决定）。
- 改造 `knowledge_chunks.category` / `faq.category` 去对齐 17 类（两者不同源，本章不迁移）。
- 多语言、多租户、权限。

---

## 2. 关键决策

| # | 决策 | 选择 | 理由 / 代价 |
|---|---|---|---|
| D1 | 多标签输出结构 | 17 个独立 sigmoid，`BCEWithLogitsLoss` | softmax 强制互斥，结构上做不到「买大了想退，运费谁出」同时命中尺码/退换货/运费；这是本章最易踩错的一处 |
| D2 | 预训练模型 | `hfl/chinese-roberta-wwm-ext`（实测 config：bert 结构 12 层 / hidden 768 / vocab 21128 / **无 id2label**），全参微调 | 需求点名；large（324M）在 8GB 上全参 + Adam fp32 状态易 OOM，base 是稳的选择 |
| D3 | 推理服务形态 | **独立 sidecar 进程**（`uvicorn app.topics.service:app --port 8100`），无状态、**不连数据库**；主应用**零 ML 依赖** | 用户 m00109 选方案 B。换来「实时链路真的碰不到模型」；代价是演示两条命令 |
| D4 | 批跑编排 | 单份 `app/topics/batch.py:run_batch()`，API 路由与 CLI 脚本共用 | 保证按钮跑的与命令行跑的是同一段逻辑，不会两边行为漂移 |
| D5 | 批跑触发 | 后台按钮 + 脚本手动触发；水位 `TOPIC_BATCH_MIN_PENDING=50` **只提示不拦截** | 用户 m00101 选「手动 + 水位提示」；不做定时器，少一个无人看管的失败面 |
| D6 | 落库语义 | 先查后插/更新（便携写法 + 单事务），不用 MySQL 专用 `ON DUPLICATE KEY UPDATE` | `UNIQUE uk_question_id` 决定重跑是覆盖；测试跑 SQLite 要能过；批量只百来行，性能无意义 |
| D7 | 去重口径 | **按行归类，不按文本去重**（「你们支持货到付款吗」3 行 → 3 条） | 分布页统计的是问题条数；同一句话被 3 个会话问到就是 3 条，写进报告脚注防误判为 bug |
| D8 | 类目表单一来源 | `app/topics/taxonomy.py`；`labels` JSON 直接存中文类目名，按类目表顺序 | 不搞英文 code 中转：「全系统统一一份」指的就是这张中文表，多一层映射只多一处对不上的地方 |
| D9 | 空预测处理 | 17 类一个都没过阈值 → 兜底 `["其他"]`，并统计兜底触发率 | 保证每行都有类目、分布页不出现空洞；触发率 >10% 说明阈值扫得过高，是个信号 |
| D10 | 「其他」语义 | **独占，不与其他类共现**；句子只要有一个诉求能落进 1–16 就不打「其他」 | 否则「其他」会变成垃圾桶标签，吃掉所有低频类的召回 |
| D11 | 测试集 gold | 我（agent）逐条人工标注 150 条并写判定理由，用户抽审 ≥30 条 | 用户 m00097 决定。训练数据是模型造的、标签也是模型预标的，测试集再用模型标，F1 只是「两个模型互相认同」 |
| D12 | 阈值来源 | 只在 val 上按类扫描（0.2–0.7 步长 0.05），**测试集只报数不看** | 拿测试集调阈值等于自欺；这条是报告可信度的根 |
| D13 | 真实问题归属 | 去重后 12 条**只进 test，不进 train/val** | 真实问题最像线上；放训练集就是背答案，评测失去意义 |
| D14 | 合成语料量 | 1500 条（25% 双/三诉求）；迭代分层 80/10/10 → train≈1200 / val 150 / test 150 | 17 类平均每类 70+，每类 F1 才有统计意义 |
| D15 | 训练循环 | 自写循环（~120 行），不用 HF `Trainer` | Trainer 的多标签 compute_metrics 与早停要绕好几层，排障成本高；自写每一步可打日志、单测里 2 条假数据可跑通 |
| D16 | 训练超参 | max_len 64 / batch 16（不够降 8 + 累积 2）/ bf16 autocast / lr 2e-5 + warmup 10% / wd 0.01 / 早停 patience 2（盯 val macro-F1@0.5）/ 梯度裁剪 1.0 | 8GB 显存 + 1200 条训练集；bf16 在 sm_120 原生支持，比 fp16 少一个 scaler |
| D17 | 混淆矩阵渲染 | **纯 HTML + CSS 色块**，不引 matplotlib | Windows 上 matplotlib 画中文要手工指定字体，缺失就出方框；CSS 零依赖零字体问题，还能当报告附件 |
| D18 | 统计聚合位置 | Python 侧聚合（读出 `labels` JSON 后计数），不用 MySQL `JSON_TABLE` | 测试跑 SQLite，函数级 SQL 不兼容；几百到几千行 Python 计数毫无压力 |
| D19 | 依赖隔离 | `[project.optional-dependencies] ml = [torch, transformers, ...]`，torch 走 cu128 index | 主应用依赖集一条不变，ch09 的测试与生产启动完全不受影响 |
| D20 | 语料入不入库 | `data/` 规则改 `data/*` + `!data/topics/`；`models/` 进 ignore 但放行 `thresholds.json` / `metrics.json` | 这一章是数据章，语料与切分文件就是交付物；权重几百 MB 绝不能进库 |

---

## 3. 架构与数据流

```
① 数据流水线（离线一次性 + 可重跑）
   飞轮低置信度池 low_confidence_questions (27 行 / 去重 12)
        │
        ├── 合成语料 build_topic_data.py --generate   （大模型照 taxonomy 造 1500 条）
        │
        ▼
   清洗 clean          data/raw/*.jsonl → data/clean/*.jsonl（带 changes）
        ▼
   归并术语表          taxonomy.py → data/taxonomy.json / taxonomy.md
        ▼
   预标 + 人工抽审     大模型照术语表预标 → data/labeled/*.jsonl（带 rationale）
        ▼
   分层抽样（迭代分层） data/splits/{train,val,test}.jsonl + strata_report.json
        ▼
   数据增强（只 train） data/splits/train_aug.jsonl（带 aug_from）

② 训练（离线）
   scripts/train_topics.py → models/topics/<run_id>/
       {config.json, model.safetensors, tokenizer/, thresholds.json, metrics.json, taxonomy.json}

③ 评测（离线）
   scripts/eval_topics.py → reports/topics/<run_id>/{report.md, metrics.json,
                              per_class_prf.csv, confusion.html, error_review.csv}

④ 推理（旁路，不在实时链路）
   ┌──────────────┐   ①读 pending      ┌──────────────────┐
   │ 主应用 8000   │  ②POST /classify   │ sidecar 8100      │
   │ 零 torch      │ ─────────────────► │ torch + 模型常驻   │
   │              │ ◄───────────────── │ 无状态 / 不连库    │
   └──────────────┘   ③标签+分数        └──────────────────┘
          │ ④upsert topic_classifications
          ▼
   web/topics/ ← GET /api/topics/distribution
```

主应用与 sidecar 的边界是硬约束：`app/api.py` 及其路由模块**永远不 import torch**，由测试钉住（见 §9.2）。

---

## 4. 类目表与标注口径

### 4.1 17 类权威表

代码位置 `app/topics/taxonomy.py`，结构 `CATEGORIES: tuple[Category, ...]`，`Category = (name, boundary, triggers)`，顺序即编号、即 `labels` 输出顺序、即混淆矩阵行列顺序。

| # | 类目 | 边界说明 | 典型触发 |
|---|---|---|---|
| 1 | 退换货 | 退货、换货、退款意图，以及退货条件、流程、时限。「修」归保修维修，「改地址 / 改规格」归订单修改 | 想退、能换吗、退款多久到 |
| 2 | 物流 | 货的运输状态。「钱」归运费 | 到哪了、几天到、哪家快递、丢件 |
| 3 | 尺码 | 选码建议、偏大偏小、尺码表。参数材质归商品信息 | 买多大、偏大吗、尺码表 |
| 4 | 发票 | 开票、抬头税号、专普票、重开。付款问题归支付 | 能开票吗、专票、发票随货 |
| 5 | 质量问题 | 到手即坏、破损、少件、发错货、瑕疵异味。用久了坏归保修维修 | 一拆就坏、发错货、有味道 |
| 6 | 运费 | 邮费多少钱、包不包邮、补运费差价、运费险。运输进度归物流 | 包邮吗、邮费多少、运费险 |
| 7 | 优惠活动 | 券、满减、秒杀、赠品、活动规则与叠加。降价补差归价保 | 券能用吗、能叠加吗、满减 |
| 8 | 价保 | 买后降价能否补差价、价保期限与申请。券和满减归优惠活动 | 降价了、补差价、价保 |
| 9 | 支付 | 付款失败、重复扣款、支付方式、分期。开票归发票 | 付不了、扣两次、能分期吗 |
| 10 | 订单修改 | 改地址、改规格颜色、合并拆分单、未发货取消。已发货要退归退换货 | 改地址、改颜色、取消订单 |
| 11 | 库存补货 | 缺货、何时补货、到货通知、预售发货时间 | 没货了、啥时候补、到货通知 |
| 12 | 商品信息 | 材质、参数、型号、功能、适配、真伪、保质期。选码归尺码 | 什么材质、支持吗、正品吗 |
| 13 | 保修维修 | 保修期、上门修、寄修、换配件、维修收费。到手即坏归质量问题 | 保修多久、怎么修、上门 |
| 14 | 账号 | 登录、密码、换绑手机、注销、实名、冻结 | 登不上、改手机号、注销 |
| 15 | 会员积分 | 会员等级、积分获取抵扣过期、权益、成长值 | 积分怎么用、会员价、过期 |
| 16 | 评价 | 晒单、追评、差评申诉、评价奖励、改评价 | 怎么晒单、能改评价吗 |
| 17 | 其他 | 以上都不涉及（投诉客服态度、转人工、闲聊）。**独占，不与其他类共现** | 转人工、你们服务太差 |

需求点名的近邻边界全部落入：修 → 保修维修、退 → 退换货；运费管钱、物流管货；价保补差价、优惠活动管券和满减。另补三对需求未提但必然冲突的：尺码 vs 商品信息、质量问题 vs 保修维修、订单修改 vs 退换货。

### 4.2 多标签标注口径（把「一个不多一个不少」写成可执行规则）

1. 只打**字面出现**的诉求，不推断隐含需求。
   - 「买大了想退，运费谁出」→ `尺码 + 退换货 + 运费`（3 个都打）
   - 「这个好用吗」→ 只 `商品信息`；不因为“可能在考虑买”而加优惠活动
2. 同义诉求合并：退货 / 退款 / 换货同属 1 类，只打一个标签。
3. 「其他」独占：句子只要有一个诉求能落进 1–16 就不打「其他」；落不进去的诉求直接忽略，不额外加标签。
4. 标签顺序固定按类目表编号升序输出，保证与混淆矩阵行列对齐。
5. 每条标注附一句 `rationale`，写清命中了哪条边界规则——这是人工抽审时唯一能快速判断对错的依据。

### 4.3 标注流程

| 语料 | 标注方式 | 复核 |
|---|---|---|
| 合成 train / val（≈1350） | 大模型照术语表预标，带 rationale | 分层抽审 60 条（每类至少 3 条），错标率 >5% 则改 prompt 重标 |
| 合成 test（138） | **我逐条人工标注** | 用户抽审 ≥30 条 |
| 真实 test（12） | **我逐条人工标注** | 同上 |

抽审记录落 `data/review/`：`train_spotcheck.csv`（大模型预标抽审）、`test_gold_review.csv`（我的标注抽审，列：id / text / 我的标签 / 我的理由 / 用户判定 / 用户意见）。抽审改过的条目 gold 以用户为准，报告脚注注明改了几条。

---

## 5. 数据流水线

| 步骤 | 实现 | 输入 → 输出 | 验证 |
|---|---|---|---|
| 清洗 | `app/topics/clean.py` | `data/raw/*.jsonl` → `data/clean/*.jsonl` | 抽 50 条人工核对 `changes` 是否合理；单测钉住脱敏正则与全半角规则 |
| 归并术语表 | `taxonomy.py` → `scripts/build_topic_data.py` 导出 | → `data/taxonomy.json` / `taxonomy.md` | 断言 prompt 片段里的类目数与编号与代码完全一致 |
| 分层抽样 | `app/topics/split.py`，多标签用**迭代分层**（按标签组合分配） | `data/labeled/*.jsonl` → `data/splits/*.jsonl` + `strata_report.json` | 报告每类在三份的计数与占比；某类在某份为 0 直接报警 |
| 数据增强 | `app/topics/augment.py`：同义词替换 + 句式微调，**只动 train** | → `data/splits/train_aug.jsonl`（带 `aug_from`） | 断言 val/test 未被污染；原句仍在扩充后的 train 里 |

### 5.1 清洗细则

- 全半角统一（标点、数字、字母）、多余空白与表情符号处理。
- 错别字：`bank/typos.json` 词典纠正确定性错字（帐号→账号、收活→收货、褪回→退回）＋ 对含疑似错字的样本走大模型校正，**保留原句与校正句两列**（`text_raw` / `text`）。
- 脱敏：手机号、身份证、订单号、邮箱、地址门牌 → `[PHONE]` / `[IDCARD]` / `[ORDER]` / `[EMAIL]` / `[ADDR]` 占位符；占位符本身保留（它是有效特征：「我的订单 [ORDER] 到哪了」仍是物流）。
- 每条记 `changes: [{"rule": "...", "from": "...", "to": "..."}]`，便于抽审与回滚。

### 5.2 合成语料

`scripts/build_topic_data.py --generate --count 1500`：

- Prompt 内嵌 `taxonomy.json` 全文 + 4.2 多标签口径 + 具体商品型号词表（沿用仓库已有商品名：记忆棉靠枕 A57、收纳箱 F98、巡检仪、门禁与探头等）+ 口语化要求（长度 8–40 字、真实客服口吻、允许口语省略）。
- 每个类目生成配额按 1–16 均分，25% 样本额外叠加一个近邻类目的第二诉求（优先挑该类的近邻对，例如退换货样本里 25% 加尺码或运费）。
- 输出即带 `labels`（生成时就知道想表达什么），再由预标环节独立复标一次做交叉检查；生成标签与预标标签不一致的样本进 `data/review/label_conflicts.csv`，人工裁决。
- 真实池 27 行先去重（精确 + 规范化后去重），得 12 条唯一文本，全部标好进 test。

---

## 6. 训练

### 6.1 模型与损失

- `AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=17, problem_type="multi_label_classification")`。
- `BCEWithLogitsLoss`，**不加 `pos_weight`**（先跑，看各类召回再定；加了会和 §6.3 的阈值扫描混在一起，说不清是谁在起作用）。
- tokenizer 用 `BertTokenizer.from_pretrained`（该 checkpoint 是 bert 结构，不是 RobertaTokenizer），`do_lower_case=True`。

### 6.2 超参与正则化

| 项 | 值 |
|---|---|
| max_length | 64 |
| batch_size | 16（OOM 则 8 + 梯度累积 2） |
| 精度 | `torch.autocast(dtype=torch.bfloat16)` |
| optimizer | AdamW，lr 2e-5，weight_decay 0.01 |
| schedule | 线性 warmup 10% + 余弦退火 |
| epochs | 上限 8，早停 patience 2（盯 val macro-F1 @0.5） |
| 梯度裁剪 | 1.0 |
| dropout | 保持 config 默认 0.1 |
| seed | 固定（42），`--seed` 可覆盖 |

正则化三道一起上：weight_decay + dropout + 早停。保留 val macro-F1 最高的 checkpoint，不是最后一个 epoch。

### 6.3 每类阈值

1. 训练过程中用固定 0.5 阈值算 val macro-F1（选 checkpoint 用），保证训练过程干净可复现。
2. 训练结束在 **val** 上按类扫 0.2–0.7（步长 0.05），每类取该类 F1 最大点，写 `thresholds.json`。
3. 测试集评测与线上推理都用这份阈值；**测试集只报数，不调参**。

不做这一步，测试集 F1 大概 0.6 且少数类接近 0；做了能到 0.8+ 且各类均衡——属于必须做，不是锦上添花。

### 6.4 产物

`models/topics/<run_id>/`（`run_id = YYYYMMDD-HHMMSS`）

```
config.json          # 超参 + git commit + 数据指纹 + 训练时长
model.safetensors    # 最优 checkpoint
tokenizer/           # vocab.txt / tokenizer_config.json
thresholds.json      # 17 类各自阈值 + val 上的每类 F1
metrics.json         # 逐 epoch 训练/验证曲线 + 最优指标
taxonomy.json        # 训练时类目表快照（含哈希）
```

`taxonomy.json` 快照防的是「线上类目表顺序变了、模型输出第 3 位对应哪个类」这种静默错位：sidecar 启动时比对快照哈希与当前 `taxonomy.py`，不一致**拒绝启动**。

---

## 7. 评测

### 7.1 指标三层

| 层 | 指标 | 落盘 |
|---|---|---|
| 整体 | micro-P/R/F1、macro-P/R/F1、weighted-F1、Hamming loss、子集完全一致率（exact match） | `report.md` + `metrics.json` |
| 每类 | precision / recall / F1 / support / TP-FP-FN-TN | `per_class_prf.csv` |
| 每类混淆矩阵 | 17 个 2×2 色块矩阵 + **最易归错的类对排行** | `confusion.html` |

类对排行直接用「某类样本被判成另一类」的计数排序，它比单类 F1 更能回答「该补哪块知识」。

### 7.2 人工抽判错复核

- 从测试集预测结果里每类最多抽 3 条错例，**优先抽假阴性**（漏判比误判更隐蔽），总样本封顶 30 条。
- 每条判成四类之一：`模型对（gold 标错）` / `人标对（模型错）` / `都错（边界争议）` / `文本本身歧义`。
- 汇总为「错误归因分布」写进 `report.md`：若多数是「gold 标错」→ 我的标注口径有问题；若多数是「边界争议」→ 17 类边界说明不够用，**回头改 §4.1 的边界规则并重标**。所以复核必须发生在报告定稿前。

### 7.3 多诉求专项（验收标准 3）

`evals/ch10_multilabel_cases.jsonl`：20 条手工设计的多诉求句，覆盖每对近邻类组合（退换货+尺码、物流+运费、价保+优惠活动、保修维修+质量问题、订单修改+退换货…），每条写明期望标签集合。评测脚本输出两个数：

- 多标签**完全命中率**（期望集合与预测集合完全一致的比例）
- **至少命中 2 个标签的比例**（目标 20/20，这是「能同时命中多个类目」的直接证据）

结果进演示命令输出，不只在报告里。

### 7.4 回归防退化

`metrics.json` 的 macro-F1 写入 `evals/ch10_baseline.json`；后续任何改动（改类目表 / 改阈值 / 重训）跑评测时自动对比，掉超 0.03 在报告标红——只提示不阻断，避免数据类任务变成「改一点点就红灯」的噪音源。

---

## 8. 推理服务与落库

### 8.1 进程职责

| 进程 | 入口 | 职责 | 依赖 |
|---|---|---|---|
| 推理 sidecar | `uvicorn app.topics.service:app --port 8100` | 无状态纯推理：加载模型、打分、按每类阈值出标签。**不连数据库** | torch / transformers（`[ml]` extra） |
| 主应用 | 现有 `uvicorn app.api:app --port 8000` | 主题分布统计、代理触发批跑、挂 `/topics/` 页面 | **零 ML 依赖**，只多 httpx |
| 批跑脚本 | `scripts/classify_topics.py` | 读 pending → 调 sidecar → upsert 写库 | 同主应用 |

### 8.2 接口

**sidecar**
- `GET /health` → `{status, run_id, taxonomy_hash, model_loaded}`
- `POST /classify`，body `{"texts": [...]}`（单请求封顶 64 条）→ `{"results": [{"labels": [...], "scores": {"退换货": 0.91, ...}}]}`

返回全部 17 个分数而不只返回过阈值的标签：低置信度问题出问题时能回溯「它贴近哪个类但差 0.03」，对补知识有用。

**主应用**
- `GET /api/topics/distribution` → `{total, by_label: [{label, count}], multi_label_share, pending, min_pending, last_classified_at}`
- `POST /api/topics/classify-batch`，body `{"limit": 100}` → `{processed, fallback_other, elapsed_ms}`；sidecar 不可达返回 **503 + 明确提示命令**（`先启动 sidecar：uvicorn app.topics.service:app --port 8100`），不返回 500 或含糊错误。

### 8.3 pending 与水位

- pending = `low_confidence_questions` LEFT JOIN `topic_classifications` 后 `tc.id IS NULL` 的行数。
- `TOPIC_BATCH_MIN_PENDING = 50`：页面显示「当前待归类 N 条，水位 50」，N < 50 时按钮仍可点（只灰字提示「未到水位，建议再攒攒」），不硬拦截——手动触发的价值就是不想等。
- `TOPIC_BATCH_LIMIT = 100`，脚本 `--limit` 可覆盖。

### 8.4 写入

- 便携 upsert（先查后插/更新，单事务），兼容 MySQL 与 SQLite。
- `classified_at` 用 Python 侧 `datetime.now()` 传入，保证可测、可断言。
- 写入的 `labels` 是过阈值后的中文类目名数组；空 → `["其他"]`（D9）。

### 8.5 配置项（`app/config.py` ch10 分节）

```
TOPIC_SERVICE_URL=http://127.0.0.1:8100
TOPIC_MODEL_DIR=models/topics/latest
TOPIC_BATCH_MIN_PENDING=50
TOPIC_BATCH_LIMIT=100
TOPIC_SERVICE_TIMEOUT=30
```

`.env.example` 加 `# ---- ch10：主题分类 ----` 分节，沿用现有风格。

### 8.6 启动自检（fail fast）

sidecar 启动时校验三件事，任一失败直接拒绝启动并打印原因：

1. `TOPIC_MODEL_DIR/taxonomy.json` 哈希 == 当前 `taxonomy.py` 哈希（防标签错位）；
2. `thresholds.json` 恰好 17 个键且集合与类目表一致；
3. 权重与 tokenizer 能加载。

---

## 9. 前端与工程接缝

### 9.1 主题分布页（Vibe Coding，不套 brainstorm / TDD / code review）

首版由我直接做，之后按用户口述效果迭代。`web/topics/{index.html,style.css,app.js}`，纯静态、无构建、无图表库（CSS 宽度画条形图），挂载照 `app/api.py:2294-2362` 现有模式。

首版布局：

```
主题分布                                     [跑一批]

┌──────────┬──────────┬──────────┐
│ 已归类   │ 待归类   │ 多标签占比 │   ← 待归类 27 条 · 水位 50（低于水位灰字提示）
└──────────┴──────────┴──────────┘

退换货   ████████████████████  34  (26.6%)
物流     ████████████████      27  (21.1%)
...
其他     ██                     4  ( 3.1%)   ← 永远最后、颜色淡化
```

- 条形按数量降序，「其他」固定最后；悬停显示精确数量。
- 空状态：「还没有归类过，点右上角跑一批」。
- 跑批时按钮禁用 + 文案「归类中…」，完成后自动刷新。
- sidecar 未起时页面顶部黄色提示条（带启动命令）。

### 9.2 接缝清单

| 要改的东西 | 怎么改 |
|---|---|
| `app/db/models.py` | 加 `TopicClassification`，字段严格对齐 `sql/ch10-ddl.sql` |
| `tests/test_db_models.py:13` | `test_metadata_has_exactly_twelve_tables` → 十三表，断言集合加 `topic_classifications` |
| `app/db/repositories.py` | 加 `TopicClassificationRepository`（pending / upsert / 标签计数）+ 行 dataclass + `Repositories` 门面属性，风格对齐 `EvalRunRepository` |
| `app/config.py` | ch10 分节 5 个字段 |
| `app/topics/` | `taxonomy.py`、`clean.py`、`split.py`、`augment.py`、`infer.py`、`service.py`、`batch.py`、`routes.py` |
| `scripts/` | `build_topic_data.py`、`train_topics.py`、`eval_topics.py`、`classify_topics.py` |
| `evals/` | `ch10_multilabel_cases.jsonl`、`ch10_baseline.json` |
| `tests/` | `test_topics_taxonomy.py`、`test_topics_clean.py`、`test_topics_split.py`、`test_topics_augment.py`、`test_topics_batch.py`、`test_topics_routes.py`、`test_topics_isolation.py` + 改 `test_db_models.py` |
| `.gitignore` | `data/` → `data/*` + `!data/topics/`；加 `models/` 但 `!models/topics/*/thresholds.json`、`!models/topics/*/metrics.json` |
| `sql/ch10-ddl.sql` | 随 ch10 一起提交（当前 untracked，表已在 MySQL 建好） |
| `pyproject.toml` | `[project.optional-dependencies] ml = ["torch", "transformers", "safetensors"]` |

**架构约束的钉住测试**（`tests/test_topics_isolation.py`）：import `app.api` 之后断言 `torch` / `transformers` 不在 `sys.modules`。否则哪天有人在路由里顺手 import 一下模型，主应用就悄悄背上 300MB。

**库用法一律先查 Context7**（需求 3）：动手顺序上先用 Context7 查 FastAPI 0.143 的 lifespan / 依赖注入 / 异常处理、SQLAlchemy 2.1 的 JSON 列与 upsert 写法、transformers 的多标签 `num_labels` 与 `BertTokenizer` 当前接口。查到与记忆不一致就改设计，不硬套。

---

## 10. 验收对照

| 验收标准 | 怎么验 | 证据 |
|---|---|---|
| ① 测试集各类目 F1 和混淆矩阵报告能跑出来 | `python scripts/eval_topics.py --run-id <id>` 一条命令出 `report.md` + `per_class_prf.csv` + `confusion.html` | 报告文件 + 命令输出 |
| ② 真实问题批量归类后分布页有统计 | 起 sidecar → `python scripts/classify_topics.py --limit 27` → 打开 `/topics/` 看条形图 | 页面截图 + `topic_classifications` 行数 |
| ③ 「买大了想退」能同时命中多个类目 | `evals/ch10_multilabel_cases.jsonl` 20 条，报告「至少命中 2 个标签的比例」 | 评测输出 + 单条演示 |

补充自检：

- `pytest -q` 全绿（含 13 表断言与零 torch 断言）。
- 人工抽错 30 条归因写进 `report.md`，用户抽审 ≥30 条写进 `data/review/test_gold_review.csv`。
- `dev-notes/ch10.md` 每阶段有追记。

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| 8GB 显存 OOM | batch 16 起步，OOM 自动降 8 + 梯度累积 2；max_len 64；bf16 |
| torch cu128 安装体积大 / 慢 | 走 `download.pytorch.org/whl/cu128`（实测可达，2.7–2.9.1 有 win/cp312）；装失败就停下来问，不自行换 CPU 版（训练会慢一个量级） |
| 17 类里少数类样本不足 | strata_report 报警；某类 train 样本 <20 时合并进「其他」或补造该类样本，在 dev-notes 记录选择 |
| 合成语料与真实分布差距大 | 真实 12 条只进 test 作真实锚点；报告里单列「真实子集 vs 合成子集」的 F1 对比 |
| 人工标注主观（尤其近邻类） | 每条带 rationale；错误归因里「边界争议」比例超 20% 就回改边界规则 |
| 测试集只有 150 条，F1 波动大 | 报告附每类 support；support <10 的类目 F1 标注「样本不足，仅供参考」 |
| 阈值扫描过拟合 val | val 只有 150 条；阈值步长 0.05 避免锯齿；报告同时给 0.5 固定阈值的对照数 |

## 12. 实施中的偏离与补充（2026-10-25 收尾定稿）

交付版为 `ch10-v3`（测试集 160 条人工 gold：macro-F1 0.9713 / micro 0.9661 / exact 0.9250 / 多诉求命中 7/16）。
相对上文设计，落地时有以下补充/偏离，均已写入代码与 `dev-notes/ch10.md`：

- **合成语料多标签比例**：prompt 只写「约三成」时模型只兑现约 10%，训练集多标签掉到 10%、模型学会只报主诉求。
  新增 `boost_multi` 步骤（`force_multi=True` 把话说死，只追加进训练集），补 600 条后训练集多标签达 28%。
- **人工复核硬例**：真机联调发现「猫窝/猫抓板 → 保修维修」成片误判，新增 `hard_cases` 流水线步骤与
  `data/topics/extra/hard_cases.jsonl`（96 条，刻意避开测试集原句防泄漏），只追加训练集。
- **阈值产物**：`thresholds.json` 增加 `max_length`，`TopicClassifier.load` 读回，避免训练/推理截断长度漂移；
  `val` 上调阈值得到的 tuned 指标改为复用 `infer.judge_scores()`，与线上「其他独占 + 兜底」同一套规则。
- **评测口径**：多诉求命中率的分母只算 gold ≥2 标签的用例；未标 gold（`needs_human_label`）的行跳过并计数；
  真实子集 macro-F1 必须附「17 类宏平均 + 真实样本集中」的结构性稀释说明。
- **落库幂等**：`upsert_many` 撞唯一键时回滚重试一次（并发批跑），DDL 补 `ON DELETE CASCADE` 与 ORM 对齐。
- **部署默认值**：sidecar 默认加载 `models/topics/latest`，可用 `TOPIC_MODEL_DIR` 或 `Settings.topic_model_dir` 覆盖。
- **测试范围**：按用户收尾要求「最简覆盖核心路径」，并发重试等边界场景不做单测，靠实现 + 手工联调验证；
  Minor 遗留（CSV 转义、地址 regex 覆盖、run_id 秒级竞态）记在 dev-notes 遗留清单。
