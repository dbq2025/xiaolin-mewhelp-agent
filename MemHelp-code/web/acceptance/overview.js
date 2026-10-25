/* 验收总览：只读 /api/acceptance/summary。
 *
 * 这一页回答三个问题：模型在留出测试集上多少分、语料是怎么分的、哪几类最弱。
 * 真实子集的结构性说明放在页脚，避免读者把 macro-F1 当成「真实业务准确率」。
 */

const $ = (id) => document.getElementById(id);

const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

function fmtInt(n) {
  return Number(n ?? 0).toLocaleString("zh-CN");
}

function showEmpty(message) {
  $("empty").hidden = false;
  $("empty").querySelector("p").textContent = message;
}

function renderDataset(dataset, multi) {
  const rows = [
    ["训练集（含增强/补造/硬例）", dataset.train, multi.train, "训模型，唯一允许做增强的一份"],
    ["验证集", dataset.val, multi.val, "早停 + 每类阈值扫描，测试集不参与调参"],
    ["测试集", dataset.test, multi.test, "留出评测，160 条全部人工 gold"],
    ["测试集 gold 存档", dataset.test_gold, multi.test, "覆盖 labels 后的快照，供复现"],
    ["人工复核硬例", dataset.hard_cases, null, "只追加训练集；与测试集原句做过去重"],
  ];
  const tbody = $("dataset-rows");
  tbody.innerHTML = "";
  for (const [name, count, multiCount, purpose] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${name}</td>
      <td class="num">${fmtInt(count)}</td>
      <td class="num">${multiCount == null ? '<span class="muted">—</span>' : fmtInt(multiCount)}</td>
      <td class="muted">${purpose}</td>`;
    tbody.appendChild(tr);
  }
}

function renderWorst(rows) {
  const tbody = $("worst-rows");
  tbody.innerHTML = "";
  if (!rows || !rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="muted">没有数据</td></tr>';
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const small = row.small_support ? '<span class="tag-small">样本不足</span>' : "";
    tr.innerHTML = `
      <td><a class="link" href="/topics/questions/?label=${encodeURIComponent(row.label)}">${row.label}</a></td>
      <td class="num">${pct(row.f1)}</td>
      <td class="num">${pct(row.precision)}</td>
      <td class="num">${pct(row.recall)}</td>
      <td class="num">${fmtInt(row.support)}</td>
      <td>${small}</td>`;
    tbody.appendChild(tr);
  }
}

function render(data) {
  const overall = data.overall || {};
  $("stat-macro").textContent = pct(overall.macro_f1);
  $("stat-micro").textContent = pct(overall.micro_f1);
  $("stat-exact").textContent = pct(overall.exact_match);

  const multi = data.multilabel_cases || {};
  if (multi.multi_cases) {
    $("stat-multi").innerHTML = `${fmtInt(multi.hit2)}<small> / ${fmtInt(multi.multi_cases)}</small>`;
    $("stat-multi-hint").textContent = `多标签句 ${multi.multi_cases} 条，gold 全部命中 ${multi.hit2} 条`;
  } else {
    $("stat-multi").textContent = "—";
  }

  $("sub").textContent = `${data.run_id || "未命名 run"} · 测试集 ${fmtInt(data.dataset.test)} 条 · ${data.generated_at || ""}`;
  $("run-meta").textContent = data.baseline && data.baseline.macro_f1
    ? `基线 ${pct(data.baseline.macro_f1)}（容差 ${pct(data.baseline.tolerance)}）`
    : "";

  renderDataset(data.dataset || {}, data.multi || {});
  renderWorst(data.worst_classes || []);

  const real = data.real_subset;
  if (real && real.samples) {
    $("footnote").textContent =
      `真实问题子集只有 ${real.samples} 条（macro-F1 ${pct(real.macro_f1)} / 完全一致 ${pct(real.exact_match)}）：` +
      "17 类宏平均会把没出现过的类目按 0 计入，数字会被结构性压低。看真实业务效果请看完全一致率与混淆矩阵。";
  }
  if (data.error_count) {
    $("footnote").textContent += ` 测试集上共 ${data.error_count} 条错例，可在「错例复核」逐条看。`;
  }
}

async function load() {
  try {
    const response = await fetch("/api/acceptance/summary", { headers: { Accept: "application/json" } });
    if (response.status === 404) {
      showEmpty("还没有评测报告，先跑 scripts/eval_topics.py。");
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (err) {
    showEmpty(`读取失败：${err.message || err}`);
  }
}

load();
