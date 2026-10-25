/* 评测详情：/api/acceptance/eval。
 *
 * 每类 P/R/F1 用「数字 + 迷你条」双呈现：数字要能读，条形要能一眼看出差距。
 * 混淆矩阵是评测脚本生成的独立 HTML，这里只给入口，不在页内重画一遍。
 */

const $ = (id) => document.getElementById(id);

const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

function fmtInt(n) {
  return Number(n ?? 0).toLocaleString("zh-CN");
}

function renderOverall(overall) {
  $("stat-macro").textContent = pct(overall.macro_f1);
  $("stat-micro").textContent = pct(overall.micro_f1);
  $("stat-hamming").textContent = Number(overall.hamming_loss ?? 0).toFixed(4);
}

function renderClasses(rows) {
  const tbody = $("class-rows");
  tbody.innerHTML = "";
  // F1 从高到低：最弱的排最后，和总览页「最弱 5 类」对得上。
  const sorted = [...rows].sort((a, b) => (b.f1 || 0) - (a.f1 || 0));
  for (const row of sorted) {
    const tr = document.createElement("tr");
    const width = Math.round((row.f1 || 0) * 100);
    const small = row.small_support ? ' <span class="tag-small">样本不足</span>' : "";
    tr.innerHTML = `
      <td><a class="link" href="/topics/questions/?label=${encodeURIComponent(row.label)}">${row.label}</a>${small}</td>
      <td class="num">${pct(row.precision)}</td>
      <td class="num">${pct(row.recall)}</td>
      <td class="bar-cell">
        <span class="num">${pct(row.f1)}</span>
        <div class="mini-track"><div class="mini-bar" style="width:${width}%"></div></div>
      </td>
      <td class="num">${fmtInt(row.support)}</td>
      <td class="num">${fmtInt(row.tp)}</td>
      <td class="num">${fmtInt(row.fp)}</td>
      <td class="num">${fmtInt(row.fn)}</td>`;
    tbody.appendChild(tr);
  }
}

function renderPairs(pairs) {
  const tbody = $("pair-rows");
  tbody.innerHTML = "";
  if (!pairs || !pairs.length) {
    $("pair-empty").hidden = false;
    tbody.innerHTML = '<tr><td colspan="3" class="muted">（空）</td></tr>';
    return;
  }
  for (const pair of pairs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="chip">${pair.gold_label}</span></td>
      <td><span class="chip gray">${pair.pred_label}</span></td>
      <td class="num">${fmtInt(pair.count)}</td>`;
    tbody.appendChild(tr);
  }
}

function render(data) {
  renderOverall(data.overall || {});
  renderClasses(data.per_class || []);
  renderPairs(data.pair_confusions || []);

  $("sub").textContent = `${data.run_id || "未命名 run"} · 17 类指标`;
  $("eval-meta").textContent = `测试集 ${fmtInt((data.overall || {}).samples)} 条`;

  const link = $("confusion-link");
  if (data.confusion_url) link.href = data.confusion_url;
}

async function load() {
  try {
    const response = await fetch("/api/acceptance/eval", { headers: { Accept: "application/json" } });
    if (response.status === 404) {
      $("empty").hidden = false;
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("p").textContent = `读取失败：${err.message || err}`;
  }
}

load();
