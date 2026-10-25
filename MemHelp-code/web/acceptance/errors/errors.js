/* 错例复核：/api/acceptance/errors。
 *
 * 测试集上的错例是「全量」而不是抽样 —— 160 条里错 13 条，逐条看得完，
 * 抽样反而会漏掉重复出现的同一类错误。
 */

const $ = (id) => document.getElementById(id);

function chips(value) {
  if (!value) return '<span class="muted">—</span>';
  return value
    .split("|")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => `<span class="chip gray">${item}</span>`)
    .join(" ");
}

function render(rows) {
  const tbody = $("error-rows");
  tbody.innerHTML = "";
  $("count-meta").textContent = `${rows.length} 条`;
  if (!rows.length) {
    $("empty").hidden = false;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.text || ""}</td>
      <td>${chips(row.gold)}</td>
      <td>${chips(row.pred)}</td>
      <td>${chips(row.missing)}</td>
      <td>${chips(row.extra)}</td>`;
    tbody.appendChild(tr);
  }
}

async function load() {
  try {
    const response = await fetch("/api/acceptance/errors", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    $("sub").textContent = `${data.run_id || "未命名 run"} · 错例 ${data.count} 条`;
    render(data.rows || []);
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("p").textContent = `读取失败：${err.message || err}`;
  }
}

load();
