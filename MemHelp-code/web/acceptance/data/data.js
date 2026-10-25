/* 数据产物：/api/acceptance/data。
 *
 * 类目表直接来自 data/topics/taxonomy.json 的快照（与训练时吃的同一份），
 * 不在页面里硬编码 —— 类目改了，这页跟着变，不会「文档说 17 类、代码变 18 类」。
 */

const $ = (id) => document.getElementById(id);

function fmtSize(bytes) {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function fmtInt(n) {
  return Number(n ?? 0).toLocaleString("zh-CN");
}

function renderTaxonomy(categories) {
  const tbody = $("taxonomy-rows");
  tbody.innerHTML = "";
  if (!categories || !categories.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">没有读类目表</td></tr>';
    return;
  }
  for (const item of categories) {
    const tr = document.createElement("tr");
    const triggers = (item.triggers || []).map((t) => `<span class="chip gray">${t}</span>`).join(" ");
    tr.innerHTML = `
      <td class="num muted">${fmtInt(item.index)}</td>
      <td><strong>${item.name}</strong></td>
      <td>
        <div>${item.boundary || ""}</div>
        <div class="muted" style="margin-top:4px;font-size:12px">${triggers}</div>
      </td>`;
    tbody.appendChild(tr);
  }
}

function renderFiles(files) {
  const tbody = $("file-rows");
  tbody.innerHTML = "";
  for (const file of files) {
    const tr = document.createElement("tr");
    const state = file.exists
      ? '<span class="chip ok">存在</span>'
      : '<span class="chip gray">未生成</span>';
    tr.innerHTML = `
      <td><code>${file.path}</code></td>
      <td class="num">${fmtSize(file.size)}</td>
      <td>${state}</td>`;
    tbody.appendChild(tr);
  }
}

function render(data) {
  renderTaxonomy(data.taxonomy || []);
  renderFiles(data.files || []);
  $("taxonomy-hash").textContent = data.taxonomy_hash ? `哈希 ${String(data.taxonomy_hash).slice(0, 12)}…` : "";
  const splits = data.splits || {};
  $("sub").textContent =
    `训练 ${fmtInt(splits.train)} · 验证 ${fmtInt(splits.val)} · 测试 ${fmtInt(splits.test)} 条`;
}

async function load() {
  try {
    const response = await fetch("/api/acceptance/data", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("p").textContent = `读取失败：${err.message || err}`;
  }
}

load();
