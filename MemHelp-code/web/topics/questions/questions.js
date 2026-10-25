/* 类目问题列表：/api/topics/questions。
 *
 * URL 带 ?label=商品信息 就只看这一类；不带就是全部已归类问题。
 * 筛选项来自接口返回的 all_labels（17 类权威表），不在 JS 里抄一份。
 */

const $ = (id) => document.getElementById(id);
const OTHER = "其他";

const params = new URLSearchParams(window.location.search);
const activeLabel = params.get("label") || "";

function fmtInt(n) {
  return Number(n ?? 0).toLocaleString("zh-CN");
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (v) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function labelLink(name) {
  return `/topics/questions/?label=${encodeURIComponent(name)}`;
}

function renderFilters(labels) {
  const box = $("filters");
  box.innerHTML = "";
  const all = document.createElement("a");
  all.className = `filter${activeLabel ? "" : " active"}`;
  all.href = "/topics/questions/";
  all.textContent = "全部";
  box.appendChild(all);

  for (const name of labels) {
    if (name === OTHER) continue;
    const link = document.createElement("a");
    link.className = `filter${name === activeLabel ? " active" : ""}`;
    link.href = labelLink(name);
    link.textContent = name;
    box.appendChild(link);
  }
  if (labels.includes(OTHER)) {
    const link = document.createElement("a");
    link.className = `filter other${activeLabel === OTHER ? " active" : ""}`;
    link.href = labelLink(OTHER);
    link.textContent = OTHER;
    box.appendChild(link);
  }
}

function renderRows(items) {
  const tbody = $("question-rows");
  tbody.innerHTML = "";
  if (!items.length) {
    $("empty").hidden = false;
    return;
  }
  for (const item of items) {
    const tr = document.createElement("tr");
    const labelChips = (item.labels || [])
      .map((name) => `<a class="chip" href="${labelLink(name)}">${name}</a>`)
      .join(" ");
    const source = item.source ? `<span class="chip gray">${item.source}</span>` : '<span class="muted">—</span>';
    tr.innerHTML = `
      <td class="num muted">${fmtInt(item.id)}</td>
      <td>${item.text || ""}</td>
      <td>${labelChips || '<span class="muted">—</span>'}</td>
      <td>${source}</td>
      <td class="muted">${fmtTime(item.classified_at)}</td>`;
    tbody.appendChild(tr);
  }
}

function render(data) {
  const total = data.total || 0;
  $("title").textContent = data.label ? `「${data.label}」下的问题` : "全部已归类问题";
  $("count-meta").textContent = `${fmtInt(total)} 条`;
  $("sub").textContent = data.label
    ? `筛选：${data.label} · 共 ${fmtInt(total)} 条`
    : `全部类目 · 共 ${fmtInt(total)} 条`;
  if (data.label === OTHER) {
    $("empty-title").textContent = "「其他」下还没有问题";
  }
  renderFilters(data.all_labels || []);
  renderRows(data.items || []);
}

async function load() {
  const query = new URLSearchParams({ limit: "200" });
  if (activeLabel) query.set("label", activeLabel);
  try {
    const response = await fetch(`/api/topics/questions?${query}`, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error((body.error && body.error.message) || `HTTP ${response.status}`);
    }
    render(await response.json());
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("p").textContent = `读取失败：${err.message || err}`;
  }
}

load();
