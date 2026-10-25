/* 主题分布页：只读两个接口，没有框架、没有构建步骤。
 *
 * 数据口径全在后端（/api/topics/distribution）：这页只负责「把数画出来」和
 * 「点一下跑一批」。类目清单也从后端拿（all_labels），不在 JS 里再抄一份 17 类 ——
 * 抄一份就一定会和 app/topics/taxonomy.py 漂移。
 */

const API = {
  distribution: "/api/topics/distribution",
  batch: "/api/topics/classify-batch",
};

const OTHER = "其他";
const $ = (id) => document.getElementById(id);

function fmtInt(n) {
  return Number(n ?? 0).toLocaleString("zh-CN");
}

function fmtTime(iso) {
  if (!iso) return "还没归过类";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (v) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/* ---- 提示条 -------------------------------------------------------- */

function showBanner(title, message, tone = "warn") {
  const banner = $("banner");
  banner.hidden = false;
  banner.classList.toggle("ok", tone === "ok");
  $("banner-title").textContent = title;
  $("banner-message").textContent = message || "";
}

function hideBanner() {
  $("banner").hidden = true;
}

/* ---- 渲染 ---------------------------------------------------------- */

function renderWatermark(pending, minPending) {
  const el = $("watermark");
  const text = $("watermark-text");
  el.hidden = false;
  el.classList.remove("ready", "low");
  if (pending === 0) {
    el.classList.add("ready");
    text.textContent = "没有待归类的问题，池子是干净的。";
  } else if (pending < minPending) {
    el.classList.add("low");
    text.textContent = `积压 ${fmtInt(pending)} 条，攒到 ${fmtInt(minPending)} 条再跑更划算（水位只提示，不拦截）。`;
  } else {
    el.classList.add("ready");
    text.textContent = `积压 ${fmtInt(pending)} 条，已过水位线 ${fmtInt(minPending)} 条，可以跑一批了。`;
  }
}

function renderChart(labels, counts) {
  const chart = $("chart");
  chart.innerHTML = "";

  const max = Math.max(1, ...labels.map((name) => counts[name] || 0));
  // 按量降序；「其他」是兜底桶，量再大也固定在最后（css 里还有淡化）。
  const sorted = labels
    .filter((name) => name !== OTHER)
    .map((name) => [name, counts[name] || 0])
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "zh-CN"));
  if (labels.includes(OTHER)) sorted.push([OTHER, counts[OTHER] || 0]);

  for (const [name, count] of sorted) {
    const row = document.createElement("div");
    row.className = "row";
    if (name === OTHER) row.classList.add("other");
    if (count === 0) row.classList.add("zero");
    row.title = `${name}：${fmtInt(count)} 条`;

    const label = document.createElement("a");
    label.className = "row-label";
    label.href = `/topics/questions/?label=${encodeURIComponent(name)}`;
    label.title = `看「${name}」下的具体问题`;
    label.textContent = name;

    const track = document.createElement("div");
    track.className = "bar-track";
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${Math.round((count / max) * 100)}%`;
    track.appendChild(bar);

    const num = document.createElement("div");
    num.className = "row-count";
    num.textContent = fmtInt(count);

    row.append(label, track, num);
    chart.appendChild(row);
  }
}

function render(data) {
  $("stat-total").textContent = fmtInt(data.total);
  $("stat-pending").textContent = fmtInt(data.pending);
  $("stat-multi").textContent = `${Math.round((data.multi_label_share || 0) * 100)}%`;
  $("stat-pending-hint").textContent =
    data.pending >= data.min_pending ? "已过水位线，可以跑一批" : `水位线 ${fmtInt(data.min_pending)} 条`;

  $("sub").textContent = `${fmtInt(data.total)} 条已归类 · 更新于 ${fmtTime(data.last_classified_at)}`;
  $("panel-meta").textContent = data.last_classified_at
    ? `最近一次归类：${fmtTime(data.last_classified_at)}`
    : "尚未归类";

  const counts = Object.create(null);
  for (const item of data.by_label || []) counts[item.label] = item.count;
  const labels = data.all_labels || Object.keys(counts);

  renderWatermark(data.pending, data.min_pending);

  const isEmpty = (data.total || 0) === 0;
  $("empty").hidden = !isEmpty;
  $("chart").hidden = isEmpty;
  $("error-panel").hidden = true;
  if (!isEmpty) renderChart(labels, counts);
}

function showError(err) {
  $("error-panel").hidden = false;
  $("error-message").textContent = String(err && err.message ? err.message : err);
  $("sub").textContent = "读取失败";
}

/* ---- 取数 ---------------------------------------------------------- */

async function load() {
  try {
    const response = await fetch(API.distribution, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`/api/topics/distribution 返回 HTTP ${response.status}`);
    }
    render(await response.json());
  } catch (err) {
    showError(err);
  }
}

/* ---- 跑一批 -------------------------------------------------------- */

async function runBatch() {
  const button = $("btn-run");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "归类中…";
  hideBanner();
  try {
    const response = await fetch(API.batch, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const body = await response.json().catch(() => ({}));
    const message = body && body.error ? body.error.message : "";

    if (response.status === 503) {
      showBanner("sidecar 没起", message || "连不上推理服务。");
    } else if (!response.ok) {
      showBanner(`批跑失败（HTTP ${response.status}）`, message || "看主应用日志。");
    } else if (!body.processed) {
      showBanner("没有待归类的问题", "池子里没有攒下的低置信度问题，等飞轮再入池一些。", "ok");
    } else {
      showBanner(
        "归类完成",
        `归了 ${fmtInt(body.processed)} 条，兜底「其他」${fmtInt(body.fallback_other)} 条，` +
          `用时 ${fmtInt(body.elapsed_ms)}ms；待归类 ${fmtInt(body.pending_before)} → ${fmtInt(body.pending_after)}。`,
        "ok"
      );
    }
    await load();
  } catch (err) {
    showBanner("批跑请求失败", String(err && err.message ? err.message : err));
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

/* ---- 启动 ---------------------------------------------------------- */

$("btn-refresh").addEventListener("click", load);
$("btn-run").addEventListener("click", runBatch);
$("btn-retry").addEventListener("click", load);
$("btn-banner-close").addEventListener("click", hideBanner);

load();
