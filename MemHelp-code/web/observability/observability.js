/* 观测与成本页：钱花在哪类意图（/api/intent-cost）+ 质量有没有退（/api/eval-runs）。
 *
 * 三项数据各自独立取，一边挂了另外两块照常显示 —— 观测页最忌讳「Langfuse 抖一下整页白」。
 * 失败原因直接写进对应面板，不吞错。
 */

'use strict';

const $ = (selector) => document.querySelector(selector);

const INTENT_LABELS = {
  logistics: '物流',
  order: '订单',
  product_consult: '商品咨询',
  refund_return: '退款退货',
  after_sales: '售后',
  complaint: '投诉',
  chitchat: '闲聊',
  other: '其他',
};

const METRICS = [
  { key: 'recall@5', section: 'retrieval', label: 'recall@5' },
  { key: 'mrr@10', section: 'retrieval', label: 'mrr@10' },
  { key: 'faithfulness', section: 'generation', label: 'faithfulness' },
  { key: 'refusal_accuracy', section: 'generation', label: 'refusal_accuracy' },
];

const state = { days: 7, loading: false };

function fmtNumber(value) {
  if (value === null || value === undefined) return '—';
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString('zh-CN') : '—';
}

function fmtFixed(value, digits) {
  if (value === null || value === undefined) return '—';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '—';
}

function fmtPercent(value, total) {
  if (!total) return '0.0%';
  return `${((Number(value) / Number(total)) * 100).toFixed(1)}%`;
}

function fmtTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
}

function labelOf(intent) {
  return INTENT_LABELS[intent] || String(intent);
}

function arrow(direction, delta) {
  if (direction === 'up') return { symbol: '↑', className: 'up', text: `+${fmtFixed(delta, 3)}` };
  if (direction === 'down') return { symbol: '↓', className: 'down', text: `-${fmtFixed(Math.abs(delta), 3)}` };
  return { symbol: '→', className: 'flat', text: '持平' };
}

function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => {
    node.hidden = true;
  }, 4000);
}

async function getJson(path) {
  const resp = await fetch(path, { headers: { Accept: 'application/json' } });
  const payload = await resp.json().catch(() => null);
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    if (payload && payload.error && payload.error.message) message = payload.error.message;
    else if (payload && payload.detail) message = String(payload.detail);
    throw new Error(message);
  }
  return payload;
}

/** 把一次请求包成 {ok, value} / {ok:false, error}，好让三块面板各自降级。 */
async function settle(task) {
  try {
    return { ok: true, value: await task() };
  } catch (error) {
    return { ok: false, error: error && error.message ? error.message : String(error) };
  }
}

function renderHero(cost, runs) {
  const sub = $('#hero-sub');
  if (cost.ok && cost.value && cost.value.configured !== false) {
    const items = Array.isArray(cost.value.items) ? cost.value.items : [];
    const total = Number(cost.value.total_tokens || 0);
    sub.textContent = `最近 ${state.days} 天 · ${items.length} 类意图 · 合计 ${fmtNumber(total)} tokens；指标看最新一轮评估。`;
    if (runs.ok && runs.value && runs.value.items && runs.value.items.length) {
      sub.textContent += ` 最近评估：${fmtTime(runs.value.items[0].created_at)}。`;
    }
  } else if (cost.ok && cost.value && cost.value.configured === false) {
    sub.textContent = `最近 ${state.days} 天 · Langfuse 未接入，成本一栏为空 —— 评估指标不受影响。`;
  }
}

function renderKpis(cost) {
  const container = $('#kpis');
  container.textContent = '';
  const items = cost.ok && cost.value && Array.isArray(cost.value.items) ? cost.value.items : [];
  const total = cost.ok && cost.value ? Number(cost.value.total_tokens || 0) : null;
  const top = items.length ? items[0] : null;
  const costSum = items.reduce((sum, item) => sum + Number(item.cost || 0), 0);
  const configured = cost.ok && cost.value ? cost.value.configured !== false : false;

  const cards = [
    {
      label: '窗口 tokens',
      value: configured ? fmtNumber(total) : '—',
      foot: configured ? `最近 ${state.days} 天 · 按意图标签聚合` : 'Langfuse 未接入',
    },
    {
      label: '覆盖意图',
      value: configured ? fmtNumber(items.length) : '—',
      foot: top ? `最烧钱：${labelOf(top.intent)}` : '窗口内没有带意图标签的 trace',
    },
    {
      label: '最烧钱意图',
      value: top ? labelOf(top.intent) : '—',
      foot: top ? `${fmtNumber(top.tokens)} tokens · 占 ${fmtPercent(top.tokens, total)}` : '—',
      text: true,
    },
    {
      label: '成本合计',
      value: configured ? fmtFixed(costSum, 4) : '—',
      foot: configured ? 'Langfuse 未配单价时记为 0' : '—',
    },
  ];

  cards.forEach((card) => {
    const node = document.createElement('article');
    node.className = 'kpi';
    const label = document.createElement('span');
    label.className = 'kpi-label';
    label.textContent = card.label;
    const value = document.createElement('strong');
    value.className = card.text ? 'kpi-value text' : 'kpi-value';
    value.textContent = card.value;
    const foot = document.createElement('span');
    foot.className = 'kpi-foot';
    foot.textContent = card.foot;
    node.append(label, value, foot);
    container.append(node);
  });
}

function showCostEmpty(title, text, icon) {
  $('#cost-table').hidden = true;
  const node = $('#cost-empty');
  node.hidden = false;
  $('#cost-empty-icon').textContent = icon;
  $('#cost-empty-title').textContent = title;
  $('#cost-empty-text').textContent = text;
  $('#cost-sub').textContent = '';
}

function hideCostEmpty() {
  $('#cost-empty').hidden = true;
  $('#cost-table').hidden = false;
}

function renderCost(cost) {
  $('#cost-rows').textContent = '';

  if (!cost.ok) {
    showCostEmpty('读取意图成本失败', `${cost.error} —— 检查服务是否在跑，或看 /api/intent-cost 的后台日志。`, '!');
    return;
  }

  const payload = cost.value || {};
  if (payload.configured === false) {
    showCostEmpty(
      'Langfuse 未接入',
      '在 .env 配好 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL 后重启服务。',
      '○',
    );
    return;
  }

  const items = Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    showCostEmpty('窗口内还没有带意图标签的 trace', `先聊几轮再回来，或把窗口调大（现在是最近 ${state.days} 天）。`, '◎');
    return;
  }

  hideCostEmpty();
  const total = Number(payload.total_tokens || items.reduce((sum, item) => sum + Number(item.tokens || 0), 0));
  $('#cost-sub').textContent = `按 intent 标签聚合 · 窗口 ${state.days} 天`;

  const tbody = $('#cost-rows');
  items.forEach((item, index) => {
    const tokens = Number(item.tokens || 0);
    const share = total ? (tokens / total) * 100 : 0;

    const tr = document.createElement('tr');
    if (index === 0) tr.className = 'top';

    const intent = document.createElement('td');
    const rank = document.createElement('span');
    rank.className = 'rank';
    rank.textContent = String(index + 1);
    intent.append(rank, document.createTextNode(labelOf(item.intent)));

    const tokensCell = document.createElement('td');
    tokensCell.className = 'num';
    tokensCell.textContent = fmtNumber(tokens);

    const shareCell = document.createElement('td');
    shareCell.className = 'share';
    const shareInner = document.createElement('div');
    shareInner.className = 'share-inner';
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('span');
    fill.className = 'bar-fill';
    fill.style.width = `${share.toFixed(1)}%`;
    bar.append(fill);
    const shareNumber = document.createElement('span');
    shareNumber.className = 'share-num';
    shareNumber.textContent = `${share.toFixed(1)}%`;
    shareInner.append(bar, shareNumber);
    shareCell.append(shareInner);

    const costCell = document.createElement('td');
    costCell.className = 'num';
    costCell.textContent = fmtFixed(item.cost, 4);

    const observationsCell = document.createElement('td');
    observationsCell.className = 'num';
    observationsCell.textContent = fmtNumber(item.observations);

    tr.append(intent, tokensCell, shareCell, costCell, observationsCell);
    tbody.append(tr);
  });
}

function renderMetrics(runs) {
  const container = $('#metric-cards');
  const empty = $('#metric-empty');
  const foot = $('#metric-foot');
  container.textContent = '';
  foot.textContent = '';

  if (!runs.ok || !runs.value || !Array.isArray(runs.value.items) || !runs.value.items.length) {
    empty.hidden = false;
    foot.textContent = runs.ok ? '' : `读取评估记录失败：${runs.error}`;
    return;
  }

  empty.hidden = true;
  const round = runs.value.items[0];
  const metrics = round.metrics || {};

  METRICS.forEach((spec) => {
    const section = metrics[spec.section] || {};
    const value = section[spec.key];
    const trend = runs.value.trend && runs.value.trend[spec.key] ? runs.value.trend[spec.key] : null;

    const card = document.createElement('div');
    card.className = 'mini';

    const label = document.createElement('span');
    label.className = 'mini-label';
    label.textContent = spec.label;

    const line = document.createElement('span');
    line.className = 'mini-value';
    line.textContent = fmtFixed(value, 2);

    if (trend && trend.direction && trend.delta !== null && trend.delta !== undefined) {
      const mark = arrow(trend.direction, trend.delta);
      const delta = document.createElement('span');
      delta.className = `delta ${mark.className}`;
      delta.textContent = `${mark.symbol} ${mark.text}`;
      line.append(delta);
    }

    card.append(label, line);
    container.append(card);
  });

  foot.textContent = `最新一轮 ${fmtTime(round.created_at)} · ${round.triggered_by === 'scheduled' ? '定时' : '手动'} · ${fmtNumber(round.dataset_size)} 条标注样本`;
}

function renderHealth(health) {
  const list = $('#health-list');
  list.textContent = '';
  $('#health-time').textContent = health.ok ? '刚刚' : '读取失败';

  const rows = [];
  if (health.ok) {
    const payload = health.value || {};
    rows.push({
      label: '服务状态',
      value: payload.status === 'ok' ? '正常' : payload.status || '未知',
      className: payload.status === 'ok' ? 'ok' : 'bad',
    });
    rows.push({ label: '模型', value: payload.model || '—' });
    rows.push({
      label: '数据库',
      value: payload.database === 'ok' ? '正常' : '不可用',
      className: payload.database === 'ok' ? 'ok' : 'bad',
    });
    rows.push({
      label: '上下文预算',
      value: payload.context_token_budget ? `${fmtNumber(payload.context_token_budget)} tokens` : '—',
    });
    rows.push({
      label: '上游地址',
      value: payload.base_url_configured ? '已配置' : '未配置',
      className: payload.base_url_configured ? 'ok' : 'bad',
    });
  } else {
    rows.push({ label: '读取失败', value: health.error, className: 'bad' });
  }

  rows.forEach((row) => {
    const item = document.createElement('li');
    const label = document.createElement('span');
    label.className = 'health-label';
    label.textContent = row.label;
    const value = document.createElement('span');
    value.className = row.className ? `health-value ${row.className}` : 'health-value';
    value.textContent = row.value;
    item.append(label, value);
    list.append(item);
  });
}

function setBusy(busy) {
  state.loading = busy;
  $('#btn-refresh').disabled = busy;
}

async function load() {
  if (state.loading) return;
  setBusy(true);

  const [cost, runs, health] = await Promise.all([
    settle(() => getJson(`/api/intent-cost?days=${state.days}`)),
    settle(() => getJson('/api/eval-runs?limit=30')),
    settle(() => getJson('/healthz')),
  ]);

  renderHero(cost, runs);
  renderKpis(cost);
  renderCost(cost);
  renderMetrics(runs);
  renderHealth(health);

  setBusy(false);
  if (!cost.ok || !health.ok) {
    toast('部分数据读取失败，详见对应面板。');
  }
}

function wire() {
  document.querySelectorAll('.seg-btn').forEach((button) => {
    button.addEventListener('click', () => {
      state.days = Number(button.dataset.days) || 7;
      document.querySelectorAll('.seg-btn').forEach((node) => {
        node.classList.toggle('active', node === button);
      });
      load();
    });
  });

  $('#btn-refresh').addEventListener('click', () => load());
}

wire();
load();
