/* 评估趋势页：GET /api/eval-runs 读行 + 趋势，POST /api/eval-runs/run 手动跑一轮。 */

'use strict';

const $ = (selector) => document.querySelector(selector);

const METRICS = [
  { key: 'recall@5', label: 'recall@5', section: 'retrieval', color: '#2563eb' },
  { key: 'mrr@10', label: 'mrr@10', section: 'retrieval', color: '#7c3aed' },
  { key: 'faithfulness', label: 'faithfulness', section: 'generation', color: '#0f766e' },
  { key: 'refusal_accuracy', label: 'refusal accuracy', section: 'generation', color: '#b45309' },
];

const state = { items: [], trend: {}, running: false };

function toast(message, bad = false) {
  const box = $('#toast');
  box.textContent = message;
  box.classList.toggle('bad', bad);
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 3400);
}

async function api(path, options) {
  const resp = await fetch(path, options);
  const payload = await resp.json().catch(() => null);
  if (!resp.ok) {
    const message = (payload && payload.error && payload.error.message)
      || (payload && payload.detail)
      || `HTTP ${resp.status}`;
    throw new Error(message);
  }
  return payload;
}

function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function metricOf(item, metric) {
  const block = (item.metrics && item.metrics[metric.section]) || {};
  const value = block[metric.key];
  return typeof value === 'number' ? value : null;
}

function pct(value) {
  return value === null || value === undefined ? '—' : value.toFixed(3);
}

function arrow(direction, delta) {
  const symbol = direction === 'up' ? '↑' : direction === 'down' ? '↓' : '→';
  const node = document.createElement('span');
  node.className = `arrow ${direction}`;
  node.textContent = symbol;
  if (typeof delta === 'number') {
    node.title = `相对上一轮 ${delta > 0 ? '+' : ''}${delta.toFixed(4)}`;
  }
  return node;
}

function sparkline(values, color) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 100 42');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.classList.add('spark');

  const points = values.length === 1 ? [values[0], values[0]] : values;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const coords = points.map((value, index) => {
    const x = points.length === 1 ? 0 : (index / (points.length - 1)) * 100;
    const y = 38 - ((value - min) / span) * 32;
    return [x, y];
  });

  const line = coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
  const area = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  area.setAttribute('points', `0,42 ${line} 100,42`);
  area.setAttribute('fill', color);
  area.setAttribute('class', 'spark-area');

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  path.setAttribute('points', line);
  path.setAttribute('stroke', color);
  path.setAttribute('class', 'sparkline');

  svg.append(area, path);
  return svg;
}

function renderCards() {
  const box = $('#cards');
  box.replaceChildren();

  if (!state.items.length) return;

  for (const metric of METRICS) {
    const latest = state.trend[metric.key] || {};
    const value = latest.latest ?? metricOf(state.items[0], metric);

    const card = document.createElement('div');
    card.className = 'card';

    const top = document.createElement('div');
    top.className = 'card-top';
    const label = document.createElement('span');
    label.className = 'card-label';
    label.textContent = metric.label;
    top.appendChild(label);
    if (latest.direction) top.appendChild(arrow(latest.direction, latest.delta));

    const number = document.createElement('div');
    number.className = 'card-value';
    number.textContent = pct(value);

    const delta = document.createElement('div');
    delta.className = 'card-delta';
    if (typeof latest.delta === 'number') {
      delta.textContent = `较上轮 ${latest.delta > 0 ? '+' : ''}${latest.delta.toFixed(4)}`;
    } else {
      delta.textContent = '还没有上一轮可比较';
    }

    // items 是新→旧，画图要旧→新。
    const history = [...state.items].reverse()
      .map((item) => metricOf(item, metric))
      .filter((item) => typeof item === 'number');

    card.append(top, number, delta);
    if (history.length) card.appendChild(sparkline(history, metric.color));
    box.appendChild(card);
  }
}

function renderTable() {
  const body = $('#rows');
  body.replaceChildren();

  if (!state.items.length) {
    $('#empty').hidden = false;
    $('#run-count').textContent = '';
    return;
  }
  $('#empty').hidden = true;
  $('#run-count').textContent = `共 ${state.items.length} 轮`;

  state.items.forEach((item, index) => {
    const row = document.createElement('tr');
    if (index === 0) row.className = 'latest';

    const time = document.createElement('td');
    time.textContent = formatTime(item.created_at);

    const trigger = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'trigger';
    badge.textContent = item.triggered_by;
    trigger.appendChild(badge);

    const size = document.createElement('td');
    size.className = 'num';
    size.textContent = String(item.dataset_size ?? '—');

    row.append(time, trigger, size);

    for (const metric of METRICS) {
      const cell = document.createElement('td');
      cell.className = 'num';
      const value = metricOf(item, metric);
      const wrap = document.createElement('span');
      wrap.className = 'metric';
      const number = document.createElement('span');
      number.textContent = pct(value);
      if (value === null) number.className = 'none';
      wrap.appendChild(number);
      if (index === 0 && state.trend[metric.key] && state.trend[metric.key].direction) {
        wrap.appendChild(arrow(
          state.trend[metric.key].direction,
          state.trend[metric.key].delta,
        ));
      }
      cell.appendChild(wrap);
      row.appendChild(cell);
    }

    body.appendChild(row);
  });
}

async function load() {
  try {
    const payload = await api('/api/eval-runs?limit=30');
    state.items = payload.items || [];
    state.trend = payload.trend || {};
    $('#error-panel').hidden = true;
    renderCards();
    renderTable();
    $('#hero-sub').textContent = state.items.length
      ? `最近 ${state.items.length} 轮 · 最新一轮 ${formatTime(state.items[0].created_at)}`
      : '最近几轮 ch04 评估集的结果，按时间从左到右。';
  } catch (error) {
    $('#rows').replaceChildren();
    $('#cards').replaceChildren();
    $('#empty').hidden = true;
    $('#error-panel').hidden = false;
    $('#error-message').textContent = error.message;
  }
}

async function waitForNewRun(baselineId) {
  const started = Date.now();
  const limit = 10 * 60 * 1000;
  while (Date.now() - started < limit) {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    const payload = await api('/api/eval-runs?limit=30');
    const items = payload.items || [];
    if (items.length && items[0].id !== baselineId) {
      state.items = items;
      state.trend = payload.trend || {};
      renderCards();
      renderTable();
      return true;
    }
  }
  return false;
}

async function triggerRun() {
  if (state.running) return;
  state.running = true;
  const button = $('#btn-run');
  button.disabled = true;
  button.textContent = '评估运行中…';
  const baselineId = state.items.length ? state.items[0].id : null;

  try {
    await api('/api/eval-runs/run', { method: 'POST' });
    toast('已开始，跑完会自动刷新（最多 10 分钟）');
    const appeared = await waitForNewRun(baselineId);
    if (appeared) {
      toast('新一轮评估已写入');
    } else {
      toast('10 分钟内没有看到新结果，稍后刷新看看', true);
      await load();
    }
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.running = false;
    button.disabled = false;
    button.textContent = '立即跑一轮';
  }
}

$('#btn-run').addEventListener('click', triggerRun);
$('#btn-retry').addEventListener('click', load);

load();
