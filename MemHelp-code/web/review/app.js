/* 待审队列后台：列表 / 详情 / 通过 / 驳回。
 *
 * 契约见 app/api.py 的 /api/review-queue 系列：
 *   GET  /api/review-queue?status=待审
 *   GET  /api/review-queue/{id}       -> {review, questions, snapshot}
 *   POST /api/review-queue/{id}/approve {approved_answer}
 *   POST /api/review-queue/{id}/reject
 */

'use strict';

const $ = (selector) => document.querySelector(selector);

const STATUS_CLASS = { '待审': 'pending', '通过': 'approved', '驳回': 'rejected' };
const SOURCE_LABEL = {
  retrieval_low_conf: '检索低置信',
  self_check: '模型自评',
  user_feedback: '用户 👎',
};

const state = {
  status: '待审',
  rows: [],
  detail: null,
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function toast(message, bad = false) {
  const box = $('#toast');
  box.textContent = message;
  box.classList.toggle('bad', bad);
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 3200);
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
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function snippet(text, length = 90) {
  const value = (text || '').replace(/\s+/g, ' ').trim();
  return value.length > length ? `${value.slice(0, length)}…` : (value || '—');
}

// ---------------------------------------------------------------- 列表

function renderRows() {
  const box = $('#rows');
  box.replaceChildren();

  if (!state.rows.length) {
    $('#empty').hidden = false;
    $('#empty-title').textContent = state.status ? `暂无${state.status}问题` : '队列还是空的';
    $('#empty-hint').textContent = state.status === '待审'
      ? '用户答不上来的问题会自动出现在这里。'
      : '换个状态看看，或等飞轮再送来新的缺口。';
    return;
  }
  $('#empty').hidden = true;

  for (const item of state.rows) {
    const row = el('div', 'row');
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-label', `查看详情：${item.normalized_question}`);

    const question = el('div', 'q');
    question.appendChild(document.createTextNode(item.normalized_question));
    if (item.updated_at) {
      question.appendChild(el('small', '', `更新于 ${formatTime(item.updated_at)}`));
    }

    const count = el('div');
    count.appendChild(el('span', 'count-pill', String(item.occurrence_count || 1)));

    const answer = el('div', 'snippet', snippet(item.approved_answer || item.ai_suggested_answer));

    const status = el('div');
    status.appendChild(el(
      'span',
      `status ${STATUS_CLASS[item.review_status] || ''}`,
      item.review_status,
    ));

    const actions = el('div', 'row-actions');
    if (item.review_status === '待审') {
      const approve = el('button', 'btn-primary', '通过');
      approve.type = 'button';
      approve.addEventListener('click', (event) => {
        event.stopPropagation();
        openApprove(item);
      });
      const reject = el('button', 'btn-ghost', '驳回');
      reject.type = 'button';
      reject.addEventListener('click', (event) => {
        event.stopPropagation();
        rejectRow(item);
      });
      actions.append(approve, reject);
    } else {
      const view = el('button', 'btn-ghost', '查看');
      view.type = 'button';
      view.addEventListener('click', (event) => {
        event.stopPropagation();
        openDetail(item.id);
      });
      actions.appendChild(view);
    }

    row.append(question, count, answer, status, actions);
    row.addEventListener('click', () => openDetail(item.id));
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openDetail(item.id);
      }
    });
    box.appendChild(row);
  }
}

async function loadRows() {
  const params = new URLSearchParams();
  if (state.status) params.set('status', state.status);
  params.set('limit', '100');

  try {
    const payload = await api(`/api/review-queue?${params}`);
    state.rows = payload.items || [];
    $('#error-panel').hidden = true;
    $('#queue-sub').textContent = `${state.rows.length} 条 · ${state.status || '全部'}`;
    renderRows();
  } catch (error) {
    $('#rows').replaceChildren();
    $('#empty').hidden = true;
    $('#error-panel').hidden = false;
    $('#error-message').textContent = error.message;
    $('#queue-sub').textContent = '读取失败';
  }
}

// ---------------------------------------------------------------- 详情

function renderDetail(detail) {
  state.detail = detail;
  const review = detail.review || {};

  $('#detail-status').textContent = `${review.review_status || '待审'} · 出现 ${review.occurrence_count || 1} 次`;
  $('#detail-question').textContent = review.normalized_question || '—';
  $('#detail-suggested').textContent = review.ai_suggested_answer || '（模型没给出建议答案）';

  const questions = detail.questions || [];
  $('#detail-count').textContent = String(questions.length);
  const list = $('#detail-questions');
  list.replaceChildren();
  if (!questions.length) {
    list.appendChild(el('li', 'hint', '这条待审问题还没有归并任何原始提问。'));
  }
  for (const question of questions) {
    const item = el('li');
    item.appendChild(el('div', '', question.raw_question));
    const meta = el('div', 'meta');
    meta.appendChild(el('span', 'badge', SOURCE_LABEL[question.source] || question.source || '未知来源'));
    meta.appendChild(el('span', '', formatTime(question.created_at)));
    if (question.reason) meta.appendChild(el('span', '', question.reason));
    item.appendChild(meta);
    list.appendChild(item);
  }

  const chunks = detail.snapshot || [];
  const chunkBox = $('#detail-chunks');
  chunkBox.replaceChildren();
  if (!chunks.length) {
    chunkBox.appendChild(el('p', 'hint', '这一轮没有召回片段（可能是闲聊/业务数据类，或事后回捞失败）。'));
  }
  for (const chunk of chunks) {
    const card = el('div', 'chunk');
    const head = el('div', 'chunk-head');
    const left = el('div');
    left.appendChild(el('strong', '', chunk.section_path || chunk.category || `片段 #${chunk.n || '?'}`));
    if (chunk.chunk_id) left.appendChild(document.createTextNode(` · #${chunk.chunk_id}`));
    head.appendChild(left);
    head.appendChild(el('span', 'score', chunk.score === null || chunk.score === undefined
      ? '无分数'
      : Number(chunk.score).toFixed(3)));
    const pre = el('pre');
    pre.textContent = chunk.quote || '（片段原文为空）';
    card.append(head, pre);
    chunkBox.appendChild(card);
  }

  const approvedBlock = $('#approved-block');
  approvedBlock.hidden = !review.approved_answer;
  $('#detail-approved').textContent = review.approved_answer || '';

  const actions = $('#detail-actions');
  actions.hidden = review.review_status !== '待审';

  $('#drawer-backdrop').hidden = false;
  $('#drawer').hidden = false;
  $('#detail-question').scrollIntoView({ block: 'nearest' });
}

async function openDetail(reviewId) {
  try {
    const detail = await api(`/api/review-queue/${reviewId}`);
    renderDetail(detail);
  } catch (error) {
    toast(error.message, true);
  }
}

function closeDrawer() {
  $('#drawer').hidden = true;
  $('#drawer-backdrop').hidden = true;
  state.detail = null;
}

// ---------------------------------------------------------------- 通过 / 驳回

function openApprove(item) {
  const current = state.detail && state.detail.review && state.detail.review.id === item.id
    ? state.detail.review
    : item;
  $('#approve-answer').value = current.approved_answer || item.ai_suggested_answer || '';
  $('#approve-modal').dataset.reviewId = String(item.id);
  $('#approve-backdrop').hidden = false;
  $('#approve-modal').hidden = false;
  $('#approve-answer').focus();
}

function closeApprove() {
  $('#approve-backdrop').hidden = true;
  $('#approve-modal').hidden = true;
  $('#btn-confirm-approve').disabled = false;
}

async function confirmApprove() {
  const reviewId = $('#approve-modal').dataset.reviewId;
  const answer = $('#approve-answer').value.trim();
  if (!answer) {
    toast('核准答案不能为空', true);
    return;
  }

  const button = $('#btn-confirm-approve');
  button.disabled = true;
  try {
    const payload = await api(`/api/review-queue/${reviewId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved_answer: answer }),
    });
    closeApprove();
    closeDrawer();
    toast(payload.vectorize_status === 'ok'
      ? '已通过并写入知识库，下次提问即可命中'
      : '已通过并写入知识库；向量化待续跑，稍后可检索');
    await loadRows();
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
  }
}

async function rejectRow(item) {
  if (!window.confirm(`驳回「${item.normalized_question}」？驳回后不会写入知识库。`)) return;
  try {
    await api(`/api/review-queue/${item.id}/reject`, { method: 'POST' });
    toast('已驳回');
    closeDrawer();
    await loadRows();
  } catch (error) {
    toast(error.message, true);
  }
}

// ---------------------------------------------------------------- 绑定

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((other) => {
      other.classList.toggle('active', other === tab);
      other.setAttribute('aria-selected', other === tab ? 'true' : 'false');
    });
    state.status = tab.dataset.status;
    loadRows();
  });
});

$('#btn-refresh').addEventListener('click', loadRows);
$('#btn-retry').addEventListener('click', loadRows);
$('#btn-close-drawer').addEventListener('click', closeDrawer);
$('#drawer-backdrop').addEventListener('click', closeDrawer);
$('#btn-approve').addEventListener('click', () => {
  if (state.detail) openApprove(state.detail.review);
});
$('#btn-reject').addEventListener('click', () => {
  if (state.detail) rejectRow(state.detail.review);
});
$('#btn-cancel-approve').addEventListener('click', closeApprove);
$('#approve-backdrop').addEventListener('click', closeApprove);
$('#btn-confirm-approve').addEventListener('click', confirmApprove);
$('#approve-answer').addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') confirmApprove();
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!$('#approve-modal').hidden) closeApprove();
  else if (!$('#drawer').hidden) closeDrawer();
});

loadRows();
