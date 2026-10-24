/* MewHelp 客服对话前端。
 *
 * 为什么不用 EventSource：后端是 POST /api/chat/completions + SSE，而
 * EventSource 只支持 GET、且不能带请求体。所以这里用 fetch 拿 ReadableStream，
 * 自己按 SSE 规范切帧（event: / data: / 空行分隔）。
 *
 * 后端事件契约（见 app/api.py::_chat_stream_body）：
 *   event: meta         data: {session_id, model}
 *   event: status       data: {tool, args, label, call_id}
 *   event: tool_result  data: {call_id, tool: {name, args, ok, duration_ms, ...}}
 *   event: tools        data: {tools}          ← 本轮结束时的权威轨迹
 *   event: token        data: {delta}          ← 逐字增量
 *   event: done         data: {finish_reason, session_id, trimmed, kept, tools}
 *   event: error        data: {type, message, session_id}
 *
 * status 帧让工具记录立刻以「执行中」出现；tool_result 帧把同一条记录翻成
 * 成功/失败；done.tools 与历史回读再用完整摘要校对一遍。历史旧数据可能没有
 * started_at / result / error，渲染函数会降级显示而不是报错。
 */

(() => {
  'use strict';

  const API = '/api/chat/completions';
  const SESSION_KEY = 'mewhelp.session_id';
  const CONVERSATION_KEY = 'mewhelp.conversation_id';
  const CONVERSATIONS_API = '/api/conversations';
  const USER_ID = 'anonymous';

  const els = {
    messages: document.getElementById('messages'),
    input: document.getElementById('input'),
    send: document.getElementById('send'),
    clear: document.getElementById('btn-clear'),
    stats: document.getElementById('show-stats'),
    status: document.getElementById('status'),
    statusText: document.getElementById('status-text'),
    sidebar: document.getElementById('sidebar'),
    sidebarList: document.getElementById('sidebar-list'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    newChat: document.getElementById('btn-new-chat'),
  };

  let sessionId = sessionStorage.getItem(SESSION_KEY) || null;
  /* 侧栏当前高亮的会话行（``conversations.id``）。**与 sessionId 是两个 id**：
     列表给的是主键，聊天用 session_id，切换时要同时换掉两个，否则「点的是 A、
     说的话却落进 B」（Review Focus 5）。 */
  let conversationId = (() => {
    const raw = sessionStorage.getItem(CONVERSATION_KEY);
    const n = raw == null ? NaN : Number(raw);
    return Number.isFinite(n) ? n : null;
  })();
  let busy = false;
  let stickToBottom = true;   // 用户往上翻时不要强行拉回底部

  // ------------------------------------------------------------------ 工具

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  /* 轻量 markdown：先转义，再只放行 **粗体**、`行内代码` 与换行。
     刻意不做完整 markdown —— 客服回复里用不到，也少一个 XSS 面。 */
  function renderMarkdown(raw) {
    return escapeHtml(raw)
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  function setStatus(text, kind) {
    els.statusText.textContent = text;
    els.status.className = 'brand-status' + (kind ? ' ' + kind : '');
  }

  function atBottom() {
    const m = els.messages;
    return m.scrollHeight - m.scrollTop - m.clientHeight < 60;
  }

  function scrollToBottom(force) {
    if (!force && !stickToBottom) return;
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  // ------------------------------------------------------------------ 渲染

  function clearWelcome() {
    const w = els.messages.querySelector('.welcome');
    if (w) w.remove();
  }

  function addNotice(text, isError) {
    clearWelcome();
    const n = el('div', 'notice' + (isError ? ' error' : ''), text);
    els.messages.appendChild(n);
    scrollToBottom(true);
    return n;
  }

  function addRow(role, { typing = false } = {}) {
    clearWelcome();
    const row = el('div', 'row ' + role);
    const avatar = el('div', 'avatar ' + (role === 'user' ? 'me' : 'bot'),
                       role === 'user' ? '我' : '美');
    const wrap = el('div', 'bubble-wrap');
    const bubble = el('div', 'bubble');
    // 工具调用记录是气泡内容的第一块，顺序与飞书截图一致：记录在上，回答在下。
    const tools = el('div', 'tool-records');
    tools.hidden = true;
    tools.setAttribute('aria-label', '工具调用记录');
    // ch06：指代补全/意图/订单选择这些流程信息放在正文上方，不混进回答文本。
    const flow = el('div', 'flow-notes');
    flow.hidden = true;
    const body = el('div', 'bubble-content');

    if (typing) {
      const dots = el('span', 'typing');
      dots.innerHTML = '<i></i><i></i><i></i>';
      body.appendChild(dots);
    }

    bubble.appendChild(tools);
    bubble.appendChild(flow);
    bubble.appendChild(body);
    wrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrap);
    els.messages.appendChild(row);
    scrollToBottom(true);

    return {
      row,
      wrap,
      bubble,
      body,
      tools,
      flow,
      // 渲染状态挂在消息节点上：多次工具调用、增量结果与最终 done 都改这一份。
      toolState: { items: [], open: new Set() },
    };
  }

  /* ------------------------------------------------------- 工具调用记录
   *
   * 一条记录三种状态：running / success / failed。所有用户可见文本都走
   * ``textContent``（工具名、参数、结果都可能来自模型），只有固定的 SVG 图标
   * 使用 innerHTML。
   *
   * 默认折叠成虚线胶囊，点击展开参数、结果、状态、耗时与调用时间；失败时
   * 自动展开，让用户不用再点一次就能看到原因。
   */
  const TOOL_LABELS = {
    query_order: '查询订单',
    query_product: '查询商品',
    query_logistics: '查询物流',
    query_faq: '检索常见问题',
    create_ticket: '创建工单',
  };

  const TOOL_ICONS = {
    wrench: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4 4 0 0 0-5.1 5.1L3 18l3 3 6.6-6.6a4 4 0 0 0 5.1-5.1l-2.3 2.3-2.3-.6-.6-2.3 2.2-2.4Z"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',
    alert: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 21h20L12 3Z"/><path d="M12 9v5m0 3h.01"/></svg>',
    chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>',
  };

  let toolRecordSeq = 0;

  function normaliseTool(raw, fallback = {}) {
    const source = raw || {};
    const ok = typeof source.ok === 'boolean'
      ? source.ok
      : (typeof fallback.ok === 'boolean' ? fallback.ok : undefined);
    const status = source.status
      || (ok === false ? 'failed' : ok === true ? 'success' : (fallback.status || 'running'));
    return {
      call_id: source.call_id || fallback.call_id || '',
      name: source.name || fallback.name || '未知工具',
      args: source.args !== undefined ? source.args : (fallback.args || {}),
      ok,
      status,
      duration_ms: typeof source.duration_ms === 'number'
        ? source.duration_ms
        : (typeof fallback.duration_ms === 'number' ? fallback.duration_ms : null),
      attempts: typeof source.attempts === 'number'
        ? source.attempts
        : (typeof fallback.attempts === 'number' ? fallback.attempts : null),
      started_at: source.started_at || fallback.started_at || null,
      result: source.result !== undefined ? source.result : (fallback.result ?? null),
      error: source.error !== undefined ? source.error : (fallback.error ?? null),
      label: source.label || fallback.label || '',
    };
  }

  function toolStatusOf(item) {
    if (item.status === 'failed') return { key: 'failed', label: '失败' };
    if (item.status === 'success') return { key: 'success', label: '成功' };
    return { key: 'running', label: '执行中' };
  }

  function toolKey(item, index = 0) {
    if (item.call_id) return 'call:' + item.call_id;
    return `${item.name}:${JSON.stringify(item.args || {})}:${index}`;
  }

  function prettyToolValue(value, emptyText = '无') {
    if (value === null || value === undefined || value === '') return emptyText;
    let parsed = value;
    if (typeof value === 'string') {
      try { parsed = JSON.parse(value); } catch { return value; }
    }
    try { return JSON.stringify(parsed, null, 2); } catch { return String(value); }
  }

  function formatDuration(ms) {
    return typeof ms === 'number' ? `${ms} ms` : '未记录';
  }

  function formatToolTime(iso) {
    if (!iso) return '未记录';
    const when = new Date(iso);
    if (Number.isNaN(when.getTime())) return String(iso);
    return when.toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });
  }

  function toolDetailRow(label, value, { pre = false, mono = false } = {}) {
    const row = el('div', 'tool-record-detail-row');
    row.appendChild(el('span', 'tool-record-detail-label', label));
    const valueEl = el(pre ? 'pre' : 'span',
      'tool-record-detail-value' + (mono ? ' mono' : ''), value);
    row.appendChild(valueEl);
    return row;
  }

  function renderToolRecord(target, item, index) {
    const state = toolStatusOf(item);
    const key = toolKey(item, index);
    const open = target.toolState.open.has(key);
    const detailId = `tool-record-detail-${++toolRecordSeq}`;

    const box = el('section',
      `tool-record is-${state.key}` + (open ? ' is-open' : ''));
    box.dataset.toolKey = key;

    const button = el('button', 'tool-record-summary');
    button.type = 'button';
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
    button.setAttribute('aria-controls', detailId);
    button.title = TOOL_LABELS[item.name] || item.name;

    const icon = el('span', 'tool-record-icon');
    icon.innerHTML = TOOL_ICONS.wrench;

    const label = el('span', 'tool-record-label');
    label.appendChild(document.createTextNode('调用了 '));
    label.appendChild(el('code', 'tool-record-name', item.name));

    const meta = el('span', 'tool-record-meta');
    const chip = el('span', `tool-record-status ${state.key}`);
    if (state.key === 'running') {
      chip.appendChild(el('span', 'tool-record-spinner'));
    } else {
      const stateIcon = el('span', 'tool-record-state-icon');
      stateIcon.innerHTML = state.key === 'success' ? TOOL_ICONS.check : TOOL_ICONS.alert;
      chip.appendChild(stateIcon);
    }
    chip.appendChild(document.createTextNode(state.label));
    meta.appendChild(chip);
    if (typeof item.duration_ms === 'number') {
      meta.appendChild(el('span', 'tool-record-duration', formatDuration(item.duration_ms)));
    }

    const chevron = el('span', 'tool-record-chevron');
    chevron.innerHTML = TOOL_ICONS.chevron;

    button.appendChild(icon);
    button.appendChild(label);
    button.appendChild(meta);
    button.appendChild(chevron);

    const detail = el('div', 'tool-record-detail');
    detail.id = detailId;
    detail.hidden = !open;

    detail.appendChild(toolDetailRow('调用参数', prettyToolValue(item.args), {
      pre: true, mono: true,
    }));
    if (state.key === 'failed') {
      detail.appendChild(toolDetailRow('失败原因',
        item.error || '工具没有返回失败原因', { pre: true }));
    } else {
      detail.appendChild(toolDetailRow('调用结果',
        item.result ? prettyToolValue(item.result) : '暂无结果快照', {
          pre: true, mono: Boolean(item.result),
        }));
    }
    detail.appendChild(toolDetailRow('状态', state.label));
    detail.appendChild(toolDetailRow('耗时', formatDuration(item.duration_ms)));
    detail.appendChild(toolDetailRow('尝试次数',
      item.attempts === null ? '未记录' : `${item.attempts} 次`));
    detail.appendChild(toolDetailRow('调用时间', formatToolTime(item.started_at)));

    button.addEventListener('click', () => {
      const isOpen = target.toolState.open.has(key);
      if (isOpen) target.toolState.open.delete(key);
      else target.toolState.open.add(key);
      box.classList.toggle('is-open', !isOpen);
      detail.hidden = isOpen;
      button.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
    });

    box.appendChild(button);
    box.appendChild(detail);
    return box;
  }

  function renderToolRecords(target) {
    const items = target.toolState.items;
    target.tools.innerHTML = '';
    target.tools.hidden = items.length === 0;
    items.forEach((item, index) => {
      target.tools.appendChild(renderToolRecord(target, item, index));
    });
  }

  /* 用权威摘要重建整组记录（done.tools 与 /messages 回读共用）。
     用下标保留用户已展开的行，刷新或流结束时不会突然收起。 */
  function setToolRecords(target, tools) {
    const previous = target.toolState.items || [];
    const previousOpen = new Set();
    previous.forEach((item, index) => {
      if (target.toolState.open.has(toolKey(item, index))) previousOpen.add(index);
    });

    const items = (tools || []).map((raw) => normaliseTool(raw));
    target.toolState.items = items;
    target.toolState.open = new Set();
    items.forEach((item, index) => {
      if (previousOpen.has(index) || item.status === 'failed') {
        target.toolState.open.add(toolKey(item, index));
      }
    });

    renderToolRecords(target);
    if (tools && tools.length) target.row.dataset.tools = JSON.stringify(tools);
    scrollToBottom();
  }

  function addRunningTool(target, data) {
    if (!data || !data.tool || data.tool === 'suggest_actions') return;
    const item = normaliseTool({
      call_id: data.call_id || '',
      name: data.tool,
      args: data.args || {},
      status: 'running',
      label: data.label || '',
    });
    const existing = item.call_id
      ? target.toolState.items.findIndex((it) => it.call_id === item.call_id)
      : -1;
    if (existing >= 0) target.toolState.items[existing] = item;
    else target.toolState.items.push(item);
    renderToolRecords(target);
    scrollToBottom();
  }

  function applyToolResult(target, data) {
    if (!data || !data.tool) return;
    const raw = data.tool;
    const callId = data.call_id || raw.call_id || '';
    let index = callId
      ? target.toolState.items.findIndex((it) => it.call_id === callId)
      : -1;
    if (index < 0) {
      index = target.toolState.items.findIndex((it) => (
        it.status === 'running'
        && it.name === raw.name
        && JSON.stringify(it.args || {}) === JSON.stringify(raw.args || {})
      ));
    }
    const fallback = index >= 0 ? target.toolState.items[index] : {};
    const item = normaliseTool({ ...raw, call_id: callId || fallback.call_id }, fallback);
    if (index < 0) {
      target.toolState.items.push(item);
      index = target.toolState.items.length - 1;
    } else {
      target.toolState.items[index] = item;
    }
    if (item.status === 'failed') target.toolState.open.add(toolKey(item, index));
    renderToolRecords(target);
    scrollToBottom();
  }

  function failRunningTools(target, message) {
    let changed = false;
    target.toolState.items.forEach((item, index) => {
      if (item.status !== 'running') return;
      item.status = 'failed';
      item.ok = false;
      item.error = item.error || message || '调用中断';
      target.toolState.open.add(toolKey(item, index));
      changed = true;
    });
    if (changed) renderToolRecords(target);
  }

  function welcome() {
    const w = el('div', 'welcome');
    w.appendChild(el('h1', null, '你好，我是 MewHelp 商城客服小美'));
    w.appendChild(el('p', null, '退换货、退款、物流、发票、商品质量都可以找我。'));

    const chips = el('div', 'chips');
    [
      '我上周买的耳机还没发货',
      '订单 20240115001 昨天就该到，现在还没收到，我要退款',
      '收到的杯子是碎的，想换一个',
    ].forEach((text) => {
      const b = el('button', 'chip', text);
      b.type = 'button';
      b.addEventListener('click', () => {
        els.input.value = text;
        autoGrow();
        send();
      });
      chips.appendChild(b);
    });

    w.appendChild(chips);
    els.messages.appendChild(w);
  }

  // ------------------------------------------------------- SSE 帧解析与应用

  /* 把一段 SSE 文本切成帧。SSE 规范用空行分隔事件，行尾可能是 \n 或 \r\n。
     用正则 exec 拿准确长度，避免手工推断分隔符宽度。 */
  const FRAME_SEP = /\r?\n\r?\n/;

  function parseFrame(block) {
    let event = 'message';
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return null;
    try {
      return { event, data: JSON.parse(dataLines.join('\n')) };
    } catch {
      return null;               // 单个坏帧不该炸掉整段回复
    }
  }

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

  function addFlowNote(target, text) {
    if (!text) return;
    target.flow.hidden = false;
    target.flow.appendChild(el('div', 'flow-note', text));
  }

  function showCoref(target, data) {
    if (!data || !data.changed || !data.resolved) return;
    addFlowNote(target, `已补全：${data.resolved}`);
  }

  function showIntent(target, data) {
    if (!data || !data.intent) return;
    const label = INTENT_LABELS[data.intent] || data.intent;
    const confidence = typeof data.confidence === 'number'
      ? ` · ${Math.round(data.confidence * 100)}%` : '';
    addFlowNote(target, `意图：${label}${confidence}`);
  }

  function formatCardTime(iso) {
    if (!iso) return '';
    const when = new Date(iso);
    if (Number.isNaN(when.getTime())) return String(iso);
    return when.toLocaleString('zh-CN', {
      month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
    });
  }

  function renderOrderPicker(target, data, ctx) {
    if (!data || !Array.isArray(data.orders) || !data.orders.length) return;
    const box = el('div', 'order-picker');
    box.appendChild(el('div', 'order-picker-title', '请选择要办理的订单'));
    data.orders.forEach((order) => {
      const card = el('button', 'order-card');
      card.type = 'button';
      card.dataset.orderId = order.order_id || '';
      const head = el('div', 'order-card-head');
      head.appendChild(el('span', 'order-card-title', order.title || order.order_id || '订单'));
      head.appendChild(el('span', 'order-card-amount',
        order.amount_yuan ? `¥${order.amount_yuan}` : ''));
      card.appendChild(head);
      card.appendChild(el('div', 'order-card-meta',
        [order.order_id, order.status, formatCardTime(order.created_at)]
          .filter(Boolean).join(' · ')));
      card.appendChild(el('span', 'order-card-hint', '点击选择这一单'));
      card.addEventListener('click', () => {
        if (card.disabled) return;
        box.querySelectorAll('button').forEach((button) => { button.disabled = true; });
        resumeRefund(ctx, order, target);
      });
      box.appendChild(card);
    });
    target.flow.hidden = false;
    target.flow.appendChild(box);
  }

  async function resumeRefund(ctx, order, target) {
    if (!sessionId || !order || !order.order_id) {
      addNotice('会话或订单信息缺失，请重新发起退款。', true);
      return;
    }
    addFlowNote(target, `已选择订单 ${order.order_id}，正在继续…`);
    setStatus('正在继续…', 'ok');
    try {
      const resp = await fetch('/api/chat/refund/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, order_id: order.order_id }),
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => null);
        const err = (payload && payload.error) || {};
        throw new Error(err.message || `HTTP ${resp.status}`);
      }
      ctx.awaiting = false;
      ctx.acc = '';
      ctx.bot.body.innerHTML = '';
      await consumeStream(resp, ctx);
      ctx.finish();
      setStatus('已连接', 'ok');
    } catch (err) {
      addNotice(`继续退款流程失败：${err && err.message ? err.message : err}`, true);
      setStatus('继续失败', 'bad');
      target.flow.querySelectorAll('.order-card').forEach((card) => { card.disabled = false; });
    } finally {
      if (!ctx.awaiting) {
        els.input.disabled = false;
        els.send.disabled = false;
        els.input.focus();
      }
    }
  }

  function renderTicketPreview(target, data, ctx) {
    const preview = (data && data.preview) || {};
    const box = el('div', 'ticket-preview');
    box.appendChild(el('div', 'ticket-preview-title', '工单预览'));
    box.appendChild(el('div', 'ticket-preview-hint', '确认后才会真正创建工单。'));

    const rows = el('div', 'ticket-preview-rows');
    [
      ['工单类型', preview.ticket_type || '—'],
      ['问题描述', preview.description || '—'],
    ].forEach(([key, value]) => {
      const row = el('div', 'ticket-preview-row');
      row.appendChild(el('span', 'ticket-preview-key', key));
      row.appendChild(el('span', 'ticket-preview-value', value));
      rows.appendChild(row);
    });
    box.appendChild(rows);

    const actions = el('div', 'ticket-preview-actions');
    const confirm = el('button', 'ticket-confirm', '确认提交');
    confirm.type = 'button';
    const cancel = el('button', 'ticket-cancel', '取消');
    cancel.type = 'button';
    confirm.addEventListener('click', () => {
      if (!confirm.disabled) confirmTicket(ctx, true, target, box);
    });
    cancel.addEventListener('click', () => {
      if (!cancel.disabled) confirmTicket(ctx, false, target, box);
    });
    actions.appendChild(confirm);
    actions.appendChild(cancel);
    box.appendChild(actions);

    target.flow.hidden = false;
    target.flow.appendChild(box);
  }

  async function confirmTicket(ctx, confirmed, target, box) {
    if (!sessionId) {
      addNotice('会话信息缺失，请刷新后重试。', true);
      return;
    }
    const buttons = box ? box.querySelectorAll('button') : [];
    buttons.forEach((button) => { button.disabled = true; });
    addFlowNote(target, confirmed ? '已确认提交，正在建单…' : '已取消建单。');
    setStatus(confirmed ? '正在建单…' : '正在取消…', 'ok');
    try {
      const resp = await fetch('/api/chat/ticket/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, confirmed: !!confirmed }),
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => null);
        const err = (payload && payload.error) || {};
        throw new Error(err.message || `HTTP ${resp.status}`);
      }
      ctx.awaiting = false;
      ctx.acc = '';
      ctx.bot.body.innerHTML = '';
      await consumeStream(resp, ctx);
      ctx.finish();
      setStatus('已连接', 'ok');
    } catch (err) {
      addNotice(`工单确认失败：${err && err.message ? err.message : err}`, true);
      setStatus('确认失败', 'bad');
      buttons.forEach((button) => { button.disabled = false; });
    } finally {
      if (!ctx.awaiting) {
        els.input.disabled = false;
        els.send.disabled = false;
        els.input.focus();
      }
    }
  }

  function attachRefundForm(target, form) {
    if (!form || !form.order_id) return;
    const existing = target.wrap.querySelector('.refund-form');
    if (existing) existing.remove();

    const box = el('form', 'refund-form');
    box.appendChild(el('div', 'refund-form-title', '申请退款'));
    box.appendChild(el('div', 'refund-form-order', form.order_label || form.order_id));

    const row = el('div', 'refund-form-row');
    const label = el('label', 'refund-form-label', '退款原因');
    const select = el('select', 'refund-select');
    (form.reasons || []).forEach((reason) => {
      const option = el('option', null, reason);
      option.value = reason;
      select.appendChild(option);
    });
    label.appendChild(select);
    row.appendChild(label);
    const submit = el('button', 'refund-submit', '提交退款申请');
    submit.type = 'submit';
    row.appendChild(submit);
    box.appendChild(row);
    box.appendChild(el('div', 'refund-form-note', '原因只用于这次申请，不会在对话里反复追问。'));

    box.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      const original = submit.textContent;
      submit.textContent = '提交中…';
      try {
        const resp = await fetch('/api/actions/create-refund', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            order_id: form.order_id,
            reason: select.value,
            idempotency_key: `${sessionId || 'anon'}:${form.order_id}`,
          }),
        });
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          const err = (payload && payload.error) || {};
          throw new Error(err.message || `HTTP ${resp.status}`);
        }
        box.replaceChildren(
          el('div', 'refund-success', `已提交退款申请，工单号 ${payload.refund_no}。`)
        );
      } catch (err) {
        submit.disabled = false;
        submit.textContent = original;
        addNotice(`提交退款单失败：${err && err.message ? err.message : err}`, true);
      }
    });

    target.wrap.appendChild(box);
    scrollToBottom();
  }

  async function consumeStream(resp, ctx) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let match;
      while ((match = FRAME_SEP.exec(buf)) !== null) {
        const block = buf.slice(0, match.index);
        buf = buf.slice(match.index + match[0].length);
        const frame = parseFrame(block);
        if (frame) applyFrame(frame, ctx);
      }
    }
    if (buf.trim()) {
      const frame = parseFrame(buf);
      if (frame) applyFrame(frame, ctx);
    }
  }

  function applyFrame(frame, ctx) {
    const { event, data } = frame;

    if (event === 'meta') {
      if (data.session_id && data.session_id !== sessionId) {
        sessionId = data.session_id;
        sessionStorage.setItem(SESSION_KEY, sessionId);
      }
      return;
    }

    if (event === 'coref') {
      showCoref(ctx.bot, data);
      return;
    }

    if (event === 'intent') {
      showIntent(ctx.bot, data);
      return;
    }

    if (event === 'order_picker') {
      renderOrderPicker(ctx.bot, data, ctx);
      return;
    }

    if (event === 'awaiting_input') {
      const kind = (data && data.kind) || 'order_picker';
      const isTicket = kind === 'ticket_preview';
      ctx.awaiting = true;
      ctx.bot.body.innerHTML = '';
      els.input.disabled = true;
      els.send.disabled = true;
      setStatus(isTicket ? '等待确认建单' : '等待选择订单', 'ok');
      if (data.error) {
        addNotice(data.error, true);
      }
      if (data.discarded_message) {
        addNotice(
          data.message || (isTicket
            ? '请先确认或取消刚才的工单；刚才的消息没有被处理。'
            : '请先选择要办理的订单；刚才的消息没有被处理。'),
          true,
        );
      } else if (isTicket) {
        if (!ctx.bot.flow.querySelector('.ticket-preview')) {
          renderTicketPreview(ctx.bot, data, ctx);
        }
      } else if (!ctx.bot.flow.querySelector('.order-picker') && data.orders) {
        renderOrderPicker(ctx.bot, data, ctx);
      }
      return;
    }

    if (event === 'status') {
      // 工具执行**之前**到达：立刻出现一条「执行中」记录。
      ctx.showStatus(data);
      return;
    }

    if (event === 'tool_result') {
      // 工具刚返回：只更新对应那条记录，不等最终 done。
      applyToolResult(ctx.bot, data);
      return;
    }

    if (event === 'tools') {
      // 权威摘要先于 done 到达；这里校对整组记录（含失败原因与结果预览）。
      setToolRecords(ctx.bot, data.tools);
      return;
    }

    if (event === 'token') {
      ctx.acc += data.delta || '';
      ctx.queuePaint();
      return;
    }

    if (event === 'done') {
      ctx.awaiting = false;
      els.input.disabled = false;
      els.send.disabled = false;
      ctx.finish();
      setToolRecords(ctx.bot, data.tools);
      // 引用与反馈都挂在最终答案上：流式过程中 cites 还没到，先不管。
      decorateCitations(ctx.bot.body, data.citations);
      attachFeedback(ctx.bot);
      attachActions(ctx.bot, data.actions, {
        route: data.route,
        sourceText: ctx.userText,
      });
      attachRefundForm(ctx.bot, data.refund_form);
      if (els.stats.checked) {
        const s = el('div', 'stats');
        s.innerHTML = `上下文：保留 <code>${data.kept}</code> 条，`
                    + `裁掉 <code>${data.trimmed}</code> 条`;
        ctx.bot.wrap.appendChild(s);
        scrollToBottom();
      }
      return;
    }

    if (event === 'error') {
      ctx.fail(data);
    }
  }

  // ------------------------------------------------------- 引用角标与证据抽屉

  /* ch04：把正文里的 [n] 变成可点角标。
   *
   * 为什么不在 renderMarkdown 里用正则替换：那是在**转义后的 HTML 字符串**上做替换，
   * 很容易把类名、属性里的方括号也换掉。这里走 DOM 文本节点 —— 只碰显示文本，
   * 天然不会破坏标签结构。
   */
  function decorateCitations(bubble, citations) {
    if (!citations || !citations.length) return;
    const byNumber = new Map(citations.map((c) => [c.n, c]));
    const walker = document.createTreeWalker(bubble, NodeFilter.SHOW_TEXT, null);
    const targets = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (/\[\d{1,2}\]/.test(node.nodeValue)) targets.push(node);
    }

    targets.forEach((node) => {
      const pieces = node.nodeValue.split(/(\[\d{1,2}\])/g);
      const frag = document.createDocumentFragment();
      pieces.forEach((piece) => {
        const match = /^\[(\d{1,2})\]$/.exec(piece);
        const cite = match ? byNumber.get(Number(match[1])) : null;
        if (!cite) {
          frag.appendChild(document.createTextNode(piece));
          return;
        }
        const button = el('button', 'cite', match[1]);
        button.type = 'button';
        button.title = cite.section_path || '查看来源';
        button.addEventListener('click', () => openEvidence(cite));
        frag.appendChild(button);
      });
      node.parentNode.replaceChild(frag, node);
    });
  }

  let drawer = null;   // 懒创建：没引用时页面上不该多一个空面板

  function ensureDrawer() {
    if (drawer) return drawer;
    drawer = el('aside', 'cite-drawer');
    drawer.hidden = true;

    const head = el('div', 'cite-drawer-head');
    head.appendChild(el('span', 'cite-drawer-title', '来源'));
    const close = el('button', 'cite-drawer-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', '关闭');
    close.addEventListener('click', () => { drawer.hidden = true; });
    head.appendChild(close);

    const path = el('div', 'cite-path');
    const meta = el('div', 'cite-meta');
    const quote = el('blockquote', 'cite-quote');

    drawer.appendChild(head);
    drawer.appendChild(path);
    drawer.appendChild(meta);
    drawer.appendChild(quote);
    document.body.appendChild(drawer);
    return drawer;
  }

  function openEvidence(cite) {
    const panel = ensureDrawer();
    panel.querySelector('.cite-path').textContent = cite.section_path || '（没有章节路径）';
    panel.querySelector('.cite-meta').textContent =
      `来源 chunk #${cite.chunk_id}` + (cite.category ? ` · ${cite.category}` : '');
    panel.querySelector('.cite-quote').textContent = cite.quote || '（没有原文快照）';
    panel.hidden = false;
  }

  // ---------------------------------------------------------- 满意度反馈
  //
  // 纯前端采集：一次点击即锁定，写进 localStorage，供后续做数据飞轮时读取。
  // 键里带序号，刷新后能对上同一条回答；序号取「当前是第几条 bot 消息」，
  // 与 restoreHistory 的渲染顺序一致。

  const FEEDBACK_PREFIX = 'mewhelp.feedback.';

  function botIndexOf(row) {
    // 转人工模拟出来的「客服小猫」气泡不参与序号：它不落库，刷新后会消失；
    // 若把它算进去，刷新后后续消息的 action/feedback 锁会整体错位。
    const rows = Array.from(
      document.querySelectorAll('.row.bot:not([data-simulated])')
    );
    return rows.indexOf(row);
  }

  function feedbackKey(index) {
    return `${FEEDBACK_PREFIX}${sessionId || 'anon'}.${index}`;
  }

  function applyFeedbackLock(box, choice) {
    box.dataset.choice = choice;
    box.querySelectorAll('button').forEach((button) => {
      const pressed = button.dataset.vote === choice;
      button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
      button.disabled = true;
      button.classList.toggle('active', pressed);
    });
    const note = box.querySelector('.feedback-note');
    if (note) note.textContent = '已反馈';
  }

  function attachFeedback(target) {
    const existing = target.wrap.querySelector('.feedback');
    if (existing) existing.remove();

    const index = botIndexOf(target.row);
    const box = el('div', 'feedback');
    box.appendChild(el('span', 'feedback-label', '这条回答有帮助吗'));

    ['up', 'down'].forEach((vote) => {
      const button = el('button', 'feedback-btn', vote === 'up' ? '👍' : '👎');
      button.type = 'button';
      button.dataset.vote = vote;
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => {
        if (box.dataset.choice) return;   // 一次性：已锁就不再写
        box.dataset.choice = vote;
        try {
          localStorage.setItem(feedbackKey(index), vote);
        } catch { /* 隐私模式写不进去也别挡住聊天 */ }
        applyFeedbackLock(box, vote);
      });
      box.appendChild(button);
    });

    box.appendChild(el('span', 'feedback-note'));
    target.wrap.appendChild(box);

    let saved = null;
    try { saved = localStorage.getItem(feedbackKey(index)); } catch { saved = null; }
    if (saved === 'up' || saved === 'down') applyFeedbackLock(box, saved);
  }

  // ---------------------------------------------------------- 建议动作按钮
  //
  // 转人工与建工单是**两件独立的事**：
  // - 转人工：纯前端模拟，不发请求；
  // - 建工单：只有用户点按钮才 POST /api/actions/create-ticket 写 tickets。
  //
  // 两个按钮互不绑定：点一个不会禁用另一个；都不点就继续正常聊天。
  // 点击状态用 localStorage 一次性锁定，刷新后从 messages.actions 还原按钮。

  const ACTION_LABELS = {
    transfer_human: '转人工',
    create_ticket: '建工单',
  };
  const ACTION_STORAGE_PREFIX = 'mewhelp.action.';

  function actionKey(index, action) {
    return `${ACTION_STORAGE_PREFIX}${sessionId || 'anon'}.${index}.${action}`;
  }

  function readActionState(index, action) {
    try {
      return localStorage.getItem(actionKey(index, action));
    } catch {
      return null;
    }
  }

  function writeActionState(index, action, value) {
    try {
      localStorage.setItem(actionKey(index, action), value);
    } catch { /* 隐私模式写不进去也别挡住聊天 */ }
  }

  function applyActionLock(button, action, value) {
    button.disabled = true;
    button.classList.add('done');
    button.textContent = action === 'transfer_human' ? '已转接' : `已建单 ${value}`;
  }

  function normaliseAction(item, route) {
    if (typeof item === 'string') {
      return { action: item, ticket_type: route === 'complaint' ? '投诉' : '售后' };
    }
    return {
      action: item && item.action,
      ticket_type: (item && item.ticket_type) || (route === 'complaint' ? '投诉' : '售后'),
    };
  }

  function simulateTransfer(target, button, index) {
    // 需求明确：本章不接真人系统，只在前端模拟效果。
    writeActionState(index, 'transfer_human', '1');
    applyActionLock(button, 'transfer_human');

    addNotice('已转接人工客服');
    const greeting = addRow('bot');
    greeting.row.dataset.simulated = '1';   // 不计入 bot 序号，避免刷新后按钮锁串位
    greeting.body.textContent = '您好，我是客服小猫，请问有什么可以帮您的？';
    scrollToBottom(true);
  }

  async function createTicketAction(target, button, index, ticketType, sourceText) {
    if (!sessionId) {
      addNotice('还没有会话，先发一句话再建工单。', true);
      return;
    }
    button.disabled = true;
    const originalLabel = button.textContent;
    button.textContent = '建单中…';
    try {
      const resp = await fetch('/api/actions/create-ticket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          description: sourceText || '',
          ticket_type: ticketType || '售后',
          // 同一个按钮位置重复提交（超时重试/刷新后再点）返回同一张工单。
          idempotency_key: actionKey(index, 'create_ticket'),
        }),
      });
      const payload = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const err = (payload && payload.error) || {};
        throw new Error(err.message || `HTTP ${resp.status}`);
      }
      writeActionState(index, 'create_ticket', payload.ticket_no);
      applyActionLock(button, 'create_ticket', payload.ticket_no);
      addNotice(`已创建工单 ${payload.ticket_no}（${payload.ticket_type}）`);
    } catch (err) {
      button.disabled = false;
      button.textContent = originalLabel;
      addNotice(`建工单失败：${err && err.message ? err.message : err}`, true);
    }
  }

  function attachActions(target, actions, { route = null, sourceText = '' } = {}) {
    if (!actions || !actions.length) return;
    const existing = target.wrap.querySelector('.chat-actions');
    if (existing) existing.remove();

    const index = botIndexOf(target.row);
    const box = el('div', 'chat-actions');
    actions.forEach((raw) => {
      const { action, ticket_type: ticketType } = normaliseAction(raw, route);
      if (!action) return;

      const button = el('button', 'action-btn', ACTION_LABELS[action] || action);
      button.type = 'button';
      button.dataset.action = action;
      button.addEventListener('click', () => {
        if (button.disabled) return;
        if (action === 'transfer_human') {
          simulateTransfer(target, button, index);
        } else if (action === 'create_ticket') {
          createTicketAction(target, button, index, ticketType, sourceText);
        }
      });
      box.appendChild(button);

      const saved = readActionState(index, action);
      if (saved) applyActionLock(button, action, saved);
    });

    if (!box.children.length) return;
    target.wrap.appendChild(box);
    scrollToBottom();
  }

  // ------------------------------------------------------------------ 历史回读

  /* 后端把角色叫 human / ai / tool，前端气泡只分「我」和「客服」两类。
   * tool 行不单独渲染成气泡 —— 它的内容已经体现在徽章里了，再画一个气泡
   * 会让用户看到一坨 JSON。 */
  // ------------------------------------------------------ 会话侧栏（ch07）

  function rememberConversation(id) {
    conversationId = id;
    try {
      if (id == null) sessionStorage.removeItem(CONVERSATION_KEY);
      else sessionStorage.setItem(CONVERSATION_KEY, String(id));
    } catch { /* 隐私模式下 storage 可能抛，功能不该因此失效 */ }
  }

  function formatSidebarTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return `${d.getMonth() + 1}月${d.getDate()}日`;
  }

  function markActiveSidebarItem() {
    if (!els.sidebarList) return;
    els.sidebarList.querySelectorAll('.sidebar-item').forEach((node) => {
      const id = Number(node.dataset.conversationId);
      node.classList.toggle('is-active', conversationId != null && id === conversationId);
    });
  }

  function renderConversations(rows) {
    if (!els.sidebarList) return;
    els.sidebarList.innerHTML = '';
    if (!rows.length) {
      const empty = el('li', 'sidebar-empty', '还没有会话，说句话就开一个。');
      els.sidebarList.appendChild(empty);
      return;
    }
    rows.forEach((row) => {
      const item = el('li', 'sidebar-item');
      item.dataset.conversationId = String(row.id);
      if (row.summary_segments > 0) item.classList.add('is-summarized');
      if (conversationId != null && row.id === conversationId) item.classList.add('is-active');

      const preview = el('div', 'sidebar-preview', row.preview || '（空会话）');
      preview.title = row.preview || '（空会话）';
      const meta = el('div', 'sidebar-meta');
      meta.appendChild(el('span', null, formatSidebarTime(row.updated_at || row.created_at)));
      if (row.message_count) meta.appendChild(el('span', null, `${row.message_count} 条`));

      item.appendChild(preview);
      item.appendChild(meta);
      item.addEventListener('click', () => {
        switchConversation(row.id, row.session_id);
      });
      els.sidebarList.appendChild(item);
    });
  }

  /* 拉侧栏。**整个函数体包在 try/catch 里**：侧栏是增强，列表挂了也必须能聊天，
     所以失败只清空列表，不弹提示、不改顶栏状态（静默降级）。 */
  async function loadConversations() {
    try {
      const resp = await fetch(
        `${CONVERSATIONS_API}?user_id=${encodeURIComponent(USER_ID)}&limit=50`,
        { method: 'GET' },
      );
      if (!resp.ok) {
        els.sidebarList.innerHTML = '';
        return;
      }
      const data = await resp.json();
      renderConversations(Array.isArray(data.conversations) ? data.conversations : []);
    } catch {
      if (els.sidebarList) els.sidebarList.innerHTML = '';
    }
  }

  function closeSidebarDrawer() {
    if (els.sidebar) els.sidebar.classList.remove('is-open');
  }

  /* 切到某个会话：先按主键取历史，成功之后才动视图与两个 id。
     先换 id 再取历史的话，取失败就变成「视图还是旧会话的，id 已经指向新会话」，
     下一句话会落进另一个会话里。 */
  async function switchConversation(id, nextSessionId) {
    if (busy) return;
    els.messages.innerHTML = '';
    welcome();
    let messages = [];
    try {
      const resp = await fetch(`${CONVERSATIONS_API}/${id}/messages`, { method: 'GET' });
      if (resp.ok) {
        const data = await resp.json();
        messages = Array.isArray(data.messages) ? data.messages : [];
      }
    } catch {
      /* 取不到就当空会话打开，不阻断聊天 */
    }

    sessionId = nextSessionId || null;
    rememberConversation(id);
    try {
      if (sessionId) sessionStorage.setItem(SESSION_KEY, sessionId);
      else sessionStorage.removeItem(SESSION_KEY);
    } catch { /* 同上 */ }

    els.messages.innerHTML = '';
    if (!messages.length) {
      welcome();
    } else {
      paintHistory(messages);
    }
    markActiveSidebarItem();
    closeSidebarDrawer();
    els.input.focus();
  }

  /* 「新对话」只开一个新会话：清视图 + 丢掉两个 id，**不删任何旧会话** ——
     旧会话仍在侧栏里，随时可以点回去。 */
  function newConversation() {
    els.messages.innerHTML = '';
    sessionId = null;
    rememberConversation(null);
    try { sessionStorage.removeItem(SESSION_KEY); } catch { /* 同上 */ }
    welcome();
    markActiveSidebarItem();
    closeSidebarDrawer();
    els.input.focus();
  }

  if (els.sidebarToggle) {
    els.sidebarToggle.addEventListener('click', () => {
      if (els.sidebar) els.sidebar.classList.toggle('is-open');
    });
  }
  if (els.newChat) els.newChat.addEventListener('click', newConversation);

  function roleToRow(role) {
    return role === 'human' ? 'user' : 'bot';
  }

  /* 把一屏历史画出来。侧栏回载与首屏恢复共用这一段 —— 两套渲染会漂：
     徽章、引用、反馈、建议动作这些细节，只要有一处忘了还原就是「刷新后少了东西」。 */
  function paintHistory(all) {
    const messages = (all || []).filter((m) => {
      if (m.role === 'tool') return false;          // 工具结果已体现在徽章里，不单独画气泡
      // 带工具调用但正文为空的那条 assistant，是「申请单」而不是回答 ——
      // 它渲染出来会变成一个多余的「（客服没有返回内容）」空气泡。
      if (m.role === 'assistant' && !(m.content || '').trim() && m.tools) return false;
      return roleToRow(m.role) != null;
    });
    if (!messages.length) return;

    clearWelcome();
    let lastUserText = '';
    messages.forEach((m) => {
      const role = roleToRow(m.role);
      const row = addRow(role);
      if (role === 'user') {
        lastUserText = m.content || '';
        row.body.textContent = m.content || '';
      } else {
        row.body.innerHTML = m.content
          ? renderMarkdown(m.content)
          : '<span style="color:#9aa2af">（客服没有返回内容）</span>';
        // 工具记录随历史一起还原 —— 这正是「刷新后记录还在」的实现。
        setToolRecords(row, m.tools);
        // 引用与反馈同理：刷新后角标仍可点、已反馈仍是已反馈
        decorateCitations(row.body, m.citations);
        attachFeedback(row);
        // ch05：建议动作也随历史还原；建单描述回退到上一条用户消息。
        attachActions(row, m.actions, { sourceText: lastUserText });
      }
    });
    scrollToBottom(true);
  }

  async function restoreHistory() {
    if (!sessionId) return;
    try {
      const resp = await fetch(`/api/chat/sessions/${sessionId}/messages`, { method: 'POST' });
      if (!resp.ok) return;
      const data = await resp.json();
      paintHistory(data.messages);
    } catch {
      /* 回读失败不该挡住聊天：静默退化成空白页，用户发一句就恢复正常 */
    }
  }

  // ------------------------------------------------------------------ 发送

  async function send() {
    const text = els.input.value.trim();
    if (!text || busy) return;

    busy = true;
    els.send.disabled = true;
    els.input.value = '';
    autoGrow();

    addRow('user').body.textContent = text;
    const bot = addRow('bot', { typing: true });

    // 流式渲染状态。累积原始文本，用 rAF 合并成一次 DOM 写入，避免逐字抖动。
    let acc = '';
    let paintQueued = false;
    let settled = false;

    const ctx = {
      bot,
      userText: text,
      awaiting: false,
      get acc() { return acc; },
      set acc(v) { acc = v; },
      showStatus(toolEvent) {
        // 兼容旧调用点传入纯文本；工具事件则直接落成一条执行中记录。
        if (typeof toolEvent === 'string') return;
        addRunningTool(bot, toolEvent);
      },
      queuePaint() {
        if (paintQueued) return;
        paintQueued = true;
        requestAnimationFrame(() => {
          paintQueued = false;
          bot.body.innerHTML = renderMarkdown(acc) + '<span class="caret"></span>';
          scrollToBottom();
        });
      },
      finish() {
        if (settled) return;
        if (ctx.awaiting) {
          // 暂停轮还没有回答：保留选择器，不要画「客服没有返回内容」。
          if (!acc) bot.body.innerHTML = '';
          return;
        }
        settled = true;
        bot.body.innerHTML = acc
          ? renderMarkdown(acc)
          : '<span style="color:#9aa2af">（客服没有返回内容）</span>';
        scrollToBottom();
      },
      fail(data) {
        settled = true;
        els.input.disabled = false;
        els.send.disabled = false;
        failRunningTools(bot, data.message || '调用中断');
        if (!acc) bot.row.remove();
        else bot.body.innerHTML = renderMarkdown(acc);
        addNotice(`出错了：${data.type || 'error'}\n${data.message || ''}`, true);
        if (data.type === 'configuration_error') setStatus('.env 未配置', 'bad');
        if (data.type === 'database_error') setStatus('数据库不可用', 'bad');
      },
    };

    try {
      const resp = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text, stream: true }),
      });

      if (!resp.ok) {
        // 非 2xx 时后端返回 {"error": {"type", "message"}} 外壳
        const payload = await resp.json().catch(() => null);
        const err = (payload && payload.error) || {};
        bot.row.remove();
        settled = true;
        addNotice(`请求失败（HTTP ${resp.status}）\n${err.type || ''} ${err.message || ''}`.trim(), true);
        if (err.type === 'configuration_error') setStatus('.env 未配置', 'bad');
        return;
      }

      setStatus('已连接', 'ok');
      bot.body.innerHTML = '';         // 清掉三点等待
      await consumeStream(resp, ctx);
      ctx.finish();
      // 这一轮可能新建了一个会话，也可能刚刚被后台摘要任务压过（✂ 标记）。
      // 失败静默：侧栏刷新不该影响刚拿到手的回复。
      loadConversations();
    } catch (err) {
      failRunningTools(bot, '连接中断，未拿到工具结果');
      if (!acc) bot.row.remove();
      settled = true;
      addNotice(`网络或脚本错误：${err && err.message ? err.message : err}`, true);
      setStatus('连接异常', 'bad');
    } finally {
      busy = false;
      if (!ctx.awaiting) {
        els.send.disabled = false;
        els.input.disabled = false;
        els.input.focus();
      }
    }
  }

  // ------------------------------------------------------------------ 输入框

  function autoGrow() {
    els.input.style.height = 'auto';
    els.input.style.height = Math.min(els.input.scrollHeight, 148) + 'px';
  }

  els.input.addEventListener('input', autoGrow);

  els.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });

  els.send.addEventListener('click', send);

  /* 顶栏「清空」与侧栏「新对话」是同一件事：只开新会话，不删旧会话。
     以前这里只清视图不换 id 的话，下一句话会继续追加进上一个会话 —— 用户看到
     「清空」但历史还在侧栏里变长。 */
  els.clear.addEventListener('click', newConversation);

  els.messages.addEventListener('scroll', () => {
    stickToBottom = atBottom();
  });

  // ------------------------------------------------------------------ 启动

  els.input.focus();
  welcome();

  // 侧栏：拉列表 + 恢复上次会话。列表加载失败静默降级，聊天照常可用。
  loadConversations();

  // 恢复上次会话（同一浏览器会话内）。徽章也一并还原 —— 因为 /messages 会
  // 把每轮的工具轨迹一起返回，见 app/api.py 的 _db_session_messages_payload。
  restoreHistory();

  // 探一次健康检查，让顶栏状态反映真实配置
  fetch('/healthz')
    .then(async (r) => ({ ok: r.ok, body: await r.json().catch(() => ({})) }))
    .then(({ ok, body }) => {
      if (!ok && body.database === 'unavailable') {
        setStatus('数据库不可用', 'bad');
      } else if (body.status === 'ok') {
        setStatus(`已就绪 · ${body.model}`, 'ok');
      } else {
        setStatus('配置缺失，请检查 .env', 'bad');
      }
    })
    .catch(() => setStatus('无法连接后端', 'bad'));
})();
