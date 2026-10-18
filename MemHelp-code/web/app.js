/* MewHelp 客服对话前端。
 *
 * 为什么不用 EventSource：后端是 POST /api/chat/completions + SSE，而
 * EventSource 只支持 GET、且不能带请求体。所以这里用 fetch 拿 ReadableStream，
 * 自己按 SSE 规范切帧（event: / data: / 空行分隔）。
 *
 * 后端事件契约（见 app/api.py::_chat_stream_body）：
 *   event: meta    data: {session_id, model}
 *   event: status  data: {tool, args, label}   ← ch02：工具执行**之前**推，显示「正在查询物流…」
 *   event: token   data: {delta}               ← 逐字增量
 *   event: done    data: {finish_reason, session_id, trimmed, kept, tools}
 *   event: error   data: {type, message, session_id}
 *
 * ch02 的 done.tools 是本轮工具轨迹的**权威来源**（含成败与耗时），
 * 徽章只用它渲染；status 帧只负责让用户立刻看到「在查数据」，不承担持久化职责。
 * 历史回读（/messages）返回的每条消息也可能带 tools 字段，复用同一个渲染函数。
 */

(() => {
  'use strict';

  const API = '/api/chat/completions';
  const SESSION_KEY = 'mewhelp.session_id';

  const els = {
    messages: document.getElementById('messages'),
    input: document.getElementById('input'),
    send: document.getElementById('send'),
    clear: document.getElementById('btn-clear'),
    stats: document.getElementById('show-stats'),
    status: document.getElementById('status'),
    statusText: document.getElementById('status-text'),
  };

  let sessionId = sessionStorage.getItem(SESSION_KEY) || null;
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
    // 工具执行状态行：在气泡**上方**，工具跑起来那一刻出现，拿到第一个 token 后隐去。
    const status = el('div', 'tool-status');
    status.hidden = true;
    const bubble = el('div', 'bubble');

    if (typing) {
      const dots = el('span', 'typing');
      dots.innerHTML = '<i></i><i></i><i></i>';
      bubble.appendChild(dots);
    }

    wrap.appendChild(status);
    wrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrap);
    els.messages.appendChild(row);
    scrollToBottom(true);

    return { row, wrap, bubble, status };
  }

  /* ---------------------------------------------------------- 工具轨迹徽章
   *
   * 形如：🔧 查询物流 · 成功 · 12ms
   *
   * 渲染成 DOM 节点（不是 innerHTML）—— 工具名与参数来自模型输出，
   * 走 innerHTML 就是一个现成的 XSS 面。textContent 天然免疫。
   *
   * label 用中文短语而不是工具名：用户看到「查询物流」比 query_logistics 有用。
   */
  const TOOL_LABELS = {
    query_order: '查询订单',
    query_product: '查询商品',
    query_logistics: '查询物流',
    query_faq: '检索常见问题',
    create_ticket: '创建工单',
  };

  function renderToolBadges(tools) {
    const box = el('div', 'tool-badges');
    (tools || []).forEach((t) => {
      const badge = el('span', 'tool-badge' + (t.ok === false ? ' failed' : ''));
      badge.appendChild(el('span', 'tool-badge-icon', '🔧'));
      badge.appendChild(el('span', 'tool-badge-name',
                           TOOL_LABELS[t.name] || t.name || '未知工具'));
      badge.appendChild(el('span', 'tool-badge-result', t.ok === false ? '失败' : '成功'));
      if (typeof t.duration_ms === 'number') {
        badge.appendChild(el('span', 'tool-badge-time', t.duration_ms + 'ms'));
      }
      if (t.ok === false && t.error) {
        badge.title = t.error;      // 失败原因放 title，不占视觉空间
      }
      box.appendChild(badge);
    });
    return box;
  }

  /* 把徽章挂到气泡下方，并把本轮轨迹写进 dataset —— 刷新后从 /messages 回读时
   * 用同一份数据渲染，所以徽章不会「刷新就消失」。
   *
   * ``target`` 就是 ``addRow()`` 的返回值 ``{row, wrap, ...}``，两个调用点都传它：
   * 流式那一轮传 ``ctx.bot``，历史回读传 ``row``。
   * 别再传「外层 ctx」——它只有 ``bot``，没有 ``wrap``/``row``，
   * 曾经因此在这里抛 TypeError，被 send() 的 catch 变成「网络或脚本错误」。 */
  function attachTools(target, tools) {
    if (!tools || !tools.length) return;
    const existing = target.wrap.querySelector('.tool-badges');
    if (existing) existing.remove();
    target.wrap.appendChild(renderToolBadges(tools));
    target.row.dataset.tools = JSON.stringify(tools);
    scrollToBottom();
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

  function applyFrame(frame, ctx) {
    const { event, data } = frame;

    if (event === 'meta') {
      if (data.session_id && data.session_id !== sessionId) {
        sessionId = data.session_id;
        sessionStorage.setItem(SESSION_KEY, sessionId);
      }
      return;
    }

    if (event === 'status') {
      // 工具执行**之前**到达：先让用户看到「在查数据」，而不是干等。
      ctx.showStatus(data.label || '正在查询…');
      return;
    }

    if (event === 'token') {
      // 第一个字到了就说明工具那段结束了，把状态行收掉。
      ctx.hideStatus();
      ctx.acc += data.delta || '';
      ctx.queuePaint();
      return;
    }

    if (event === 'done') {
      ctx.finish();
      attachTools(ctx.bot, data.tools);
      // 引用与反馈都挂在最终答案上：流式过程中 cites 还没到，先不管。
      decorateCitations(ctx.bot.bubble, data.citations);
      attachFeedback(ctx.bot);
      attachActions(ctx.bot, data.actions, {
        route: data.route,
        sourceText: ctx.userText,
      });
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
    greeting.bubble.textContent = '您好，我是客服小猫，请问有什么可以帮您的？';
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
  function roleToRow(role) {
    return role === 'human' ? 'user' : 'bot';
  }

  async function restoreHistory() {
    if (!sessionId) return;
    try {
      const resp = await fetch(`/api/chat/sessions/${sessionId}/messages`, { method: 'POST' });
      if (!resp.ok) return;
      const data = await resp.json();
      const messages = (data.messages || []).filter((m) => {
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
          row.bubble.textContent = m.content || '';
        } else {
          row.bubble.innerHTML = m.content
            ? renderMarkdown(m.content)
            : '<span style="color:#9aa2af">（客服没有返回内容）</span>';
          // 徽章随历史一起还原 —— 这正是「刷新后徽章还在」的实现
          attachTools(row, m.tools);
          // 引用与反馈同理：刷新后角标仍可点、已反馈仍是已反馈
          decorateCitations(row.bubble, m.citations);
          attachFeedback(row);
          // ch05：建议动作也随历史还原；建单描述回退到上一条用户消息。
          attachActions(row, m.actions, { sourceText: lastUserText });
        }
      });
      scrollToBottom(true);
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

    addRow('user').bubble.textContent = text;
    const bot = addRow('bot', { typing: true });

    // 流式渲染状态。累积原始文本，用 rAF 合并成一次 DOM 写入，避免逐字抖动。
    let acc = '';
    let paintQueued = false;
    let settled = false;

    const ctx = {
      bot,
      userText: text,
      get acc() { return acc; },
      set acc(v) { acc = v; },
      showStatus(text) {
        bot.status.textContent = text;
        bot.status.hidden = false;
        scrollToBottom();
      },
      hideStatus() {
        bot.status.hidden = true;
        bot.status.textContent = '';
      },
      queuePaint() {
        if (paintQueued) return;
        paintQueued = true;
        requestAnimationFrame(() => {
          paintQueued = false;
          bot.bubble.innerHTML = renderMarkdown(acc) + '<span class="caret"></span>';
          scrollToBottom();
        });
      },
      finish() {
        if (settled) return;
        settled = true;
        bot.bubble.innerHTML = acc
          ? renderMarkdown(acc)
          : '<span style="color:#9aa2af">（客服没有返回内容）</span>';
        scrollToBottom();
      },
      fail(data) {
        settled = true;
        ctx.hideStatus();
        if (!acc) bot.row.remove();
        else bot.bubble.innerHTML = renderMarkdown(acc);
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
      bot.bubble.innerHTML = '';       // 清掉三点等待

      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let m;
        while ((m = FRAME_SEP.exec(buf)) !== null) {
          const block = buf.slice(0, m.index);
          buf = buf.slice(m.index + m[0].length);
          const frame = parseFrame(block);
          if (frame) applyFrame(frame, ctx);
        }
      }

      // 收尾：流结束时若最后一段没有以空行结束，也要处理掉
      if (buf.trim()) {
        const frame = parseFrame(buf);
        if (frame) applyFrame(frame, ctx);
      }
      ctx.finish();
    } catch (err) {
      if (!acc) bot.row.remove();
      settled = true;
      addNotice(`网络或脚本错误：${err && err.message ? err.message : err}`, true);
      setStatus('连接异常', 'bad');
    } finally {
      busy = false;
      els.send.disabled = false;
      els.input.focus();
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

  els.clear.addEventListener('click', () => {
    els.messages.innerHTML = '';
    sessionStorage.removeItem(SESSION_KEY);
    sessionId = null;
    welcome();
    els.input.focus();
  });

  els.messages.addEventListener('scroll', () => {
    stickToBottom = atBottom();
  });

  // ------------------------------------------------------------------ 启动

  els.input.focus();
  welcome();

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
