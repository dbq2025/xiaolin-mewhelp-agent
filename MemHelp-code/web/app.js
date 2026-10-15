/* MewHelp 客服对话前端。
 *
 * 为什么不用 EventSource：后端是 POST /api/chat/completions + SSE，而
 * EventSource 只支持 GET、且不能带请求体。所以这里用 fetch 拿 ReadableStream，
 * 自己按 SSE 规范切帧（event: / data: / 空行分隔）。
 *
 * 后端事件契约（见 app/api.py::_chat_stream_body）：
 *   event: meta   data: {session_id, model}
 *   event: token  data: {delta}                ← 逐字增量
 *   event: done   data: {finish_reason, session_id, trimmed, kept}
 *   event: error  data: {type, message, session_id}
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
    const bubble = el('div', 'bubble');

    if (typing) {
      const dots = el('span', 'typing');
      dots.innerHTML = '<i></i><i></i><i></i>';
      bubble.appendChild(dots);
    }

    wrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrap);
    els.messages.appendChild(row);
    scrollToBottom(true);

    return { row, wrap, bubble };
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

    if (event === 'token') {
      ctx.acc += data.delta || '';
      ctx.queuePaint();
      return;
    }

    if (event === 'done') {
      ctx.finish();
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
      get acc() { return acc; },
      set acc(v) { acc = v; },
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
        if (!acc) bot.row.remove();
        else bot.bubble.innerHTML = renderMarkdown(acc);
        addNotice(`出错了：${data.type || 'error'}\n${data.message || ''}`, true);
        if (data.type === 'configuration_error') setStatus('.env 未配置', 'bad');
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

  // 探一次健康检查，让顶栏状态反映真实配置
  fetch('/healthz')
    .then((r) => r.json())
    .then((h) => {
      if (h.status === 'ok') setStatus(`已就绪 · ${h.model}`, 'ok');
      else setStatus('配置缺失，请检查 .env', 'bad');
    })
    .catch(() => setStatus('无法连接后端', 'bad'));
})();
