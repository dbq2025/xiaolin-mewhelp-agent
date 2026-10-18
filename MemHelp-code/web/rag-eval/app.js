/* MewHelp · ch04 RAG 评估看板
 *
 * 数据来源只有两个：
 *   GET  /api/rag-eval       归一后的看板契约（报告 + 题面 + 运行状态）
 *   POST /api/rag-eval/rerun 启动后台评估；GET /api/rag-eval/run 轮询
 *
 * 页面不直接读 evals/reports/ch04.json：报告字段会随评估口径演进，
 * 归一放在后端（app/rag_eval.py），这里只负责展示与交互。
 */

(() => {
  'use strict';

  const STRATEGY_COLORS = {
    vector: 'var(--c-vector)',
    bm25: 'var(--c-bm25)',
    hybrid: 'var(--c-hybrid)',
    hybrid_rerank: 'var(--c-rerank)',
  };

  const TAB_META = {
    mrr: {
      title: '检索排序 · MRR@10',
      sub: '越大越好。MRR 只看第一个正确证据排在第几位，没跑的策略显示 —。',
      source: 'retrieval',
      key: 'mrr@10',
      foot: 'MRR@10：第一处正确证据的排名倒数取平均。它比 Recall 更在意「排得靠不靠前」。',
    },
    recall: {
      title: '检索召回 · Recall@5',
      sub: '前 5 条里至少命中一条金标证据的题占比，越大越好。',
      source: 'retrieval',
      key: 'recall@5',
      foot: 'Recall@5 偏宽松：同一事实被多条 chunk（含挖矿问答对）陈述时，gold 会变大。区分度看 MRR 与 Recall@1。',
    },
    evidence: {
      title: '证据覆盖 · Faithfulness',
      sub: '回答里的 claim 被引用证据支持的比例；只有跑过生成段的策略有值。',
      source: 'generation',
      key: 'faithfulness',
      foot: '证据覆盖 = 支持的 claim 数 ÷ 总 claim 数。它衡量「答案有没有超出证据」，不是检索命中率。',
    },
  };

  const els = {
    loading: document.getElementById('loading'),
    errorPanel: document.getElementById('error-panel'),
    errorMessage: document.getElementById('error-message'),
    errorHint: document.getElementById('error-hint'),
    btnRetry: document.getElementById('btn-retry'),
    emptyPanel: document.getElementById('empty-panel'),
    emptyPath: document.getElementById('empty-report-path'),
    btnEmptyRun: document.getElementById('btn-empty-run'),
    btnEmptyRefresh: document.getElementById('btn-empty-refresh'),

    dashboard: document.getElementById('dashboard'),
    reportTime: document.getElementById('report-time'),
    runStatus: document.getElementById('run-status'),
    runStatusText: document.getElementById('run-status-text'),
    btnRefresh: document.getElementById('btn-refresh'),
    btnRerun: document.getElementById('btn-rerun'),

    bestStrategy: document.getElementById('best-strategy'),
    acceptanceBadge: document.getElementById('acceptance-badge'),
    metaGrid: document.getElementById('meta-grid'),
    kpiGrid: document.getElementById('kpi-grid'),

    chartTitle: document.getElementById('chart-title'),
    chartSub: document.getElementById('chart-sub'),
    tabs: document.querySelectorAll('[data-tab]'),
    legend: document.getElementById('legend'),
    yAxis: document.getElementById('y-axis'),
    gridlines: document.getElementById('gridlines'),
    groups: document.getElementById('groups'),
    xAxis: document.getElementById('x-axis'),
    chartEmpty: document.getElementById('chart-empty'),
    chartEmptyText: document.getElementById('chart-empty-text'),
    chartFoot: document.getElementById('chart-foot'),

    acceptancePill: document.getElementById('acceptance-pill'),
    metricBody: document.getElementById('metric-body'),

    warningsPanel: document.getElementById('warnings-panel'),
    warningsList: document.getElementById('warnings-list'),
    warningsToggle: document.getElementById('warnings-toggle'),
    warningsCount: document.getElementById('warnings-count'),

    caseTabs: document.querySelectorAll('[data-case-tab]'),
    filterStrategy: document.getElementById('filter-strategy'),
    filterBucket: document.getElementById('filter-bucket'),
    filterBehavior: document.getElementById('filter-behavior'),
    casesCount: document.getElementById('cases-count'),
    casesHead: document.getElementById('cases-head'),
    casesBody: document.getElementById('cases-body'),
    casesEmpty: document.getElementById('cases-empty'),

    notesPanel: document.getElementById('notes-panel'),
    notesList: document.getElementById('notes-list'),
    reportPath: document.getElementById('report-path'),

    runBanner: document.getElementById('run-banner'),
    runBannerText: document.getElementById('run-banner-text'),
    runLog: document.getElementById('run-log'),
    btnToggleLog: document.getElementById('btn-toggle-log'),

    drawer: document.getElementById('case-drawer'),
    drawerBackdrop: document.getElementById('drawer-backdrop'),
    drawerEyebrow: document.getElementById('drawer-eyebrow'),
    drawerTitle: document.getElementById('drawer-title'),
    drawerBody: document.getElementById('drawer-body'),
    drawerClose: document.getElementById('drawer-close'),

    toast: document.getElementById('toast'),
  };

  const state = {
    data: null,
    tab: 'mrr',
    caseTab: 'retrieval',
    filters: { strategy: '', bucket: '', behavior: '' },
    hiddenSeries: new Set(),
    run: null,
    polling: null,
    toastTimer: null,
    flashCaseId: null,
    warningsExpanded: false,
  };

  // ------------------------------------------------------------------ 工具

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmt(value, digits = 3) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  function signed(value, digits = 3) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    return (value >= 0 ? '+' : '') + value.toFixed(digits);
  }

  function pct(value, digits = 1) {
    return typeof value === 'number' && Number.isFinite(value) ? (value * 100).toFixed(digits) + '%' : '—';
  }

  function formatTime(raw) {
    if (!raw || typeof raw !== 'string') return '—';
    return raw.replace('T', ' ').slice(0, 16);
  }

  function strategyLabel(name) {
    return (state.data?.labels?.strategies || {})[name] || name || '—';
  }

  function bucketLabel(name) {
    return (state.data?.labels?.buckets || {})[name] || name || '—';
  }

  function pill(text, kind) {
    return el('span', 'pill' + (kind ? ' ' + kind : ''), text);
  }

  function showToast(text, isError) {
    els.toast.textContent = text;
    els.toast.className = 'toast' + (isError ? ' bad' : '');
    els.toast.hidden = false;
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => { els.toast.hidden = true; }, isError ? 5200 : 3200);
  }

  class ApiError extends Error {
    constructor(status, type, message) {
      super(message || ('HTTP ' + status));
      this.status = status;
      this.type = type || '';
    }
  }

  async function api(path, options = {}) {
    let resp;
    try {
      resp = await fetch(path, {
        headers: { Accept: 'application/json', ...(options.headers || {}) },
        ...options,
      });
    } catch (err) {
      throw new ApiError(0, 'network_error', '无法连接后端：' + (err && err.message ? err.message : err));
    }
    let payload = null;
    try { payload = await resp.json(); } catch { /* 非 JSON 响应交给下面统一报错 */ }
    if (!resp.ok) {
      const error = (payload && payload.error) || {};
      throw new ApiError(resp.status, error.type || 'http_error', error.message || ('请求失败（HTTP ' + resp.status + '）'));
    }
    return payload;
  }

  // ------------------------------------------------------------------ 状态切换

  function showOnly(which) {
    els.loading.hidden = which !== 'loading';
    els.errorPanel.hidden = which !== 'error';
    els.emptyPanel.hidden = which !== 'empty';
    els.dashboard.hidden = which !== 'dashboard';
  }

  function showError(err) {
    showOnly('error');
    els.runStatus.className = 'run-status bad';
    els.runStatusText.textContent = '读取失败';
    els.errorMessage.textContent = err.message || '未知错误';
    if (err.type === 'report_invalid') {
      els.errorHint.textContent = '报告 JSON 可能写坏了。重新跑一次评估会覆盖它，或直接检查 evals/reports/ch04.json。';
    } else if (err.status === 0) {
      els.errorHint.textContent = '确认 uvicorn 还在运行，然后打开 /rag-eval/ 重试。';
    } else {
      els.errorHint.textContent = '可以点重试；如果一直失败，先看服务端日志。';
    }
  }

  function showEmpty(err) {
    showOnly('empty');
    if (err && err.message) els.emptyPath.textContent = 'evals/reports/ch04.json';
  }

  // ------------------------------------------------------------------ 顶栏与运行状态

  function renderReportTime() {
    const meta = state.data?.meta || {};
    const parts = ['报告生成于 ' + formatTime(meta.generated_at)];
    if (meta.cases != null) parts.push(meta.cases + ' 题');
    els.reportTime.textContent = parts.join(' · ');
  }

  function runStatusKind(run) {
    if (!state.data && !(run && (run.running || run.status === 'running'))) {
      return { text: '还没有报告', cls: 'bad' };
    }
    if (!run) return { text: '报告就绪', cls: 'ok' };
    if (run.running || run.status === 'running') return { text: '评估运行中…', cls: 'running' };
    if (run.status === 'succeeded') return { text: '上次重跑通过', cls: 'ok' };
    if (run.status === 'failed') return { text: '上次重跑未通过', cls: 'bad' };
    if (run.status === 'cancelled') return { text: '上次重跑被中止', cls: 'bad' };
    return { text: '报告就绪', cls: 'ok' };
  }

  function renderRunStatus(run) {
    const kind = runStatusKind(run);
    els.runStatus.className = 'run-status ' + kind.cls;
    els.runStatusText.textContent = kind.text;
    const running = Boolean(run && (run.running || run.status === 'running'));
    const finishedBadly = Boolean(run && (run.status === 'failed' || run.status === 'cancelled'));
    els.btnRerun.disabled = running;
    els.btnRerun.textContent = running ? '评估运行中…' : '重跑评估';
    els.runBanner.classList.toggle('bad', finishedBadly);
    if (running || finishedBadly) {
      els.runBanner.hidden = false;
      els.runBannerText.textContent = running
        ? '正在重跑评估，通常需要几分钟；页面会自动刷新报告。'
        : (run.message || '上次重跑未通过，报告可能已经更新，可查看日志定位。');
      if (run.log_tail) {
        els.runLog.textContent = run.log_tail;
        if (!els.runLog.hidden) els.runLog.scrollTop = els.runLog.scrollHeight;
      }
    } else {
      els.runBanner.hidden = true;
      els.runLog.hidden = true;
      els.btnToggleLog.textContent = '查看日志';
    }
  }

  // ------------------------------------------------------------------ 元信息 / KPI

  function metaCard(label, value, sub) {
    const card = el('div', 'meta-card');
    card.appendChild(el('div', 'meta-card-label', label));
    card.appendChild(el('div', 'meta-card-value', value));
    card.appendChild(el('div', 'meta-card-sub', sub));
    return card;
  }

  function renderMeta() {
    const meta = state.data.meta || {};
    clear(els.metaGrid);

    const genLabels = (meta.generation_strategies || []).map(strategyLabel).join(' · ') || '—';
    const rerankShort = (meta.rerank_model || '').split('/').pop() || '—';
    const casesSub = meta.cases != null
      ? '四桶 × 12 条 · 10 可答 + 2 拒答'
      : '评估集题目数未知';

    els.metaGrid.appendChild(metaCard('评估集', meta.cases != null ? meta.cases + ' 题' : '—', casesSub));
    els.metaGrid.appendChild(metaCard('知识库', meta.milvus_rows != null ? meta.milvus_rows + ' 块' : '—', 'Milvus 索引副本'));
    els.metaGrid.appendChild(metaCard('重排模型', rerankShort, '阈值 ' + fmt(meta.min_rerank_score, 1)));
    els.metaGrid.appendChild(metaCard('生成段策略', (meta.generation_strategies || []).length + ' / ' + (meta.strategies || []).length, genLabels));
    els.metaGrid.appendChild(metaCard('上次跑于', formatTime(meta.generated_at), meta.report_path || '—'));
  }

  function kpiCard(label, value, valueSuffix, sub, color) {
    const card = el('div', 'kpi-card');
    card.style.setProperty('--kpi-color', color || 'var(--accent)');
    card.appendChild(el('div', 'kpi-label', label));
    const valueNode = el('div', 'kpi-value');
    valueNode.textContent = value;
    if (valueSuffix) valueNode.appendChild(el('small', null, valueSuffix));
    card.appendChild(valueNode);
    card.appendChild(el('div', 'kpi-sub', sub));
    return card;
  }

  function renderKpis() {
    const s = state.data.summary || {};
    clear(els.kpiGrid);

    const gain = s.colloquial_gain;
    els.kpiGrid.appendChild(kpiCard(
      '最佳整体 MRR',
      fmt(s.best_mrr),
      '',
      (s.best_strategy_label || '—') + ' · MRR@10',
      'var(--c-rerank)',
    ));
    els.kpiGrid.appendChild(kpiCard(
      '口语桶 MRR 提升',
      signed(gain),
      '',
      '混合 + 重排 vs 纯 BM25 · 口语问法',
      typeof gain === 'number' && gain >= 0 ? 'var(--ok)' : 'var(--danger)',
    ));
    els.kpiGrid.appendChild(kpiCard(
      '端到端答案覆盖',
      pct(s.answer_coverage != null ? s.answer_coverage : s.citation_coverage),
      '',
      (s.answered_total != null ? s.answered_total : '—') + ' / ' +
        (s.answerable_total != null ? s.answerable_total : '—') +
        ' 作答 · 引用覆盖 ' + pct(s.citation_coverage) + ' · ' +
        (s.final_generation_label || s.final_generation_strategy || '—'),
      'var(--c-vector)',
    ));
    els.kpiGrid.appendChild(kpiCard(
      '意外问题拒答',
      pct(s.refusal_accuracy),
      '',
      (s.refusal_correct != null ? s.refusal_correct : '—') + ' / ' + (s.refusal_total != null ? s.refusal_total : '—') + ' 正确拒答并落池',
      'var(--c-bm25)',
    ));
  }

  // ------------------------------------------------------------------ 图表

  function currentTabStrategies(tab) {
    const meta = state.data.meta || {};
    return tab === 'evidence'
      ? (meta.generation_strategies || [])
      : (meta.strategies || []);
  }

  function metricValue(strategy, bucket, tab) {
    const meta = TAB_META[tab];
    const source = state.data[meta.source] || {};
    return source[strategy]?.[bucket]?.[meta.key];
  }

  function renderLegend(tab) {
    clear(els.legend);
    currentTabStrategies(tab).forEach((strategy) => {
      const item = el('button', 'legend-item' + (state.hiddenSeries.has(strategy) ? ' off' : ''));
      item.type = 'button';
      item.dataset.series = strategy;
      const swatch = el('span', 'legend-swatch');
      swatch.style.setProperty('--swatch', STRATEGY_COLORS[strategy] || 'var(--ink-faint)');
      item.appendChild(swatch);
      item.appendChild(el('span', null, strategyLabel(strategy)));
      item.title = state.hiddenSeries.has(strategy) ? '点击显示该策略' : '点击隐藏该策略';
      item.addEventListener('click', () => {
        if (state.hiddenSeries.has(strategy)) state.hiddenSeries.delete(strategy);
        else state.hiddenSeries.add(strategy);
        renderChart();
      });
      els.legend.appendChild(item);
    });
  }

  function renderChart() {
    const meta = TAB_META[state.tab];
    const buckets = state.data.labels?.buckets_order || ['A_policy', 'B_model', 'C_colloquial', 'E_multi', 'overall'];
    const strategies = currentTabStrategies(state.tab).filter((name) => !state.hiddenSeries.has(name));

    els.tabs.forEach((button) => {
      const active = button.dataset.tab === state.tab;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    els.chartTitle.textContent = meta.title;
    els.chartSub.textContent = meta.sub;
    els.chartFoot.textContent = meta.foot;
    renderLegend(state.tab);

    const hasValue = strategies.some((strategy) => buckets.some((bucket) => {
      const value = metricValue(strategy, bucket, state.tab);
      return typeof value === 'number' && Number.isFinite(value);
    }));
    if (!strategies.length || !hasValue) {
      els.chartEmpty.hidden = false;
      els.chartEmptyText.textContent = !strategies.length
        ? '所有策略都被图例隐藏了；点一下上方的策略名可以重新显示。'
        : state.tab === 'evidence'
          ? '这份报告没有生成段数据（本轮没跑 --generation）。'
          : '这份报告没有该指标的数据。';
      clear(els.groups);
      clear(els.xAxis);
      clear(els.yAxis);
      clear(els.gridlines);
      return;
    }
    els.chartEmpty.hidden = true;

    // Y 轴与网格（0 / 0.25 / 0.5 / 0.75 / 1.00）
    clear(els.yAxis);
    clear(els.gridlines);
    for (let i = 0; i <= 4; i += 1) {
      const ratio = i / 4;
      const label = el('span', 'y-label', ratio.toFixed(2));
      label.style.bottom = (ratio * 100) + '%';
      els.yAxis.appendChild(label);

      const line = el('div', 'grid-line' + (i === 0 || i === 4 ? ' solid' : ''));
      line.style.bottom = (ratio * 100) + '%';
      els.gridlines.appendChild(line);
    }

    clear(els.groups);
    clear(els.xAxis);
    buckets.forEach((bucket) => {
      const group = el('div', 'bar-group');
      const bars = el('div', 'group-bars');
      strategies.forEach((strategy) => {
        const value = metricValue(strategy, bucket, state.tab);
        const col = el('div', 'bar-col');
        const bar = el('div', 'bar');
        bar.style.setProperty('--swatch', STRATEGY_COLORS[strategy] || 'var(--ink-faint)');
        const label = el('span', 'bar-value');

        if (typeof value !== 'number' || !Number.isFinite(value)) {
          bar.classList.add('unavailable');
          label.classList.add('unavailable');
          label.textContent = '—';
          bar.title = strategyLabel(strategy) + ' · ' + bucketLabel(bucket) + '：未运行';
        } else {
          const height = Math.max(0, Math.min(1, value)) * 92;
          bar.style.setProperty('--h', height + '%');
          label.style.setProperty('--h', height + '%');
          label.textContent = value.toFixed(2);
          bar.title = strategyLabel(strategy) + ' · ' + bucketLabel(bucket) + '：' + fmt(value);
        }
        bar.setAttribute('aria-label',
          strategyLabel(strategy) + ' ' + bucketLabel(bucket) + ' ' +
          (typeof value === 'number' ? fmt(value) : '未运行'));

        col.appendChild(bar);
        col.appendChild(label);
        bars.appendChild(col);
      });

      group.appendChild(bars);
      group.title = bucketLabel(bucket);
      els.groups.appendChild(group);
    });

    // x 轴单独一层，和柱组共用同一组 flex 宽度
    buckets.forEach((bucket) => {
      const cell = el('div', 'x-cell');
      cell.appendChild(el('b', null, bucketLabel(bucket)));
      cell.appendChild(el('span', null, bucket));
      els.xAxis.appendChild(cell);
    });
  }

  // ------------------------------------------------------------------ 指标表

  function overallValue(strategy, source, key) {
    return state.data[source]?.[strategy]?.overall?.[key];
  }

  function renderMetricTable() {
    const meta = state.data.meta || {};
    const strategies = meta.strategies || Object.keys(state.data.retrieval || {});
    const rows = strategies.map((strategy) => ({
      strategy,
      recall1: overallValue(strategy, 'retrieval', 'recall@1'),
      recall5: overallValue(strategy, 'retrieval', 'recall@5'),
      mrr: overallValue(strategy, 'retrieval', 'mrr@10'),
      faithfulness: overallValue(strategy, 'generation', 'faithfulness'),
      answerCoverage: overallValue(strategy, 'generation', 'answer_coverage'),
      coverage: overallValue(strategy, 'generation', 'citation_coverage'),
      refusal: overallValue(strategy, 'generation', 'refusal_accuracy'),
      degraded: state.data.rerank_degraded?.[strategy],
    }));

    const keys = ['recall1', 'recall5', 'mrr', 'faithfulness', 'answerCoverage', 'coverage', 'refusal'];
    const best = {};
    keys.forEach((key) => {
      const values = rows.map((row) => row[key]).filter((value) => typeof value === 'number');
      best[key] = values.length ? Math.max(...values) : null;
    });

    clear(els.metricBody);
    rows.forEach((row) => {
      const tr = el('tr');
      const nameCell = el('td', 'strategy-cell');
      const dot = el('span', 'strategy-dot');
      dot.style.setProperty('--swatch', STRATEGY_COLORS[row.strategy] || 'var(--ink-faint)');
      nameCell.appendChild(dot);
      nameCell.appendChild(el('span', null, strategyLabel(row.strategy)));
      tr.appendChild(nameCell);

      [['recall1', row.recall1], ['recall5', row.recall5], ['mrr', row.mrr],
       ['faithfulness', row.faithfulness], ['answerCoverage', row.answerCoverage],
       ['coverage', row.coverage], ['refusal', row.refusal]]
        .forEach(([key, value]) => {
          const td = el('td', 'num');
          td.textContent = key === 'coverage' || key === 'refusal' || key === 'answerCoverage'
            ? pct(value)
            : fmt(value);
          if (value == null) td.classList.add('muted');
          if (value != null && best[key] != null && value === best[key]) td.classList.add('best');
          tr.appendChild(td);
        });

      const degraded = el('td', 'num');
      degraded.textContent = row.degraded ? row.degraded + ' 次' : '—';
      if (!row.degraded) degraded.classList.add('muted');
      tr.appendChild(degraded);
      els.metricBody.appendChild(tr);
    });

    const acceptance = state.data.summary?.acceptance || {};
    els.acceptancePill.textContent = acceptance.status === 'pass'
      ? '验收通过'
      : acceptance.status === 'fail' ? '验收未通过' : '未判定';
    els.acceptancePill.className = 'pill ' + (acceptance.status === 'pass' ? 'ok' : acceptance.status === 'fail' ? 'bad' : '');
    els.acceptancePill.title = acceptance.message || '';
  }

  // ------------------------------------------------------------------ 锚串警告 / 阅读口径

  function renderWarnings() {
    const warnings = state.data.warnings || [];
    if (!warnings.length) {
      els.warningsPanel.hidden = true;
      return;
    }
    els.warningsPanel.hidden = false;
    els.warningsCount.textContent = warnings.length + ' 条';
    els.warningsList.classList.toggle('collapsed', !state.warningsExpanded);
    els.warningsToggle.hidden = warnings.length <= 12;
    els.warningsToggle.textContent = state.warningsExpanded
      ? '收起'
      : '展开全部 ' + warnings.length + ' 条';
    clear(els.warningsList);
    warnings.forEach((warning) => {
      const chip = el('button', 'warning-chip');
      chip.type = 'button';
      chip.title = '查看 ' + warning.case_id + ' 的命中明细';
      chip.appendChild(el('b', null, warning.case_id));
      chip.appendChild(el('span', 'warning-anchor', warning.anchor));
      chip.appendChild(el('span', null, '命中 ' + warning.count + ' 块'));
      chip.addEventListener('click', () => {
        const found = (state.data.cases?.retrieval || []).find((row) =>
          row.case_id === warning.case_id && row.strategy === 'hybrid_rerank')
          || (state.data.cases?.retrieval || []).find((row) => row.case_id === warning.case_id);
        if (found) openDrawer(found, 'retrieval');
        else showToast('这条题在报告的逐条明细里没有记录');
      });
      els.warningsList.appendChild(chip);
    });
  }

  function renderNotes() {
    const notes = state.data.notes || [];
    if (!notes.length) {
      els.notesPanel.hidden = true;
      return;
    }
    els.notesPanel.hidden = false;
    clear(els.notesList);
    notes.forEach((note) => {
      const item = el('div', 'note-item');
      item.appendChild(el('div', 'note-label', note.label));
      item.appendChild(el('div', 'note-text', note.text));
      els.notesList.appendChild(item);
    });
  }

  // ------------------------------------------------------------------ 逐条明细

  function caseStrategies() {
    return state.caseTab === 'generation'
      ? (state.data.meta?.generation_strategies || [])
      : (state.data.meta?.strategies || []);
  }

  function renderFilters() {
    clear(els.filterStrategy);
    const all = el('option', null, '全部策略');
    all.value = '';
    els.filterStrategy.appendChild(all);
    caseStrategies().forEach((strategy) => {
      const option = el('option', null, strategyLabel(strategy));
      option.value = strategy;
      els.filterStrategy.appendChild(option);
    });
    els.filterStrategy.value = caseStrategies().includes(state.filters.strategy) ? state.filters.strategy : '';

    clear(els.filterBucket);
    const allBucket = el('option', null, '全部分桶');
    allBucket.value = '';
    els.filterBucket.appendChild(allBucket);
    (state.data.labels?.buckets_order || []).forEach((bucket) => {
      const option = el('option', null, bucketLabel(bucket));
      option.value = bucket;
      els.filterBucket.appendChild(option);
    });
    els.filterBucket.value = state.filters.bucket;
    els.filterBehavior.value = state.filters.behavior;
  }

  function filterCases(rows) {
    return rows.filter((row) => {
      if (state.filters.strategy && row.strategy !== state.filters.strategy) return false;
      if (state.filters.bucket && row.bucket !== state.filters.bucket) return false;
      if (state.filters.behavior && row.behavior !== state.filters.behavior) return false;
      return true;
    });
  }

  function buildHead(cells) {
    const tr = el('tr');
    cells.forEach((cell) => {
      const th = el('th', null, cell.label);
      th.scope = 'col';
      if (cell.alignLeft) th.style.textAlign = 'left';
      tr.appendChild(th);
    });
    return tr;
  }

  function tag(text, kind) {
    return el('span', 'tag ' + (kind || ''), text);
  }

  function renderCases() {
    els.caseTabs.forEach((button) => {
      const active = button.dataset.caseTab === state.caseTab;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    renderFilters();

    const isGeneration = state.caseTab === 'generation';
    const rows = filterCases(state.data.cases?.[state.caseTab] || []);
    clear(els.casesHead);
    clear(els.casesBody);
    els.casesCount.textContent = '共 ' + rows.length + ' 条';

    if (isGeneration) {
      els.casesHead.appendChild(buildHead([
        { label: '题号' }, { label: '策略' }, { label: '分桶' }, { label: '行为' },
        { label: '问题', alignLeft: true }, { label: '忠实度' }, { label: 'Claim' },
        { label: '引用' }, { label: '结果' }, { label: '答案摘要', alignLeft: true },
      ]));
    } else {
      els.casesHead.appendChild(buildHead([
        { label: '题号' }, { label: '策略' }, { label: '分桶' }, { label: '难度' },
        { label: '问题', alignLeft: true }, { label: '首个命中' }, { label: 'Top5' },
        { label: 'Gold' }, { label: '降级' },
      ]));
    }

    rows.forEach((row) => {
      const tr = el('tr');
      tr.dataset.caseId = row.case_id;
      tr.tabIndex = 0;
      const open = () => openDrawer(row, state.caseTab);
      tr.addEventListener('click', open);
      tr.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
      });

      tr.appendChild(el('td', null, row.case_id));
      const strategyCell = el('td');
      strategyCell.appendChild(el('span', null, row.strategy_label || strategyLabel(row.strategy)));
      tr.appendChild(strategyCell);
      tr.appendChild(el('td', null, row.bucket_label || bucketLabel(row.bucket)));

      if (isGeneration) {
        const behavior = el('td');
        behavior.appendChild(tag(row.behavior === 'refuse' ? '应拒答' : '应作答', row.behavior === 'refuse' ? 'refuse' : 'answer'));
        tr.appendChild(behavior);

        const query = el('td', 'case-query', row.query || row.case_id);
        query.title = row.query || '';
        tr.appendChild(query);

        tr.appendChild(el('td', 'num' + (row.faithfulness == null ? ' muted' : ''), fmt(row.faithfulness)));
        tr.appendChild(el('td', 'num', row.claims_supported + ' / ' + row.claims_total));
        tr.appendChild(el('td', 'num', String((row.citations?.valid || []).length)));

        const result = el('td');
        result.appendChild(
          row.refused ? tag('拒答', 'refuse') : tag('作答', 'ok')
        );
        tr.appendChild(result);

        const answer = el('td', 'answer-preview', (row.answer || '').replace(/\s+/g, ' ').slice(0, 72) || '—');
        answer.title = (row.answer || '').slice(0, 500);
        tr.appendChild(answer);
      } else {
        tr.appendChild(el('td', null, row.difficulty || '—'));

        const query = el('td', 'case-query', row.query || row.case_id);
        query.title = row.query || '';
        tr.appendChild(query);

        tr.appendChild(el('td', 'num' + (row.first_gold_rank == null ? ' muted' : ''),
          row.first_gold_rank == null ? '未命中' : '第 ' + row.first_gold_rank + ' 位'));
        const top5 = el('td');
        top5.appendChild(tag(row.top5_hit ? '命中' : '未命中', row.top5_hit ? 'ok' : 'fail'));
        tr.appendChild(top5);
        tr.appendChild(el('td', 'num', String(row.gold_count || 0)));
        tr.appendChild(el('td', 'num', row.rerank_degraded ? '是' : '—'));
      }

      if (state.flashCaseId === row.case_id) {
        tr.style.background = '#fff7d6';
        setTimeout(() => { tr.style.background = ''; }, 2200);
        state.flashCaseId = null;
      }
      els.casesBody.appendChild(tr);
    });

    els.casesEmpty.hidden = rows.length > 0;
    els.casesEmpty.textContent = (state.data.cases?.[state.caseTab] || []).length
      ? '没有符合筛选条件的记录。'
      : (isGeneration ? '这份报告没有生成段逐条记录（本轮没跑 --generation）。' : '这份报告没有检索段逐条记录。');
  }

  // ------------------------------------------------------------------ 抽屉

  function closeDrawer() {
    els.drawer.hidden = true;
    els.drawerBackdrop.hidden = true;
    document.body.style.overflow = '';
  }

  function openDrawer(row, kind) {
    els.drawer.hidden = false;
    els.drawerBackdrop.hidden = false;
    document.body.style.overflow = 'hidden';
    clear(els.drawerBody);

    const isGeneration = kind === 'generation';
    els.drawerEyebrow.textContent = (isGeneration ? '生成明细' : '检索明细') +
      ' · ' + (row.strategy_label || strategyLabel(row.strategy));
    els.drawerTitle.textContent = row.query || row.case_id;

    const metaRow = el('div', 'drawer-meta');
    [row.case_id, row.bucket_label || bucketLabel(row.bucket),
     row.behavior === 'refuse' ? '应拒答' : '应作答',
     row.difficulty ? '难度 ' + row.difficulty : '',
     row.rerank_degraded ? '重排已降级' : '']
      .filter(Boolean)
      .forEach((text) => metaRow.appendChild(pill(text)));
    els.drawerBody.appendChild(metaRow);

    if (isGeneration) {
      const answerSection = el('section', 'drawer-section');
      answerSection.appendChild(el('h3', null, '答案'));
      answerSection.appendChild(el('pre', 'drawer-answer', row.answer || '（没有答案）'));
      els.drawerBody.appendChild(answerSection);

      if (row.refused) {
        const refusal = el('section', 'drawer-section');
        refusal.appendChild(el('h3', null, '拒答判定'));
        const box = el('div', 'drawer-answer');
        box.textContent = '来源：' + (row.refusal_source || '—') + '\n原因：' + (row.refusal_reason || '—');
        refusal.appendChild(box);
        els.drawerBody.appendChild(refusal);
      }

      const claimsSection = el('section', 'drawer-section');
      claimsSection.appendChild(el('h3', null,
        'Claim 判定 · ' + row.claims_supported + ' / ' + row.claims_total + ' 支持'));
      if (!row.claims.length) {
        claimsSection.appendChild(el('p', 'empty-inline', '这条没有 claim 判定（拒答题不计忠实度）。'));
      } else {
        const list = el('div', 'claim-list');
        row.claims.forEach((claim) => {
          const item = el('div', 'claim' + (claim.supported ? '' : ' unsupported'));
          item.appendChild(el('span', 'claim-icon', claim.supported ? '✓' : '✗'));
          item.appendChild(el('span', null, claim.text || '—'));
          item.appendChild(el('span', 'claim-evidence',
            claim.evidence_n == null ? '无证据' : '证据 [' + claim.evidence_n + ']'));
          list.appendChild(item);
        });
        claimsSection.appendChild(list);
      }
      els.drawerBody.appendChild(claimsSection);

      const citationsSection = el('section', 'drawer-section');
      citationsSection.appendChild(el('h3', null, '引用编号'));
      const rowNode = el('div', 'citation-row');
      const valid = row.citations?.valid || [];
      const dropped = row.citations?.dropped || [];
      if (valid.length) {
        valid.forEach((n) => rowNode.appendChild(el('span', 'citation-badge', '[' + n + ']')));
      } else {
        rowNode.appendChild(el('span', 'empty-inline', '没有有效引用'));
      }
      if (dropped.length) {
        rowNode.appendChild(el('span', 'empty-inline', '已丢弃越界编号：' + dropped.join('、')));
      }
      citationsSection.appendChild(rowNode);
      els.drawerBody.appendChild(citationsSection);
    } else {
      const rank = el('section', 'drawer-section');
      rank.appendChild(el('h3', null, '命中概览'));
      const box = el('div', 'drawer-answer');
      box.textContent = row.first_gold_rank == null
        ? '前 ' + (row.hit_count || 0) + ' 条命中里没有金标证据。'
        : '第一个金标证据排在第 ' + row.first_gold_rank + ' 位；Top-5 ' +
          (row.top5_hit ? '命中' : '未命中') + '。';
      rank.appendChild(box);
      els.drawerBody.appendChild(rank);

      const hitsSection = el('section', 'drawer-section');
      hitsSection.appendChild(el('h3', null, 'Top-10 命中（★ 为金标）'));
      const list = el('div', 'claim-list');
      const goldSet = new Set(row.gold || []);
      (row.hits || []).forEach((chunkId, index) => {
        const item = el('div', 'claim');
        item.appendChild(el('span', 'claim-icon', goldSet.has(chunkId) ? '★' : '·'));
        item.appendChild(el('span', null, 'chunk #' + chunkId));
        item.appendChild(el('span', 'claim-evidence', '第 ' + (index + 1) + ' 位'));
        if (goldSet.has(chunkId)) item.style.background = '#f0fdf4';
        list.appendChild(item);
      });
      if (!list.childNodes.length) list.appendChild(el('p', 'empty-inline', '没有命中记录。'));
      hitsSection.appendChild(list);
      els.drawerBody.appendChild(hitsSection);

      const goldSection = el('section', 'drawer-section');
      goldSection.appendChild(el('h3', null, '金标证据（' + (row.gold_count || 0) + ' 块）'));
      const goldRow = el('div', 'citation-row');
      (row.gold || []).forEach((chunkId) => goldRow.appendChild(el('span', 'citation-badge', '#' + chunkId)));
      if (!(row.gold || []).length) goldRow.appendChild(el('span', 'empty-inline', '拒答题没有金标。'));
      goldSection.appendChild(goldRow);
      els.drawerBody.appendChild(goldSection);
    }
  }

  // ------------------------------------------------------------------ 渲染总入口

  function renderDashboard() {
    renderReportTime();
    renderMeta();
    renderKpis();
    renderChart();
    renderMetricTable();
    renderWarnings();
    renderCases();
    renderNotes();

    const summary = state.data.summary || {};
    els.bestStrategy.textContent = summary.best_strategy_label || '—';
    const acceptance = summary.acceptance || {};
    els.acceptanceBadge.textContent = acceptance.status === 'pass'
      ? '验收通过'
      : acceptance.status === 'fail' ? '验收未通过' : (acceptance.message || '未判定');
    els.acceptanceBadge.title = acceptance.message || '';
    els.reportPath.textContent = state.data.meta?.report_path || 'evals/reports/ch04.json';
    renderRunStatus(state.data.run || state.run);
  }

  // ------------------------------------------------------------------ 数据加载

  async function refresh(options = {}) {
    const silent = Boolean(options.silent);
    if (!state.data && !silent) showOnly('loading');
    try {
      const payload = await api('/api/rag-eval');
      state.data = payload;
      state.run = payload.run || null;
      showOnly('dashboard');
      renderDashboard();
      if (state.run && state.run.running) startPolling();
    } catch (err) {
      if (state.data) {
        showToast('刷新失败：' + (err.message || err), true);
        return;
      }
      if (err.status === 404) {
        showEmpty(err);
        try {
          const runPayload = await api('/api/rag-eval/run');
          state.run = runPayload.run;
          renderRunStatus(state.run);
          if (state.run && state.run.running) startPolling();
        } catch {
          /* 空状态下的状态探测失败不额外打扰，但顶栏别一直停在「读取中」 */
          renderRunStatus(null);
        }
      } else {
        showError(err);
      }
    }
  }

  // ------------------------------------------------------------------ 重跑

  function stopPolling() {
    if (state.polling) {
      clearInterval(state.polling);
      state.polling = null;
    }
  }

  function startPolling() {
    if (state.polling) return;
    state.polling = setInterval(pollRun, 2000);
  }

  async function pollRun() {
    try {
      const payload = await api('/api/rag-eval/run');
      state.run = payload.run;
      if (state.data) state.data.run = payload.run;
      renderRunStatus(payload.run);
      if (!payload.run.running) {
        stopPolling();
        if (payload.run.status === 'succeeded') showToast('评估完成，报告已更新');
        else if (payload.run.status === 'failed') showToast(payload.run.message || '评估跑完了，但验收未通过', true);
        await refresh({ silent: true });
      }
    } catch (err) {
      stopPolling();
      els.btnRerun.disabled = false;
      els.btnRerun.textContent = '重跑评估';
      els.runBanner.classList.add('bad');
      els.runBannerText.textContent = '轮询评估状态失败：' + (err.message || err);
      showToast('轮询评估状态失败：' + (err.message || err), true);
    }
  }

  async function rerun() {
    if (state.run && state.run.running) {
      showToast('已经有一个评估任务在跑，等它结束再点。');
      return;
    }
    els.btnRerun.disabled = true;
    try {
      const payload = await api('/api/rag-eval/rerun', { method: 'POST' });
      state.run = payload.run;
      if (state.data) state.data.run = payload.run;
      renderRunStatus(payload.run);
      showToast('评估已启动，跑完会自动刷新');
      startPolling();
    } catch (err) {
      if (err.status === 409) {
        showToast('已有评估任务在跑，页面会继续跟踪它的状态。');
        startPolling();
      } else {
        showToast('启动评估失败：' + (err.message || err), true);
        els.btnRerun.disabled = false;
      }
    }
  }

  // ------------------------------------------------------------------ 事件绑定

  els.tabs.forEach((button) => {
    button.addEventListener('click', () => {
      state.tab = button.dataset.tab;
      state.hiddenSeries.clear();
      renderChart();
    });
  });

  els.caseTabs.forEach((button) => {
    button.addEventListener('click', () => {
      state.caseTab = button.dataset.caseTab;
      state.filters.strategy = '';
      renderCases();
    });
  });

  els.filterStrategy.addEventListener('change', () => {
    state.filters.strategy = els.filterStrategy.value;
    renderCases();
  });
  els.filterBucket.addEventListener('change', () => {
    state.filters.bucket = els.filterBucket.value;
    renderCases();
  });
  els.filterBehavior.addEventListener('change', () => {
    state.filters.behavior = els.filterBehavior.value;
    renderCases();
  });

  els.btnRefresh.addEventListener('click', () => refresh({ silent: true }));
  els.btnRerun.addEventListener('click', rerun);
  els.btnRetry.addEventListener('click', () => refresh());
  els.btnEmptyRefresh.addEventListener('click', () => refresh());
  els.btnEmptyRun.addEventListener('click', rerun);
  els.warningsToggle.addEventListener('click', () => {
    state.warningsExpanded = !state.warningsExpanded;
    renderWarnings();
  });

  els.btnToggleLog.addEventListener('click', () => {
    els.runLog.hidden = !els.runLog.hidden;
    if (!els.runLog.hidden) els.runLog.scrollTop = els.runLog.scrollHeight;
    els.btnToggleLog.textContent = els.runLog.hidden ? '查看日志' : '收起日志';
  });

  els.drawerClose.addEventListener('click', closeDrawer);
  els.drawerBackdrop.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !els.drawer.hidden) closeDrawer();
  });

  window.addEventListener('beforeunload', stopPolling);

  // ------------------------------------------------------------------ 启动

  refresh();
})();
