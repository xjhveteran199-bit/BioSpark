/**
 * Real-time streaming inference UI — WebSocket-driven live signal monitoring.
 *
 * Features:
 *   - Scrolling real-time signal chart (Plotly.js)
 *   - Live prediction display with confidence bars
 *   - Prediction history timeline
 *   - Alert system for anomalous detections
 *   - Session statistics dashboard
 *   - Demo mode (synthetic ECG/EEG) and device mode
 */

const Streaming = {
    ws: null,
    config: null,
    running: false,

    // Signal buffer for the scrolling chart
    signalBuffer: [],
    maxDisplaySamples: 2000,  // ~5.5s of ECG @ 360Hz

    // Prediction history
    predictions: [],
    maxPredictions: 200,
    alerts: [],
    maxAlerts: 50,

    // Class distribution counts
    classCounts: {},

    // Stats
    stats: { total_samples: 0, total_predictions: 0, elapsed_sec: 0 },

    // Chart update throttle
    _chartTimer: null,
    _chartInterval: 80, // ms (~12 FPS for smooth scrolling)

    // ─── Connection management ───────────────────────────

    start(modelId, mode, options) {
        if (this.ws) this.stop();

        const lang = window.App?.lang || 'en';
        this._setStatus('stream-status', 'loading',
            `<span class="spinner"></span>${lang === 'zh' ? '正在连接...' : 'Connecting...'}`);

        // Build WS URL
        const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
            ? 'localhost:8000' : location.host;
        const wsUrl = `${wsProto}//${wsHost}/api/stream/ws`;

        this.ws = new WebSocket(wsUrl);
        this.running = false;

        // Reset state
        this.signalBuffer = [];
        this.predictions = [];
        this.alerts = [];
        this.classCounts = {};
        this.stats = { total_samples: 0, total_predictions: 0, elapsed_sec: 0 };
        this._clearAlertLog();

        this.ws.onopen = () => {
            this.ws.send(JSON.stringify({
                type: 'start',
                model_id: modelId,
                mode: mode || 'demo',
                sampling_rate: options?.sampling_rate || null,
                heart_rate: options?.heart_rate || 72,
            }));
        };

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this._handleMessage(msg);
        };

        this.ws.onerror = () => {
            this._setStatus('stream-status', 'error',
                lang === 'zh' ? 'WebSocket 连接失败' : 'WebSocket connection failed');
        };

        this.ws.onclose = () => {
            this.running = false;
            this._stopChartUpdates();
            this._updateControlButtons();
        };
    },

    stop() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop' }));
        }
        this.running = false;
        this._stopChartUpdates();
        this._updateControlButtons();
    },

    disconnect() {
        this.stop();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    },

    // ─── Message handler ─────────────────────────────────

    _handleMessage(msg) {
        const lang = window.App?.lang || 'en';

        switch (msg.type) {
            case 'config':
                this.config = msg;
                this.running = true;
                this._updateControlButtons();
                this._setStatus('stream-status', 'success',
                    lang === 'zh'
                        ? `已连接 — ${msg.model_description} @ ${msg.sampling_rate} Hz (${msg.mode === 'demo' ? '演示模式' : '设备模式'})`
                        : `Connected — ${msg.model_description} @ ${msg.sampling_rate} Hz (${msg.mode} mode)`);
                this._initCharts();
                this._startChartUpdates();
                break;

            case 'samples':
                // Append to signal buffer
                const newSamples = msg.data;
                this.signalBuffer.push(...newSamples);
                // Trim to max display window
                if (this.signalBuffer.length > this.maxDisplaySamples) {
                    this.signalBuffer = this.signalBuffer.slice(-this.maxDisplaySamples);
                }
                break;

            case 'prediction':
                this._onPrediction(msg);
                break;

            case 'alert':
                this._onAlert(msg);
                // Also handle as a prediction
                if (msg.prediction) this._onPrediction(msg);
                break;

            case 'stats':
                this.stats = msg;
                this._updateStatsPanel();
                break;

            case 'stopped':
                this.running = false;
                this.stats = { ...this.stats, ...msg };
                this._updateStatsPanel();
                this._stopChartUpdates();
                this._updateControlButtons();
                this._setStatus('stream-status', 'success',
                    lang === 'zh'
                        ? `已停止 — 共处理 ${msg.total_samples || 0} 个样本, ${msg.total_predictions || 0} 次预测`
                        : `Stopped — ${msg.total_samples || 0} samples, ${msg.total_predictions || 0} predictions`);
                break;

            case 'error':
                this._setStatus('stream-status', 'error', `Error: ${msg.message}`);
                break;
        }
    },

    _onPrediction(pred) {
        this.predictions.push(pred);
        if (this.predictions.length > this.maxPredictions) {
            this.predictions = this.predictions.slice(-this.maxPredictions);
        }
        // Update class distribution
        const cls = pred.prediction;
        this.classCounts[cls] = (this.classCounts[cls] || 0) + 1;

        this._updateLatestPrediction(pred);
        this._updateDistributionChart();
    },

    _onAlert(alert) {
        this.alerts.push(alert);
        if (this.alerts.length > this.maxAlerts) {
            this.alerts = this.alerts.slice(-this.maxAlerts);
        }
        this._appendAlertLog(alert);
    },

    // ─── Chart initialization ────────────────────────────

    _initCharts() {
        const signalColor = {
            ecg: '#ef4444', eeg: '#3b82f6', emg: '#22c55e'
        }[this.config.signal_type] || '#3b82f6';

        // Signal chart
        Plotly.newPlot('stream-signal-chart', [{
            y: [],
            type: 'scattergl',
            mode: 'lines',
            line: { color: signalColor, width: 1.2 },
            name: this.config.signal_type.toUpperCase(),
        }], {
            margin: { t: 10, r: 10, b: 35, l: 50 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(15,23,42,0.3)',
            font: { size: 11, color: '#e2e8f0' },
            xaxis: {
                title: '',
                gridcolor: 'rgba(148,163,184,0.1)',
                showticklabels: false,
            },
            yaxis: {
                title: '',
                gridcolor: 'rgba(148,163,184,0.1)',
            },
        }, { responsive: true, displayModeBar: false, staticPlot: false });

        // Distribution chart
        Plotly.newPlot('stream-dist-chart', [{
            labels: this.config.classes,
            values: this.config.classes.map(() => 0),
            type: 'pie',
            hole: 0.5,
            marker: { colors: this._classColors() },
            textinfo: 'percent',
            textfont: { size: 10, color: '#fff' },
            hoverinfo: 'label+value+percent',
        }], {
            margin: { t: 5, r: 5, b: 5, l: 5 },
            paper_bgcolor: 'transparent',
            showlegend: false,
            font: { size: 10, color: '#e2e8f0' },
        }, { responsive: true, displayModeBar: false });
    },

    // ─── Periodic chart updates ──────────────────────────

    _startChartUpdates() {
        this._stopChartUpdates();
        this._chartTimer = setInterval(() => this._updateSignalChart(), this._chartInterval);
    },

    _stopChartUpdates() {
        if (this._chartTimer) {
            clearInterval(this._chartTimer);
            this._chartTimer = null;
        }
    },

    _updateSignalChart() {
        const chartEl = document.getElementById('stream-signal-chart');
        if (!chartEl || !chartEl.data) return;

        Plotly.update('stream-signal-chart', {
            y: [this.signalBuffer],
        }, {}, [0]);
    },

    _updateDistributionChart() {
        const chartEl = document.getElementById('stream-dist-chart');
        if (!chartEl || !chartEl.data || !this.config) return;

        const values = this.config.classes.map(cls => this.classCounts[cls] || 0);
        Plotly.update('stream-dist-chart', {
            values: [values],
        }, {}, [0]);
    },

    // ─── Live prediction display ─────────────────────────

    _updateLatestPrediction(pred) {
        const el = document.getElementById('stream-latest-pred');
        if (!el || !this.config) return;

        const lang = window.App?.lang || 'en';
        const classes = this.config.classes;
        const probs = pred.probabilities;
        const ts = pred.timestamp ? pred.timestamp.toFixed(1) : '?';

        // Build probability bars
        let barsHtml = '';
        const probEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        const colors = this._classColors();

        probEntries.forEach(([cls, p], i) => {
            const pct = (p * 100).toFixed(1);
            const isTop = cls === pred.prediction;
            const clsIdx = classes.indexOf(cls);
            const color = clsIdx >= 0 ? colors[clsIdx] : '#64748b';
            barsHtml += `
                <div class="stream-prob-row${isTop ? ' stream-prob-top' : ''}">
                    <span class="stream-prob-label">${cls}</span>
                    <div class="stream-prob-bar-bg">
                        <div class="stream-prob-bar-fill" style="width:${pct}%;background:${color};"></div>
                    </div>
                    <span class="stream-prob-value">${pct}%</span>
                </div>`;
        });

        el.innerHTML = `
            <div class="stream-pred-header">
                <span class="stream-pred-class${pred.is_alert ? ' stream-pred-alert' : ''}">${pred.prediction}</span>
                <span class="stream-pred-conf">${(pred.confidence * 100).toFixed(1)}%</span>
                <span class="stream-pred-time">#${pred.seq} · ${ts}s</span>
            </div>
            ${barsHtml}
        `;
    },

    // ─── Alert log ───────────────────────────────────────

    _appendAlertLog(alert) {
        const log = document.getElementById('stream-alert-log');
        if (!log) return;

        const lang = window.App?.lang || 'en';
        const ts = alert.timestamp ? alert.timestamp.toFixed(1) : '?';
        const conf = (alert.confidence * 100).toFixed(1);

        const entry = document.createElement('div');
        entry.className = 'stream-alert-entry';
        entry.innerHTML = `
            <span class="stream-alert-dot"></span>
            <span class="stream-alert-text">
                <strong>${alert.prediction}</strong> (${conf}%)
                <span class="stream-alert-ts">${ts}s</span>
            </span>
        `;
        log.prepend(entry);

        // Update alert counter badge
        const badge = document.getElementById('stream-alert-count');
        if (badge) {
            badge.textContent = this.alerts.length;
            badge.classList.toggle('hidden', this.alerts.length === 0);
        }

        // Keep only maxAlerts entries in DOM
        while (log.children.length > this.maxAlerts) {
            log.removeChild(log.lastChild);
        }
    },

    _clearAlertLog() {
        const log = document.getElementById('stream-alert-log');
        if (log) log.innerHTML = '';
        const badge = document.getElementById('stream-alert-count');
        if (badge) { badge.textContent = '0'; badge.classList.add('hidden'); }
    },

    // ─── Stats panel ─────────────────────────────────────

    _updateStatsPanel() {
        const el = document.getElementById('stream-stats');
        if (!el) return;

        const lang = window.App?.lang || 'en';
        const s = this.stats;
        const elapsed = s.elapsed_sec || 0;
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        const timeStr = `${mins}:${secs.toString().padStart(2, '0')}`;

        el.innerHTML = `
            <div class="stream-stat-item">
                <span class="stream-stat-value">${timeStr}</span>
                <span class="stream-stat-label">${lang === 'zh' ? '运行时间' : 'Elapsed'}</span>
            </div>
            <div class="stream-stat-item">
                <span class="stream-stat-value">${(s.total_samples || 0).toLocaleString()}</span>
                <span class="stream-stat-label">${lang === 'zh' ? '总样本数' : 'Samples'}</span>
            </div>
            <div class="stream-stat-item">
                <span class="stream-stat-value">${s.total_predictions || 0}</span>
                <span class="stream-stat-label">${lang === 'zh' ? '预测次数' : 'Predictions'}</span>
            </div>
            <div class="stream-stat-item">
                <span class="stream-stat-value">${(s.effective_sr || 0).toFixed(0)}</span>
                <span class="stream-stat-label">${lang === 'zh' ? '有效采样率' : 'Eff. SR (Hz)'}</span>
            </div>
            <div class="stream-stat-item">
                <span class="stream-stat-value">${this.alerts.length}</span>
                <span class="stream-stat-label">${lang === 'zh' ? '异常警报' : 'Alerts'}</span>
            </div>
        `;
    },

    // ─── UI control helpers ──────────────────────────────

    _updateControlButtons() {
        const startBtn = document.getElementById('stream-start-btn');
        const stopBtn = document.getElementById('stream-stop-btn');
        if (startBtn) startBtn.disabled = this.running;
        if (stopBtn) stopBtn.disabled = !this.running;
    },

    _setStatus(id, type, html) {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = `status ${type}`;
        el.innerHTML = html;
        el.classList.remove('hidden');
    },

    _classColors() {
        return [
            '#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6', '#22c55e',
            '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1',
            '#14b8a6', '#e11d48', '#a855f7', '#0ea5e9', '#d946ef',
        ];
    },
};

window.Streaming = Streaming;
