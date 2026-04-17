/**
 * Grad-CAM attention heatmap visualization.
 *
 * Overlays CNN attention weights on the original signal using Plotly.js,
 * showing which signal regions the model focuses on for its prediction.
 */

const GradCAM = {
    lastResult: null,
    currentSegment: 0,
    mode: 'analyze', // 'analyze' | 'train'

    // ─── Analyze mode: fetch Grad-CAM for uploaded file ───

    async fetchForAnalysis(fileId, modelId, channel) {
        const section = document.getElementById('gradcam-section');
        const status = document.getElementById('gradcam-status');
        section.classList.remove('hidden');

        status.className = 'status loading';
        const lang = window.App?.lang || 'en';
        status.innerHTML = `<span class="spinner"></span>${lang === 'zh' ? '正在计算 Grad-CAM 注意力热力图...' : 'Computing Grad-CAM attention heatmaps...'}`;
        status.classList.remove('hidden');

        try {
            const headers = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(
                `${API_BASE}/gradcam/${fileId}?model_id=${modelId}&channel=${channel}&max_segments=20`,
                { method: 'POST', headers }
            );

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Grad-CAM failed');
            }

            const data = await resp.json();
            this.mode = 'analyze';
            this.show(data);
            status.className = 'status success';
            const msg = lang === 'zh'
                ? `Grad-CAM 完成: ${data.computed_segments}/${data.total_segments} 个片段`
                : `Grad-CAM complete: ${data.computed_segments}/${data.total_segments} segments`;
            status.textContent = msg;
        } catch (err) {
            status.className = 'status error';
            status.textContent = `Grad-CAM error: ${err.message}`;
        }
    },

    // ─── Train mode: fetch Grad-CAM for training job ───

    async fetchForTraining(jobId) {
        const section = document.getElementById('train-gradcam-section');
        const status = document.getElementById('train-gradcam-status');
        section.classList.remove('hidden');

        status.className = 'status loading';
        const lang = window.App?.lang || 'en';
        status.innerHTML = `<span class="spinner"></span>${lang === 'zh' ? '正在计算 Grad-CAM...' : 'Computing Grad-CAM...'}`;
        status.classList.remove('hidden');

        try {
            const headers = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(
                `${API_BASE}/train/${jobId}/gradcam?max_segments=20`,
                { headers }
            );

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Grad-CAM failed');
            }

            const data = await resp.json();
            this.mode = 'train';
            this.show(data, 'train');
            status.className = 'status success';
            const msg = lang === 'zh'
                ? `Grad-CAM 完成: ${data.computed_segments} 个验证样本`
                : `Grad-CAM complete: ${data.computed_segments} validation samples`;
            status.textContent = msg;
        } catch (err) {
            status.className = 'status error';
            status.textContent = `Grad-CAM error: ${err.message}`;
        }
    },

    // ─── Display results ───

    show(data, prefix) {
        prefix = prefix || '';
        this.lastResult = data;
        this.currentSegment = 0;

        const chartId = prefix ? 'train-gradcam-chart' : 'gradcam-chart';
        const controlsId = prefix ? 'train-gradcam-controls' : 'gradcam-controls';
        const infoId = prefix ? 'train-gradcam-info' : 'gradcam-info';

        // Build segment selector
        const controls = document.getElementById(controlsId);
        if (controls) {
            const gradcamData = data.gradcam;
            const lang = window.App?.lang || 'en';

            let html = `<label><span>${lang === 'zh' ? '片段:' : 'Segment:'}</span>
                <select id="${prefix ? 'train-' : ''}gradcam-segment-select">`;
            gradcamData.forEach((seg, i) => {
                const className = data.classes
                    ? data.classes[seg.predicted_class]
                    : (seg.predicted_class_name || `Class ${seg.predicted_class}`);
                const conf = (seg.confidence * 100).toFixed(1);
                const trueLabel = seg.true_class_name ? ` | True: ${seg.true_class_name}` : '';
                html += `<option value="${i}">#${i + 1} — ${className} (${conf}%)${trueLabel}</option>`;
            });
            html += `</select></label>`;

            // Target class selector
            const classNames = data.classes || data.class_names || [];
            if (classNames.length > 0) {
                html += `<label style="margin-left:1rem;"><span>${lang === 'zh' ? '目标类别:' : 'Target class:'}</span>
                    <select id="${prefix ? 'train-' : ''}gradcam-class-select">
                        <option value="-1">${lang === 'zh' ? '预测类别' : 'Predicted class'}</option>`;
                classNames.forEach((cls, i) => {
                    html += `<option value="${i}">${cls}</option>`;
                });
                html += `</select></label>`;
            }

            controls.innerHTML = html;

            // Bind events
            const segSelect = document.getElementById(`${prefix ? 'train-' : ''}gradcam-segment-select`);
            if (segSelect) {
                segSelect.addEventListener('change', (e) => {
                    this.currentSegment = parseInt(e.target.value);
                    this.plotSegment(chartId, infoId);
                });
            }
            const clsSelect = document.getElementById(`${prefix ? 'train-' : ''}gradcam-class-select`);
            if (clsSelect) {
                clsSelect.addEventListener('change', () => {
                    // Re-fetch would be needed for a different target class;
                    // for now just re-plot with stored data
                    this.plotSegment(chartId, infoId);
                });
            }
        }

        this.plotSegment(chartId, infoId);
    },

    plotSegment(chartId, infoId) {
        const data = this.lastResult;
        if (!data || !data.gradcam || data.gradcam.length === 0) return;

        const seg = data.gradcam[this.currentSegment];
        const signal = seg.signal;
        const heatmap = seg.heatmap;
        const n = signal.length;
        const lang = window.App?.lang || 'en';

        // Time axis (sample indices)
        const xAxis = Array.from({ length: n }, (_, i) => i);

        // Signal trace
        const signalTrace = {
            x: xAxis,
            y: signal,
            type: 'scatter',
            mode: 'lines',
            name: lang === 'zh' ? '信号' : 'Signal',
            line: { color: '#64748b', width: 1.5 },
            yaxis: 'y',
        };

        // Heatmap as filled area trace (background)
        // Scale heatmap to signal range for visual overlay
        const sigMin = Math.min(...signal);
        const sigMax = Math.max(...signal);
        const sigRange = sigMax - sigMin || 1;

        // Create colored segments using heatmap values
        // We'll use a scatter trace with marker coloring for the heatmap
        const heatmapTrace = {
            x: xAxis,
            y: heatmap,
            type: 'scatter',
            mode: 'lines',
            name: lang === 'zh' ? '注意力' : 'Attention',
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 107, 107, 0.15)',
            line: { color: 'rgba(255, 107, 107, 0.6)', width: 1 },
            yaxis: 'y2',
        };

        // Heatmap colorbar trace (invisible scatter for colorbar)
        const colorbarTrace = {
            x: xAxis,
            y: heatmap.map(v => sigMin + v * sigRange),
            type: 'scatter',
            mode: 'markers',
            marker: {
                size: 6,
                color: heatmap,
                colorscale: [
                    [0, 'rgba(59, 130, 246, 0.5)'],
                    [0.25, 'rgba(34, 197, 94, 0.5)'],
                    [0.5, 'rgba(234, 179, 8, 0.6)'],
                    [0.75, 'rgba(249, 115, 22, 0.7)'],
                    [1, 'rgba(239, 68, 68, 0.9)'],
                ],
                colorbar: {
                    title: lang === 'zh' ? '注意力' : 'Attention',
                    titleside: 'right',
                    thickness: 12,
                    len: 0.6,
                },
                showscale: true,
            },
            name: lang === 'zh' ? '热力图' : 'Heatmap',
            yaxis: 'y',
            showlegend: false,
        };

        // Highlight top-attention regions
        const threshold = 0.7;
        const highAttentionX = [];
        const highAttentionY = [];
        for (let i = 0; i < n; i++) {
            if (heatmap[i] >= threshold) {
                highAttentionX.push(i);
                highAttentionY.push(signal[i]);
            }
        }

        const highlightTrace = {
            x: highAttentionX,
            y: highAttentionY,
            type: 'scatter',
            mode: 'markers',
            marker: {
                size: 3,
                color: 'rgba(239, 68, 68, 0.8)',
                symbol: 'circle',
            },
            name: lang === 'zh' ? '高注意力区域' : 'High attention',
            yaxis: 'y',
        };

        const classNames = data.classes || data.class_names || [];
        const predName = classNames[seg.predicted_class] || `Class ${seg.predicted_class}`;
        const conf = (seg.confidence * 100).toFixed(1);

        const layout = {
            title: {
                text: lang === 'zh'
                    ? `Grad-CAM 注意力热力图 — 片段 #${seg.segment_idx + 1} — ${predName} (${conf}%)`
                    : `Grad-CAM Attention Heatmap — Segment #${seg.segment_idx + 1} — ${predName} (${conf}%)`,
                font: { size: 14 },
            },
            xaxis: {
                title: lang === 'zh' ? '样本点' : 'Sample Index',
                gridcolor: '#e2e8f0',
            },
            yaxis: {
                title: lang === 'zh' ? '信号幅值' : 'Signal Amplitude',
                gridcolor: '#e2e8f0',
            },
            yaxis2: {
                title: lang === 'zh' ? '注意力权重' : 'Attention Weight',
                overlaying: 'y',
                side: 'right',
                range: [0, 1.05],
                showgrid: false,
            },
            margin: { t: 50, r: 80, b: 50, l: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'white',
            legend: {
                x: 0.01, y: 0.99,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
            },
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
            shapes: this._buildAttentionShapes(heatmap, n),
        };

        Plotly.newPlot(chartId, [signalTrace, heatmapTrace, colorbarTrace, highlightTrace], layout, {
            responsive: true,
            displayModeBar: true,
        });

        // Info panel
        this._renderInfo(infoId, seg, data);
    },

    _buildAttentionShapes(heatmap, n) {
        // Build vertical highlight rectangles for high-attention regions
        const shapes = [];
        const threshold = 0.6;
        let inRegion = false;
        let regionStart = 0;

        for (let i = 0; i <= n; i++) {
            const val = i < n ? heatmap[i] : 0;
            if (val >= threshold && !inRegion) {
                inRegion = true;
                regionStart = i;
            } else if (val < threshold && inRegion) {
                inRegion = false;
                shapes.push({
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: regionStart, x1: i,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(239, 68, 68, 0.08)',
                    line: { width: 0 },
                    layer: 'below',
                });
            }
        }
        return shapes;
    },

    _renderInfo(infoId, seg, data) {
        const info = document.getElementById(infoId);
        if (!info) return;

        const lang = window.App?.lang || 'en';
        const classNames = data.classes || data.class_names || [];
        const predName = classNames[seg.predicted_class] || `Class ${seg.predicted_class}`;
        const conf = (seg.confidence * 100).toFixed(1);

        // Find top-3 attention regions
        const heatmap = seg.heatmap;
        const n = heatmap.length;
        const regionSize = Math.max(1, Math.floor(n / 10));
        const regionScores = [];
        for (let i = 0; i <= n - regionSize; i += Math.floor(regionSize / 2)) {
            const end = Math.min(i + regionSize, n);
            const slice = heatmap.slice(i, end);
            const avgAttn = slice.reduce((a, b) => a + b, 0) / slice.length;
            regionScores.push({ start: i, end, avgAttn });
        }
        regionScores.sort((a, b) => b.avgAttn - a.avgAttn);
        const topRegions = regionScores.slice(0, 3);

        // Probability bars
        let probsHtml = '';
        if (seg.probabilities) {
            const probs = seg.probabilities;
            const sortedIdx = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
            sortedIdx.slice(0, 5).forEach(({ p, i }) => {
                const clsName = classNames[i] || `Class ${i}`;
                const pct = (p * 100).toFixed(1);
                const isTop = i === seg.predicted_class;
                probsHtml += `
                    <div style="margin-bottom: 4px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                            <span${isTop ? ' style="font-weight:700;"' : ''}>${clsName}</span>
                            <span>${pct}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:${pct}%; background: ${isTop ? 'var(--primary)' : '#94a3b8'};"></div>
                        </div>
                    </div>`;
            });
        }

        // Attention statistics
        const maxAttn = Math.max(...heatmap).toFixed(3);
        const avgAttn = (heatmap.reduce((a, b) => a + b, 0) / n).toFixed(3);
        const highPct = ((heatmap.filter(v => v >= 0.5).length / n) * 100).toFixed(1);

        const trueInfo = seg.true_class_name
            ? `<div class="gradcam-stat"><strong>${lang === 'zh' ? '真实标签:' : 'True label:'}</strong> ${seg.true_class_name}</div>`
            : '';

        info.innerHTML = `
            <div class="gradcam-info-grid">
                <div class="gradcam-info-col">
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '预测:' : 'Prediction:'}</strong> ${predName}</div>
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '置信度:' : 'Confidence:'}</strong> ${conf}%</div>
                    ${trueInfo}
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '特征图大小:' : 'Feature map size:'}</strong> ${seg.feature_map_size}</div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid var(--border);">
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '最大注意力:' : 'Max attention:'}</strong> ${maxAttn}</div>
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '平均注意力:' : 'Mean attention:'}</strong> ${avgAttn}</div>
                    <div class="gradcam-stat"><strong>${lang === 'zh' ? '高注意力占比:' : 'High attn ratio:'}</strong> ${highPct}%</div>
                </div>
                <div class="gradcam-info-col">
                    <div style="font-weight:600;margin-bottom:6px;font-size:0.85rem;">${lang === 'zh' ? '类别概率分布' : 'Class Probabilities'}</div>
                    ${probsHtml}
                </div>
                <div class="gradcam-info-col">
                    <div style="font-weight:600;margin-bottom:6px;font-size:0.85rem;">${lang === 'zh' ? '高注意力区域 (Top-3)' : 'Top-3 Attention Regions'}</div>
                    ${topRegions.map((r, i) => `
                        <div class="gradcam-region">
                            <span class="gradcam-region-badge">#${i + 1}</span>
                            ${lang === 'zh' ? '样本' : 'Samples'} ${r.start}–${r.end}
                            <span style="color:var(--primary);font-weight:600;margin-left:auto;">
                                ${(r.avgAttn * 100).toFixed(1)}%
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },
};

window.GradCAM = GradCAM;
