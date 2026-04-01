/**
 * Results display and export.
 */

const Results = {
    lastResult: null,

    show(result) {
        this.lastResult = result;
        document.getElementById('results-section').classList.remove('hidden');

        // Demo mode banner
        const banner = document.getElementById('demo-banner');
        if (result.demo_mode) {
            banner.classList.remove('hidden');
            banner.textContent = result.demo_note;
        } else {
            banner.classList.add('hidden');
        }

        // Summary panel
        this.renderSummary(result);

        // Distribution chart
        this.renderDistributionChart(result);

        // Details table
        this.renderDetailsTable(result);
    },

    renderSummary(result) {
        const summary = result.summary;
        const panel = document.getElementById('result-summary');

        let html = `
            <div class="summary-dominant">${summary.dominant_class}</div>
            <div class="summary-stat">Dominant prediction across ${summary.total_segments} segments</div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;">
            <div class="summary-stat"><strong>Model:</strong> ${result.model_info.description}</div>
            <div class="summary-stat"><strong>Channel:</strong> ${result.channel}</div>
            <div class="summary-stat"><strong>Preprocessing:</strong> ${result.preprocessing.preprocessing}</div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;">
            <div class="summary-stat" style="font-weight:600; margin-bottom:6px;">Class Distribution:</div>
        `;

        // Class distribution bars
        const total = summary.total_segments;
        for (const [cls, count] of Object.entries(summary.class_distribution)) {
            const pct = ((count / total) * 100).toFixed(1);
            const conf = summary.average_confidences[cls] ? (summary.average_confidences[cls] * 100).toFixed(1) : '0.0';
            html += `
                <div style="margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span>${cls}</span>
                        <span>${count}/${total} (${pct}%) — avg conf: ${conf}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }

        panel.innerHTML = html;
    },

    renderDistributionChart(result) {
        const summary = result.summary;
        const classes = Object.keys(summary.class_distribution);
        const counts = Object.values(summary.class_distribution);

        const colors = [
            '#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed',
            '#0891b2', '#be185d', '#4d7c0f', '#9333ea', '#ea580c',
        ];

        const trace = {
            labels: classes,
            values: counts,
            type: 'pie',
            hole: 0.4,
            marker: { colors: colors.slice(0, classes.length) },
            textinfo: 'label+percent',
            textfont: { size: 11 },
        };

        const layout = {
            title: { text: 'Prediction Distribution', font: { size: 13 } },
            margin: { t: 40, r: 10, b: 10, l: 10 },
            paper_bgcolor: 'transparent',
            showlegend: false,
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 },
        };

        Plotly.newPlot('result-chart', [trace], layout, { responsive: true, displayModeBar: false });
    },

    renderDetailsTable(result) {
        const container = document.getElementById('result-details');
        const predictions = result.predictions;

        if (predictions.length === 0) {
            container.innerHTML = '<p>No predictions available.</p>';
            return;
        }

        // Show per-segment predictions
        let html = '<table><thead><tr><th>Segment</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody>';
        predictions.forEach((pred, i) => {
            const confPct = (pred.confidence * 100).toFixed(1);
            const confColor = pred.confidence > 0.8 ? '#16a34a' : pred.confidence > 0.5 ? '#d97706' : '#dc2626';
            html += `<tr>
                <td>#${i + 1}</td>
                <td>${pred.class}</td>
                <td style="color: ${confColor}; font-weight: 600;">${confPct}%</td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    exportJSON() {
        if (!this.lastResult) return;
        const blob = new Blob([JSON.stringify(this.lastResult, null, 2)], { type: 'application/json' });
        this._download(blob, 'biosignal_results.json');
    },

    exportCSV() {
        if (!this.lastResult) return;
        const predictions = this.lastResult.predictions;
        let csv = 'segment,class,confidence\n';
        predictions.forEach((pred, i) => {
            csv += `${i + 1},"${pred.class}",${pred.confidence.toFixed(4)}\n`;
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        this._download(blob, 'biosignal_results.csv');
    },

    _download(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    },
};
