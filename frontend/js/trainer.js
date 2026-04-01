/**
 * Trainer — Phase 1: Labeled dataset upload + summary visualization.
 *
 * Handles the "Train" mode of BioSpark:
 *   1. Drag-and-drop / file picker for CSV or ZIP uploads
 *   2. POST to /api/train/upload
 *   3. Renders dataset summary: info bar + Plotly bar chart + class table
 */

const Trainer = (() => {
    // State
    let datasetId = null;
    let datasetSummary = null;

    // Colour palette for class bars
    const CLASS_COLORS = [
        '#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626',
        '#0891b2', '#65a30d', '#c026d3', '#ea580c', '#0f766e',
    ];

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------

    function init() {
        const dropZone = document.getElementById('train-drop-zone');
        const fileInput = document.getElementById('train-file-input');

        // Drag-and-drop
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });

        // File picker
        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) handleFile(fileInput.files[0]);
        });

        // Buttons
        document.getElementById('train-reset-btn').addEventListener('click', resetTrainer);
        document.getElementById('train-next-btn').addEventListener('click', () => {
            // Phase 2 will attach here
            alert('Phase 2 (Training Engine) — coming soon!');
        });

        // Re-bind browse link after i18n updates
        _bindBrowseLink();
    }

    function openFilePicker() {
        document.getElementById('train-file-input').click();
    }

    function _bindBrowseLink() {
        const link = document.querySelector('#train-drop-zone .browse-link');
        if (link) {
            link.onclick = e => {
                e.stopPropagation();
                document.getElementById('train-file-input').click();
            };
        }
    }

    // -----------------------------------------------------------------------
    // File Upload
    // -----------------------------------------------------------------------

    async function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['csv', 'txt', 'zip'].includes(ext)) {
            _setStatus('error', `Unsupported format ".${ext}". Use .csv or .zip`);
            return;
        }

        _setStatus('loading', `<span class="spinner"></span>Parsing dataset…`);

        const form = new FormData();
        form.append('file', file);

        try {
            const resp = await fetch(`${API_BASE}/train/upload`, {
                method: 'POST',
                body: form,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await resp.json();
            datasetId = data.dataset_id;
            datasetSummary = data;

            _setStatus('success', `Dataset parsed: ${data.total_samples} samples, ${data.class_names.length} classes`);
            _showSummary(data);

        } catch (err) {
            _setStatus('error', `Error: ${err.message}`);
        }
    }

    // -----------------------------------------------------------------------
    // Summary Rendering
    // -----------------------------------------------------------------------

    function _showSummary(data) {
        document.getElementById('train-summary-section').classList.remove('hidden');

        _renderInfoBar(data);
        _renderClassChart(data);
        _renderClassTable(data);

        document.getElementById('train-next-btn').disabled = false;
        document.getElementById('train-summary-section').scrollIntoView({ behavior: 'smooth' });
    }

    function _renderInfoBar(data) {
        const bar = document.getElementById('train-info-bar');
        const items = [
            { label: 'Format',   value: data.format === 'zip_folder' ? 'ZIP (folder-per-class)' : 'CSV (labeled)' },
            { label: 'Classes',  value: data.class_names.length },
            { label: 'Samples',  value: data.total_samples.toLocaleString() },
            { label: 'Length',   value: `${data.signal_length} pts` },
            { label: 'Channels', value: data.n_channels },
        ];
        bar.innerHTML = items.map(it =>
            `<div class="info-item">
                <span class="label">${it.label}:</span>
                <span class="value">${it.value}</span>
             </div>`
        ).join('');
    }

    function _renderClassChart(data) {
        const { class_names, class_counts } = data;
        const counts = class_names.map(c => class_counts[c] ?? 0);
        const colors = class_names.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);

        const trace = {
            type: 'bar',
            x: class_names,
            y: counts,
            marker: { color: colors },
            text: counts.map(String),
            textposition: 'outside',
            hovertemplate: '<b>%{x}</b><br>Samples: %{y}<extra></extra>',
        };

        const layout = {
            margin: { t: 20, r: 10, b: 60, l: 50 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { size: 12, color: '#1e293b' },
            xaxis: { title: 'Class', tickangle: class_names.length > 6 ? -35 : 0 },
            yaxis: { title: 'Sample Count' },
            bargap: 0.3,
        };

        Plotly.newPlot('train-class-chart', [trace], layout, { responsive: true, displayModeBar: false });
    }

    function _renderClassTable(data) {
        const { class_names, class_counts, total_samples } = data;
        const colors = class_names.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);

        const rows = class_names.map((cls, i) => {
            const count = class_counts[cls] ?? 0;
            const pct = total_samples > 0 ? ((count / total_samples) * 100).toFixed(1) : '0.0';
            return `
                <tr>
                    <td>
                        <span class="class-dot" style="background:${colors[i]}"></span>
                        ${cls}
                    </td>
                    <td>${count.toLocaleString()}</td>
                    <td>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:${pct}%;background:${colors[i]}"></div>
                        </div>
                        <span style="font-size:0.8rem;color:var(--text-secondary)">${pct}%</span>
                    </td>
                </tr>`;
        }).join('');

        document.getElementById('train-class-table').innerHTML = `
            <table class="train-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Samples</th>
                        <th>Distribution</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    function _setStatus(type, html) {
        const el = document.getElementById('train-upload-status');
        el.className = `status ${type}`;
        el.innerHTML = html;
        el.classList.remove('hidden');
    }

    function resetTrainer() {
        datasetId = null;
        datasetSummary = null;

        document.getElementById('train-summary-section').classList.add('hidden');
        document.getElementById('train-upload-status').classList.add('hidden');
        document.getElementById('train-file-input').value = '';
        document.getElementById('train-next-btn').disabled = true;

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Public API
    return { init, openFilePicker, handleFile, reset: resetTrainer, _bindBrowseLink };
})();

window.Trainer = Trainer;
