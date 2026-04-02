/**
 * Main application controller with i18n support.
 * Updated for Vercel deployment - uses stateless combined upload+analyze endpoint.
 */

const API_BASE = window.location.origin + '/api';

const App = {
    selectedModel: null,
    lang: 'en', // 'en' or 'zh'

    init() {
        Uploader.init();
        Visualizer.init();
        this.bindEvents();
        this.loadModels();

        // Detect browser language
        if (navigator.language.startsWith('zh')) {
            this.lang = 'zh';
            this.applyLang();
        }
    },

    bindEvents() {
        document.getElementById('analyze-btn').addEventListener('click', () => this.runAnalysis());
        document.getElementById('export-json-btn').addEventListener('click', () => Results.exportJSON());
        document.getElementById('export-csv-btn').addEventListener('click', () => Results.exportCSV());
        document.getElementById('new-analysis-btn').addEventListener('click', () => this.reset());
    },

    // --- Language Toggle ---
    toggleLang() {
        this.lang = this.lang === 'en' ? 'zh' : 'en';
        this.applyLang();
    },

    applyLang() {
        const lang = this.lang;
        document.querySelectorAll('[data-en][data-zh]').forEach(el => {
            const text = el.getAttribute(`data-${lang}`);
            if (text) el.innerHTML = text;
        });
        // Re-bind browse link after innerHTML change
        const browseLink = document.querySelector('.browse-link');
        if (browseLink) {
            browseLink.addEventListener('click', (e) => {
                e.stopPropagation();
                document.getElementById('file-input').click();
            });
        }
    },

    // --- Models ---
    async loadModels() {
        try {
            const resp = await fetch(`${API_BASE}/models`);
            if (resp.ok) {
                const data = await resp.json();
                this.renderModels(data.models);
            }
        } catch (e) {
            this.renderFallbackModels();
        }
    },

    renderModels(models) {
        const grid = document.getElementById('model-list');
        grid.innerHTML = models.map(m => `
            <div class="model-card" data-model-id="${m.id}" data-signal-type="${m.signal_type}">
                <span class="model-type-badge badge-${m.signal_type}">${m.signal_type.toUpperCase()}</span>
                <div class="model-name">${m.description}</div>
                <div class="model-classes">${m.classes.length} classes: ${m.classes.slice(0, 4).join(', ')}${m.classes.length > 4 ? '...' : ''}</div>
            </div>
        `).join('');

        grid.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => this.selectModel(card));
        });
    },

    renderFallbackModels() {
        const models = [
            { id: 'ecg_arrhythmia', signal_type: 'ecg', description: 'ECG Arrhythmia Detection (5-class)', classes: ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)'] },
            { id: 'eeg_sleep', signal_type: 'eeg', description: 'EEG Sleep Staging (5-class)', classes: ['Wake (W)', 'N1', 'N2', 'N3', 'REM'] },
            { id: 'emg_gesture', signal_type: 'emg', description: 'EMG Gesture Recognition', classes: ['Gesture 1', 'Gesture 2', '...', 'Gesture 52'] },
        ];
        this.renderModels(models);
    },

    selectModel(card) {
        document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        this.selectedModel = card.dataset.modelId;
        document.getElementById('analyze-btn').disabled = false;
    },

    // --- File Upload Callback ---
    onFileUploaded(data) {
        Visualizer.showSignal(data);

        document.getElementById('analysis-section').classList.remove('hidden');

        // Filter models by signal type
        document.querySelectorAll('.model-card').forEach(card => {
            if (card.dataset.signalType === data.signal_type) {
                card.style.opacity = '1';
                card.style.pointerEvents = 'auto';
            } else {
                card.style.opacity = '0.4';
                card.style.pointerEvents = 'none';
            }
        });

        // Auto-select first compatible model
        const compatible = document.querySelector(`.model-card[data-signal-type="${data.signal_type}"]`);
        if (compatible) this.selectModel(compatible);

        document.getElementById('preview-section').scrollIntoView({ behavior: 'smooth' });
    },

    // --- Analysis ---
    async runAnalysis() {
        if (!Uploader.fileContent || !this.selectedModel) return;

        const status = document.getElementById('analyze-status');
        const btn = document.getElementById('analyze-btn');

        btn.disabled = true;
        const loadingMsg = this.lang === 'zh' ? '正在预处理和模型推理...' : 'Preprocessing and running model inference...';
        status.className = 'status loading';
        status.innerHTML = `<span class="spinner"></span>${loadingMsg}`;
        status.classList.remove('hidden');

        try {
            // Build form data with file + model_id for stateless Vercel API
            const formData = new FormData();
            // Create a File-like object from the stored content
            const file = new File([Uploader.fileContent], Uploader.fileName, { type: 'application/octet-stream' });
            formData.append('file', file);
            formData.append('model_id', this.selectedModel);

            const resp = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(err.error || 'Analysis failed');
            }

            const result = await resp.json();

            const successMsg = this.lang === 'zh'
                ? `分析完成: ${result.summary.total_segments} 个片段已分析`
                : `Analysis complete: ${result.summary.total_segments} segments analyzed`;
            status.className = 'status success';
            status.textContent = successMsg;

            Results.show(result);
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });

        } catch (err) {
            status.className = 'status error';
            status.textContent = `Error: ${err.message}`;
        } finally {
            btn.disabled = false;
        }
    },

    // --- Reset ---
    reset() {
        ['preview-section', 'analysis-section', 'results-section'].forEach(id => {
            document.getElementById(id).classList.add('hidden');
        });

        Uploader.fileId = null;
        Uploader.fileData = null;
        Uploader.fileContent = null;
        Uploader.fileName = null;
        this.selectedModel = null;
        document.getElementById('analyze-btn').disabled = true;
        document.getElementById('upload-status').classList.add('hidden');
        document.getElementById('analyze-status').classList.add('hidden');
        document.getElementById('file-input').value = '';

        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('selected');
            card.style.opacity = '1';
            card.style.pointerEvents = 'auto';
        });

        window.scrollTo({ top: 0, behavior: 'smooth' });
    },
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => App.init());
window.App = App;
