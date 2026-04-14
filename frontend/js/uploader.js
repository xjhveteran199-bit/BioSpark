/**
 * File upload handler with drag-and-drop support.
 * Uploads file to backend API for proper parsing and analysis.
 */

// Global API base — used by all JS modules
var API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000/api'
    : `${window.location.origin}/api`;

const Uploader = {
    fileId: null,
    fileData: null,
    fileName: null,

    init() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const browseLink = dropZone.querySelector('.browse-link');

        // Click to browse
        browseLink.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
        dropZone.addEventListener('click', () => fileInput.click());

        // File selected
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.uploadFile(e.target.files[0]);
            }
        });

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.uploadFile(e.dataTransfer.files[0]);
            }
        });
    },

    async uploadFile(file) {
        const status = document.getElementById('upload-status');
        const signalTypeSelect = document.getElementById('signal-type-select');
        const samplingRateInput = document.getElementById('sampling-rate-input');

        // Show uploading state
        const lang = window.App ? App.lang : 'en';
        status.className = 'status loading';
        status.innerHTML = `<span class="spinner"></span>${lang === 'zh' ? '正在上传并解析...' : 'Uploading and parsing...'}`;
        status.classList.remove('hidden');

        try {
            // Build FormData and POST to backend
            const form = new FormData();
            form.append('file', file);

            // Add signal type hint if user selected one
            let url = `${API_BASE}/upload`;
            const params = new URLSearchParams();
            if (signalTypeSelect && signalTypeSelect.value) {
                params.set('signal_type', signalTypeSelect.value);
            }
            if (params.toString()) url += `?${params}`;

            const headers = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(url, { method: 'POST', body: form, headers });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || `Upload failed (${resp.status})`);
            }

            const data = await resp.json();

            // Store results from backend
            this.fileId = data.file_id;
            this.fileName = data.filename;
            this.fileData = data;

            // Success message
            const sr = data.sampling_rate ? Math.round(data.sampling_rate) : '?';
            const dur = data.duration_sec ? data.duration_sec.toFixed(1) : '?';
            status.className = 'status success';
            status.textContent = `${lang === 'zh' ? '就绪' : 'Ready'}: ${data.filename} | ${data.signal_type.toUpperCase()} | ${data.n_channels} ch | ${dur}s | ${sr} Hz`;

            // Trigger visualization and model selection
            if (window.App) {
                window.App.onFileUploaded(data);
            }

        } catch (err) {
            status.className = 'status error';
            status.textContent = `Error: ${err.message}`;
        }
    },
};
