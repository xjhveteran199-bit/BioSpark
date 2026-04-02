/**
 * File upload handler with drag-and-drop support.
 * For Vercel deployment - stores file content in memory for stateless analysis.
 */

const API_BASE = window.location.origin + '/api';

const Uploader = {
    fileId: null,
    fileData: null,
    fileContent: null,  // Store file content for stateless Vercel API
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

        // Show loading
        status.className = 'status loading';
        status.innerHTML = '<span class="spinner"></span>Reading file...';
        status.classList.remove('hidden');

        try {
            // Read file as ArrayBuffer
            const arrayBuffer = await file.arrayBuffer();
            this.fileContent = new Uint8Array(arrayBuffer);
            this.fileName = file.name;
            this.fileId = 'vercel-' + Math.random().toString(36).substr(2, 8);

            // Parse file locally for preview (basic CSV parsing)
            const text = new TextDecoder().decode(this.fileContent);
            const parsed = this.parseLocalCSV(text);
            
            this.fileData = {
                file_id: this.fileId,
                filename: file.name,
                signal_type: parsed.signal_type,
                format: parsed.format,
                channels: parsed.channels,
                n_channels: parsed.n_channels,
                n_samples: parsed.n_samples,
                sampling_rate: parsed.sampling_rate,
                duration_sec: parsed.duration_sec,
                preview_data: parsed.preview_data,
            };

            // Success
            status.className = 'status success';
            status.textContent = `Ready: ${file.name} | ${parsed.signal_type.toUpperCase()} | ${parsed.n_channels} channels | ${parsed.duration_sec}s | ${parsed.sampling_rate} Hz`;

            // Trigger visualization
            if (window.App) {
                window.App.onFileUploaded(this.fileData);
            }
        } catch (err) {
            status.className = 'status error';
            status.textContent = `Error: ${err.message}`;
        }
    },

    parseLocalCSV(text) {
        const lines = text.trim().split('\n');
        if (!lines.length) throw new Error("Empty file");

        const header = lines[0].toLower();
        const hasTime = header.includes('time') || header.includes('sample');
        const dataLines = lines.slice(1);

        let signal = [];
        let sampling_rate = 360;

        if (hasTime) {
            const times = [];
            const values = [];
            for (const line of dataLines) {
                const parts = line.split(',');
                if (parts.length >= 2) {
                    const t = parseFloat(parts[0].trim());
                    const v = parseFloat(parts[1].trim());
                    if (!isNaN(t) && !isNaN(v)) {
                        times.push(t);
                        values.push(v);
                    }
                }
            }
            signal = values;
            if (times.length > 1) {
                const dt = (times[times.length - 1] - times[0]) / (times.length - 1);
                sampling_rate = dt > 0 ? 1 / dt : 360;
            }
        } else {
            for (const line of dataLines) {
                const parts = line.trim().split(',');
                for (const p of parts) {
                    const v = parseFloat(p.trim());
                    if (!isNaN(v)) signal.push(v);
                }
            }
        }

        if (!signal.length) throw new Error("No valid numeric data found");

        // Auto-detect signal type
        let signal_type = 'ecg';
        if (sampling_rate >= 100 && sampling_rate <= 256) signal_type = 'eeg';
        else if (sampling_rate > 500) signal_type = 'emg';

        const n_samples = signal.length;
        const duration_sec = n_samples / sampling_rate;

        // Downsample for preview
        let preview_data = [signal];
        const max_preview = 5000;
        if (n_samples > max_preview) {
            const step = Math.ceil(n_samples / max_preview);
            preview_data = [signal.filter((_, i) => i % step === 0)];
        }

        return {
            signal_type,
            format: 'csv',
            channels: ['signal'],
            n_channels: 1,
            n_samples,
            sampling_rate,
            duration_sec,
            preview_data,
        };
    },
};
