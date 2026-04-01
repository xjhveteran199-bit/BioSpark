/**
 * File upload handler with drag-and-drop support.
 */

const API_BASE = window.location.origin + '/api';

const Uploader = {
    fileId: null,
    fileData: null,

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
        const signalType = document.getElementById('signal-type-select').value;

        // Show loading
        status.className = 'status loading';
        status.innerHTML = '<span class="spinner"></span>Uploading and parsing...';
        status.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        let url = `${API_BASE}/upload`;
        if (signalType) {
            url += `?signal_type=${signalType}`;
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await response.json();
            this.fileId = data.file_id;
            this.fileData = data;

            // Success
            status.className = 'status success';
            status.textContent = `Uploaded: ${data.filename} | ${data.signal_type.toUpperCase()} | ${data.n_channels} channels | ${data.duration_sec}s | ${data.sampling_rate} Hz`;

            // Trigger visualization
            if (window.App) {
                window.App.onFileUploaded(data);
            }
        } catch (err) {
            status.className = 'status error';
            status.textContent = `Error: ${err.message}`;
        }
    },
};
