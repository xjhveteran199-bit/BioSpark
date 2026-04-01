/**
 * Signal visualization using Plotly.js — time domain + FFT spectrum.
 */

const Visualizer = {
    currentData: null,
    currentChannel: 0,
    viewMode: 'time',

    init() {
        const viewSelect = document.getElementById('view-mode-select');
        if (viewSelect) {
            viewSelect.addEventListener('change', (e) => {
                this.viewMode = e.target.value;
                this.plotChannel(this.currentChannel);
            });
        }
    },

    showSignal(data) {
        this.currentData = data;
        this.currentChannel = 0;
        this.viewMode = 'time';

        document.getElementById('preview-section').classList.remove('hidden');

        // Info bar
        const infoBar = document.getElementById('signal-info');
        const typeLabel = { ecg: 'ECG', eeg: 'EEG', emg: 'EMG' }[data.signal_type] || data.signal_type.toUpperCase();
        infoBar.innerHTML = `
            <span class="info-item"><span class="label">Type:</span> <span class="value">${typeLabel}</span></span>
            <span class="info-item"><span class="label">Format:</span> <span class="value">${data.format.toUpperCase()}</span></span>
            <span class="info-item"><span class="label">Channels:</span> <span class="value">${data.n_channels}</span></span>
            <span class="info-item"><span class="label">Samples:</span> <span class="value">${data.n_samples.toLocaleString()}</span></span>
            <span class="info-item"><span class="label">Rate:</span> <span class="value">${data.sampling_rate} Hz</span></span>
            <span class="info-item"><span class="label">Duration:</span> <span class="value">${data.duration_sec}s</span></span>
        `;

        // Channel selector
        const channelSelect = document.getElementById('channel-select');
        channelSelect.innerHTML = data.channels.map((ch, i) =>
            `<option value="${i}">${ch}</option>`
        ).join('');
        channelSelect.onchange = (e) => {
            this.currentChannel = parseInt(e.target.value);
            this.plotChannel(this.currentChannel);
        };

        // Reset view mode
        const viewSelect = document.getElementById('view-mode-select');
        if (viewSelect) viewSelect.value = 'time';

        this.plotChannel(0);
    },

    plotChannel(channelIdx) {
        if (this.viewMode === 'fft') {
            this.plotFFT(channelIdx);
        } else {
            this.plotTimeDomain(channelIdx);
        }
    },

    plotTimeDomain(channelIdx) {
        const data = this.currentData;
        if (!data || !data.preview_data || !data.preview_data[channelIdx]) return;

        const signal = data.preview_data[channelIdx];
        const n = signal.length;
        const dt = data.duration_sec / n;
        const time = Array.from({ length: n }, (_, i) => i * dt);

        const signalColor = { ecg: '#dc2626', eeg: '#2563eb', emg: '#16a34a' }[data.signal_type] || '#2563eb';

        const trace = {
            x: time,
            y: signal,
            type: 'scattergl',
            mode: 'lines',
            name: data.channels[channelIdx],
            line: { color: signalColor, width: 1 },
        };

        const layout = {
            title: { text: `${data.signal_type.toUpperCase()} — ${data.channels[channelIdx]}`, font: { size: 14 } },
            xaxis: { title: 'Time (s)', gridcolor: '#e2e8f0' },
            yaxis: { title: 'Amplitude', gridcolor: '#e2e8f0' },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'white',
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
        };

        Plotly.newPlot('signal-chart', [trace], layout, { responsive: true, displayModeBar: true });
    },

    plotFFT(channelIdx) {
        const data = this.currentData;
        if (!data || !data.preview_data || !data.preview_data[channelIdx]) return;

        const signal = data.preview_data[channelIdx];
        const n = signal.length;
        const sr = data.sampling_rate * (n / data.n_samples); // Effective sample rate of preview data

        // Simple FFT using DFT (for preview-sized data this is fast enough)
        const fftMag = this._computeFFTMagnitude(signal);
        const freqs = Array.from({ length: Math.floor(n / 2) }, (_, i) => (i * sr / n).toFixed(2));
        const magnitudes = fftMag.slice(0, Math.floor(n / 2));

        // Frequency band annotations
        const bands = data.signal_type === 'eeg' ? [
            { name: 'Delta', range: [0.5, 4], color: 'rgba(239,68,68,0.1)' },
            { name: 'Theta', range: [4, 8], color: 'rgba(249,115,22,0.1)' },
            { name: 'Alpha', range: [8, 13], color: 'rgba(34,197,94,0.1)' },
            { name: 'Beta', range: [13, 30], color: 'rgba(59,130,246,0.1)' },
            { name: 'Gamma', range: [30, 100], color: 'rgba(168,85,247,0.1)' },
        ] : [];

        const signalColor = { ecg: '#dc2626', eeg: '#2563eb', emg: '#16a34a' }[data.signal_type] || '#2563eb';

        const trace = {
            x: freqs,
            y: magnitudes,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Magnitude',
            line: { color: signalColor, width: 1.5 },
            fillcolor: signalColor.replace(')', ',0.1)').replace('rgb', 'rgba'),
        };

        const shapes = bands.map(b => ({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: b.range[0], x1: b.range[1], y0: 0, y1: 1,
            fillcolor: b.color, line: { width: 0 },
        }));

        const annotations = bands.map(b => ({
            x: (b.range[0] + b.range[1]) / 2, y: 1.02,
            xref: 'x', yref: 'paper',
            text: b.name, showarrow: false,
            font: { size: 10, color: '#64748b' },
        }));

        const maxFreq = data.signal_type === 'ecg' ? 50 : data.signal_type === 'emg' ? 500 : 60;

        const layout = {
            title: { text: `Frequency Spectrum — ${data.channels[channelIdx]}`, font: { size: 14 } },
            xaxis: { title: 'Frequency (Hz)', gridcolor: '#e2e8f0', range: [0, Math.min(maxFreq, sr / 2)] },
            yaxis: { title: 'Magnitude', gridcolor: '#e2e8f0' },
            margin: { t: 50, r: 20, b: 50, l: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'white',
            shapes,
            annotations,
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
        };

        Plotly.newPlot('signal-chart', [trace], layout, { responsive: true, displayModeBar: true });
    },

    _computeFFTMagnitude(signal) {
        // Simple magnitude spectrum via DFT (works for preview data ~1000-5000 points)
        const n = signal.length;
        const mag = new Float64Array(n);

        // Use a simple approach: compute |FFT| using real DFT
        // For performance, use radix-2 if power of 2, else truncate
        const N = Math.pow(2, Math.floor(Math.log2(n)));
        const trimmed = signal.slice(0, N);

        // Cooley-Tukey FFT
        const re = new Float64Array(N);
        const im = new Float64Array(N);
        for (let i = 0; i < N; i++) re[i] = trimmed[i];

        this._fft(re, im, N);

        const result = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            result[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]) / N;
        }
        return result;
    },

    _fft(re, im, N) {
        // Bit-reversal permutation
        for (let i = 1, j = 0; i < N; i++) {
            let bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                [re[i], re[j]] = [re[j], re[i]];
                [im[i], im[j]] = [im[j], im[i]];
            }
        }
        // FFT butterfly
        for (let len = 2; len <= N; len <<= 1) {
            const angle = -2 * Math.PI / len;
            const wRe = Math.cos(angle), wIm = Math.sin(angle);
            for (let i = 0; i < N; i += len) {
                let curRe = 1, curIm = 0;
                for (let j = 0; j < len / 2; j++) {
                    const tRe = curRe * re[i + j + len / 2] - curIm * im[i + j + len / 2];
                    const tIm = curRe * im[i + j + len / 2] + curIm * re[i + j + len / 2];
                    re[i + j + len / 2] = re[i + j] - tRe;
                    im[i + j + len / 2] = im[i + j] - tIm;
                    re[i + j] += tRe;
                    im[i + j] += tIm;
                    const newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
    },
};
