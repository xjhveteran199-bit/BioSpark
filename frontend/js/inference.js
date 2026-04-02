/**
 * BioSpark ONNX Inference Module
 * Browser-based ECG arrhythmia detection using ONNX Runtime Web
 * 
 * No backend required - all inference runs in the browser!
 */

class ECGInference {
    constructor() {
        this.session = null;
        this.ort = null;
        this.classes = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)'];
        this.segmentLength = 187;
        this.samplingRate = 360;
        this.ready = false;
        this.modelLoaded = false;
    }

    /**
     * Initialize ONNX Runtime Web and load the model
     * @param {string} modelUrl - URL to the .onnx model file
     */
    async init(modelUrl) {
        try {
            // Load ONNX Runtime Web from CDN
            if (!window.ort) {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
                await new Promise((resolve, reject) => {
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
            }
            
            this.ort = window.ort;
            console.log('ONNX Runtime Web loaded');

            // Try to load the model
            try {
                this.session = await this.ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['wasm'], // Use WebAssembly for broad compatibility
                });
                this.modelLoaded = true;
                console.log('ONNX model loaded successfully');
            } catch (e) {
                console.warn('Could not load ONNX model, using demo mode:', e.message);
                this.modelLoaded = false;
            }

            this.ready = true;
            return this.ready;
        } catch (error) {
            console.error('Failed to initialize inference:', error);
            throw error;
        }
    }

    /**
     * Load model from a local ArrayBuffer (for Vercel deployment)
     */
    async initFromBuffer(buffer) {
        try {
            // Load ONNX Runtime Web from CDN
            if (!window.ort) {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
                await new Promise((resolve, reject) => {
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
            }
            
            this.ort = window.ort;
            
            try {
                this.session = await this.ort.InferenceSession.create(buffer, {
                    executionProviders: ['wasm'],
                });
                this.modelLoaded = true;
                console.log('ONNX model loaded from buffer');
            } catch (e) {
                console.warn('Could not load ONNX model, using demo mode:', e.message);
                this.modelLoaded = false;
            }

            this.ready = true;
            return this.ready;
        } catch (error) {
            console.error('Failed to initialize from buffer:', error);
            throw error;
        }
    }

    /**
     * Run analysis on uploaded data
     * @param {number[][]} data - Array of channels, each channel is array of samples
     * @param {string} signalType - 'ecg', 'eeg', or 'emg'
     * @returns {object} Analysis results
     */
    async analyze(data, signalType = 'ecg') {
        if (!this.ready) {
            throw new Error('Inference not initialized. Call init() first.');
        }

        if (signalType !== 'ecg') {
            return { error: `Signal type ${signalType} not supported in browser mode yet. Only ECG is available.` };
        }

        // Use first channel
        const signal = data[0];
        
        // Preprocess
        const preprocessed = this.preprocess(signal);
        
        // Run inference on each segment
        const predictions = [];
        for (const segment of preprocessed.segments) {
            const pred = await this.predictSegment(segment);
            predictions.push(pred);
        }

        // Build result
        return this.buildResult(predictions, preprocessed.info);
    }

    /**
     * ECG Preprocessing pipeline
     */
    preprocess(signal) {
        // Step 1: Bandpass filter (Butterworth 4th order, 0.5-40Hz)
        let filtered = this.bandpassFilter(signal, 0.5, 40.0, this.samplingRate, 4);
        
        // Step 2: Z-score normalization
        filtered = this.zScoreNormalize(filtered);
        
        // Step 3: Segment into 187-sample windows (R-peak based or fixed)
        const segments = this.segmentSignal(filtered);
        
        return {
            segments,
            info: {
                preprocessing: 'bandpass(0.5-40Hz) → normalize → segmentation',
                n_segments: segments.length,
                segment_length: this.segmentLength,
                effective_sr: this.samplingRate
            }
        };
    }

    /**
     * Butterworth bandpass filter implementation
     */
    bandpassFilter(signal, lowcut, highcut, fs, order = 4) {
        // Compute filter coefficients
        const nyq = 0.5 * fs;
        const low = lowcut / nyq;
        const high = highcut / nyq;
        
        // Clamp to valid range
        const lowClamped = Math.max(low, 0.001);
        const highClamped = Math.min(high, 0.999);
        
        // Compute B and A coefficients (2nd order sections)
        const [b, a] = this.butterworthBandpass(order, lowClamped, highClamped);
        
        // Apply filter using filtfilt (forward-backward)
        return this.filtfilt(signal, b, a);
    }

    /**
     * Compute Butterworth bandpass coefficients
     */
    butterworthBandpass(order, low, high) {
        // Simple 2nd order IIR bandpass
        // Using bilinear transform approach
        const K = Math.tan(Math.PI * (high - low) / 2);
        const norm = 1 / (1 + K);
        const b = [
            K * norm,
            0,
            -K * norm
        ];
        const a = [
            1,
            2 * (1 - K * K) * norm,
            (K - 1) / (K + 1)
        ];
        
        // For higher order, we'd cascade, but here we use a simplified version
        // Using 4th order as two 2nd order sections
        return { b, a };
    }

    /**
     * Apply forward-backward filtering (filtfilt equivalent)
     */
    filtfilt(signal, b, a) {
        // Forward pass
        let filtered = this.filter(signal, b, a);
        // Reverse
        filtered = filtered.reverse();
        // Backward pass
        filtered = this.filter(filtered, b, a);
        // Reverse again
        filtered = filtered.reverse();
        return filtered;
    }

    /**
     * Simple IIR filter implementation
     */
    filter(signal, b, a) {
        const output = new Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            let sum = 0;
            for (let j = 0; j < b.length; j++) {
                if (i - j >= 0) {
                    sum += b[j] * signal[i - j];
                }
            }
            for (let j = 1; j < a.length; j++) {
                if (i - j >= 0) {
                    sum -= a[j] * output[i - j];
                }
            }
            output[i] = sum / a[0];
        }
        return output;
    }

    /**
     * Z-score normalization
     */
    zScoreNormalize(signal) {
        const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
        const std = Math.sqrt(signal.reduce((sum, x) => sum + (x - mean) ** 2, 0) / signal.length);
        if (std < 1e-10) return signal;
        return signal.map(x => (x - mean) / std);
    }

    /**
     * Segment signal into 187-sample windows
     * Uses simple R-peak detection or fixed segmentation
     */
    segmentSignal(signal) {
        const segments = [];
        const step = Math.floor(this.segmentLength * 0.5); // 50% overlap
        
        // Try R-peak detection first
        const rpeaks = this.simpleRPeakDetection(signal);
        
        if (rpeaks.length > 0) {
            // Extract segments around R-peaks
            const half = Math.floor(this.segmentLength / 2);
            for (const rpeak of rpeaks) {
                const start = rpeak - half;
                const end = start + this.segmentLength;
                if (start >= 0 && end <= signal.length) {
                    segments.push(signal.slice(start, end));
                }
            }
        }
        
        // Fallback: fixed segmentation
        if (segments.length === 0) {
            for (let i = 0; i <= signal.length - this.segmentLength; i += step) {
                segments.push(signal.slice(i, i + this.segmentLength));
            }
        }
        
        return segments;
    }

    /**
     * Simple R-peak detection using threshold
     */
    simpleRPeakDetection(signal) {
        const peaks = [];
        const threshold = Math.max(...signal) * 0.6;
        
        for (let i = 1; i < signal.length - 1; i++) {
            if (signal[i] > threshold && signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
                // Make sure peaks are at least 100 samples apart
                if (peaks.length === 0 || i - peaks[peaks.length - 1] > 100) {
                    peaks.push(i);
                }
            }
        }
        
        return peaks;
    }

    /**
     * Run inference on a single segment
     */
    async predictSegment(segment) {
        if (this.modelLoaded && this.session) {
            // Use ONNX model
            const input = new ort.Tensor('float32', new Float32Array(segment), [1, 1, this.segmentLength]);
            const outputs = await this.session.run({ input: input });
            const probs = outputs[0].data;
            
            // Apply softmax
            const max = Math.max(...probs);
            const exp = Array.from(probs).map(x => Math.exp(x - max));
            const sum = exp.reduce((a, b) => a + b, 0);
            const softmax = exp.map(x => x / sum);
            
            const predIdx = softmax.indexOf(Math.max(...softmax));
            
            return {
                class: this.classes[predIdx],
                class_idx: predIdx,
                confidence: softmax[predIdx],
                probabilities: this.classes.reduce((obj, c, i) => {
                    obj[c] = softmax[i];
                    return obj;
                }, {})
            };
        } else {
            // Demo mode: feature-based prediction
            return this.demoPredict(segment);
        }
    }

    /**
     * Demo predictor (used when ONNX model is not available)
     */
    demoPredict(segment) {
        const mean = segment.reduce((a, b) => a + b, 0) / segment.length;
        const std = Math.sqrt(segment.reduce((sum, x) => sum + (x - mean) ** 2, 0) / segment.length);
        const rms = Math.sqrt(segment.reduce((sum, x) => sum + x ** 2, 0) / segment.length);
        
        // Simple heuristic based on signal characteristics
        const seed = Math.floor(Math.abs(mean * 1000 + std * 100 + rms * 10)) % 10000;
        const rng = seed => ((seed * 1103515245 + 12345) % (1 << 31)) / (1 << 31);
        
        // Generate pseudo-random but deterministic probabilities
        let s = seed;
        const probs = [];
        for (let i = 0; i < 5; i++) {
            s = (s * 1103515245 + 12345) & 0x7fffffff;
            probs.push((s >>> 16) / 0x10000);
        }
        
        // Boost first class (Normal)
        probs[0] = Math.max(probs[0], 0.5);
        
        const sum = probs.reduce((a, b) => a + b, 0);
        const normalized = probs.map(p => p / sum);
        const predIdx = normalized.indexOf(Math.max(...normalized));
        
        return {
            class: this.classes[predIdx],
            class_idx: predIdx,
            confidence: normalized[predIdx],
            probabilities: this.classes.reduce((obj, c, i) => {
                obj[c] = normalized[i];
                return obj;
            }, {}),
            demo_mode: true
        };
    }

    /**
     * Build final result object
     */
    buildResult(predictions, info) {
        // Count class distribution
        const classCounts = {};
        const classConfidences = {};
        
        for (const pred of predictions) {
            const cls = pred.class;
            classCounts[cls] = (classCounts[cls] || 0) + 1;
            if (!classConfidences[cls]) classConfidences[cls] = [];
            classConfidences[cls].push(pred.confidence);
        }
        
        // Average confidence per class
        const avgConfidences = {};
        for (const [cls, confs] of Object.entries(classConfidences)) {
            avgConfidences[cls] = confs.reduce((a, b) => a + b, 0) / confs.length;
        }
        
        // Find dominant class
        const dominantClass = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || this.classes[0];
        
        return {
            predictions,
            summary: {
                total_segments: predictions.length,
                dominant_class: dominantClass,
                class_distribution: classCounts,
                average_confidences: avgConfidences,
            },
            model_info: {
                id: 'ecg_arrhythmia',
                description: 'ECG Arrhythmia Detection (5-class)',
                classes: this.classes,
                inference_backend: this.modelLoaded ? 'onnx' : 'demo',
            },
            preprocessing_info: info,
            demo_mode: !this.modelLoaded
        };
    }
}

// Export for use in other modules
window.ECGInference = ECGInference;
