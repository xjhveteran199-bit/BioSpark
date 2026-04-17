"""Real-time streaming inference engine for wearable biosignal devices.

Maintains a sliding window buffer, applies lightweight preprocessing,
and runs model inference on complete windows.  Designed for WebSocket
data ingestion at device sampling rates (e.g. 360 Hz ECG, 256 Hz EEG).
"""

import time
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt

from backend.config import MODEL_REGISTRY
from backend.services.predictor import predictor


# ── Filter bank (pre-computed SOS coefficients for low latency) ──────────

_SOS_CACHE: dict[tuple, np.ndarray] = {}


def _get_sos(lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Return cached second-order sections for Butterworth bandpass."""
    key = (lowcut, highcut, fs, order)
    if key not in _SOS_CACHE:
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        _SOS_CACHE[key] = butter(order, [low, high], btype="band", output="sos")
    return _SOS_CACHE[key]


# ── Streaming session ────────────────────────────────────────────────────

# Signal-type specific parameters
_STREAM_PARAMS = {
    "ecg": {
        "lowcut": 0.5,
        "highcut": 40.0,
        "segment_length": 187,  # samples per heartbeat @ 360 Hz
        "stride": 93,           # ~50% overlap for smoother updates
        "default_sr": 360,
    },
    "eeg": {
        "lowcut": 0.5,
        "highcut": 45.0,
        "segment_length": 3000,  # 30s epoch @ 100 Hz
        "stride": 500,           # 5s slide
        "default_sr": 100,
    },
    "emg": {
        "lowcut": 20.0,
        "highcut": 450.0,
        "segment_length": 80,    # 400ms @ 200 Hz
        "stride": 40,            # 200ms slide
        "default_sr": 200,
    },
}


class StreamingSession:
    """Manages a single real-time inference session.

    Usage:
        session = StreamingSession("ecg_arrhythmia", sampling_rate=360)
        session.push_samples([0.1, 0.3, -0.2, ...])
        results = session.get_new_predictions()
    """

    def __init__(self, model_id: str, sampling_rate: Optional[float] = None):
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        self.model_id = model_id
        self.model_info = MODEL_REGISTRY[model_id]
        self.signal_type = self.model_info["signal_type"]

        params = _STREAM_PARAMS[self.signal_type]
        self.sampling_rate = sampling_rate or params["default_sr"]
        self.segment_length = self.model_info["input_length"]
        self.stride = params["stride"]
        self.lowcut = params["lowcut"]
        self.highcut = min(params["highcut"], self.sampling_rate * 0.49)

        # SOS filter coefficients
        self._sos = _get_sos(self.lowcut, self.highcut, self.sampling_rate)

        # Raw sample buffer — keep enough for 2 full segments + filter warmup
        buf_size = self.segment_length * 4
        self._buffer = deque(maxlen=buf_size)

        # Track how many samples have been consumed (for stride stepping)
        self._samples_since_last = 0
        self._total_samples = 0

        # Pending predictions not yet read by the consumer
        self._pending: list[dict] = []

        # Session stats
        self.start_time = time.time()
        self.total_predictions = 0

        # Alert thresholds
        self._alert_classes: set[int] = set()
        self._alert_threshold: float = 0.5

    def configure_alerts(self, class_indices: list[int], threshold: float = 0.5):
        """Set which class predictions trigger alerts."""
        self._alert_classes = set(class_indices)
        self._alert_threshold = threshold

    def push_samples(self, samples: list[float] | np.ndarray):
        """Ingest new samples from the device stream.

        Automatically triggers inference when enough new samples accumulate.
        """
        for s in samples:
            self._buffer.append(float(s))
        self._samples_since_last += len(samples)
        self._total_samples += len(samples)

        # Check if we should run inference
        while (self._samples_since_last >= self.stride
               and len(self._buffer) >= self.segment_length):
            self._run_inference()
            self._samples_since_last -= self.stride

    def _run_inference(self):
        """Run inference on the latest window in the buffer."""
        buf_arr = np.array(list(self._buffer), dtype=np.float64)

        # Take the last segment_length samples
        window = buf_arr[-self.segment_length:]

        # Lightweight preprocessing: bandpass → normalize
        try:
            filtered = sosfilt(self._sos, window)
        except Exception:
            filtered = window

        std = np.std(filtered)
        if std > 1e-10:
            normalized = (filtered - np.mean(filtered)) / std
        else:
            normalized = filtered - np.mean(filtered)

        segment = normalized.astype(np.float32)

        # Run prediction
        try:
            result = predictor.predict(self.model_id, [segment])
            if result["predictions"]:
                pred = result["predictions"][0]
                self.total_predictions += 1

                # Check for alert
                is_alert = (
                    pred["class_idx"] in self._alert_classes
                    and pred["confidence"] >= self._alert_threshold
                )

                self._pending.append({
                    "type": "prediction",
                    "timestamp": time.time() - self.start_time,
                    "sample_idx": self._total_samples,
                    "prediction": pred["class"],
                    "class_idx": pred["class_idx"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "signal_window": segment.tolist(),
                    "is_alert": is_alert,
                    "seq": self.total_predictions,
                })
        except Exception as e:
            self._pending.append({
                "type": "error",
                "timestamp": time.time() - self.start_time,
                "message": str(e),
            })

    def get_new_predictions(self) -> list[dict]:
        """Return and clear pending predictions."""
        results = self._pending
        self._pending = []
        return results

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "total_samples": self._total_samples,
            "total_predictions": self.total_predictions,
            "elapsed_sec": round(elapsed, 2),
            "effective_sr": round(self._total_samples / max(elapsed, 0.001), 1),
            "buffer_fill": len(self._buffer),
            "buffer_capacity": self._buffer.maxlen,
        }


# ── Demo signal generators ───────────────────────────────────────────────

def generate_ecg_samples(n_samples: int, sr: float = 360, heart_rate: float = 72,
                         noise: float = 0.05, anomaly_prob: float = 0.05) -> np.ndarray:
    """Generate synthetic ECG-like samples for demo streaming.

    Produces a realistic PQRST waveform with optional PVC-like anomalies.
    """
    t = np.arange(n_samples) / sr
    beat_period = 60.0 / heart_rate
    phase = (t % beat_period) / beat_period  # 0..1 within each beat

    signal = np.zeros(n_samples)

    # P wave (small positive bump at ~0.1-0.2 of beat)
    p_mask = (phase >= 0.05) & (phase < 0.18)
    p_phase = (phase[p_mask] - 0.05) / 0.13
    signal[p_mask] += 0.15 * np.sin(np.pi * p_phase)

    # QRS complex (sharp spike at ~0.25-0.35)
    # Q dip
    q_mask = (phase >= 0.22) & (phase < 0.27)
    q_phase = (phase[q_mask] - 0.22) / 0.05
    signal[q_mask] -= 0.1 * np.sin(np.pi * q_phase)
    # R peak
    r_mask = (phase >= 0.27) & (phase < 0.33)
    r_phase = (phase[r_mask] - 0.27) / 0.06
    signal[r_mask] += 1.0 * np.sin(np.pi * r_phase)
    # S dip
    s_mask = (phase >= 0.33) & (phase < 0.38)
    s_phase = (phase[s_mask] - 0.33) / 0.05
    signal[s_mask] -= 0.2 * np.sin(np.pi * s_phase)

    # T wave (broad positive bump at ~0.45-0.65)
    t_mask = (phase >= 0.42) & (phase < 0.65)
    t_phase = (phase[t_mask] - 0.42) / 0.23
    signal[t_mask] += 0.3 * np.sin(np.pi * t_phase)

    # Inject anomalies (widened QRS / PVC-like)
    rng = np.random.RandomState(int(time.time()) % 2**31)
    beat_starts = np.where(np.diff((t % beat_period) < 0.01) > 0)[0]
    for bs in beat_starts:
        if rng.random() < anomaly_prob:
            anomaly_len = min(int(0.15 * sr), n_samples - bs)
            if anomaly_len > 0:
                anom_t = np.arange(anomaly_len) / sr
                signal[bs:bs + anomaly_len] += 0.8 * np.sin(2 * np.pi * 8 * anom_t) * np.exp(-3 * anom_t)

    # Add noise
    signal += rng.randn(n_samples) * noise

    return signal.astype(np.float32)


def generate_eeg_samples(n_samples: int, sr: float = 100, noise: float = 0.1) -> np.ndarray:
    """Generate synthetic EEG-like samples with mixed frequency bands."""
    t = np.arange(n_samples) / sr
    rng = np.random.RandomState(int(time.time()) % 2**31)

    # Mix of EEG bands
    signal = (
        0.4 * np.sin(2 * np.pi * 2.0 * t + rng.uniform(0, 2 * np.pi))   # Delta
        + 0.3 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))  # Theta
        + 0.5 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi)) # Alpha
        + 0.2 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi)) # Beta
        + rng.randn(n_samples) * noise
    )
    return signal.astype(np.float32)
