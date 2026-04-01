"""Generate sample biosignal data for testing.

Run: python sample_data/generate_samples.py
"""

import numpy as np
import os

output_dir = os.path.dirname(os.path.abspath(__file__))


def generate_ecg_sample():
    """Generate a synthetic ECG-like signal (single lead, MIT-BIH format)."""
    sr = 360  # Hz
    duration = 10  # seconds
    t = np.arange(0, duration, 1 / sr)

    # Simulate ECG with simple sinusoidal components
    heart_rate = 72  # bpm
    f_hr = heart_rate / 60

    # P-QRS-T morphology approximation
    signal = np.zeros_like(t)
    for beat_num in range(int(duration * f_hr)):
        beat_center = beat_num / f_hr
        # P wave
        signal += 0.15 * np.exp(-((t - beat_center + 0.16) ** 2) / (2 * 0.01 ** 2))
        # QRS complex
        signal += -0.1 * np.exp(-((t - beat_center + 0.04) ** 2) / (2 * 0.005 ** 2))
        signal += 1.0 * np.exp(-((t - beat_center) ** 2) / (2 * 0.008 ** 2))
        signal += -0.2 * np.exp(-((t - beat_center - 0.04) ** 2) / (2 * 0.005 ** 2))
        # T wave
        signal += 0.3 * np.exp(-((t - beat_center - 0.22) ** 2) / (2 * 0.025 ** 2))

    # Add noise
    signal += np.random.normal(0, 0.02, len(signal))

    # Save as CSV
    header = "time,MLII"
    data = np.column_stack([t, signal])
    np.savetxt(os.path.join(output_dir, "ecg_sample.csv"), data, delimiter=",",
               header=header, comments="", fmt="%.6f")
    print(f"Generated ecg_sample.csv: {len(t)} samples, {sr} Hz, {duration}s")


def generate_emg_sample():
    """Generate a synthetic EMG-like signal."""
    sr = 1000  # Hz
    duration = 5  # seconds
    t = np.arange(0, duration, 1 / sr)

    # Simulate muscle activation bursts
    signal = np.random.normal(0, 0.05, len(t))

    # Add activation bursts
    for start_sec in [0.5, 1.5, 2.5, 3.5]:
        burst_dur = 0.4
        mask = (t >= start_sec) & (t < start_sec + burst_dur)
        amplitude = np.random.uniform(0.5, 1.0)
        signal[mask] += np.random.normal(0, amplitude, mask.sum())

    # Save as CSV
    header = "time,EMG_Ch1"
    data = np.column_stack([t, signal])
    np.savetxt(os.path.join(output_dir, "emg_sample.csv"), data, delimiter=",",
               header=header, comments="", fmt="%.6f")
    print(f"Generated emg_sample.csv: {len(t)} samples, {sr} Hz, {duration}s")


def generate_eeg_sample():
    """Generate a synthetic multi-channel EEG-like signal."""
    sr = 256  # Hz
    duration = 60  # seconds (2 epochs of 30s)
    t = np.arange(0, duration, 1 / sr)
    n_channels = 4
    channel_names = ["Fp1", "Fp2", "C3", "C4"]

    signals = []
    for ch in range(n_channels):
        # Mix of frequency bands
        delta = 0.5 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2 * np.pi))   # 2 Hz
        theta = 0.3 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2 * np.pi))   # 6 Hz
        alpha = 0.4 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))  # 10 Hz
        beta = 0.15 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2 * np.pi))  # 20 Hz
        noise = np.random.normal(0, 0.1, len(t))
        signals.append(delta + theta + alpha + beta + noise)

    # Save as CSV
    header = "time," + ",".join(channel_names)
    data = np.column_stack([t] + signals)
    np.savetxt(os.path.join(output_dir, "eeg_sample.csv"), data, delimiter=",",
               header=header, comments="", fmt="%.6f")
    print(f"Generated eeg_sample.csv: {len(t)} samples, {sr} Hz, {duration}s, {n_channels} channels")


if __name__ == "__main__":
    generate_ecg_sample()
    generate_emg_sample()
    generate_eeg_sample()
    print("All sample files generated.")
