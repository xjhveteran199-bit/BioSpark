"""
Generate realistic EEG and EMG test data for BioSpark platform testing.
"""
import numpy as np
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# EEG Sleep Data — single channel, 100 Hz, ~60 seconds
# Simulates a real EEG with mixed sleep stage characteristics
# ============================================================
def generate_eeg_sleep_test():
    sr = 100  # Hz (matches Sleep-EDF / model expectation)
    duration = 90  # seconds — enough for 3x 30-sec epochs
    t = np.arange(0, duration, 1/sr)
    n = len(t)

    signal = np.zeros(n)

    # Epoch 1 (0-30s): Wake-like — alpha rhythm 8-13 Hz
    seg1 = slice(0, 30*sr)
    signal[seg1] = (
        40 * np.sin(2*np.pi*10*t[seg1]) +       # alpha 10 Hz
        15 * np.sin(2*np.pi*12*t[seg1]) +        # alpha 12 Hz
        8 * np.sin(2*np.pi*20*t[seg1]) +         # beta
        np.random.randn(30*sr) * 10              # noise
    )

    # Epoch 2 (30-60s): N2-like — sleep spindles + K-complex
    seg2 = slice(30*sr, 60*sr)
    signal[seg2] = (
        20 * np.sin(2*np.pi*4*t[seg2]) +         # theta 4 Hz
        10 * np.sin(2*np.pi*14*t[seg2]) *         # spindle 14 Hz
            np.exp(-0.5*((t[seg2]-45)**2)/2) * 30 +  # spindle burst at 45s
        np.random.randn(30*sr) * 8
    )
    # K-complex at 38s
    kc_center = int(38 * sr)
    kc_len = int(0.5 * sr)
    if kc_center + kc_len < 60*sr:
        kc_t = np.arange(kc_len) / sr
        signal[kc_center:kc_center+kc_len] += -80 * np.sin(2*np.pi*1*kc_t)

    # Epoch 3 (60-90s): REM-like — mixed frequency, low amplitude
    seg3 = slice(60*sr, 90*sr)
    signal[seg3] = (
        15 * np.sin(2*np.pi*6*t[seg3]) +         # theta
        10 * np.sin(2*np.pi*2*t[seg3]) +          # delta
        12 * np.sin(2*np.pi*11*t[seg3]) +         # alpha
        np.random.randn(30*sr) * 12               # noise
    )

    # Scale to microvolts range
    signal = signal * 0.5  # ~±50 µV

    # Save as CSV
    filepath = os.path.join(output_dir, "eeg_sleep_test.csv")
    with open(filepath, 'w') as f:
        f.write("time,EEG\n")
        for i in range(n):
            f.write(f"{t[i]:.4f},{signal[i]:.4f}\n")

    print(f"[OK] EEG test data: {filepath}")
    print(f"   {n} samples, {sr} Hz, {duration}s, 1 channel")
    print(f"   Epoch 1 (0-30s): Wake-like alpha rhythm")
    print(f"   Epoch 2 (30-60s): N2-like with spindles")
    print(f"   Epoch 3 (60-90s): REM-like mixed frequency")


# ============================================================
# EMG Gesture Data — 16 channels, 200 Hz, ~10 seconds
# Simulates NinaPro DB5 sEMG from Thalmic Myo armband
# ============================================================
def generate_emg_gesture_test():
    sr = 200  # Hz (matches NinaPro DB5)
    duration = 10  # seconds
    t = np.arange(0, duration, 1/sr)
    n = len(t)
    n_channels = 16

    signals = np.zeros((n_channels, n))

    # Base EMG noise (all channels)
    for ch in range(n_channels):
        signals[ch] = np.random.randn(n) * 5  # baseline noise

    # Gesture 1 (0-2s): Rest — low amplitude baseline
    # Already covered by noise

    # Gesture 2 (2-4s): Index finger flexion — channels 0-3 activate
    seg = slice(2*sr, 4*sr)
    for ch in [0, 1, 2, 3]:
        burst = np.random.randn(2*sr) * 30 + 20 * np.sin(2*np.pi*80*t[seg])
        envelope = np.ones(2*sr)
        # Ramp up/down
        ramp = int(0.2*sr)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        signals[ch, seg] += burst * envelope

    # Gesture 3 (4-6s): Fist — all channels activate
    seg = slice(4*sr, 6*sr)
    for ch in range(n_channels):
        intensity = 25 + 10 * np.random.rand()
        burst = np.random.randn(2*sr) * intensity
        burst += 15 * np.sin(2*np.pi*(60+ch*5)*t[seg])
        envelope = np.ones(2*sr)
        ramp = int(0.3*sr)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        signals[ch, seg] += burst * envelope

    # Gesture 4 (6-8s): Wrist extension — channels 8-13 activate
    seg = slice(6*sr, 8*sr)
    for ch in [8, 9, 10, 11, 12, 13]:
        burst = np.random.randn(2*sr) * 35 + 25 * np.sin(2*np.pi*90*t[seg])
        envelope = np.ones(2*sr)
        ramp = int(0.25*sr)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        signals[ch, seg] += burst * envelope

    # Gesture 5 (8-10s): Thumb flexion — channels 4-6 activate
    seg = slice(8*sr, 10*sr)
    for ch in [4, 5, 6]:
        burst = np.random.randn(2*sr) * 28 + 18 * np.sin(2*np.pi*70*t[seg])
        envelope = np.ones(2*sr)
        ramp = int(0.2*sr)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        signals[ch, seg] += burst * envelope

    # Save as CSV — 16-channel format
    filepath = os.path.join(output_dir, "emg_gesture_test.csv")
    ch_names = [f"sEMG_Ch{i+1}" for i in range(n_channels)]

    with open(filepath, 'w') as f:
        f.write("time," + ",".join(ch_names) + "\n")
        for i in range(n):
            values = ",".join(f"{signals[ch, i]:.4f}" for ch in range(n_channels))
            f.write(f"{t[i]:.4f},{values}\n")

    print(f"\n[OK] EMG test data: {filepath}")
    print(f"   {n} samples × {n_channels} channels, {sr} Hz, {duration}s")
    print(f"   0-2s: Rest (baseline noise)")
    print(f"   2-4s: Index flexion (Ch1-4 active)")
    print(f"   4-6s: Fist clench (all channels)")
    print(f"   6-8s: Wrist extension (Ch9-14 active)")
    print(f"   8-10s: Thumb flexion (Ch5-7 active)")


if __name__ == "__main__":
    generate_eeg_sleep_test()
    generate_emg_gesture_test()
    print("\n[DONE] Test data generation complete!")
