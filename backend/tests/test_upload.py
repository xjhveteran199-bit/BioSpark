"""Tests for file upload and format parsing."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from backend.services.format_parser import parse_file


def _create_csv(tmpdir, data, header=None):
    """Helper to create a temp CSV file."""
    path = os.path.join(tmpdir, "test.csv")
    if header:
        np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    else:
        np.savetxt(path, data, delimiter=",", fmt="%.6f")
    return path


class TestCSVParsing:
    def test_single_channel_csv(self, tmp_path):
        """Test parsing single-column CSV."""
        signal = np.random.randn(1000)
        path = tmp_path / "test.csv"
        np.savetxt(str(path), signal.reshape(-1, 1), delimiter=",", header="MLII", comments="", fmt="%.6f")

        result = parse_file(str(path), "ecg")
        assert result["data"].shape[0] == 1  # 1 channel
        assert result["data"].shape[1] == 1000
        assert result["signal_type"] == "ecg"
        assert result["format"] == "csv"

    def test_multi_channel_csv(self, tmp_path):
        """Test parsing multi-column CSV."""
        data = np.random.randn(500, 4)
        path = tmp_path / "test.csv"
        np.savetxt(str(path), data, delimiter=",", header="Fp1,Fp2,C3,C4", comments="", fmt="%.6f")

        result = parse_file(str(path), "eeg")
        assert result["data"].shape[0] == 4  # 4 channels
        assert result["data"].shape[1] == 500
        assert result["signal_type"] == "eeg"

    def test_csv_with_time_column(self, tmp_path):
        """Test that time column is excluded from signal data."""
        t = np.arange(0, 1, 0.001)
        signal = np.sin(2 * np.pi * 10 * t)
        data = np.column_stack([t, signal])
        path = tmp_path / "test.csv"
        np.savetxt(str(path), data, delimiter=",", header="time,ECG", comments="", fmt="%.6f")

        result = parse_file(str(path), "ecg")
        assert result["data"].shape[0] == 1  # time excluded, only 1 signal channel

    def test_auto_detect_ecg(self, tmp_path):
        """Test auto-detection of ECG from channel names."""
        data = np.random.randn(1000, 2)
        path = tmp_path / "test.csv"
        np.savetxt(str(path), data, delimiter=",", header="MLII,V1", comments="", fmt="%.6f")

        result = parse_file(str(path))
        assert result["signal_type"] == "ecg"

    def test_auto_detect_eeg(self, tmp_path):
        """Test auto-detection of EEG from channel names."""
        data = np.random.randn(500, 4)
        path = tmp_path / "test.csv"
        np.savetxt(str(path), data, delimiter=",", header="Fp1,Fp2,C3,C4", comments="", fmt="%.6f")

        result = parse_file(str(path))
        assert result["signal_type"] == "eeg"


class TestPreprocessing:
    def test_ecg_preprocess(self):
        from backend.services.preprocess import preprocess

        data = np.random.randn(1, 3600)  # 10 seconds at 360 Hz
        result = preprocess(data, "ecg", 360.0)
        assert len(result["segments"]) > 0
        assert result["info"]["effective_sr"] == 360.0

    def test_eeg_preprocess(self):
        from backend.services.preprocess import preprocess

        data = np.random.randn(1, 15360)  # 60 seconds at 256 Hz
        result = preprocess(data, "eeg", 256.0, target_sr=100.0)
        assert len(result["segments"]) > 0
        assert result["info"]["effective_sr"] == 100.0

    def test_emg_preprocess(self):
        from backend.services.preprocess import preprocess

        data = np.random.randn(1, 5000)  # 5 seconds at 1000 Hz
        result = preprocess(data, "emg", 1000.0)
        assert len(result["segments"]) > 0


class TestPredictor:
    def test_demo_prediction(self):
        from backend.services.predictor import predictor

        segments = [np.random.randn(187) for _ in range(5)]
        result = predictor.predict("ecg_arrhythmia", segments)

        assert result["demo_mode"] is True
        assert len(result["predictions"]) == 5
        assert "summary" in result
        assert result["summary"]["total_segments"] == 5

    def test_prediction_classes(self):
        from backend.services.predictor import predictor

        segments = [np.random.randn(3000)]
        result = predictor.predict("eeg_sleep", segments)

        pred = result["predictions"][0]
        assert "class" in pred
        assert "confidence" in pred
        assert 0 <= pred["confidence"] <= 1
        assert "probabilities" in pred
