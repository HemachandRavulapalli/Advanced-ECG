# backend/src/feature_extraction.py

import numpy as np
from scipy.signal import find_peaks

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False


def extract_ecg_features(signal, fs=250):
    """
    Extract clinically meaningful ECG features
    for classical ML models.
    Returns a 1D numpy array.
    """

    features = []

    signal = np.asarray(signal).flatten()

    # -------------------------
    # Basic statistical features
    # -------------------------
    features.extend([
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.var(signal),
    ])

    # -------------------------
    # Peak-based features (heart rate)
    # -------------------------
    peaks, _ = find_peaks(signal, distance=fs * 0.3)
    rr_intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([0])

    features.extend([
        len(peaks),                      # number of beats
        np.mean(rr_intervals),           # mean RR
        np.std(rr_intervals),            # HRV
        np.min(rr_intervals),
        np.max(rr_intervals),
    ])

    # -------------------------
    # Frequency-domain features
    # -------------------------
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)

    def band_energy(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(fft[mask] ** 2)

    features.extend([
        band_energy(0.5, 4),    # baseline / P-wave
        band_energy(4, 15),     # QRS complex
        band_energy(15, 40),    # noise / muscle
    ])

    # -------------------------
    # NeuroKit features (if available)
    # -------------------------
    if NEUROKIT_AVAILABLE:
        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            hrv = nk.hrv_time(info, sampling_rate=fs)

            features.extend([
                hrv.get("HRV_SDNN", [0])[0],
                hrv.get("HRV_RMSSD", [0])[0],
            ])
        except Exception:
            features.extend([0, 0])
    else:
        features.extend([0, 0])

    return np.array(features, dtype=np.float32)
