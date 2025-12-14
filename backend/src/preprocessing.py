# preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
import pywt


# =====================================================
# Bandpass Filter (ECG-safe)
# =====================================================
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=500, order=4):
    nyq = 0.5 * fs
    highcut = min(highcut, 0.45 * fs)  # prevent instability
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# =====================================================
# Notch Filter (Powerline)
# =====================================================
def notch_filter(signal, freq=50, fs=500, Q=30):
    w0 = freq / (fs / 2)
    if w0 >= 1:
        return signal  # skip if invalid
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)


# =====================================================
# Baseline Wander Removal (cheap & effective)
# =====================================================
def remove_baseline(signal, kernel_size=201):
    if kernel_size >= len(signal):
        return signal
    baseline = medfilt(signal, kernel_size)
    return signal - baseline


# =====================================================
# Wavelet Denoising (ECG-standard)
# =====================================================
def wavelet_denoise(signal, wavelet="db4", level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)

    # Ensure length consistency
    if len(rec) > len(signal):
        rec = rec[:len(signal)]
    elif len(rec) < len(signal):
        rec = np.pad(rec, (0, len(signal) - len(rec)))

    return rec


# =====================================================
# Normalization
# =====================================================
def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# =====================================================
# MAIN PREPROCESSING PIPELINE
# =====================================================
def preprocess_ecg(signal, fs=500, window_size=1000):
    signal = np.asarray(signal).astype(np.float32).flatten()

    # Enforce fixed length
    if len(signal) < window_size:
        signal = np.pad(signal, (0, window_size - len(signal)))
    else:
        signal = signal[:window_size]

    # Filtering pipeline
    signal = bandpass_filter(signal, fs=fs)
    signal = notch_filter(signal, fs=fs)
    signal = remove_baseline(signal)
    signal = wavelet_denoise(signal)

    # Normalize
    signal = normalize(signal)

    # Shapes for DL
    signal_1d = signal                      # (1000,)
    signal_2d = signal.reshape(100, 10)     # (100, 10)

    return {
        "filtered_signal": signal_1d,
        "signal_2d": signal_2d
    }
