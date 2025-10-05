import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import pywt


# ------------------------
# Empirical Mode Decomposition (EMD)
# ------------------------
try:
    from PyEMD import EMD   # ✅ provided by emd-signal
    print("⚡ Using EMD-signal (fast)")


    def apply_emd(signal):
        emd = EMD()
        imfs = emd(signal)
        return imfs


except ImportError:
    print("❌ PyEMD not found — please install `emd-signal`")
    def apply_emd(signal):
        raise RuntimeError("No EMD backend available")


# ------------------------
# Bandpass filter
# ------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500, order=4):
    nyq = 0.5 * fs
    high = min(highcut, nyq - 0.1)  # prevent highcut hitting Nyquist
    low = max(lowcut, 0.01)


    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)




# ------------------------
# Notch filter (50/60Hz powerline)
# ------------------------
def notch_filter(signal, freq=50.0, fs=500, quality=30):
    b, a = iirnotch(w0=freq / (fs / 2), Q=quality)
    return filtfilt(b, a, signal)


# ------------------------
# Wavelet Denoising
# ------------------------
def wavelet_denoise(signal, wavelet="db4", level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode="soft") for i in coeffs[1:]]  # fixed: use list not generator
    return pywt.waverec(coeffs, wavelet)


# ------------------------
# Normalization
# ------------------------
def normalize(signal):
    signal = np.array(signal)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# ------------------------
# Main Preprocessing Pipeline
# ------------------------
def preprocess_ecg(signal, fs=500):
    # Sanity check for sampling rate
    if fs <= 0:
        raise ValueError("Invalid sampling rate fs <= 0")

    # Step 1: Bandpass
    filtered = bandpass_filter(signal, fs=fs)

    # Step 2: Notch filter
    filtered = notch_filter(filtered, fs=fs)

    # Step 3: Wavelet denoising
    denoised = wavelet_denoise(filtered)

    # Step 4: Normalization
    normalized = normalize(denoised)

    # Step 5: EMD decomposition
    imfs = apply_emd(normalized)

    return {
        "filtered_signal": normalized,
        "imfs": imfs
    }
