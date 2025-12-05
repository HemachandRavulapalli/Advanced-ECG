#!/usr/bin/env python3
"""
pdf_to_signal.py — Extract ECG signal from a PDF or image.
Works for Kardia6L PDFs and regular ECG images.
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
import tempfile
import os

# ---------- Helpers ----------

def pdf_to_image(pdf_path):
    """Convert first page of a PDF into an image (RGB numpy array)."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x scaling
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def validate_ecg_signal(signal):
    """
    Validate if extracted signal is a valid ECG signal.
    Returns (is_valid, reason)
    """
    if signal is None or len(signal) == 0:
        return False, "Empty signal"
    
    # Check signal length
    if len(signal) < 100:
        return False, f"Signal too short ({len(signal)} samples, expected >= 100)"
    
    # Check for variation (ECG should have variation)
    if np.std(signal) < 0.01:
        return False, "Signal has no variation (constant or near-constant)"
    
    # Check for NaN or Inf values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False, "Signal contains NaN or Inf values"
    
    # Check signal range (after normalization, should be reasonable)
    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 0.1:
        return False, f"Signal range too small ({signal_range:.4f})"
    
    # Check for ECG-like characteristics (some periodicity or variation)
    # ECG signals typically have some autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:len(autocorr)//2]
    if len(autocorr) > 0 and np.max(autocorr[1:]) / autocorr[0] < 0.1:
        # Very low autocorrelation suggests noise or non-ECG signal
        return False, "Signal lacks ECG-like characteristics (low autocorrelation)"
    
    return True, "Valid ECG signal"


def extract_waveform_from_image(img, target_length=1000):
    """
    Convert an ECG image into a 1-D waveform array.
    Steps:
      - Convert to grayscale
      - Denoise & edge-detect
      - Find the ECG trace (largest contour)
      - Map y-coordinates along x-axis to waveform values
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 30, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("❌ No contours detected — cannot extract ECG waveform. The image may not contain an ECG trace.")
    
    # Try to find ECG-like contour (should have horizontal spread)
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / max(h, 1)  # ECG traces are typically wide
        if aspect_ratio > 2:  # Prefer wide contours
            valid_contours.append((c, w * h))
    
    if valid_contours:
        contour = max(valid_contours, key=lambda x: x[1])[0]
    else:
        contour = max(contours, key=lambda c: cv2.boundingRect(c)[2])

    h, w = gray.shape
    x_vals = contour[:, 0, 0]
    y_vals = contour[:, 0, 1]
    signal = np.zeros(w)
    for x, y in zip(x_vals, y_vals):
        if 0 <= x < w:
            signal[x] = h - y  # flip vertically (higher = stronger)

    # Normalize & resize to fixed length
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    signal = cv2.resize(signal.reshape(-1, 1), (1, target_length), interpolation=cv2.INTER_AREA).flatten()
    
    # Validate the extracted signal
    is_valid, reason = validate_ecg_signal(signal)
    if not is_valid:
        raise ValueError(f"❌ Invalid ECG signal extracted: {reason}. The file may not be a valid ECG image/PDF.")
    
    return signal


def extract_signal_from_file(file_path, target_length=1000):
    """
    Main entry: extract waveform from PDF or image.
    Validates file type and extracted signal.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"❌ File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[-1].lower()
    supported_formats = [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    
    if ext not in supported_formats:
        raise ValueError(f"❌ Unsupported file format: {ext}. Supported formats: {', '.join(supported_formats)}")
    
    try:
        if ext in [".pdf"]:
            print("📄 Converting PDF → image…")
            img = pdf_to_image(file_path)
        else:
            print("🖼 Loading image directly…")
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("❌ Unable to read image file. File may be corrupted or in unsupported format.")
        
        if img is None or img.size == 0:
            raise ValueError("❌ Image is empty or corrupted")
        
        print("📈 Extracting waveform…")
        signal = extract_waveform_from_image(img, target_length=target_length)
        print(f"✅ Extracted waveform of length {len(signal)}")
        
        # Final validation
        is_valid, reason = validate_ecg_signal(signal)
        if not is_valid:
            raise ValueError(f"❌ Invalid ECG signal: {reason}. Please ensure the file contains a valid ECG trace.")
        
        return signal
        
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        raise ValueError(f"❌ Error processing file: {str(e)}. Please ensure the file is a valid ECG PDF or image.")


# ---------- CLI test ----------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Convert ECG PDF/Image → waveform")
    parser.add_argument("file", help="Path to ECG PDF or image")
    parser.add_argument("--show", action="store_true", help="Show the extracted waveform")
    args = parser.parse_args()

    signal = extract_signal_from_file(args.file)
    if args.show:
        plt.plot(signal)
        plt.title("Extracted ECG Waveform")
        plt.show()
