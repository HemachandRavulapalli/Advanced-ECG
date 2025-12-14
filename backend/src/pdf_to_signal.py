#!/usr/bin/env python3
"""
pdf_to_signal.py
Robust ECG waveform extraction from Kardia 6L PDFs and ECG images.
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
import os


# ======================================================
# PDF → Image
# ======================================================
def pdf_to_image(pdf_path, scale=5):
    doc = fitz.open(pdf_path)

    # Try first 2 pages (Kardia ECG may be on page 1)
    for i in range(min(2, len(doc))):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if img.size > 0:
            return img

    raise ValueError("No valid page found in PDF")


# ======================================================
# Grid Removal
# ======================================================
def remove_grid(gray):
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    grid_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_h)
    grid_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_v)

    grid = cv2.add(grid_h, grid_v)
    return cv2.subtract(bw, grid)


# ======================================================
# ECG Waveform Extraction
# ======================================================
def extract_waveform_from_image(img, target_length=1000):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Simple thresholding for black traces
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    signal = np.zeros(w)

    for x in range(w):
        column = thresh[:, x]
        black_ys = np.where(column > 128)[0]  # white in inverted, so black in original
        if len(black_ys) > 0:
            signal[x] = h - np.mean(black_ys)

    idx = np.where(signal != 0)[0]
    print(f"Total points: {len(idx)} out of {w}")
    if len(idx) < 10:
        raise ValueError("No valid ECG traces found")

    signal = np.interp(np.arange(len(signal)), idx, signal[idx])
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    signal = cv2.resize(
        signal.reshape(-1, 1),
        (1, target_length),
        interpolation=cv2.INTER_AREA
    ).flatten()

    return signal


# ======================================================
# Validation
# ======================================================
def validate_ecg_signal(signal):
    if signal is None or len(signal) < 200:
        return False, "Signal too short"
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False, "NaN/Inf detected"
    if np.std(signal) < 0.05:
        return False, "Low signal variance"
    return True, "Valid ECG"


# ======================================================
# Main API
# ======================================================
def extract_signal_from_file(file_path, target_length=1000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        img = pdf_to_image(file_path)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Invalid image file")

    signal = extract_waveform_from_image(img, target_length)

    ok, reason = validate_ecg_signal(signal)
    if not ok:
        raise ValueError(f"Invalid ECG signal: {reason}")

    return signal
