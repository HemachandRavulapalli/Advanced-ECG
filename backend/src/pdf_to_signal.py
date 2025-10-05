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
        raise ValueError("❌ No contours detected — cannot extract ECG waveform.")
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
    return signal


def extract_signal_from_file(file_path, target_length=1000):
    """Main entry: extract waveform from PDF or image."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".pdf"]:
        print("📄 Converting PDF → image…")
        img = pdf_to_image(file_path)
    else:
        print("🖼 Loading image directly…")
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("❌ Unable to read image file.")

    print("📈 Extracting waveform…")
    signal = extract_waveform_from_image(img, target_length=target_length)
    print(f"✅ Extracted waveform of length {len(signal)}")
    return signal


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
