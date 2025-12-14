# backend/src/predict_ecg.py
import os
import sys
import json
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from scipy.signal import find_peaks

from pdf_to_signal import extract_signal_from_file
from hybrid_model import HybridEnsemble
from feature_extraction import extract_ecg_features

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "saved_models"))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
RESULTS_FILE = os.path.join(LOG_DIR, "results_history.csv")

# ------------------------
# Run selection
# ------------------------
def get_latest_run():
    if not os.path.exists(MODEL_DIR):
        return None
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def get_best_run(by="advanced_hybrid_acc"):
    if not os.path.exists(RESULTS_FILE):
        return get_latest_run()

    try:
        df = pd.read_csv(RESULTS_FILE)
        if by not in df.columns or df.empty:
            return get_latest_run()
        best_row = df.loc[df[by].idxmax()]
        return os.path.join(MODEL_DIR, os.path.basename(best_row["run_folder"]))
    except Exception:
        return get_latest_run()


# ------------------------
# Load models
# ------------------------
def load_models(run_dir):
    import tensorflow as tf

    ml_models, dl_models = {}, {}

    for f in os.listdir(run_dir):
        path = os.path.join(run_dir, f)

        if f.endswith(".joblib"):
            ml_models[f.replace(".joblib", "")] = joblib.load(path)

        elif f.endswith(".keras"):
            name = Path(f).stem.replace("_best", "")
            dl_models[name] = tf.keras.models.load_model(path, safe_mode=False)

    return ml_models, dl_models


# ------------------------
# Prediction
# ------------------------
def predict_ecg(file_path):
    # Optional: Check for ECG-related keywords in PDF or image
    def contains_ecg_keywords(file_path):
        import re
        ecg_keywords = [
            r"ecg", r"ekg", r"kardia", r"lead", r"i", r"ii", r"iii", r"avl", r"avf", r"avr", r"v1", r"v2", r"v3", r"v4", r"v5", r"v6"
        ]
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".pdf":
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = " ".join([page.get_text() for page in doc])
                # If no text found, try OCR on all pages
                if not text.strip():
                    import pytesseract
                    import numpy as np
                    import cv2
                    ocr_texts = []
                    for i in range(len(doc)):
                        page = doc.load_page(i)
                        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        if pix.n == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        ocr_texts.append(pytesseract.image_to_string(img))
                    text = " ".join(ocr_texts)
            else:
                import pytesseract
                import cv2
                img = cv2.imread(file_path)
                if img is None:
                    return False
                # Advanced preprocessing for OCR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize to improve OCR on small text
                scale_factor = 2
                gray = cv2.resize(gray, (img.shape[1]*scale_factor, img.shape[0]*scale_factor), interpolation=cv2.INTER_CUBIC)
                # Denoise
                gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
                # Morphological opening to remove small noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                # Adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
                )
                # Tesseract config: --psm 6 (assume a block of text), --oem 3 (default engine), whitelist for I, II, III
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=Iiv '
                text = pytesseract.image_to_string(thresh, config=custom_config)
        except Exception:
            return False
        print("[ECG Keyword Debug] Extracted text:\n", text)
        text_norm = ' '.join(text.lower().split())  # normalize whitespace and case
        if ext == ".pdf":
            # Accept if all of aVR, aVL, aVF are present together, or 'kardia' or 'ecg' is present
            has_leads = all(lead in text_norm for lead in ["avr", "avl", "avf"])
            has_kardia = "kardia" in text_norm
            has_ecg = "ecg" in text_norm
            print(f"[ECG Keyword Debug] aVR/aVL/aVF: {has_leads}, Kardia: {has_kardia}, ECG: {has_ecg}")
            if has_leads or has_kardia or has_ecg:
                return True
            return False
        else:
            # For images, require at least two of I, II, III
            required_leads = ["i", "ii", "iii"]
            found = 0
            missing = []
            for lead in required_leads:
                present = lead in text_norm
                print(f"[ECG Keyword Debug] Image required lead '{lead}': {present}")
                if present:
                    found += 1
                else:
                    missing.append(lead)
            if found >= 2:
                return True
            print(f"[ECG Keyword Debug] Image missing required leads: {missing}")
            return False

    if not contains_ecg_keywords(file_path):
        raise ValueError("File does not contain ECG-related keywords. Only ECG images or PDFs are supported.")

    import mimetypes
    print(f"📄 Processing: {os.path.basename(file_path)}")

    # File type validation
    mime, _ = mimetypes.guess_type(file_path)
    allowed_image_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp"]
    allowed_pdf_type = "application/pdf"
    if mime is None:
        raise ValueError("Could not determine file type. Only image files (PNG, JPG, BMP) and PDFs are supported.")
    if not (mime in allowed_image_types or mime == allowed_pdf_type):
        raise ValueError(f"Unsupported file type: {mime}. Only image files (PNG, JPG, BMP) and PDFs are supported.")

    # 1️⃣ Extract ECG signal
    print("📊 Extracting ECG signal from file...")

    try:
        signal = extract_signal_from_file(file_path)
    except Exception as e:
        raise ValueError(f"Failed to extract ECG signal: {e}")
    if signal is None:
        raise ValueError("Failed to extract ECG signal")
    print(f"✅ Signal extracted: {len(signal)} samples")

    # Stricter signal validation
    if len(signal) < 500:
        raise ValueError(f"Signal too short to be ECG: {len(signal)} samples")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or Inf values")
    if np.std(signal) < 0.05:
        raise ValueError("Signal variance too low to be ECG")

    # Peak-based validation (stricter)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=np.std(signal) * 0.1, distance=40, prominence=0.1)
    duration_sec = 10  # 1000 samples at 100 Hz
    bpm = len(peaks) * 60 / duration_sec
    if len(peaks) < 5 or bpm < 30 or bpm > 220:
        raise ValueError(f"Invalid ECG: {len(peaks)} peaks, {bpm:.1f} BPM (not physiological)")
    print(f"✅ Stricter signal check passed: {len(peaks)} peaks, {bpm:.1f} BPM")

    # 2️⃣ Peak-based validation
    print("🔍 Analyzing signal for R-peaks...")
    peaks, _ = find_peaks(
        signal,
        height=np.std(signal) * 0.1,
        distance=40
    )

    duration_sec = 10  # 1000 samples at 100 Hz
    bpm = len(peaks) * 60 / duration_sec

    if len(peaks) < 3 or bpm < 30 or bpm > 220:
        raise ValueError(f"Invalid ECG: {len(peaks)} peaks, {bpm:.1f} BPM")

    print(f"✅ Signal OK: {len(peaks)} peaks, {bpm:.1f} BPM")

    # 3️⃣ Normalize & reshape
    print("🔧 Normalizing and reshaping signal...")
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    X_ml = extract_ecg_features(signal, fs=100).reshape(1, -1)
    X_dl = signal.reshape(1, 1000, 1)

    # 4️⃣ Load models
    print("🤖 Loading trained models...")
    run_dir = get_best_run()
    if run_dir is None:
        raise ValueError("No trained models found")
    print(f"📁 Using model run: {os.path.basename(run_dir)}")

    ml_models, dl_models = load_models(run_dir)
    if not ml_models and not dl_models:
        raise ValueError("No models loaded")
    print(f"✅ Loaded {len(ml_models)} ML models and {len(dl_models)} DL models")

    classes = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "Bradycardia",
        "Tachycardia",
        "Ventricular Arrhythmias",
    ]

    # 5️⃣ Hybrid ensemble
    print("🧠 Running hybrid ensemble classification...")
    ensemble = HybridEnsemble(
        ml_models=ml_models,
        dl_models=dl_models,
        classes=classes,
        weights={}
    )

    probs = ensemble.predict_proba(X_ml, X_dl)
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-8)

    idx = int(np.argmax(probs))
    predicted_class = classes[idx]
    confidence = round(float(np.max(probs)), 4)
    
    print(f"🎯 Prediction complete: {predicted_class} (confidence: {confidence})")
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            classes[i]: round(float(p), 4) for i, p in enumerate(probs[0])
        }
    }


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ecg.py <ecg_pdf_or_image>")
        sys.exit(1)

    try:
        result = predict_ecg(sys.argv[1])
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
