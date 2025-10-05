#!/usr/bin/env python3
import os, json
import numpy as np
import fitz  # PyMuPDF
from backend.src.pdf_to_signal import extract_signal_from_file

LABEL_KEYWORDS = {
    "normal": "Normal Sinus Rhythm",
    "atrial fibrillation": "Atrial Fibrillation",
    "bradycardia": "Bradycardia",
    "tachycardia": "Tachycardia",
    "ventricular": "Ventricular Arrhythmias",
}

def extract_label_from_pdf(pdf_path):
    """Read text from PDF and match against known labels."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text").lower()
    for key, label in LABEL_KEYWORDS.items():
        if key in text:
            return label
    return None  # unrecognized

def load_kardia_folder(folder_path, target_len=1000):
    """Return signals + labels from a folder of Kardia PDFs."""
    X, y = [], []
    for f in os.listdir(folder_path):
        if f.lower().endswith(".pdf"):
            path = os.path.join(folder_path, f)
            signal = extract_signal_from_file(path)
            label = extract_label_from_pdf(path)
            if signal is not None and label is not None:
                # resample to fixed length
                signal = np.interp(
                    np.linspace(0, len(signal), target_len),
                    np.arange(len(signal)),
                    signal
                )
                X.append(signal)
                y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing Kardia 6L PDFs")
    parser.add_argument("--out_dir", default="kardia_data", help="Where to save npy outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y = load_kardia_folder(args.data_dir)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(list(np.unique(y)), f, indent=2)

    print(f"✅ Extracted {len(y)} samples to {args.out_dir}")
