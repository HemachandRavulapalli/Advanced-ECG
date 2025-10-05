#!/usr/bin/env python3
import os
import json
import numpy as np
import scipy.io

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_KARDIA_DIR = os.path.join(BASE_DIR, "data", "raw", "kardia")
OUT_DIR = os.path.join(BASE_DIR, "data")

# Output files (combined)
X_OUT = os.path.join(OUT_DIR, "X.npy")
Y_OUT = os.path.join(OUT_DIR, "Y.npy")
LABELS_OUT = os.path.join(OUT_DIR, "labels.json")

def extract_ecg_from_mat(mat_path):
    """Extract ECG_1 signal from nested .mat file."""
    try:
        mat = scipy.io.loadmat(mat_path)
        data_struct = mat.get("Data")
        if data_struct is None:
            return None
        nested = data_struct[0, 0]
        if len(nested) < 5:
            return None
        ecg_struct = nested[4][0, 0]  # the ECG struct with ECG_1, ECG_2, etc.
        if "ECG_1" in ecg_struct.dtype.names:
            ecg = ecg_struct["ECG_1"][0, 0].flatten()
        else:
            ecg = None
        return ecg
    except Exception as e:
        print(f"⚠️ Error reading {mat_path}: {e}")
        return None

def scan_kardia_files(base_folder):
    """Recursively scan for .mat files and optional label .txt files."""
    pairs = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.endswith(".mat"):
                mat_path = os.path.join(root, f)
                txt_path = mat_path.replace(".mat", ".txt")
                pairs.append((mat_path, txt_path if os.path.exists(txt_path) else None))
    return pairs

def main():
    print(f"📂 Scanning Kardia folder recursively: {RAW_KARDIA_DIR}")
    pairs = scan_kardia_files(RAW_KARDIA_DIR)
    print(f"🔍 Found {len(pairs)} candidate (.mat, .txt) pairs.")

    signals, labels = [], []
    for mat_path, txt_path in pairs:
        ecg = extract_ecg_from_mat(mat_path)
        if ecg is None or len(ecg) == 0:
            continue
        label = "Unknown"
        if txt_path and os.path.exists(txt_path):
            try:
                with open(txt_path) as f:
                    label = f.read().strip()
            except Exception:
                pass
        signals.append(ecg)
        labels.append(label)

    if len(signals) == 0:
        print("❌ No valid ECG signals extracted. Exiting.")
        return

    # Combine with existing Kardia data if available
    if os.path.exists(X_OUT) and os.path.exists(Y_OUT):
        print("📦 Combining with existing Kardia X.npy/Y.npy...")
        X_existing = np.load(X_OUT, allow_pickle=True)
        Y_existing = np.load(Y_OUT, allow_pickle=True)
        X_combined = np.concatenate([X_existing, np.array(signals, dtype=object)])
        Y_combined = np.concatenate([Y_existing, np.array(labels, dtype=object)])
    else:
        X_combined = np.array(signals, dtype=object)
        Y_combined = np.array(labels, dtype=object)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(X_OUT, X_combined)
    np.save(Y_OUT, Y_combined)
    json.dump(list(set(Y_combined.tolist())), open(LABELS_OUT, "w"), indent=2)

    print(f"✅ Combined dataset saved:")
    print(f"   → {X_OUT}")
    print(f"   → {Y_OUT}")
    print(f"   → {LABELS_OUT}")
    print(f"📊 Total signals: {len(X_combined)}, unique labels: {len(set(Y_combined))}")

if __name__ == "__main__":
    main()
