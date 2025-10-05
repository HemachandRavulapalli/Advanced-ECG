# backend/src/data_loader.py
import os
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# -----------------------------
# Base paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
KARDIA_DIR = os.path.join(BASE_DIR, "data", "kardia")

# -----------------------------
# Unified target classes
# -----------------------------
TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]

# -----------------------------
# Utility
# -----------------------------
def segment_signal(signal, window_size=1000, step=1000):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

# -----------------------------
# Label map
# -----------------------------
LABEL_MAP = {
    # MIT-BIH
    "N": "Normal Sinus Rhythm",
    "L": "Normal Sinus Rhythm",
    "R": "Normal Sinus Rhythm",
    "A": "Atrial Fibrillation",
    "a": "Atrial Fibrillation",
    "S": "Tachycardia",
    "F": "Ventricular Arrhythmias",
    "V": "Ventricular Arrhythmias",
    "B": "Bradycardia",

    # PTB-XL
    "NORM": "Normal Sinus Rhythm",
    "SR": "Normal Sinus Rhythm",
    "SBRAD": "Bradycardia",
    "STACH": "Tachycardia",
    "AFIB": "Atrial Fibrillation",
    "VEB": "Ventricular Arrhythmias",
    "VT": "Ventricular Arrhythmias",
}

def map_label_to_target(lbl):
    if lbl is None:
        return None
    s = str(lbl).strip()
    if s in TARGET_CLASSES:
        return s
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    su = s.upper()
    if su in LABEL_MAP:
        return LABEL_MAP[su]
    sl = s.lower()
    for k, v in LABEL_MAP.items():
        if k.lower() == sl:
            return v
    if "atrial" in s.lower() and "fibril" in s.lower():
        return "Atrial Fibrillation"
    if "brady" in s.lower():
        return "Bradycardia"
    if "tachy" in s.lower():
        return "Tachycardia"
    if "ventr" in s.lower() or "vt" in s.lower():
        return "Ventricular Arrhythmias"
    if "sinus" in s.lower() and "normal" in s.lower():
        return "Normal Sinus Rhythm"
    return None

# -----------------------------
# MIT-BIH loader
# -----------------------------
def load_mitdb(limit=None, window_size=1000):
    mit_dir = os.path.join(RAW_DIR, "mitdb")
    ann_file = os.path.join(RAW_DIR, "mitdb_annotations.csv")
    if not os.path.exists(ann_file):
        print("⚠️ MIT-BIH annotations not found, skipping.")
        return np.empty((0, window_size)), np.array([])

    df = pd.read_csv(ann_file)
    signals, labels = [], []
    for _, row in df.iterrows():
        record_path = os.path.join(mit_dir, str(row["record"]))
        try:
            sig, _ = wfdb.rdsamp(record_path)
        except:
            continue
        sig = sig[:, 0] if sig.ndim > 1 else sig
        segs = segment_signal(sig, window_size)
        mapped = map_label_to_target(row.get("label") or row.get("annotation"))
        if mapped:
            signals.extend(segs)
            labels.extend([mapped] * len(segs))
        if limit and len(signals) >= limit:
            break
    return np.array(signals), np.array(labels)

# -----------------------------
# MIT-BIH Ventricular Arrhythmias loader (example, adjust paths/annotations)
# -----------------------------
def load_mitdb_veb(limit=None, window_size=1000):
    veb_dir = os.path.join(RAW_DIR, "mitdb_veb")  # path to Ventricular Arrhythmias dataset
    ann_file = os.path.join(veb_dir, "annotations.csv")  # example annotation file

    if not os.path.exists(ann_file):
        print("⚠️ MIT-BIH VEB annotations not found, skipping.")
        return np.empty((0, window_size)), np.array([])

    df = pd.read_csv(ann_file)
    signals, labels = [], []
    for _, row in df.iterrows():
        record_path = os.path.join(veb_dir, str(row["record"]))
        try:
            sig, _ = wfdb.rdsamp(record_path)
        except:
            continue
        sig = sig[:, 0] if sig.ndim > 1 else sig
        segs = segment_signal(sig, window_size)
        mapped = "Ventricular Arrhythmias"  # All samples mapped to this class
        signals.extend(segs)
        labels.extend([mapped] * len(segs))
        if limit and len(signals) >= limit:
            break
    return np.array(signals), np.array(labels)

# -----------------------------
# PTB-XL loader (patient-safe)
# -----------------------------
def load_ptbxl(limit=None, window_size=1000):
    ann_file = os.path.join(RAW_DIR, "ptbxl", "ptbxl_database.csv")
    scp_file = os.path.join(RAW_DIR, "ptbxl", "scp_statements.csv")
    if not os.path.exists(ann_file):
        print("⚠️ PTB-XL annotations not found, skipping.")
        return (np.empty((0, window_size)), np.array([])), (np.empty((0, window_size)), np.array([]))

    df = pd.read_csv(ann_file)
    scp_df = pd.read_csv(scp_file, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def map_superclass(scp_codes):
        try:
            codes = eval(scp_codes) if isinstance(scp_codes, str) else scp_codes
        except Exception:
            return None
        found = []
        for code in codes.keys():
            if code in LABEL_MAP:
                found.append(LABEL_MAP[code])
        if not found:
            return None
        priority = [
            "Atrial Fibrillation",
            "Ventricular Arrhythmias",
            "Tachycardia",
            "Bradycardia",
            "Normal Sinus Rhythm",
        ]
        for p in priority:
            if p in found:
                return p
        return found[0]

    df["mapped"] = df["scp_codes"].apply(map_superclass)
    df = df.dropna(subset=["mapped"])

    train_df = df[df.strat_fold < 9]
    test_df = df[df.strat_fold == 10]

    def load_rows(sub_df):
        X, y = [], []
        for _, row in sub_df.iterrows():
            npz_path = os.path.join(PROCESSED_DIR, "ptbxl", f"{row['ecg_id']}.npz")
            if not os.path.exists(npz_path):
                continue
            try:
                data = np.load(npz_path)
                sig = data["signal"][:window_size]
                if sig.shape[0] < window_size:
                    sig = np.pad(sig, (0, window_size - len(sig)))
                X.append(sig)
                y.append(row["mapped"])
                if limit and len(X) >= limit:
                    break
            except Exception:
                continue
        return np.array(X), np.array(y)

    return load_rows(train_df), load_rows(test_df)

# -----------------------------
# Kardia loader
# -----------------------------
def load_kardia(folder_path, window_size=1000):
    X_path, y_path = os.path.join(folder_path, "X.npy"), os.path.join(folder_path, "Y.npy")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print("⚠️ Kardia data not found, skipping.")
        return np.empty((0, window_size)), np.array([])

    X = np.load(X_path, allow_pickle=True)
    y_raw = np.load(y_path, allow_pickle=True)

    X_fixed, y_fixed = [], []
    for sig, lbl in zip(X, y_raw):
        mapped = map_label_to_target(lbl)
        if not mapped:
            continue
        if len(sig) != window_size:
            sig = np.interp(np.linspace(0, len(sig), window_size), np.arange(len(sig)), sig)
        X_fixed.append(sig)
        y_fixed.append(mapped)
    return np.array(X_fixed), np.array(y_fixed)

# -----------------------------
# Duplicate check utility
# -----------------------------
def check_duplicates(X, name="Dataset"):
    unique_rows = np.unique(X, axis=0)
    if len(unique_rows) < len(X):
        print(f"⚠️ Warning: {name} contains {len(X) - len(unique_rows)} duplicate samples.")
    else:
        print(f"{name} contains no duplicates.")

# -----------------------------
# Combined loader with VEB included and no leakage
# -----------------------------
def load_all_datasets(limit=None, one_hot=True, window_size=1000):
    print("📥 Loading PTB-XL...")
    (X_ptb_train, y_ptb_train), (X_ptb_test, y_ptb_test) = load_ptbxl(limit, window_size)
    print(f"✅ PTB-XL: Train={X_ptb_train.shape}, Test={X_ptb_test.shape}")

    print("📥 Loading MIT-BIH...")
    X_mit, y_mit = load_mitdb(limit, window_size)
    print(f"✅ MIT-BIH: {X_mit.shape}")

    print("📥 Loading MIT-BIH VEB...")
    X_veb, y_veb = load_mitdb_veb(limit, window_size)
    print(f"✅ MIT-BIH VEB: {X_veb.shape}")

    print("📥 Loading Kardia...")
    X_kardia, y_kardia = load_kardia(KARDIA_DIR, window_size)
    print(f"✅ Kardia: {X_kardia.shape}")

    # Training set combines PTB-XL train + MIT-BIH + MIT-BIH VEB + Kardia
    X_train = np.concatenate([X_ptb_train, X_mit, X_veb, X_kardia])
    y_train = np.concatenate([y_ptb_train, y_mit, y_veb, y_kardia])

    # Test set is PTB-XL test fold
    X_test = X_ptb_test
    y_test = y_ptb_test

    # Clean labels to target classes only
    y_train = np.array([lbl for lbl in y_train if lbl in TARGET_CLASSES])
    y_test = np.array([lbl for lbl in y_test if lbl in TARGET_CLASSES])
    X_train = X_train[:len(y_train)]
    X_test = X_test[:len(y_test)]

    # Encode labels
    le = LabelEncoder()
    le.fit(TARGET_CLASSES)
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)

    if one_hot:
        ohe = OneHotEncoder(sparse_output=False)
        y_train_enc = ohe.fit_transform(y_train_int.reshape(-1, 1))
        y_test_enc = ohe.transform(y_test_int.reshape(-1, 1))
    else:
        y_train_enc, y_test_enc = y_train_int, y_test_int

    print(f"📊 Combined dataset: Train={X_train.shape}, Test={X_test.shape}, Classes={list(le.classes_)}")

    # Duplication check example
    check_duplicates(X_train, "Training set")
    check_duplicates(X_test, "Test set")

    return (X_train, y_train_enc), (X_test, y_test_enc), list(le.classes_)
