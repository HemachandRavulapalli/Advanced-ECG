# preprocess_data.py
import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import datetime
import json

from preprocessing import preprocess_ecg


# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ======================================================
# Target Classes (FINAL – CONSISTENT)
# ======================================================
TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]


# ======================================================
# Helpers
# ======================================================
def format_eta(elapsed, done, total):
    if done == 0:
        return "estimating..."
    rate = elapsed / done
    remaining = (total - done) * rate
    return str(datetime.timedelta(seconds=int(remaining)))


def save_preprocessed(signal, label, fs, dataset, save_path):
    np.savez_compressed(
        save_path,
        signal=signal,
        label=label,
        fs=fs,
        dataset=dataset,
    )


# ======================================================
# MIT-BIH Worker (TOP-LEVEL ✅)
# ======================================================
def process_mit_record(args):
    record_path, save_path = args
    try:
        if os.path.exists(save_path):
            return "skip"

        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]
        fs = record.fs

        result = preprocess_ecg(signal, fs=fs, window_size=1000)

        # Safe default label (MIT-BIH annotations handled later)
        label = "Normal Sinus Rhythm"

        save_preprocessed(
            result["filtered_signal"],
            label,
            fs,
            "MIT-BIH",
            save_path,
        )
        return "done"
    except Exception as e:
        return f"❌ MIT error {record_path}: {e}"


# ======================================================
# PTB-XL Worker (TOP-LEVEL ✅ FIXED)
# ======================================================
def process_ptbxl_worker(args):
    record_path, label, fs, save_path = args
    try:
        if os.path.exists(save_path):
            return "skip"

        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]
        result = preprocess_ecg(signal, fs=fs, window_size=1000)

        save_preprocessed(
            result["filtered_signal"],
            label,
            fs,
            "PTB-XL",
            save_path,
        )
        return "done"
    except Exception as e:
        return f"❌ PTB-XL error {record_path}: {e}"


# ======================================================
# MIT-BIH Runner
# ======================================================
def process_mitdb():
    mit_dir = os.path.join(RAW_DIR, "mitdb")
    save_dir = os.path.join(PROCESSED_DIR, "mitdb")
    os.makedirs(save_dir, exist_ok=True)

    tasks = []
    for f in os.listdir(mit_dir):
        if f.endswith(".dat"):
            rec = f.replace(".dat", "")
            tasks.append((
                os.path.join(mit_dir, rec),
                os.path.join(save_dir, f"{rec}.npz"),
            ))

    print(f"📊 MIT-BIH records: {len(tasks)}")

    start = time.time()
    with Pool(min(cpu_count(), 8)) as pool:
        for i, status in enumerate(
            pool.imap_unordered(process_mit_record, tasks), 1
        ):
            eta = format_eta(time.time() - start, i, len(tasks))
            tqdm.write(f"{status} ({i}/{len(tasks)}) | ETA {eta}")


# ======================================================
# PTB-XL Runner (FULLY FIXED)
# ======================================================
def process_ptbxl(limit=None):
    ptb_dir = os.path.join(RAW_DIR, "ptbxl")
    save_dir = os.path.join(PROCESSED_DIR, "ptbxl")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ptb_dir, "ptbxl_database.csv"))

    def map_label(scp_codes):
        codes = json.loads(scp_codes.replace("'", '"'))
        if "AFIB" in codes:
            return "Atrial Fibrillation"
        if "SBRAD" in codes:
            return "Bradycardia"
        if "STACH" in codes:
            return "Tachycardia"
        if "VEB" in codes or "VT" in codes or "PVC" in codes:
            return "Ventricular Arrhythmias"
        if "NORM" in codes:
            return "Normal Sinus Rhythm"
        return None

    tasks = []
    for _, row in df.iterrows():
        label = map_label(row["scp_codes"])
        if label not in TARGET_CLASSES:
            continue

        # ✅ CRITICAL FIX: use WFDB record path
        record_path = os.path.join(ptb_dir, row["filename_lr"])
        if not os.path.exists(record_path + ".dat"):
            continue

        save_path = os.path.join(save_dir, f"{row['ecg_id']}.npz")
        tasks.append((record_path, label, 100, save_path))

        if limit and len(tasks) >= limit:
            break

    print(f"📊 PTB-XL records: {len(tasks)}")

    start = time.time()
    with Pool(min(cpu_count(), 8)) as pool:
        for i, status in enumerate(
            pool.imap_unordered(process_ptbxl_worker, tasks), 1
        ):
            eta = format_eta(time.time() - start, i, len(tasks))
            tqdm.write(f"{status} ({i}/{len(tasks)}) | ETA {eta}")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    print("📥 Starting ECG preprocessing...")
    process_mitdb()
    process_ptbxl()
    print("🎉 Preprocessing completed successfully!")
