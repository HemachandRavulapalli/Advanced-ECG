import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import datetime

from preprocessing import preprocess_ecg

# ------------------------
# Paths (top-level D4/data)
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # D4/
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Target classes to filter by (must match your data loader's TARGET_CLASSES)
TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]

# ------------------------
# ETA Helper
# ------------------------
def format_eta(elapsed, done, total):
    if done == 0:
        return "estimating..."
    rate = elapsed / done
    remaining = (total - done) * rate
    return str(datetime.timedelta(seconds=int(remaining)))

# ------------------------
# Save function
# ------------------------
def save_preprocessed(signal, imfs, fs, save_path):
    np.savez_compressed(save_path, signal=signal, imfs=imfs, fs=fs)

# ------------------------
# Worker
# ------------------------
def process_record(args):
    record_path, save_path = args
    try:
        # Skip if already processed (resume support)
        if os.path.exists(save_path):
            return "skip"

        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # lead 1
        fs = record.fs

        result = preprocess_ecg(signal, fs=fs)
        save_preprocessed(result["filtered_signal"], result["imfs"], fs, save_path)
        return "done"
    except Exception as e:
        return f"❌ Error in {record_path}: {e}"

# ------------------------
# MIT-BIH
# ------------------------
def process_mitdb():
    mitdb_dir = os.path.join(RAW_DIR, "mitdb")
    save_dir = os.path.join(PROCESSED_DIR, "mitdb")
    os.makedirs(save_dir, exist_ok=True)

    records = [os.path.join(mitdb_dir, r) for r in os.listdir(mitdb_dir) if r.endswith(".dat")]
    tasks = [(r.replace(".dat", ""), os.path.join(save_dir, os.path.basename(r).replace(".dat", ".npz"))) for r in records]

    print(f"📊 Processing MIT-BIH ({len(tasks)} records) using {cpu_count()} cores (resume enabled)...")

    start = time.time()
    with Pool(processes=min(8, cpu_count())) as pool:
        for i, status in enumerate(pool.imap_unordered(process_record, tasks), 1):
            elapsed = time.time() - start
            eta = format_eta(elapsed, i, len(tasks))
            if status == "skip":
                tqdm.write(f"⏩ Skipped ({i}/{len(tasks)}) | ETA {eta}")
            elif status == "done":
                tqdm.write(f"✅ Done ({i}/{len(tasks)}) | ETA {eta}")
            else:
                tqdm.write(f"{status} ({i}/{len(tasks)}) | ETA {eta}")

    elapsed = time.time() - start
    print(f"✅ MIT-BIH completed in {elapsed/60:.2f} min")

# ------------------------
# PTB-XL
# ------------------------
def process_ptbxl(limit=None):
    ptbxl_dir = os.path.join(RAW_DIR, "ptbxl")
    save_dir = os.path.join(PROCESSED_DIR, "ptbxl")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"))

    # Map or filter records by target classes from diagnostic info
    # Adjust based on your actual dataframe column for diagnostic class
    if "diagnostic_class" in df.columns:
        df = df[df['diagnostic_class'].isin(TARGET_CLASSES)]
    else:
        # If diagnostic_class doesn't exist, keep all or add other filtering logic here
        pass

    if limit:
        df = df.head(limit)

    tasks = []
    for _, row in df.iterrows():
        record_path = os.path.join(ptbxl_dir, row["filename_lr"].replace(".npy", ""))
        save_path = os.path.join(save_dir, f"{row['ecg_id']}.npz")
        tasks.append((record_path, save_path))

    print(f"📊 Processing PTB-XL ({len(tasks)} records) using {cpu_count()} cores (resume enabled)...")

    start = time.time()
    with Pool(processes=min(8, cpu_count())) as pool:
        for i, status in enumerate(pool.imap_unordered(process_record, tasks), 1):
            elapsed = time.time() - start
            eta = format_eta(elapsed, i, len(tasks))
            if status == "skip":
                tqdm.write(f"⏩ Skipped ({i}/{len(tasks)}) | ETA {eta}")
            elif status == "done":
                tqdm.write(f"✅ Done ({i}/{len(tasks)}) | ETA {eta}")
            else:
                tqdm.write(f"{status} ({i}/{len(tasks)}) | ETA {eta}")

    elapsed = time.time() - start
    print(f"✅ PTB-XL completed in {elapsed/60:.2f} min")

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("📥 Starting ECG preprocessing (resume enabled)...")
    process_mitdb()
    process_ptbxl(limit=None)  # ⚡ full PTB-XL (remove limit now)
    print("🎉 Preprocessing completed! Data saved in:", PROCESSED_DIR)
