import os
import wfdb
import pandas as pd

# Go 2 levels up (~/D4)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # D4/
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MITDB_DIR = os.path.join(RAW_DIR, "mitdb")   # ✅ FIXED
ann_file = os.path.join(RAW_DIR, "mitdb_annotations.csv")


def generate_annotations():
    if not os.path.exists(MITDB_DIR):
        raise FileNotFoundError(f"❌ MITDB directory not found at {MITDB_DIR}")

    records = [f.replace(".dat", "") for f in os.listdir(MITDB_DIR) if f.endswith(".dat")]
    annotations = []

    for rec in records:
        try:
            ann = wfdb.rdann(os.path.join(MITDB_DIR, rec), "atr")
            for sym in ann.symbol:
                annotations.append({"record": rec, "label": sym})
        except Exception as e:
            print(f"❌ Error reading {rec}: {e}")

    df = pd.DataFrame(annotations)
    df.to_csv(ann_file, index=False)
    print(f"✅ Saved annotations: {ann_file} ({len(df)} rows)")

if __name__ == "__main__":
    generate_annotations()
