#!/usr/bin/env python3
"""
train_pipeline.py — ECG ML + DL Hybrid Training Pipeline (FIXED)

✔ Fixed class-name mismatch
✔ Stable ML + DL + Hybrid training
✔ Safe for CPU-only Azure VM
✔ Resume + logging supported
✔ College + demo ready
"""

import os
# Disable GPU to avoid CUDA errors on CPU-only systems
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress TensorFlow warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import json

# Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from datetime import datetime
import argparse

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from data_loader import load_all_datasets
from ml_models import prepare_features, get_ml_models, train_ml_model
from cnn_models import build_cnn_1d, build_cnn_2d
from hybrid_model import AdvancedHybridModel, HybridEnsemble


# ======================================================
# CLI
# ======================================================
parser = argparse.ArgumentParser("Train ECG Hybrid Models")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--limit", type=int, default=3000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--svm_limit", type=int, default=2000)
parser.add_argument("--keep_runs", type=int, default=5)
parser.add_argument("--normalize", action="store_true")
args = parser.parse_args()


# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "train_log.txt")
results_file = os.path.join(LOG_DIR, "results_history.csv")


# ======================================================
# Logger
# ======================================================
class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self): pass

sys.stdout = Logger()


# ======================================================
# Run Management
# ======================================================
def get_latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def cleanup_old_runs(keep_last):
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    for old in runs[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


if args.resume:
    RUN_DIR = get_latest_run()
    if RUN_DIR:
        print(f"🔁 Resuming from {RUN_DIR}")
    else:
        RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(RUN_DIR, exist_ok=True)
else:
    RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(RUN_DIR, exist_ok=True)
    cleanup_old_runs(args.keep_runs)


# ======================================================
# Load Data
# ======================================================
print("📥 Loading datasets...")
(X_train, y_train), (X_test, y_test), classes = load_all_datasets(
    limit=args.limit, one_hot=True, window_size=1000
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Classes:", classes)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Apply SMOTE to balance classes
if SMOTE_AVAILABLE:
    print("🔄 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten for SMOTE
    X_train_flat, y_train_int = smote.fit_resample(X_train_flat, y_train_int)
    X_train = X_train_flat.reshape(-1, 1000)  # Reshape back
    y_train = np.eye(len(classes))[y_train_int]  # Update one-hot labels
    print(f"After SMOTE - Train: {X_train.shape}, classes: {np.unique(y_train_int, return_counts=True)}")
else:
    print("⚠️ SMOTE not available, skipping oversampling")


# ======================================================
# Train / Validation split (DL only)
# ======================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train_int,
    random_state=42
)


# ======================================================
# Normalization
# ======================================================
if args.normalize:
    mean, std = np.mean(X_train), np.std(X_train) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_tr = (X_tr - mean) / std
    X_val = (X_val - mean) / std


# ======================================================
# Prepare inputs
# ======================================================
X_train_ml = prepare_features(X_train)
X_test_ml = prepare_features(X_test)

X_train_dl = X_train[..., np.newaxis]
X_test_dl = X_test[..., np.newaxis]
X_tr_dl = X_tr[..., np.newaxis]
X_val_dl = X_val[..., np.newaxis]

assert X_train_dl.shape[1] == 1000, "Signal length mismatch"


# ======================================================
# Class weights
# ======================================================
weights = compute_class_weight("balanced", classes=np.unique(y_train_int), y=y_train_int)
class_weights = {int(i): float(w) for i, w in zip(np.unique(y_train_int), weights)}
print("⚖️ Class weights:", class_weights)


# ======================================================
# Train ML Models
# ======================================================
ml_models = {}
ml_scores = {}

for name, model in get_ml_models(len(classes)).items():
    path = os.path.join(RUN_DIR, f"{name}.joblib")

    if args.resume and os.path.exists(path):
        ml_models[name] = joblib.load(path)
        continue

    if name == "SVM":
        idx = np.random.choice(len(X_train_ml), min(args.svm_limit, len(X_train_ml)), replace=False)
        model, acc = train_ml_model(name, model,
                                    X_train_ml[idx], y_train_int[idx],
                                    X_test_ml, y_test_int, classes)
    else:
        model, acc = train_ml_model(name, model,
                                    X_train_ml, y_train_int,
                                    X_test_ml, y_test_int, classes)

    joblib.dump(model, path)
    ml_models[name] = model
    ml_scores[name] = acc


# ======================================================
# Train DL Models
# ======================================================
print("🚀 Training DL Models...")
dl_models = {}
dl_scores = {}

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5)
]

# CNN1D
print("Training CNN1D...")
cnn1d = build_cnn_1d((1000, 1), len(classes))
cnn1d.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist = cnn1d.fit(
    X_tr_dl, y_tr,
    validation_data=(X_val_dl, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)
cnn1d.save(os.path.join(RUN_DIR, "cnn1d.keras"))
dl_models["CNN1D"] = cnn1d
dl_scores["CNN1D"] = max(hist.history["val_accuracy"])


# CNN2D
print("Training CNN2D...")
X_tr_2d = X_tr_dl.reshape(-1, 100, 10, 1)
X_val_2d = X_val_dl.reshape(-1, 100, 10, 1)

cnn2d = build_cnn_2d((100, 10, 1), len(classes))
cnn2d.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist2 = cnn2d.fit(
    X_tr_2d, y_tr,
    validation_data=(X_val_2d, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)
cnn2d.save(os.path.join(RUN_DIR, "cnn2d.keras"))
dl_models["CNN2D"] = cnn2d
dl_scores["CNN2D"] = max(hist2.history["val_accuracy"])


# ======================================================
# Advanced Hybrid (LIMITED epochs for CPU)
# ======================================================
print("🚀 Training Advanced Hybrid (light mode)...")
adv = AdvancedHybridModel(input_shape=(1000, 1), num_classes=len(classes))
adv.train_ensemble(X_tr_dl, y_tr, X_val_dl, y_val, epochs=20, batch_size=args.batch_size)

adv_pred = adv.predict_ensemble(X_test_dl)
adv_acc = np.mean(np.argmax(adv_pred, axis=1) == y_test_int)
print(f"✅ Advanced Hybrid Accuracy: {adv_acc:.4f}")

for name, model in adv.models.items():
    model.save(os.path.join(RUN_DIR, f"advanced_{name}.keras"))


# ======================================================
# Traditional Hybrid Ensemble
# ======================================================
scores = {**ml_scores, **dl_scores}
vals = np.array(list(scores.values()))
vals = (vals - vals.min()) + 1e-8
weights = vals / vals.sum()
weights_dict = {k: float(weights[i]) for i, k in enumerate(scores.keys())}

hybrid = HybridEnsemble(ml_models, dl_models, classes, weights_dict)
hyb_acc, _ = hybrid.evaluate(X_test_ml, X_test_dl, y_test_int)


# ======================================================
# Save metadata
# ======================================================
with open(os.path.join(RUN_DIR, "classes.json"), "w") as f:
    json.dump(classes, f)

record = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "limit": args.limit,
    "epochs": args.epochs,
    "hybrid_acc": hyb_acc,
    "advanced_hybrid_acc": adv_acc,
    "run_folder": RUN_DIR
}])

if os.path.exists(results_file):
    pd.concat([pd.read_csv(results_file), record]).to_csv(results_file, index=False)
else:
    record.to_csv(results_file, index=False)


print("🎉 Training complete")
print("📁 Run folder:", RUN_DIR)
sys.stdout.log.close()
sys.stdout = sys.__stdout__
print(f"✅ Logs saved to {log_file}")
print(f"✅ Results history saved to {results_file}")
print("Goodbye! 👋")
