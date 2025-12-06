#!/usr/bin/env python3
"""
train_pipeline.py — Unified ECG ML + DL Hybrid Training
Now with proper PTB-XL patient-level split and no data leakage.
"""

import sys
import os
import time
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
from datetime import datetime
from collections import Counter
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from data_loader import load_all_datasets
from ml_models import prepare_features, get_ml_models, train_ml_model
from cnn_models import build_cnn_1d, build_cnn_2d
from hybrid_model import AdvancedHybridModel, HybridEnsemble

# ------------------------
# CLI Arguments
# ------------------------
parser = argparse.ArgumentParser(description="Train ECG hybrid ML + DL models")
parser.add_argument("--resume", action="store_true", help="Resume from latest run")
parser.add_argument("--limit", type=int, default=3000, help="Limit number of samples to load per dataset")
parser.add_argument("--epochs", type=int, default=50, help="Epochs for DL training (default: 50 for better accuracy)")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DL training")
parser.add_argument("--max_per_class", type=int, default=5000, help="Max samples per class if balancing")
parser.add_argument("--normalize", action="store_true", help="Apply global normalization (recommended)")
parser.add_argument("--sample_norm", action="store_true", help="Apply per-sample normalization")
parser.add_argument("--undersample", action="store_true", help="Undersample majority classes")
parser.add_argument("--svm_limit", type=int, default=2000, help="Subset size for SVM training")
parser.add_argument("--keep_runs", type=int, default=5, help="Keep last N model runs")
args = parser.parse_args()

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "train_log.txt")
results_file = os.path.join(LOG_DIR, "results_history.csv")

# ------------------------
# Logger
# ------------------------
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

# ------------------------
# Config
# ------------------------
LIMIT = args.limit
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SVM_LIMIT = args.svm_limit
MAX_PER_CLASS = args.max_per_class
APPLY_GLOBAL_NORM = args.normalize
APPLY_SAMPLE_NORM = args.sample_norm
UNDERSAMPLE = args.undersample
KEEP_RUNS = args.keep_runs

# ------------------------
# Run Management
# ------------------------
def get_latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None

def cleanup_old_runs(keep_last=KEEP_RUNS):
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    if len(runs) > keep_last:
        for old_run in runs[:-keep_last]:
            shutil.rmtree(old_run, ignore_errors=True)
            print(f"🧹 Deleted old run folder: {old_run}")

if args.resume:
    RUN_DIR = get_latest_run()
    if RUN_DIR:
        print(f"🔁 Resuming training from {RUN_DIR}")
    else:
        RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(RUN_DIR, exist_ok=True)
else:
    RUN_DIR = os.path.join(MODEL_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(RUN_DIR, exist_ok=True)
    cleanup_old_runs()

# ------------------------
# Load Data
# ------------------------
print("📥 Loading datasets (PTB-XL train/test + others)...")
(X_train, y_train), (X_test, y_test), classes = load_all_datasets(limit=LIMIT, one_hot=True, window_size=1000)
print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}, Classes={classes}")

if y_train.ndim > 1 and y_train.shape[1] > 1:
    y_train_int = np.argmax(y_train, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
else:
    y_train_int, y_test_int = y_train, y_test

# Print class distribution for diagnosis
print("\n📊 Class Distribution:")
train_dist = Counter(y_train_int)
test_dist = Counter(y_test_int)
print(f"Train: {dict(train_dist)}")
print(f"Test: {dict(test_dist)}")

# ------------------------
# Create Train/Validation Split (for model selection)
# ------------------------
print("\n🔀 Creating train/validation split from training data...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train_int
)
y_train_split_int = np.argmax(y_train_split, axis=1) if y_train_split.ndim > 1 else y_train_split
y_val_split_int = np.argmax(y_val_split, axis=1) if y_val_split.ndim > 1 else y_val_split

print(f"✅ Split: Train={X_train_split.shape}, Val={X_val_split.shape}, Test={X_test.shape}")

# ------------------------
# Normalization (use training split for statistics)
# ------------------------
if APPLY_SAMPLE_NORM and APPLY_GLOBAL_NORM:
    print("⚠️ Both sample and global normalization chosen, using global only.")
    APPLY_SAMPLE_NORM = False

if APPLY_SAMPLE_NORM:
    print("🔬 Applying per-sample z-score normalization")
    X_train_split = (X_train_split - np.mean(X_train_split, axis=1, keepdims=True)) / (np.std(X_train_split, axis=1, keepdims=True) + 1e-8)
    X_val_split = (X_val_split - np.mean(X_val_split, axis=1, keepdims=True)) / (np.std(X_val_split, axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / (np.std(X_test, axis=1, keepdims=True) + 1e-8)

if APPLY_GLOBAL_NORM:
    print("🔬 Applying global normalization (using training split statistics)")
    mean, std = np.mean(X_train_split), np.std(X_train_split) + 1e-8
    X_train_split = (X_train_split - mean) / std
    X_val_split = (X_val_split - mean) / std
    X_test = (X_test - mean) / std

# ------------------------
# Prepare Data for Models
# ------------------------
X_train_ml = prepare_features(X_train_split)
X_val_ml = prepare_features(X_val_split)
X_test_ml = prepare_features(X_test)
X_train_dl = X_train_split[..., np.newaxis]
X_val_dl = X_val_split[..., np.newaxis]
X_test_dl = X_test[..., np.newaxis]

# ------------------------
# Class Weights (based on training split)
# ------------------------
class_weights = None
if len(np.unique(y_train_split_int)) > 1:
    weights = compute_class_weight("balanced", classes=np.unique(y_train_split_int), y=y_train_split_int)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train_split_int), weights)}
    print("⚖️ Class weights:", class_weights)

# ------------------------
# Train ML Models
# ------------------------
ml_models = {}
ml_defs = get_ml_models(num_classes=len(classes))
ml_val_scores = {}

for name, model in ml_defs.items():
    path = os.path.join(RUN_DIR, f"{name}.joblib")
    if args.resume and os.path.exists(path):
        print(f"📂 Resuming {name} - loading and re-evaluating...")
        ml_models[name] = joblib.load(path)
        # Re-evaluate on validation set even when resuming
        if name == "SVM":
            acc = ml_models[name].score(X_val_ml, y_val_split_int)
        else:
            acc = ml_models[name].score(X_val_ml, y_val_split_int)
        ml_val_scores[name] = acc
        print(f"✅ {name} validation accuracy: {acc:.4f}")
        continue
    print(f"🚀 Training {name}...")
    start = time.time()
    if name == "SVM":
        idx = np.random.choice(len(X_train_ml), min(SVM_LIMIT, len(X_train_ml)), replace=False)
        X_sub, y_sub = X_train_ml[idx], y_train_split_int[idx]
        model, acc = train_ml_model(name, model, X_sub, y_sub, X_val_ml, y_val_split_int)
    else:
        model, acc = train_ml_model(name, model, X_train_ml, y_train_split_int, X_val_ml, y_val_split_int)
    joblib.dump(model, path)
    ml_models[name] = model
    ml_val_scores[name] = acc
    print(f"✅ {name} finished in {time.time() - start:.2f}s (val_acc={acc:.4f})")

# ------------------------
# Train DL Models
# ------------------------
dl_models = {}
dl_val_scores = {}

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def make_callbacks(name):
    ckpt = os.path.join(RUN_DIR, f"{name}_best.keras")
    # Adjust patience based on epochs - early stopping should trigger
    early_stop_patience = max(5, min(EPOCHS // 3, 15))
    lr_patience = max(3, early_stop_patience // 2)
    return [
        ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=lr_patience, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=early_stop_patience, restore_best_weights=True, verbose=1),
    ]

# CNN1D
cnn1d_path = os.path.join(RUN_DIR, "cnn1d.keras")
if args.resume and os.path.exists(cnn1d_path):
    print("📂 Resuming CNN1D - loading and re-evaluating...")
    cnn1d = tf.keras.models.load_model(cnn1d_path)
    # Re-evaluate on validation set
    val_loss, val_acc = cnn1d.evaluate(X_val_dl, y_val_split, verbose=0)
    dl_val_scores["CNN1D"] = val_acc
    print(f"✅ CNN1D validation accuracy: {val_acc:.4f}")
else:
    print("🚀 Training CNN1D...")
    cnn1d = build_cnn_1d((1000, 1), num_classes=len(classes))
    cnn1d.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history = cnn1d.fit(X_train_dl, y_train_split, validation_data=(X_val_dl, y_val_split),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                        callbacks=make_callbacks("cnn1d"),
                        class_weight=class_weights)
    cnn1d.save(cnn1d_path)
    dl_val_scores["CNN1D"] = max(history.history.get("val_accuracy", [0.0]))
dl_models["CNN1D"] = cnn1d

# CNN2D
cnn2d_path = os.path.join(RUN_DIR, "cnn2d.keras")
X_train_2d = X_train_dl.reshape(-1, 100, 10, 1)
X_val_2d = X_val_dl.reshape(-1, 100, 10, 1)
X_test_2d = X_test_dl.reshape(-1, 100, 10, 1)

if args.resume and os.path.exists(cnn2d_path):
    print("📂 Resuming CNN2D - loading and re-evaluating...")
    cnn2d = tf.keras.models.load_model(cnn2d_path)
    # Re-evaluate on validation set
    val_loss, val_acc = cnn2d.evaluate(X_val_2d, y_val_split, verbose=0)
    dl_val_scores["CNN2D"] = val_acc
    print(f"✅ CNN2D validation accuracy: {val_acc:.4f}")
else:
    print("🚀 Training CNN2D...")
    cnn2d = build_cnn_2d((100, 10, 1), num_classes=len(classes))
    cnn2d.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history2 = cnn2d.fit(X_train_2d, y_train_split, validation_data=(X_val_2d, y_val_split),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                         callbacks=make_callbacks("cnn2d"),
                         class_weight=class_weights)
    cnn2d.save(cnn2d_path)
    dl_val_scores["CNN2D"] = max(history2.history.get("val_accuracy", [0.0]))
dl_models["CNN2D"] = cnn2d

# ------------------------
# Hybrid Ensemble
# ------------------------
print("⚖️ Computing weights for hybrid ensemble...")
scores = {**ml_val_scores, **dl_val_scores}
vals = np.array(list(scores.values()), dtype=float)
vals = (vals - vals.min()) + 1e-8
weights = vals / (vals.sum() + 1e-8)
weights = {k: float(weights[i]) for i, k in enumerate(scores.keys())}
print("🔢 Ensemble weights:", weights)

# ------------------------
# Advanced Hybrid Model (99%+ accuracy target)
# ------------------------
print("🚀 Training Advanced Hybrid Model for 99%+ accuracy...")
advanced_hybrid = AdvancedHybridModel(input_shape=(1000, 1), num_classes=len(classes))

# Train the advanced ensemble (use validation split, not test)
advanced_hybrid.train_ensemble(
    X_train_dl, y_train_split, 
    X_val_dl, y_val_split,
    epochs=min(30, EPOCHS * 2),  # Scale with EPOCHS but cap at 30
    batch_size=BATCH_SIZE
)

# Evaluate advanced hybrid model on validation set
advanced_acc, advanced_predictions = advanced_hybrid.evaluate(X_val_dl, y_val_split)

# Save advanced models
advanced_hybrid.save_models(os.path.join(RUN_DIR, "advanced_hybrid"))

# Save classes for prediction
import json
classes_file = os.path.join(RUN_DIR, "classes.json")
with open(classes_file, "w") as f:
    json.dump(classes, f)
print(f"💾 Saved classes to {classes_file}")

print("\n🤝 Building Traditional Hybrid Ensemble...")
print(f"📊 Validation scores for weighting: {ml_val_scores}")
print(f"📊 DL validation scores: {dl_val_scores}")

hybrid = HybridEnsemble(ml_models=ml_models, dl_models=dl_models, classes=classes, weights=weights)

# Evaluate on validation set (for model selection) and test set (final evaluation)
print("\n📈 Evaluating on validation set...")
val_acc, _ = hybrid.evaluate(X_val_ml, X_val_dl, np.argmax(y_val_split, axis=1))

print("\n📈 Final evaluation on test set...")
test_acc, _ = hybrid.evaluate(X_test_ml, X_test_dl, np.argmax(y_test, axis=1))
acc = test_acc  # Use test accuracy for logging

# ------------------------
# Log Results
# ------------------------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
record = pd.DataFrame([{
    "timestamp": timestamp,
    "limit": LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "normalize": APPLY_GLOBAL_NORM,
    "hybrid_acc": acc,
    "advanced_hybrid_acc": advanced_acc,
    "run_folder": RUN_DIR
}])

if os.path.exists(results_file):
    results = pd.concat([pd.read_csv(results_file), record], ignore_index=True)
else:
    results = record
results.to_csv(results_file, index=False)

best = results.loc[results["hybrid_acc"].idxmax()]
best_advanced = results.loc[results["advanced_hybrid_acc"].idxmax()]
print(f"📊 Results logged: {results_file}")
print(f"🏆 Best traditional hybrid accuracy: {best.hybrid_acc:.4f} ({best.timestamp})")
print(f"🚀 Best advanced hybrid accuracy: {best_advanced.advanced_hybrid_acc:.4f} ({best_advanced.timestamp})")
print(f"📁 Model folder: {best.run_folder}")
print("🎉 Training completed successfully!")
