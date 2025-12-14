# backend/src/data_loader.py
import os
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MIT_DIR = os.path.join(PROCESSED_DIR, "mitdb")
PTBXL_DIR = os.path.join(PROCESSED_DIR, "ptbxl")
KARDIA_DIR = os.path.join(PROCESSED_DIR, "kardia")

# ======================================================
# Data Augmentation
# ======================================================
def augment_ecg_signals(X, augment_factor=2):
    """
    Augment ECG signals with noise, scaling, and shifting
    """
    augmented_X = [X]
    
    for _ in range(augment_factor):
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, X.shape)
        X_noise = X + noise
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1, X.shape[0])
        X_scaled = X * scale[:, np.newaxis]
        
        # Random shifting
        shift = np.random.uniform(-0.05, 0.05, X.shape[0])
        X_shifted = X + shift[:, np.newaxis]
        
        augmented_X.extend([X_noise, X_scaled, X_shifted])
    
    return np.concatenate(augmented_X)
TARGET_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Ventricular Arrhythmias",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}

# ======================================================
# Utility
# ======================================================
def load_npz_folder(folder):
    X, y = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".npz"):
            continue

        data = np.load(os.path.join(folder, fname))
        signal = data["signal"]
        label = data["label"].item()
        if isinstance(label, (int, np.integer)):
            label = TARGET_CLASSES[label]

        if signal.shape[0] != 1000:
            continue

        if label not in CLASS_TO_IDX:
            continue

        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)

# ======================================================
# Load datasets
# ======================================================
def load_all_datasets(limit=None, one_hot=True, window_size=None):
    print("📥 Loading MIT-BIH...")
    X_mit, y_mit = load_npz_folder(MIT_DIR)

    print("📥 Loading PTB-XL...")
    X_ptb, y_ptb = load_npz_folder(PTBXL_DIR)

    print("📥 Loading Kardia...")
    kardia_X_path = os.path.join(KARDIA_DIR, "X.npy")
    kardia_y_path = os.path.join(KARDIA_DIR, "y.npy")
    if os.path.exists(kardia_X_path) and os.path.exists(kardia_y_path):
        X_kardia = np.load(kardia_X_path)
        y_kardia = np.load(kardia_y_path)
        has_kardia = True
    else:
        print("⚠️ Kardia data not found, skipping")
        X_kardia = np.array([])
        y_kardia = np.array([])
        has_kardia = False

    # Combine
    datasets = [X_mit, X_ptb]
    labels = [y_mit, y_ptb]
    if has_kardia:
        datasets.append(X_kardia)
        labels.append(y_kardia)

    X = np.concatenate(datasets) if datasets else np.array([])
    y = np.concatenate(labels) if labels else np.array([])

    print(f"📊 Total samples: {X.shape[0]}")

    # Encode labels to integers
    y_int = np.array([TARGET_CLASSES.index(label) for label in y])

    # Apply limit if specified
    if limit:
        X = X[:limit]
        y_int = y_int[:limit]
        print(f"📊 Limited to {limit} samples")

    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    # Augment training data
    print("🔄 Applying data augmentation...")
    X_train_aug = augment_ecg_signals(X_train, augment_factor=1)  # Add 3x augmented versions
    y_train_int_aug = np.tile(y_train_int, 4)  # Repeat labels 4 times (original + 3 aug)
    X_train = X_train_aug
    y_train_int = y_train_int_aug
    print(f"📊 After augmentation: {X_train.shape[0]} samples")

    # Encode labels
    if one_hot:
        y_train_enc = np.eye(len(TARGET_CLASSES))[y_train_int]
        y_test_enc = np.eye(len(TARGET_CLASSES))[y_test_int]
    else:
        y_train_enc = y_train_int
        y_test_enc = y_test_int

    return (X_train, y_train_enc), (X_test, y_test_enc), TARGET_CLASSES
