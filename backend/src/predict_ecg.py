# backend/src/predict_ecg.py
import os
import sys
import numpy as np
import joblib
import json

# Try to import tensorflow, but handle if missing
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ Warning: TensorFlow not available. Deep learning models will not work.")

try:
    from pdf_to_signal import extract_signal_from_file
except ImportError as e:
    print(f"⚠️ Warning: Cannot import pdf_to_signal: {e}")
    extract_signal_from_file = None

try:
    from hybrid_model import HybridEnsemble
    HYBRID_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Cannot import HybridEnsemble: {e}")
    HybridEnsemble = None
    HYBRID_ENSEMBLE_AVAILABLE = False


# ------------------------
# Label map (5 target classes)
# ------------------------
LABEL_MAP = {
    "Class_0": "Normal Sinus Rhythm",
    "Class_1": "Atrial Fibrillation",
    "Class_2": "Bradycardia",
    "Class_3": "Tachycardia",
    "Class_4": "Ventricular Arrhythmias"
}

BASE_DIR = os.path.dirname(__file__)

# Allow overriding model directory via environment variable so we can
# mount a persistent volume in Railway (or other hosts) and point to it.
# Default remains the local "saved_models" folder for local development.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "saved_models"))

def get_latest_run():
    runs = sorted(
        [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("run_")],
        key=os.path.getmtime,
    )
    return runs[-1] if runs else None


def load_models(run_dir):
    ml_models = {}
    dl_models = {}

    for f in os.listdir(run_dir):
        if f.endswith(".joblib"):
            name = f.replace(".joblib", "")
            try:
                ml_models[name] = joblib.load(os.path.join(run_dir, f))
                print(f"✅ Loaded ML model: {name}")
            except Exception as e:
                print(f"⚠️ Skipping {f}: Could not load model - {e}")
                continue
        elif f.endswith(".keras") or f.endswith(".h5"):
            if not TENSORFLOW_AVAILABLE:
                print(f"⚠️ Skipping {f}: TensorFlow not available")
                continue
            name = f.replace(".keras", "").replace(".h5", "")
            try:
                dl_models[name.upper()] = tf.keras.models.load_model(os.path.join(run_dir, f))
                print(f"✅ Loaded DL model: {name}")
            except Exception as e:
                print(f"⚠️ Skipping {f}: Could not load model - {e}")
                continue

    return ml_models, dl_models


def validate_input_signal(signal):
    """
    Additional runtime validation to catch non-ECG inputs that slip past
    the image/PDF extraction heuristics.
    """
    if signal is None or len(signal) == 0:
        raise ValueError("❌ Empty signal extracted.")

    # Require meaningful variation and dynamic range
    if np.std(signal) < 0.05 or (np.max(signal) - np.min(signal)) < 0.5:
        raise ValueError("❌ Signal lacks ECG-like variation (too flat).")

    # Require that enough points are non-zero (reject mostly blank scans)
    nonzero_ratio = np.count_nonzero(signal) / len(signal)
    if nonzero_ratio < 0.05:
        raise ValueError("❌ Signal appears blank/non-ECG (too many zero values).")

    # Reject signals that are almost monotonic/line-like
    mean_step = np.mean(np.abs(np.diff(signal)))
    if mean_step < 1e-3:
        raise ValueError("❌ Signal changes are too small to represent an ECG trace.")


def predict_ecg(pdf_path):
    if extract_signal_from_file is None:
        raise ValueError("❌ PDF signal extraction not available. Missing dependencies.")
    
    if not HYBRID_ENSEMBLE_AVAILABLE or HybridEnsemble is None:
        raise ValueError("❌ Hybrid ensemble model not available. Missing dependencies.")
    
    print("📄 Converting PDF → ECG signal...")
    signal = extract_signal_from_file(pdf_path)
    if signal is None:
        raise ValueError("❌ Could not extract signal from PDF")

    # Additional validation to reject non-ECG uploads early
    validate_input_signal(signal)

    # preprocess for models
    X_ml = signal.reshape(1, -1)
    X_dl = signal.reshape(1, -1, 1)

    # normalization (z-score)
    X_ml = (X_ml - np.mean(X_ml)) / (np.std(X_ml) + 1e-8)
    X_dl = (X_dl - np.mean(X_dl)) / (np.std(X_dl) + 1e-8)

    # load models
    best_run = get_latest_run()
    if best_run is None:
        raise ValueError("❌ No trained models found. Please train models first.")
    print(f"📂 Loading models from: {best_run}")

    # Sanity check: ensure all models belong to new 4-class setup
    ml_models, dl_models = load_models(best_run)
    print(f"📦 Loaded ML models: {list(ml_models.keys())}")
    print(f"📦 Loaded DL models: {list(dl_models.keys())}")
    
    # Check if we have at least some models
    if not ml_models and not dl_models:
        raise ValueError("❌ No models loaded. Please check that model files exist in the saved_models directory.")
    
    if not ml_models:
        print("⚠️ Warning: No ML models loaded. Predictions may be less accurate.")
    if not dl_models:
        print("⚠️ Warning: No DL models loaded (TensorFlow not available). Using ML models only.")

    # Load class list (always use 5 classes)
    classes = ["Normal Sinus Rhythm", "Atrial Fibrillation", "Bradycardia", "Tachycardia", "Ventricular Arrhythmias"]
    
    # Load saved classes if available
    class_file = os.path.join(best_run, "classes.json")
    if os.path.exists(class_file):
        try:
            with open(class_file, "r") as f:
                saved_classes = json.load(f)
            if len(saved_classes) == 5:
                classes = saved_classes
                print(f"📋 Using saved classes: {classes}")
        except Exception as e:
            print(f"⚠️ Could not load classes.json: {e}, using default 5 classes")
    
    # Ensure signal is exactly 1000 samples (required by models)
    if X_dl.shape[1] != 1000:
        if X_dl.shape[1] < 1000:
            # Pad if too short
            pad_length = 1000 - X_dl.shape[1]
            X_dl = np.pad(X_dl, ((0, 0), (0, pad_length), (0, 0)), mode='constant')
            X_ml = X_dl.reshape(1, -1)
            print(f"⚠️ Signal was {X_dl.shape[1]} samples, padded to 1000")
        else:
            # Truncate if too long
            X_dl = X_dl[:, :1000, :]
            X_ml = X_dl.reshape(1, -1)
            print(f"⚠️ Signal was {X_dl.shape[1]} samples, truncated to 1000")
    
    # hybrid ensemble
    hybrid = HybridEnsemble(ml_models=ml_models, dl_models=dl_models, classes=classes, weights={})

    print("🧠 Predicting...")
    try:
        probs = hybrid.predict_proba(X_ml, X_dl)
        
        # Ensure probabilities are for 5 classes
        if probs.shape[1] > 5:
            probs = probs[:, :5]
        elif probs.shape[1] < 5:
            # Pad with zeros if fewer classes
            pad_probs = np.zeros((probs.shape[0], 5))
            pad_probs[:, :probs.shape[1]] = probs
            probs = pad_probs
        
        # Normalize probabilities
        probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-8)
        
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx] if pred_idx < len(classes) else classes[0]
        pred_conf = float(np.max(probs))
        
        # Check if confidence is too low (might indicate invalid input)
        if pred_conf < 0.3:
            print(f"⚠️ Warning: Low confidence ({pred_conf:.4f}). Input may not be a valid ECG signal.")
        
    except Exception as e:
        raise ValueError(f"❌ Prediction failed: {str(e)}. Models may not be compatible.")

    # use human-readable labels
    readable_pred = pred_class  # Already using readable labels
    readable_probs = {
        classes[i]: round(float(p), 4)
        for i, p in enumerate(probs[0].tolist())
    }

    results = {
        "predicted_class": readable_pred,
        "confidence": round(pred_conf, 4),
        "probabilities": readable_probs
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_ecg.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    predict_ecg(pdf_path)
