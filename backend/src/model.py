# model.py
import os
import numpy as np
import joblib
import tensorflow as tf
from hybrid_model import HybridEnsemble

# ------------------------
# Paths
# ------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")

# ------------------------
# Loader
# ------------------------
def load_saved_models():
    models = {}

    # ML models
    for name in ["SVM", "RandomForest", "KNN", "XGBoost"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"✅ Loaded ML model: {name}")

    # DL models
    cnn1d_path = os.path.join(ARTIFACTS_DIR, "cnn1d.h5")
    if os.path.exists(cnn1d_path):
        models["CNN1D"] = tf.keras.models.load_model(cnn1d_path)
        print("✅ Loaded DL model: CNN1D")

    cnn2d_path = os.path.join(ARTIFACTS_DIR, "cnn2d.h5")
    if os.path.exists(cnn2d_path):
        models["CNN2D"] = tf.keras.models.load_model(cnn2d_path)
        print("✅ Loaded DL model: CNN2D")

    # Hybrid Ensemble
    hybrid_path = os.path.join(ARTIFACTS_DIR, "hybrid.pkl")
    if os.path.exists(hybrid_path):
        hybrid = joblib.load(hybrid_path)
        print("✅ Loaded Hybrid Ensemble")
        models["Hybrid"] = hybrid

    return models


# ------------------------
# Predict (single signal)
# ------------------------
def predict_signal(models, signal, classes):
    """
    Run prediction using Hybrid Ensemble if available,
    else fall back to CNN1D.
    """
    signal = np.array(signal)

    # Ensure correct shape
    X_ml = signal.reshape(1, -1)   # for ML
    X_dl = signal.reshape(1, -1, 1)  # for CNN1D

    if "Hybrid" in models:
        hybrid = models["Hybrid"]
        probs = hybrid.predict_proba(X_ml, X_dl)
        pred_idx = np.argmax(probs, axis=1)[0]
        return classes[pred_idx], float(np.max(probs))

    elif "CNN1D" in models:
        probs = models["CNN1D"].predict(X_dl, verbose=0)
        pred_idx = np.argmax(probs, axis=1)[0]
        return classes[pred_idx], float(np.max(probs))

    else:
        raise RuntimeError("❌ No trained model available for prediction!")


# ------------------------
# Quick Test
# ------------------------
if __name__ == "__main__":
    models = load_saved_models()

    # Dummy ECG input (1000 samples)
    dummy_signal = np.random.randn(1000)
    classes = ["NORM", "AFIB", "MI", "STTC", "HYP", "V"]  # replace with real from training

    pred, conf = predict_signal(models, dummy_signal, classes)
    print(f"🫀 Prediction: {pred} (confidence {conf:.2f})")
