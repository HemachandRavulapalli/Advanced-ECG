# ml_models.py
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from feature_extraction import extract_ecg_features


# =====================================================
# Feature preparation for ML models
# =====================================================
def prepare_features(X):
    """
    Extract features from raw ECG signals for ML models.
    X: (n_samples, n_timesteps) array of ECG signals
    Returns: (n_samples, n_features) array of feature vectors
    """
    features = []
    for signal in X:
        feat = extract_ecg_features(signal)
        features.append(feat)
    return np.array(features, dtype=np.float32)


# =====================================================
# Classical ML models for ECG FEATURE VECTORS
# =====================================================
def get_ml_models(num_classes: int):
    """
    Returns dictionary of ML models.
    These models EXPECT feature vectors, not raw ECG.
    """

    models = {

        # -----------------------
        # Support Vector Machine
        # -----------------------
        "SVM": SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            probability=True,        # REQUIRED for hybrid ensemble
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ),

        # -----------------------
        # Random Forest
        # -----------------------
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        # -----------------------
        # K-Nearest Neighbors
        # -----------------------
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="euclidean"
        ),

        # -----------------------
        # XGBoost
        # -----------------------
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        ),
    }

    return models


# =====================================================
# Train + Evaluate helper
# =====================================================
def train_ml_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    classes=None
):
    """
    Train and evaluate a classical ML model
    Inputs must be FEATURE VECTORS.
    """

    print(f"🚀 Training ML model: {name}")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    print(f"✅ {name} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if classes:
        y_pred = model.predict(X_val)
        unique_labels = np.unique(np.concatenate([y_val, y_pred]))
        filtered_names = [classes[i] for i in unique_labels if i < len(classes)]
        print(f"📊 {name} Classification Report:")
        print(classification_report(y_val, y_pred, target_names=filtered_names, labels=unique_labels, zero_division=0))

    return model, val_acc
