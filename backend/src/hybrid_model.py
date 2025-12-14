# hybrid_model.py
import os
import glob
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


class HybridEnsemble:
    """
    Hybrid ML + DL Ensemble for ECG Classification

    ML models  → handcrafted ECG features
    DL models  → raw ECG waveform
    """

    def __init__(self, ml_models=None, dl_models=None, classes=None, weights=None):
        self.ml_models = ml_models or {}
        self.dl_models = dl_models or {}
        self.classes = classes or []
        self.weights = weights or {}

        # Unified view (optional convenience)
        self.models = {**self.ml_models, **self.dl_models}

        # Meta-model for stacking
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict_proba(self, X_ml, X_dl):
        """
        Parameters
        ----------
        X_ml : np.ndarray (N, n_features)
            Feature vectors for ML models
        X_dl : np.ndarray (N, 1000, 1)
            Raw ECG input for DL models

        Returns
        -------
        np.ndarray (N, num_classes)
            Ensemble probability predictions
        """

        predictions = []
        weights = []

        # ---- ML predictions ----
        for name, model in self.ml_models.items():
            try:
                proba = model.predict_proba(X_ml)
                predictions.append(proba)
                weights.append(self.weights.get(name, 1.0))
            except Exception as e:
                print(f"⚠️ ML model {name} failed: {e}")

        # ---- DL predictions ----
        for name, model in self.dl_models.items():
            try:
                if "CNN2D" in name.upper():
                    # Reshape for 2D CNN: (N, 1000, 1) -> (N, 100, 10, 1)
                    X_dl_reshaped = X_dl.reshape(X_dl.shape[0], 100, 10, 1)
                    proba = model.predict(X_dl_reshaped, verbose=0)
                else:
                    proba = model.predict(X_dl, verbose=0)
                predictions.append(proba)
                weights.append(self.weights.get(name, 1.0))
            except Exception as e:
                print(f"⚠️ DL model {name} failed: {e}")

        if not predictions:
            raise ValueError("❌ No models produced predictions")

        # ---- Weighted average ----
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        ensemble_proba = np.zeros_like(predictions[0])
        for p, w in zip(predictions, weights):
            # Safety: align class dimensions
            min_c = min(p.shape[1], ensemble_proba.shape[1])
            ensemble_proba[:, :min_c] += w * p[:, :min_c]

        return ensemble_proba

    # --------------------------------------------------
    # Helper for getting predictions from all models
    # --------------------------------------------------
    def _get_predictions(self, X_ml, X_dl):
        predictions = []

        # ---- ML predictions ----
        for name, model in self.ml_models.items():
            try:
                proba = model.predict_proba(X_ml)
                predictions.append(proba)
            except Exception as e:
                print(f"⚠️ ML model {name} failed: {e}")

        # ---- DL predictions ----
        for name, model in self.dl_models.items():
            try:
                if name == "CNN2D":
                    # Reshape for 2D CNN: (N, 1000, 1) -> (N, 100, 10, 1)
                    X_dl_reshaped = X_dl.reshape(X_dl.shape[0], 100, 10, 1)
                    proba = model.predict(X_dl_reshaped, verbose=0)
                else:
                    proba = model.predict(X_dl, verbose=0)
                predictions.append(proba)
            except Exception as e:
                print(f"⚠️ DL model {name} failed: {e}")

        if not predictions:
            raise ValueError("❌ No models produced predictions")

        return predictions

    # --------------------------------------------------
    # Final prediction
    # --------------------------------------------------
    def predict(self, X_ml, X_dl):
        probs = self.predict_proba(X_ml, X_dl)
        idx = np.argmax(probs, axis=1)
        return idx, probs

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def evaluate(self, X_ml, X_dl, y_true):
        """
        Evaluate ensemble on test data
        """

        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        probs = self.predict_proba(X_ml, X_dl)
        y_pred = np.argmax(probs, axis=1)

        acc = accuracy_score(y_true, y_pred)
        print(f"\n🎯 Hybrid Ensemble Accuracy: {acc:.4f}\n")

        print("📊 Classification Report:")
        # Filter target_names to only include present classes
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        filtered_names = [self.classes[i] for i in unique_labels if i < len(self.classes)]
        print(classification_report(y_true, y_pred, target_names=filtered_names, labels=unique_labels, zero_division=0))

        return acc, probs

    # --------------------------------------------------
    # Save models
    # --------------------------------------------------
    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # Save DL models
        for name, model in self.dl_models.items():
            path = os.path.join(save_dir, f"{name}.keras")
            model.save(path)
            print(f"💾 Saved DL model: {path}")

        # Save ML models
        for name, model in self.ml_models.items():
            path = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(model, path)
            print(f"💾 Saved ML model: {path}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    def load_models(self, load_dir):
        """
        Load previously saved ML & DL models
        """

        # Load DL models
        for model_path in glob.glob(os.path.join(load_dir, "*.keras")):
            name = os.path.basename(model_path).replace(".keras", "")
            self.dl_models[name] = tf.keras.models.load_model(
                model_path,
                safe_mode=False
            )
            print(f"📂 Loaded DL model: {name}")

        # Load ML models
        for model_path in glob.glob(os.path.join(load_dir, "*.joblib")):
            name = os.path.basename(model_path).replace(".joblib", "")
            self.ml_models[name] = joblib.load(model_path)
            print(f"📂 Loaded ML model: {name}")

        self.models = {**self.ml_models, **self.dl_models}
# ======================================================
# End of hybrid_model.py
# ======================================================


class AdvancedHybridModel:
    """
    Advanced Hybrid Model with multiple CNN architectures
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}

        # Build multiple CNN models
        from cnn_models import build_cnn_1d, build_cnn_2d

        self.models["CNN1D"] = build_cnn_1d(input_shape, num_classes)
        self.models["CNN2D"] = build_cnn_2d((100, 10, 1), num_classes)

        # Compile models
        for model in self.models.values():
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == "CNN2D":
                # Reshape for 2D CNN
                X_train_reshaped = X_train.reshape(X_train.shape[0], 100, 10, 1)
                X_val_reshaped = X_val.reshape(X_val.shape[0], 100, 10, 1)
            else:
                X_train_reshaped = X_train
                X_val_reshaped = X_val
            model.fit(
                X_train_reshaped, y_train,
                validation_data=(X_val_reshaped, y_val),
                epochs=epochs, batch_size=batch_size, verbose=1
            )

    def predict_ensemble(self, X):
        predictions = []
        for name, model in self.models.items():
            if name == "CNN2D":
                X_reshaped = X.reshape(X.shape[0], 100, 10, 1)
            else:
                X_reshaped = X
            pred = model.predict(X_reshaped, verbose=0)
            predictions.append(pred)

        # Average predictions
        return np.mean(predictions, axis=0)