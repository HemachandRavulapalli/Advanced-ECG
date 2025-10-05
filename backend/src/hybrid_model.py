import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class HybridEnsemble:
    """
    Hybrid Ensemble: Combines ML + DL model predictions
    Fusion via weighted probability averaging (softmax averaging).
    """

    def __init__(self, ml_models=None, dl_models=None, classes=None, weights=None):
        self.ml_models = ml_models or {}
        self.dl_models = dl_models or {}
        # ensure classes is a list of strings
        try:
            self.classes = list(classes) if classes is not None else []
        except Exception:
            self.classes = []
        # weights: dict {model_name: weight}
        self.weights = weights or {}

    def _get_model_weight(self, name):
        # default weight 1.0 if not specified
        return float(self.weights.get(name, 1.0))

    def predict_proba(self, X_ml, X_dl):
        """
        Get probability predictions from all models and weighted-average them.
        X_ml = preprocessed flat features for ML models
        X_dl = tensor input for DL models
        """
        probs = []
        wts = []

        # ML models
        for name, model in self.ml_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_ml)
                else:
                    pred = model.predict(X_ml)
                    p = np.eye(len(self.classes))[pred]
                probs.append(p)
                wts.append(self._get_model_weight(name))
            except Exception as e:
                print(f"❌ Error getting proba from ML model {name}: {e}")

        # DL models
        for name, model in self.dl_models.items():
            try:
                if "2D" in name or "cnn2d" in name.lower():
                    X_in = X_dl.reshape(-1, 100, 10, 1)
                else:
                    X_in = X_dl
                p = model.predict(X_in, verbose=0)
                probs.append(p)
                wts.append(self._get_model_weight(name))
            except Exception as e:
                print(f"❌ Error getting proba from DL model {name}: {e}")

        if len(probs) == 0:
            raise RuntimeError("No model probabilities available to combine.")

        try:
            stacked = np.stack(probs, axis=0).astype(float)
        except Exception as e:
            min_c = min(p.shape[1] for p in probs)
            stacked = np.stack([p[:, :min_c] for p in probs], axis=0)
            print("⚠️ Models had differing class counts, truncated to", min_c)

        weights = np.array(wts).reshape(-1, 1, 1)
        weights = weights / (np.sum(weights) + 1e-9)

        weighted = stacked * weights
        avg_probs = np.sum(weighted, axis=0)
        return avg_probs

    def predict(self, X_ml, X_dl):
        avg_probs = self.predict_proba(X_ml, X_dl)
        preds = np.argmax(avg_probs, axis=1)
        return preds

    def evaluate(self, X_ml, X_dl, y_true):
        preds = self.predict(X_ml, X_dl)
        acc = accuracy_score(y_true, preds)
        print(f"✅ Hybrid Accuracy: {acc:.4f}")
        target_names = [str(c) for c in self.classes] if len(self.classes) > 0 else None
        try:
            print(classification_report(y_true, preds, target_names=target_names, zero_division=0))
        except Exception:
            print("ℹ️ Could not print full classification report (mismatch in classes).")
        return acc
