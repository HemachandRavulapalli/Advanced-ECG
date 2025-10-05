from fastapi import FastAPI
import numpy as np
from src.preprocessing import preprocess_ecg
from src.ml_models import predict_with_ml
from src.cnn_models import predict_with_cnn
from src.hybrid_model import hybrid_predict
from src.data_loader import load_ecg_record

app = FastAPI()

CLASSES = ["Normal", "AFib", "Bradycardia", "Tachycardia", "Ventricular Arrhythmia"]

@app.get("/")
def root():
    return {"message": "Welcome to ECG API 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/predict")
def predict_ecg(record: str = "100"):
    # Load real ECG
    signal, fs = load_ecg_record(record=record)

    # Preprocess
    processed = preprocess_ecg(signal, fs=fs)

    # Predictions
    ml_pred, ml_conf = predict_with_ml(processed["filtered_signal"], CLASSES)
    cnn_pred, cnn_conf = predict_with_cnn(processed["filtered_signal"], CLASSES)
    final_pred, final_conf = hybrid_predict([ml_conf, cnn_conf], CLASSES)

    return {
        "record": record,
        "ml_prediction": ml_pred,
        "cnn_prediction": cnn_pred,
        "final_prediction": final_pred,
        "confidence": final_conf,
        "imf_count": len(processed["imfs"])
    }
