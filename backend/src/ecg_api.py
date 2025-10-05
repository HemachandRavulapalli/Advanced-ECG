import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from backend.src.predict_ecg import predict_ecg
import os

app = FastAPI()

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        # Create a temp file securely
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Call prediction
        result = predict_ecg(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        file.file.close()
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return result
