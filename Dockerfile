# Dockerfile for Hugging Face Spaces (FastAPI backend)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, PDF/image processing, and OCR
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# Copy all code
COPY . .

EXPOSE 7860 8000

# Start FastAPI app (Spaces expects app:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
