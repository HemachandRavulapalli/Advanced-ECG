#!/bin/bash

echo "🚀 Starting ECG Classification Backend..."

# Correct venv path
echo "✅ Activating virtual environment"
source venv/bin/activate

cd backend || exit 1

echo "📦 Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "🌐 Starting FastAPI server on http://localhost:8000"
uvicorn app:app --host 0.0.0.0 --port 8000
