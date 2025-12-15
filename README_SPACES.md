# Deploying on Hugging Face Spaces

This project can be deployed on [Hugging Face Spaces](https://huggingface.co/spaces) using the FastAPI backend.

## Steps

1. **Push this repository to your Hugging Face Space.**
2. **Ensure the following files are present at the repo root:**
   - `app.py` (FastAPI wrapper)
   - `requirements.txt` (Python dependencies)
3. **Spaces will automatically install dependencies and launch the FastAPI app.**

## Notes
- The main backend code is in `backend/`.
- The wrapper in `app.py` exposes the FastAPI app for Spaces.
- If you need system dependencies (e.g., Tesseract), use a `Dockerfile` or request support from Hugging Face.
- For custom system dependencies, see `backend/Dockerfile` as a reference.

## Troubleshooting
- If you see errors about missing libraries (e.g., Tesseract), try switching to Docker-based Spaces and use the provided Dockerfile.
- For issues with model files or data, ensure all required files are committed and tracked in git.

---
For further help, see the main `README.md` or open an issue.
