# ECG Classification System

A hybrid machine learning and deep learning system for classifying ECG signals into 5 cardiac rhythm categories:
- Normal Sinus Rhythm
- Atrial Fibrillation
- Bradycardia
- Tachycardia
- Ventricular Arrhythmias

## 🚀 Quick Start Guide

This guide will help you set up and run the project on a new laptop from scratch.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)
- ~10GB free disk space for datasets

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd D4

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
cd backend/src
python3 verify_setup.py
```

### Step 2: Download and Prepare Datasets

The system uses multiple ECG datasets. Follow these steps to download and prepare them:

#### 2.1 Download MIT-BIH Arrhythmia Database

1. **Register and Download**:
   - Visit: https://physionet.org/content/mitdb/1.0.0/
   - Sign up for a free account on PhysioNet
   - Download the complete dataset

2. **Extract and Organize**:
   ```bash
   # Create data directory structure
   mkdir -p data/raw/mitdb
   
   # Extract downloaded files to data/raw/mitdb/
   # You should have files like 100.dat, 100.hea, 100.atr, etc.
   ```

3. **Generate Annotations** (if not already present):
   ```bash
   cd backend/src
   python3 generate_mitdb_annotations.py
   ```
   This will create `data/raw/mitdb_annotations.csv` with record labels.

#### 2.2 Download PTB-XL Dataset

1. **Download**:
   - Visit: https://physionet.org/content/ptb-xl/1.0.1/
   - Download the dataset (requires PhysioNet account)

2. **Extract and Organize**:
   ```bash
   # Create PTB-XL directory
   mkdir -p data/raw/ptbxl
   
   # Extract files to data/raw/ptbxl/
   # You should have:
   #   - ptbxl_database.csv
   #   - scp_statements.csv
   #   - records100/ (or records500/, records1000/)
   ```

#### 2.3 Prepare Kardia Dataset (Optional)

If you have Kardia ECG PDFs:

```bash
# Place PDF files in data/kardia/ or data/raw/kardia/
mkdir -p data/kardia

# If you have extracted signals, place them in data/ directory as:
#   - X.npy (signals)
#   - Y.npy or y.npy (labels)
```

**Note**: If datasets are missing, the system will automatically generate synthetic data to ensure all 5 classes are available.

### Step 3: Verify Dataset Structure

Your `data/` directory should look like this:

```
data/
├── raw/
│   ├── mitdb/
│   │   ├── 100.dat
│   │   ├── 100.hea
│   │   ├── 100.atr
│   │   ├── ...
│   │   └── mitdb_annotations.csv
│   ├── ptbxl/
│   │   ├── ptbxl_database.csv
│   │   ├── scp_statements.csv
│   │   └── records100/  (or records500/, records1000/)
│   └── kardia/  (optional PDF files)
├── processed/  (auto-generated during preprocessing)
└── X.npy, Y.npy  (optional, if you have extracted Kardia data)
```

### Step 4: Train the Models

Train the hybrid ensemble models:

```bash
cd backend/src

# Basic training (quick test)
python3 train_pipeline.py --limit 1000 --epochs 10 --normalize

# Full training (recommended for best accuracy)
python3 train_pipeline.py --limit 5000 --epochs 50 --normalize --batch_size 64

# For 99%+ accuracy (advanced ensemble)
python3 train_pipeline.py --limit 10000 --epochs 100 --normalize --batch_size 32
```

**Training Parameters**:
- `--limit`: Number of samples per dataset (default: 3000)
- `--epochs`: Training epochs (default: 5)
- `--batch_size`: Batch size (default: 64)
- `--normalize`: Apply global normalization (highly recommended)
- `--resume`: Resume from latest run if interrupted

**Training Time**:
- Quick test (1000 samples, 10 epochs): ~15-30 minutes
- Full training (5000 samples, 50 epochs): ~2-4 hours
- Advanced ensemble (10000 samples, 100 epochs): ~6-12 hours

### Step 5: Verify Model Training

After training completes, check:

1. **Model files** should be saved in:
   ```
   backend/src/saved_models/run_YYYY-MM-DD_HH-MM-SS/
   ├── CNN1D.keras
   ├── CNN2D.keras
   ├── SVM.joblib
   ├── RandomForest.joblib
   ├── XGBoost.joblib
   ├── KNN.joblib
   ├── classes.json
   └── advanced_hybrid/
       ├── cnn1d_residual_cnn.keras
       ├── cnn1d_densenet_cnn.keras
       └── ...
   ```

2. **Check training results**:
   ```bash
   cat backend/logs/results_history.csv
   ```

3. **Verify model outputs**:
   - Look for accuracy scores > 0.90 for good models
   - Advanced hybrid should approach 0.99 (99%)

### Step 6: Make Predictions

#### Using Python Script

```bash
cd backend/src

# Predict from PDF
python3 predict_ecg.py path/to/ecg_file.pdf

# Predict from image
python3 predict_ecg.py path/to/ecg_image.png
```

#### Using FastAPI Server

1. **Start the server**:
   ```bash
   cd backend
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Make a prediction**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/ecg_file.pdf"
   ```

## 🔍 Troubleshooting

### Issue: "No trained models found"

**Solution**:
```bash
# Train models first
cd backend/src
python3 train_pipeline.py --limit 2000 --epochs 20 --normalize
```

### Issue: "Dataset not found, skipping"

**Solution**: The system will automatically generate synthetic data for missing classes. However, for best results:

1. Download datasets as described in Step 2
2. Ensure proper directory structure
3. Re-run training

### Issue: Models always predict the same class (e.g., "Bradycardia")

**Possible Causes & Solutions**:

1. **Models not properly trained**:
   ```bash
   # Re-train with more data and epochs
   python3 train_pipeline.py --limit 5000 --epochs 50 --normalize
   ```

2. **Invalid input file** (not an ECG):
   - The system validates ECG signals
   - Non-ECG images/PDFs will be rejected with error message
   - Ensure input files contain valid ECG traces

3. **Model mismatch** (4 classes vs 5 classes):
   ```bash
   # Check if models were trained with 5 classes
   cat backend/src/saved_models/run_*/classes.json
   
   # Re-train if needed
   python3 train_pipeline.py --limit 3000 --epochs 30 --normalize
   ```

4. **Normalization issues**:
   - Always use `--normalize` flag during training
   - Prediction code handles normalization automatically

### Issue: "Invalid ECG signal" error

**Solutions**:

1. **File is not an ECG**:
   - Only PDF/image files with visible ECG waveforms are supported
   - The system validates that extracted signals have ECG-like characteristics

2. **Corrupted file**:
   - Try converting PDF to high-resolution image first
   - Ensure image is clear and ECG trace is visible

3. **Format issues**:
   - Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF
   - Ensure file is not password-protected

### Issue: Low accuracy or poor predictions

**Solutions**:

1. **Increase training data**:
   ```bash
   python3 train_pipeline.py --limit 10000 --epochs 100 --normalize
   ```

2. **Use advanced hybrid model**:
   - The advanced hybrid ensemble trains automatically
   - Check `advanced_hybrid_acc` in results_history.csv

3. **Check class balance**:
   ```bash
   # Verify all 5 classes are present
   cd backend/src
   python3 -c "from data_loader import load_all_datasets; (X,y,classes)=load_all_datasets(limit=100); print(f'Classes: {classes}'); print(f'Samples: {len(X)}')"
   ```

### Issue: "TensorFlow not available"

**Solution**:
```bash
pip install tensorflow
# Or for GPU support:
pip install tensorflow-gpu
```

### Issue: Out of memory during training

**Solutions**:

1. **Reduce batch size**:
   ```bash
   python3 train_pipeline.py --limit 3000 --epochs 30 --batch_size 32 --normalize
   ```

2. **Reduce dataset size**:
   ```bash
   python3 train_pipeline.py --limit 2000 --epochs 20 --normalize
   ```

3. **Train models separately** (modify train_pipeline.py to skip advanced ensemble temporarily)

## 📊 Model Architecture

### Traditional Models
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: 500 trees with balanced class weights
- **XGBoost**: Gradient boosting with regularization
- **KNN**: K-nearest neighbors with distance weighting

### Deep Learning Models
- **CNN1D**: 1D Convolutional Neural Network with residual connections
- **CNN2D**: 2D CNN for spectrogram-like representations with attention

### Advanced Hybrid Ensemble
- **Residual CNN**: Deep network with skip connections
- **DenseNet CNN**: Dense block architecture
- **Attention CNN**: Self-attention mechanism
- **Multi-scale CNN**: Multiple kernel sizes for feature extraction

## 📁 Project Structure

```
D4/
├── backend/
│   ├── src/
│   │   ├── train_pipeline.py      # Main training script
│   │   ├── predict_ecg.py         # Prediction script
│   │   ├── data_loader.py         # Dataset loading
│   │   ├── cnn_models.py          # CNN architectures
│   │   ├── ml_models.py           # ML model definitions
│   │   ├── hybrid_model.py        # Hybrid ensemble
│   │   ├── pdf_to_signal.py       # PDF/image to signal conversion
│   │   └── saved_models/          # Trained models (auto-generated)
│   ├── logs/                      # Training logs and results
│   └── app.py                     # FastAPI server
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed data (auto-generated)
└── requirements.txt               # Python dependencies
```

## 🔬 Validation and Quality Checks

The system includes automatic validation:

1. **Input File Validation**:
   - File format check
   - File existence and readability
   - Image/PDF validity

2. **ECG Signal Validation**:
   - Signal length check
   - Variation detection (constant signals rejected)
   - ECG-like characteristics verification
   - Autocorrelation analysis

3. **Model Validation**:
   - All 5 classes must be present
   - Class balance checking
   - Model compatibility verification

## 📈 Performance Metrics

- **Target Accuracy**: 99%+ (advanced hybrid ensemble)
- **Traditional Hybrid**: Typically 85-95%
- **Individual Models**: 70-90%

Check `backend/logs/results_history.csv` for detailed metrics.

## 🆘 Getting Help

1. Check this README troubleshooting section
2. Review training logs in `backend/logs/train_log.txt`
3. Verify dataset structure matches requirements
4. Ensure all dependencies are installed correctly

## 📝 Notes

- First-time setup may take 1-2 hours including dataset downloads
- Training time depends on hardware (CPU/GPU) and dataset size
- Synthetic data is automatically generated if datasets are missing
- All 5 classes are guaranteed to be present after data loading

## ✅ Verification Checklist

Before running predictions, verify:

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Datasets downloaded (or synthetic data will be used)
- [ ] Models trained successfully (check `saved_models/run_*/`)
- [ ] `classes.json` file exists in latest run directory
- [ ] At least one `.keras` or `.joblib` model file exists
- [ ] Training completed without errors (check logs)

---

**Last Updated**: 2025-01-23

For issues or questions, please check the troubleshooting section or review the training logs.