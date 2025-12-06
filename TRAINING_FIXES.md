# Training Pipeline Fixes - ML & DL Accuracy Issues

## Problems Identified

### 1. **ML Models Showing Same Accuracy**

**Root Causes:**
- All models were evaluated on the same test set (`X_test_ml, y_test_int`)
- Resume logic skipped evaluation entirely, leaving `ml_val_scores` empty
- No proper train/validation split for model selection
- If classes were imbalanced, all models might predict the majority class

**Impact:** All models appeared to have identical (often low) accuracy because:
- They were evaluated on identical test data
- Resume mode didn't re-evaluate models, showing missing scores
- No differentiation between model performance

### 2. **DL Models Have Very Low Accuracy**

**Root Causes:**
- **Data Leakage**: Using test set as validation set (lines 213, 233)
- **Too Few Epochs**: Default 5 epochs is way too low for complex CNNs
- **Early Stopping Never Triggers**: Patience of 15 with only 5 epochs
- **Overfitting on Test Set**: Models indirectly optimized on test data

**Impact:** Models couldn't learn properly and showed artificially low accuracy

## Fixes Applied

### 1. **Proper Train/Validation/Test Split**

```python
# Before: Used test set for validation (data leakage!)
validation_data=(X_test_dl, y_test)

# After: Created proper 80/20 split from training data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train_int
)
```

**Benefits:**
- No data leakage - test set remains untouched until final evaluation
- Proper model selection on validation set
- More realistic accuracy estimates

### 2. **Fixed ML Model Evaluation**

**Changes:**
- All models now evaluated on validation set (not test) for model selection
- Resume logic now re-evaluates models on validation set
- Each model gets unique validation scores
- Class distribution logging added for diagnosis

```python
# Before: Resume skipped evaluation
if args.resume and os.path.exists(path):
    ml_models[name] = joblib.load(path)
    continue  # No evaluation!

# After: Resume re-evaluates
if args.resume and os.path.exists(path):
    ml_models[name] = joblib.load(path)
    acc = ml_models[name].score(X_val_ml, y_val_split_int)  # Re-evaluate
    ml_val_scores[name] = acc
```

### 3. **Fixed DL Model Training**

**Changes:**
- Use validation split for training monitoring (no test set leakage)
- Increased default epochs from 5 to 50
- Dynamic early stopping patience (scales with epochs)
- Proper normalization using training split statistics only

```python
# Before: 5 epochs, patience 15 (never triggers)
epochs=5
patience=15  # Won't trigger!

# After: 50 epochs, dynamic patience
epochs=50  # Default increased
early_stop_patience = max(5, min(EPOCHS // 3, 15))  # Scales with epochs
```

### 4. **Added Class Distribution Logging**

```python
print("\n📊 Class Distribution:")
train_dist = Counter(y_train_int)
test_dist = Counter(y_test_int)
print(f"Train: {dict(train_dist)}")
print(f"Test: {dict(test_dist)}")
```

This helps diagnose if class imbalance is causing all models to predict the same class.

### 5. **Improved Normalization**

- Normalization statistics computed only from training split
- Applied consistently to validation and test sets
- Prevents data leakage through statistics

### 6. **Better Evaluation Strategy**

- **Validation Set**: Used for model selection and hyperparameter tuning
- **Test Set**: Only used for final evaluation (reported in logs)
- Both validations and test scores are reported for transparency

## Expected Results After Fixes

### ML Models
- **Different accuracies** for each model (SVM, RF, KNN, XGBoost)
- **Realistic scores** based on validation performance
- **Proper model selection** for ensemble weighting

### DL Models
- **Higher accuracy** with proper training duration (50 epochs)
- **Better convergence** with appropriate early stopping
- **No overfitting** on test set

### Overall
- **Training**: Uses training split (80% of training data)
- **Validation**: Uses validation split (20% of training data) for model selection
- **Test**: Uses test set only for final evaluation

## How to Train With Fixes

```bash
cd backend/src

# Basic training (uses new defaults: 50 epochs)
python3 train_pipeline.py --normalize

# Full training with more data
python3 train_pipeline.py --limit 5000 --epochs 50 --normalize --batch_size 64

# Resume training (now properly re-evaluates)
python3 train_pipeline.py --resume --normalize
```

## Key Changes Summary

| Issue | Before | After |
|-------|--------|-------|
| Validation Set | Test set (data leakage) | 20% split from training data |
| ML Evaluation | Same test set, resume skipped | Validation set, resume re-evaluates |
| DL Epochs | 5 (too few) | 50 (default) |
| Early Stopping | Never triggers (patience 15, epochs 5) | Dynamic (scales with epochs) |
| Normalization | Leaked test stats | Training-only statistics |
| Class Distribution | Not shown | Logged for diagnosis |

## Verification

After training, you should see:
1. ✅ Different validation accuracies for each ML model
2. ✅ Increasing DL model accuracy over epochs
3. ✅ Early stopping actually triggering when appropriate
4. ✅ Class distribution printed for diagnosis
5. ✅ Separate validation and test scores reported

## Notes

- **Retraining Required**: Old models were trained with data leakage, so retraining is necessary
- **Default Epochs**: Changed from 5 to 50 for better accuracy (can override with `--epochs`)
- **Validation Set**: 20% of training data is now reserved for validation
- **Test Set**: Only used for final evaluation, not during training

