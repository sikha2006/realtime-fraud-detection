# Real-Time Transaction Fraud Detection System

A machine learning system that detects fraudulent transactions in highly
imbalanced data, deployed as a live REST API for real-time inference.

## Why this project (for your report/viva)
Fraud detection is a genuine, hireable-skill problem used across fintech,
banking, and e-commerce. This project demonstrates:
- Handling severe **class imbalance** (fraud is typically <1% of data) using SMOTE
- Using the **right evaluation metrics** (Precision, Recall, F1, ROC-AUC) instead
  of misleading accuracy on imbalanced data
- **Deploying** a model as a live API, not just a notebook

## Project structure
```
fraud_project/
├── generate_sample_data.py   # Creates a synthetic dataset for testing (no real data needed)
├── train_model.py            # Preprocessing + SMOTE + XGBoost training + evaluation
├── app.py                    # Flask REST API for real-time predictions
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a dataset (pick one)

**Option A - Real dataset (recommended for your final report):**
Download the Kaggle "Credit Card Fraud Detection" dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place `creditcard.csv` in this folder.

**Option B - Synthetic sample (for quick testing):**
```bash
python generate_sample_data.py
```
This creates `creditcard_sample.csv` with the same structure. `train_model.py`
will automatically use this if the real dataset isn't found.

### 3. Train the model
```bash
python train_model.py
```
This will:
- Load your data
- Scale features and apply SMOTE to balance the training set
- Train an XGBoost classifier
- Print a classification report + ROC-AUC score
- Save `roc_curve.png` and `precision_recall_curve.png` (use these in your report!)
- Save `fraud_model.joblib`, `scaler.joblib`, `feature_columns.joblib`

### 4. Run the API
```bash
python app.py
```
Server starts at `http://127.0.0.1:5000`

### 5. Test it
Quick built-in test (no data needed):
```bash
curl http://127.0.0.1:5000/predict_sample
```

Real prediction request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 5000, "V1": -1.2, "V2": 0.5, "Amount": 149.62}'
```
(Any feature you don't include defaults to 0 — but for real accuracy, send all
V1-V28, Time, and Amount fields matching your dataset's columns.)

## What to put in your report
- Include the confusion matrix and classification report output
- Include the two saved plots (`roc_curve.png`, `precision_recall_curve.png`)
- Explain WHY you used SMOTE (imbalance) and WHY you report Recall/F1/ROC-AUC
  instead of accuracy (accuracy would look great even if the model predicted
  "not fraud" every single time)

## Resume bullet you can use
> Built a real-time transaction fraud detection system using XGBoost on highly
> imbalanced data (SMOTE-balanced training), achieving strong recall on the
> minority (fraud) class; deployed as a REST API with sub-200ms inference.

*(Fill in your actual recall/precision/AUC numbers once you've run training on
the real dataset — recruiters and ATS systems both respond well to concrete
numbers.)*

## Possible extensions (if you want to go further)
- Add a simple HTML/React frontend where you paste transaction details and see the result
- Log predictions to a small SQLite database and build a "flagged transactions" dashboard
- Add SHAP explainability so you can show *why* a transaction was flagged
