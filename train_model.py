"""
train_model.py

End-to-end training pipeline for a real-time transaction fraud detection
system. Steps:
    1. Load transaction data (real Kaggle dataset, or the synthetic sample)
    2. Preprocess & scale features
    3. Handle severe class imbalance with SMOTE
    4. Train an XGBoost classifier
    5. Evaluate with metrics that matter for fraud (Precision, Recall, F1, ROC-AUC)
       -- NOT plain accuracy, which is misleading on imbalanced data
    6. Save the trained model + scaler for the Flask API to use

Run:
    python train_model.py
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH_REAL = "creditcard.csv"
DATA_PATH_SAMPLE = "creditcard_sample.csv"
MODEL_PATH = "fraud_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "feature_columns.joblib"


def load_data() -> pd.DataFrame:
    """Load the real dataset if present, else fall back to the synthetic sample."""
    if os.path.exists(DATA_PATH_REAL):
        print(f"Loading real dataset: {DATA_PATH_REAL}")
        return pd.read_csv(DATA_PATH_REAL)
    elif os.path.exists(DATA_PATH_SAMPLE):
        print(f"Real dataset not found. Loading synthetic sample: {DATA_PATH_SAMPLE}")
        print("NOTE: For your actual report, download the real Kaggle dataset:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return pd.read_csv(DATA_PATH_SAMPLE)
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python generate_sample_data.py` first, "
            "or place the real `creditcard.csv` in this folder."
        )


def preprocess(df: pd.DataFrame):
    """Split features/target and scale numeric columns."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler, list(X.columns)


def train():
    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}% of total)\n")

    X, y, scaler, feature_columns = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Before SMOTE -> Train class distribution:\n{y_train.value_counts()}\n")

    # SMOTE only on training data (never touch the test set - avoids data leakage)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE -> Train class distribution:\n{pd.Series(y_train_res).value_counts()}\n")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
    )

    print("Training XGBoost model...")
    start = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s\n")

    # ---- Evaluation ----
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("CLASSIFICATION REPORT (on untouched, real-world-like test set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3, target_names=["Legit", "Fraud"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}\n")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted Legit   Predicted Fraud")
    print(f"Actual Legit     {cm[0][0]:<17} {cm[0][1]}")
    print(f"Actual Fraud     {cm[1][0]:<17} {cm[1][1]}")

    # ---- Save plots (useful for your project report) ----
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()
    print("\nSaved ROC curve plot -> roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Fraud Detection Model")
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=150)
    plt.close()
    print("Saved Precision-Recall plot -> precision_recall_curve.png")

    # ---- Persist model artifacts for the Flask API ----
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved scaler     -> {SCALER_PATH}")
    print(f"Saved feature list -> {FEATURES_PATH}")

    return model, scaler, feature_columns, auc


if __name__ == "__main__":
    train()
