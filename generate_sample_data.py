"""
generate_sample_data.py

Generates a synthetic, imbalanced transaction dataset that mimics the
structure of the popular Kaggle "Credit Card Fraud Detection" dataset
(Time, V1-V28 PCA-like features, Amount, Class).

Use this ONLY if you don't have the real dataset yet, so you can test
the pipeline end-to-end. For your actual project/report, download the
real dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it as `creditcard.csv` in this folder (same column names:
Time, V1...V28, Amount, Class).
"""

import numpy as np
import pandas as pd

def generate_sample_data(n_samples: int = 20000, fraud_ratio: float = 0.0017, seed: int = 42):
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    n_features = 28  # mimics V1..V28

    # Legit transactions: centered around 0, tighter spread
    legit = rng.normal(loc=0, scale=1.0, size=(n_legit, n_features))
    # Fraudulent transactions: shifted mean + higher variance (mimics anomalies)
    fraud = rng.normal(loc=1.5, scale=3.0, size=(n_fraud, n_features))

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, n_features + 1)])
    df["Time"] = np.sort(rng.integers(0, 172800, size=len(df)))  # 2 days in seconds
    df["Amount"] = np.round(np.abs(rng.normal(loc=88, scale=250, size=len(df))), 2)
    df["Class"] = y

    # Reorder columns to match the real dataset's layout
    cols = ["Time"] + [f"V{i}" for i in range(1, n_features + 1)] + ["Amount", "Class"]
    df = df[cols]
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv("creditcard_sample.csv", index=False)
    print(f"Saved synthetic dataset: creditcard_sample.csv")
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
