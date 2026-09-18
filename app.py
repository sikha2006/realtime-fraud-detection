"""
app.py

Flask REST API that serves the trained fraud detection model for
real-time inference. This is what makes your project "deployed", not
just a notebook -- a genuinely strong resume/viva talking point.

Run:
    python app.py

Then test with:
    curl -X POST http://127.0.0.1:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"Time": 5000, "V1": -1.2, "V2": 0.5, ..., "Amount": 149.62}'

Or use the /predict_sample endpoint (no body needed) to sanity-check
that the server + model are working.
"""

import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "fraud_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "feature_columns.joblib"

app = Flask(__name__)

# Load model artifacts once at startup (not per-request -> keeps latency low)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print(f"Model loaded successfully. Expecting {len(feature_columns)} features.")
except FileNotFoundError:
    model, scaler, feature_columns = None, None, None
    print("WARNING: Model files not found. Run `python train_model.py` first.")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Fraud Detection API",
        "status": "running" if model is not None else "model not loaded",
        "endpoints": {
            "/predict": "POST - send a transaction JSON to get a fraud prediction",
            "/predict_sample": "GET - runs a quick built-in sanity check",
            "/health": "GET - health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON body received"}), 400

    # Build a single-row DataFrame in the exact column order the model expects
    try:
        row = {col: data.get(col, 0.0) for col in feature_columns}
        X = pd.DataFrame([row], columns=feature_columns)
    except Exception as e:
        return jsonify({"error": f"Malformed input: {str(e)}"}), 400

    X_scaled = scaler.transform(X)

    start = time.time()
    proba = float(model.predict_proba(X_scaled)[0][1])
    pred = int(proba >= 0.5)
    latency_ms = round((time.time() - start) * 1000, 2)

    return jsonify({
        "is_fraud": bool(pred),
        "fraud_probability": round(proba, 4),
        "risk_level": (
            "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW"
        ),
        "inference_time_ms": latency_ms
    })


@app.route("/predict_sample", methods=["GET"])
def predict_sample():
    """Quick sanity check using a random synthetic transaction."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    rng = np.random.default_rng()
    sample = {col: float(rng.normal(0, 1)) for col in feature_columns}
    X = pd.DataFrame([sample], columns=feature_columns)
    X_scaled = scaler.transform(X)
    proba = float(model.predict_proba(X_scaled)[0][1])

    return jsonify({
        "note": "This used a random synthetic transaction, not a real one.",
        "sample_input": sample,
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(proba >= 0.5)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
