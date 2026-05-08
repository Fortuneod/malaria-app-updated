"""
app.py — Malaria Severity Prediction API
=========================================
REST API built with Flask. Loads the trained pipeline once at startup,
then serves predictions at POST /predict.

Endpoints:
  GET  /health    — liveness check
  GET  /info      — model metadata and expected features
  POST /predict   — predict severe malaria for one patient
  GET  /stats     — live request statistics

Run locally:
  python app.py

Test with curl:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":5,"sex":1,"fever":1,"cold":1,"rigor":1,"fatigue":1,
         "headache":1,"bitter_tongue":0,"vomitting":0,"diarrhea":1,
         "convulsion":0,"Anemia":0,"jaundice":0,"cocacola_urine":0,
         "hypoglycemia":1,"prostration":0,"hyperpyrexia":0}'
"""

import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from datetime import UTC, datetime

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── App & model ───────────────────────────────────────────────────────────────
app      = Flask(__name__)
pipeline = joblib.load("model/pipeline.joblib")
FEATURES = joblib.load("model/features.joblib")

# ── Live stats tracker ────────────────────────────────────────────────────────
stats = {
    "total_requests":  0,
    "severe_predicted": 0,
    "not_severe_predicted": 0,
    "errors": 0,
    "started_at": datetime.now(UTC).isoformat() + "Z"
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Liveness check — returns 200 if the server is up."""
    return jsonify({"status": "ok", "model_loaded": True}), 200


@app.route("/info")
def info():
    """Returns model metadata and the list of expected input features."""
    return jsonify({
        "model":       "Malaria Severity Classifier",
        "version":     "1.0.0",
        "description": "Predicts whether a patient has severe malaria based on age and symptoms.",
        "features":    FEATURES,
        "target":      "severe_maleria  (0 = not severe, 1 = severe)",
        "endpoint":    "POST /predict  — send JSON with all features listed above"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with patient features.
    Returns the prediction (0 or 1), label, and probability.
    """
    stats["total_requests"] += 1

    # ── Parse input ──────────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if body is None:
        stats["errors"] += 1
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # ── Validate all features are present ────────────────────────────────────
    missing = [f for f in FEATURES if f not in body]
    if missing:
        stats["errors"] += 1
        return jsonify({
            "error":   "Missing features in request.",
            "missing": missing,
            "required": FEATURES
        }), 422

    # ── Build feature vector ──────────────────────────────────────────────────
    try:
        X = np.array([[float(body[f]) for f in FEATURES]])
    except (ValueError, TypeError) as e:
        stats["errors"] += 1
        return jsonify({"error": f"Feature values must be numeric. Detail: {str(e)}"}), 422

    # ── Predict ───────────────────────────────────────────────────────────────
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])
    label = "Severe Malaria" if prediction == 1 else "Not Severe Malaria"

    if prediction == 1:
        stats["severe_predicted"] += 1
    else:
        stats["not_severe_predicted"] += 1

    logger.info(f"Prediction → {label}  (prob={probability:.3f})  | input={body}")

    return jsonify({
        "prediction":        prediction,
        "label":             label,
        "probability_severe": round(probability, 4),
        "severity_risk":     (
            "HIGH"   if probability >= 0.70 else
            "MEDIUM" if probability >= 0.40 else
            "LOW"
        )
    }), 200


@app.route("/stats")
def get_stats():
    """Returns live prediction statistics since server start."""
    return jsonify(stats), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Malaria Severity Prediction API on port 3100 ...")
    app.run(host="0.0.0.0", port=3100, debug=False)
