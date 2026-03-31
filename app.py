# ============================================
# FLASK API
# ============================================

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Load model & scaler from project directory
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
label_encoder_path = BASE_DIR / "label_encoder.pkl"
metadata_path = BASE_DIR / "model_metadata.json"

if label_encoder_path.exists():
    label_encoder = joblib.load(label_encoder_path)
    CLASS_LABELS = [str(item) for item in label_encoder.classes_]
else:
    CLASS_LABELS = ["Average", "Good", "Poor"]

if metadata_path.exists():
    MODEL_METADATA = json.loads(metadata_path.read_text(encoding="utf-8"))
else:
    MODEL_METADATA = {}

FEATURE_COLUMNS = MODEL_METADATA.get(
    "feature_columns",
    [
        "attendance_percentage",
        "tasks_completed",
        "tasks_pending",
        "avg_task_score",
        "mentor_feedback_score",
        "communication_score",
        "teamwork_score",
        "punctuality_score",
        "learning_progress",
        "task_completion_rate",
        "performance_score",
    ],
)

LOGS_DIR = BASE_DIR / "monitoring_logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"
FEEDBACK_LOG = LOGS_DIR / "feedback.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def to_float(data: dict, key: str) -> float:
    try:
        return float(data[key])
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"Invalid numeric field: {key}")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def prepare_features(data: dict) -> tuple[pd.DataFrame, dict]:
    attendance = to_float(data, "attendance_percentage")
    completed = to_float(data, "tasks_completed")
    pending = to_float(data, "tasks_pending")
    avg_score = to_float(data, "avg_task_score")
    mentor = to_float(data, "mentor_feedback_score")
    comm = to_float(data, "communication_score")
    team = to_float(data, "teamwork_score")
    punctual = to_float(data, "punctuality_score")
    learning = to_float(data, "learning_progress")

    total_tasks = completed + pending
    task_rate = completed / total_tasks if total_tasks else 0
    perf_score = (avg_score + mentor + comm + team + punctual + learning) / 6

    feature_dict = {
        "attendance_percentage": attendance,
        "tasks_completed": completed,
        "tasks_pending": pending,
        "avg_task_score": avg_score,
        "mentor_feedback_score": mentor,
        "communication_score": comm,
        "teamwork_score": team,
        "punctuality_score": punctual,
        "learning_progress": learning,
        "task_completion_rate": task_rate,
        "performance_score": perf_score,
    }

    features = pd.DataFrame([[feature_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return features, feature_dict


def estimate_drift(feature_values: dict) -> dict:
    means = MODEL_METADATA.get("feature_means", {})
    stds = MODEL_METADATA.get("feature_stds", {})
    z_scores: dict[str, float] = {}
    for feature_name, live_value in feature_values.items():
        if feature_name not in means or feature_name not in stds:
            continue
        std = stds[feature_name] if stds[feature_name] != 0 else 1e-8
        z = abs((live_value - means[feature_name]) / std)
        z_scores[feature_name] = round(float(z), 4)

    top_drift = sorted(z_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return {
        "drift_score": round(float(np.mean(list(z_scores.values()))) if z_scores else 0.0, 4),
        "top_drift_features": [{"feature": key, "z_score": value} for key, value in top_drift],
    }

@app.route('/')
def home():
    return "🚀 Intern Performance Prediction API is Running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return jsonify({
                "message": "Use POST with JSON body to get a prediction.",
                "required_fields": [
                    "attendance_percentage",
                    "tasks_completed",
                    "tasks_pending",
                    "avg_task_score",
                    "mentor_feedback_score",
                    "communication_score",
                    "teamwork_score",
                    "punctuality_score",
                    "learning_progress"
                ]
            })

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        required_fields = [
            "attendance_percentage",
            "tasks_completed",
            "tasks_pending",
            "avg_task_score",
            "mentor_feedback_score",
            "communication_score",
            "teamwork_score",
            "punctuality_score",
            "learning_progress"
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        raw_features, feature_dict = prepare_features(data)

        # Scale
        features = scaler.transform(raw_features)

        # Predict
        pred = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        prediction_id = str(uuid.uuid4())

        response_payload = {
            "prediction_id": prediction_id,
            "prediction": CLASS_LABELS[pred] if pred < len(CLASS_LABELS) else str(pred),
            "confidence": round(confidence, 4),
            "class_probabilities": {
                CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i): round(float(probabilities[i]), 4)
                for i in range(len(probabilities))
            },
            "behavior_signals": {
                "task_completion_rate": round(feature_dict["task_completion_rate"], 4),
                "performance_score": round(feature_dict["performance_score"], 4),
            },
            "drift_monitor": estimate_drift(feature_dict),
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }

        append_jsonl(
            PREDICTIONS_LOG,
            {
                "prediction_id": prediction_id,
                "prediction_index": pred,
                "prediction": response_payload["prediction"],
                "confidence": response_payload["confidence"],
                "class_probabilities": response_payload["class_probabilities"],
                "features": feature_dict,
                "requested_at": response_payload["requested_at"],
            },
        )

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        prediction_id = data.get("prediction_id")
        actual_performance = data.get("actual_performance")
        if not prediction_id or actual_performance is None:
            return jsonify({"error": "Fields required: prediction_id, actual_performance"}), 400

        predictions = load_jsonl(PREDICTIONS_LOG)
        matched = next((row for row in reversed(predictions) if row.get("prediction_id") == prediction_id), None)
        if not matched:
            return jsonify({"error": "prediction_id not found"}), 404

        predicted_label = matched.get("prediction")
        is_correct = str(predicted_label) == str(actual_performance)

        entry = {
            "prediction_id": prediction_id,
            "predicted": predicted_label,
            "actual": actual_performance,
            "is_correct": is_correct,
            "confidence": matched.get("confidence"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(FEEDBACK_LOG, entry)
        return jsonify({"message": "Feedback recorded", "is_correct": is_correct})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/monitor/performance', methods=['GET'])
def monitor_performance():
    try:
        window = int(request.args.get("window", 100))
        predictions = load_jsonl(PREDICTIONS_LOG)
        recent_predictions = predictions[-window:] if window > 0 else predictions

        if not recent_predictions:
            return jsonify({"message": "No predictions logged yet"})

        confidences = [float(item.get("confidence", 0.0)) for item in recent_predictions]
        low_conf_count = sum(1 for c in confidences if c < 0.55)

        by_label: dict[str, int] = {}
        for row in recent_predictions:
            label = str(row.get("prediction", "unknown"))
            by_label[label] = by_label.get(label, 0) + 1

        return jsonify(
            {
                "window_size": len(recent_predictions),
                "prediction_distribution": by_label,
                "avg_confidence": round(float(np.mean(confidences)), 4),
                "low_confidence_rate": round(low_conf_count / len(recent_predictions), 4),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/monitor/errors', methods=['GET'])
def monitor_errors():
    try:
        feedback_rows = load_jsonl(FEEDBACK_LOG)
        if not feedback_rows:
            return jsonify({"message": "No feedback records yet"})

        total = len(feedback_rows)
        correct = sum(1 for row in feedback_rows if bool(row.get("is_correct")))
        avg_conf = float(np.mean([float(row.get("confidence", 0.0)) for row in feedback_rows]))

        confusion: dict[str, int] = {}
        for row in feedback_rows:
            key = f"{row.get('actual')} -> {row.get('predicted')}"
            confusion[key] = confusion.get(key, 0) + 1

        return jsonify(
            {
                "samples": total,
                "accuracy": round(correct / total, 4),
                "avg_confidence": round(avg_conf, 4),
                "confusion_pairs": confusion,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)