# ============================================
# FLASK API
# ============================================

from flask import Flask, request, jsonify
import numpy as np
import joblib
from pathlib import Path

# Load model & scaler from project directory
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

app = Flask(__name__)

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

        attendance = data['attendance_percentage']
        completed = data['tasks_completed']
        pending = data['tasks_pending']
        avg_score = data['avg_task_score']
        mentor = data['mentor_feedback_score']
        comm = data['communication_score']
        team = data['teamwork_score']
        punctual = data['punctuality_score']
        learning = data['learning_progress']

        # Feature Engineering
        total_tasks = completed + pending
        task_rate = completed / total_tasks if total_tasks else 0
        perf_score = (avg_score + mentor + comm + team + punctual + learning) / 6

        features = np.array([
            attendance, completed, pending,
            avg_score, mentor, comm,
            team, punctual, learning,
            task_rate, perf_score
        ]).reshape(1, -1)

        # Scale
        features = scaler.transform(features)

        # Predict
        pred = model.predict(features)[0]

        labels = ["Average", "Good", "Poor"]

        return jsonify({"prediction": labels[pred]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)