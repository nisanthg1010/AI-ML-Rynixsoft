# ============================================
# TRAIN & SAVE MODEL
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("intern_performance_500_dataset.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode target labels
le = LabelEncoder()
df['final_performance'] = le.fit_transform(df['final_performance'])

# Feature Engineering
df['task_completion_rate'] = df['tasks_completed'] / (df['tasks_completed'] + df['tasks_pending'])

df['performance_score'] = (
    df['avg_task_score'] +
    df['mentor_feedback_score'] +
    df['communication_score'] +
    df['teamwork_score'] +
    df['punctuality_score'] +
    df['learning_progress']
) / 6

df.replace([np.inf, -np.inf], 0, inplace=True)

# Features & target
# Keep feature set aligned with API fields by excluding department.
X = df.drop(['intern_id', 'name', 'department', 'final_performance'], axis=1)
y = df['final_performance']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved!")