# ============================================
# TRAIN / RETRAIN & SAVE MODEL
# ============================================

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TARGET_COL = "final_performance"


def load_data(primary_data: Path, additional_data: Path | None) -> pd.DataFrame:
    df = pd.read_csv(primary_data)
    if additional_data:
        new_df = pd.read_csv(additional_data)
        df = pd.concat([df, new_df], ignore_index=True)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must include target column: {TARGET_COL}")

    # Fill missing values before encoding and feature engineering.
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    label_encoder = LabelEncoder()
    df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])

    denominator = df["tasks_completed"] + df["tasks_pending"]
    df["task_completion_rate"] = np.where(denominator == 0, 0, df["tasks_completed"] / denominator)
    df["performance_score"] = (
        df["avg_task_score"]
        + df["mentor_feedback_score"]
        + df["communication_score"]
        + df["teamwork_score"]
        + df["punctuality_score"]
        + df["learning_progress"]
    ) / 6
    df.replace([np.inf, -np.inf], 0, inplace=True)

    drop_cols = ["intern_id", "name", "department", TARGET_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_COL]
    return X, y, label_encoder


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, StandardScaler, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return model, scaler, metrics


def build_metadata(X: pd.DataFrame, row_count: int) -> dict:
    return {
        "rows_used": int(row_count),
        "feature_columns": list(X.columns),
        "feature_means": {k: float(v) for k, v in X.mean().to_dict().items()},
        "feature_stds": {k: float(v) for k, v in X.std(ddof=0).replace(0, 1e-8).to_dict().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or retrain intern performance model")
    parser.add_argument("--data", default="intern_performance_500_dataset.csv", help="Primary CSV dataset path")
    parser.add_argument("--new-data", default=None, help="Optional additional CSV to append for retraining")
    parser.add_argument("--model-path", default="model.pkl")
    parser.add_argument("--scaler-path", default="scaler.pkl")
    parser.add_argument("--label-encoder-path", default="label_encoder.pkl")
    parser.add_argument("--metrics-path", default="sprint5_model_results.json")
    parser.add_argument("--metadata-path", default="model_metadata.json")
    args = parser.parse_args()

    data_path = (BASE_DIR / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    new_data_path = None
    if args.new_data:
        new_data_path = (BASE_DIR / args.new_data).resolve() if not Path(args.new_data).is_absolute() else Path(args.new_data)

    df = load_data(data_path, new_data_path)
    X, y, label_encoder = preprocess(df)
    model, scaler, metrics = train_and_evaluate(X, y)
    metadata = build_metadata(X, row_count=len(df))

    model_path = BASE_DIR / args.model_path
    scaler_path = BASE_DIR / args.scaler_path
    label_encoder_path = BASE_DIR / args.label_encoder_path
    metrics_path = BASE_DIR / args.metrics_path
    metadata_path = BASE_DIR / args.metadata_path

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved label encoder to: {label_encoder_path}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()