"""Train and evaluate the phishing website Random Forest model.

Example:
    python train_model.py --data phishing.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
FEATURE_FILE = BASE_DIR / "feature_columns.json"
MODEL_FILE = BASE_DIR / "phishing_rf_model.pkl"

with FEATURE_FILE.open(encoding="utf-8") as file:
    FEATURE_COLUMNS = json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the phishing website classifier.")
    parser.add_argument("--data", required=True, help="Path to the phishing dataset CSV file.")
    parser.add_argument(
        "--target",
        default="Result",
        help="Target column name. Default: Result",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def prepare_target(series):
    """Convert common phishing-dataset target encodings to -1 / 1."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = set(numeric.astype(int).unique())
        if values.issubset({-1, 1}):
            return numeric.astype(int)
        if values.issubset({0, 1}):
            return numeric.map({0: -1, 1: 1}).astype(int)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "phishing": -1,
        "legitimate": 1,
        "legit": 1,
        "-1": -1,
        "1": 1,
        "0": -1,
    }
    converted = text.map(mapping)
    if converted.isna().any():
        unknown = sorted(text[converted.isna()].unique())
        raise ValueError(f"Unsupported target values: {unknown}")
    return converted.astype(int)


def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(
            "The dataset is missing required features: "
            + ", ".join(missing_features)
        )

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found.")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = prepare_target(df[args.target])

    invalid_rows = X.isna().any(axis=1) | y.isna()
    if invalid_rows.any():
        removed = int(invalid_rows.sum())
        print(f"Removing {removed} rows with invalid values.")
        X = X.loc[~invalid_rows]
        y = y.loc[~invalid_rows]

    if len(X) < 10:
        raise ValueError("Not enough valid rows to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)

    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Phishing (-1)", "Legitimate (1)"],
        )
    )

    if -1 in classes and 1 in classes:
        positive_index = classes.index(1)
        roc_auc = roc_auc_score(y_test, probabilities[:, positive_index])
        print(f"ROC-AUC (Legitimate=positive class): {roc_auc:.4f}")

    joblib.dump(model, MODEL_FILE)
    with FEATURE_FILE.open("w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)

    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved feature order to: {FEATURE_FILE}")


if __name__ == "__main__":
    main()
