"""Core churn model: synthetic data generation, training, prediction."""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["tenure_months", "monthly_charge", "total_calls", "support_tickets"]
TARGET = "churn"


def generate_synthetic_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate a small synthetic telecom-style dataset.

    make_classification produces 4 informative features, which we rescale
    into plausible ranges for each column. This is synthetic data, not real
    customer records.
    """
    X, y = make_classification(
        n_samples=rows,
        n_features=len(FEATURES),
        n_informative=4,
        n_redundant=0,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    # Rescale each column into a plausible range for its meaning.
    df["tenure_months"] = np.clip((df["tenure_months"] * 12 + 24).round(), 1, 72)
    df["monthly_charge"] = np.clip((df["monthly_charge"] * 15 + 60).round(2), 20, 120)
    df["total_calls"] = np.clip((df["total_calls"] * 20 + 60).round(), 0, 200)
    df["support_tickets"] = np.clip((df["support_tickets"] * 2 + 2).round(), 0, 12)
    df[TARGET] = y
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV with the expected feature + target columns."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    return df[FEATURES + [TARGET]]


def build_model() -> Pipeline:
    """StandardScaler + LogisticRegression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(df: pd.DataFrame, random_state: int = 42) -> tuple[Pipeline, dict]:
    """Train on 80% of the data, return the model and held-out metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=random_state
    )
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, metrics


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_churn(model: Pipeline, features: dict) -> tuple[float, bool]:
    """Return (churn probability, churn yes/no) for one customer."""
    row = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(row)[0][1])
    return prob, prob >= 0.5
