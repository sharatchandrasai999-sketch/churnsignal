"""Tests for churnsignal."""

import os

import pandas as pd
import pytest

from churnsignal.cli import main
from churnsignal.model import (
    FEATURES,
    generate_synthetic_data,
    load_model,
    predict_churn,
    train_model,
)


def test_training_accuracy_above_chance():
    df = generate_synthetic_data(rows=200)
    _, metrics = train_model(df)
    assert metrics["accuracy"] > 0.5
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_predict_probability_in_range(tmp_path):
    df = generate_synthetic_data(rows=200)
    model, _ = train_model(df)
    prob, churn = predict_churn(
        model,
        {"tenure_months": 12, "monthly_charge": 65.0,
         "total_calls": 40, "support_tickets": 2},
    )
    assert 0.0 <= prob <= 1.0
    assert churn == (prob >= 0.5)


def test_predict_missing_model_errors_cleanly(capsys):
    rc = main(["predict", "--tenure-months", "12", "--monthly-charge", "65.0",
               "--total-calls", "40", "--support-tickets", "2",
               "--model", "no-such-model.joblib"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert "train" in err


def test_train_with_provided_csv(tmp_path, capsys):
    csv_path = tmp_path / "customers.csv"
    df = generate_synthetic_data(rows=120)
    df.to_csv(csv_path, index=False)
    out = tmp_path / "model.joblib"
    rc = main(["train", "--data", str(csv_path), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "accuracy" in capsys.readouterr().out


def test_saved_model_loads_and_predicts(tmp_path):
    df = generate_synthetic_data(rows=200)
    model, _ = train_model(df)
    path = tmp_path / "model.joblib"
    from churnsignal.model import save_model

    save_model(model, str(path))
    loaded = load_model(str(path))
    prob, _ = predict_churn(
        loaded,
        {"tenure_months": 6, "monthly_charge": 90.0,
         "total_calls": 10, "support_tickets": 5},
    )
    assert 0.0 <= prob <= 1.0


def test_cli_train_and_predict_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["train", "--rows", "200", "--out", "model.joblib"]) == 0
    assert main(["predict", "--tenure-months", "12", "--monthly-charge", "65.0",
                 "--total-calls", "40", "--support-tickets", "2"]) == 0
    out = capsys.readouterr().out
    assert "Churn probability" in out
    assert "Prediction" in out


def test_synthetic_data_has_expected_columns():
    df = generate_synthetic_data(rows=50)
    assert list(df.columns) == FEATURES + ["churn"]
    assert set(df["churn"].unique()) <= {0, 1}
