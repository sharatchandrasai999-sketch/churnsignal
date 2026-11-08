# churnsignal

Small churn-prediction tool built with scikit-learn. It trains a
logistic-regression model on telecom-style customer data and predicts
whether a customer is likely to churn.

**Data note:** by default the training data is *synthetic*, generated with
`sklearn.datasets.make_classification` (fixed seed, so results are
reproducible). It is not real customer data. You can also train on your own
CSV with `--data` (columns: `tenure_months, monthly_charge, total_calls,
support_tickets, churn`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Train a model (synthetic data, 500 rows):

```bash
churnsignal train --rows 500 --out model.joblib
```

Train on your own CSV:

```bash
churnsignal train --data customers.csv --out model.joblib
```

Predict churn for one customer:

```bash
churnsignal predict --tenure-months 12 --monthly-charge 65.0 \
    --total-calls 40 --support-tickets 2
```

## Example output

```
$ churnsignal train --rows 500 --out model.joblib
Training on 500 synthetic rows (400 train / 100 test)
accuracy:  0.7000
precision: 0.6600
recall:    0.7174
Saved model to model.joblib

$ churnsignal predict --tenure-months 12 --monthly-charge 65.0 --total-calls 40 --support-tickets 2
Churn probability: 0.45
Prediction: no churn
```

Metrics are measured on the held-out 20% test split, not hardcoded —
your numbers will vary slightly with `--rows`.

## How it works

- Features: `tenure_months`, `monthly_charge`, `total_calls`,
  `support_tickets` (kept in one shared `FEATURES` list so train and
  predict always agree on order).
- Model: `Pipeline(StandardScaler + LogisticRegression)`, 80/20 split,
  `random_state=42`.

## Tests

```bash
pytest
```
