"""Command-line interface for churnsignal."""

import argparse
import os
import sys

from churnsignal.model import (
    FEATURES,
    generate_synthetic_data,
    load_data,
    load_model,
    predict_churn,
    save_model,
    train_model,
)

DEFAULT_MODEL = "model.joblib"


def _train(args: argparse.Namespace) -> int:
    if args.data:
        df = load_data(args.data)
        print(f"Training on {len(df)} rows from {args.data}")
    else:
        df = generate_synthetic_data(rows=args.rows)
        print(f"Training on {len(df)} synthetic rows (fixed seed)")
    model, metrics = train_model(df)
    print(f"train: {metrics['n_train']} rows, test: {metrics['n_test']} rows")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    save_model(model, args.out)
    print(f"Saved model to {args.out}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    if not os.path.exists(args.model):
        print(
            f"error: model file '{args.model}' not found. "
            "Run 'churnsignal train' first to create it.",
            file=sys.stderr,
        )
        return 1
    model = load_model(args.model)
    features = {
        "tenure_months": args.tenure_months,
        "monthly_charge": args.monthly_charge,
        "total_calls": args.total_calls,
        "support_tickets": args.support_tickets,
    }
    prob, churn = predict_churn(model, features)
    print(f"Churn probability: {prob:.2f}")
    print(f"Prediction: {'churn' if churn else 'no churn'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churnsignal", description="Predict customer churn with scikit-learn."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a churn model")
    p_train.add_argument("--rows", type=int, default=500,
                         help="Rows of synthetic data to generate (default: 500)")
    p_train.add_argument("--out", default=DEFAULT_MODEL,
                         help="Where to save the model (default: model.joblib)")
    p_train.add_argument("--data", default=None,
                         help="Optional CSV to train on instead of synthetic data")
    p_train.set_defaults(func=_train)

    p_pred = sub.add_parser("predict", help="Predict churn for one customer")
    p_pred.add_argument("--tenure-months", type=int, required=True)
    p_pred.add_argument("--monthly-charge", type=float, required=True)
    p_pred.add_argument("--total-calls", type=int, required=True)
    p_pred.add_argument("--support-tickets", type=int, required=True)
    p_pred.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model file (default: model.joblib)")
    p_pred.set_defaults(func=_predict)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
