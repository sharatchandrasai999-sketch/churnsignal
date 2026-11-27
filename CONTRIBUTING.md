# Contributing

Thanks for helping. Small churn-prediction CLI built with scikit-learn. This is a small, personal project, so keep changes small and easy to review.

## Before you send a change

- Run the tests from any folder: `pytest -q` (tests must not depend on the current working directory)
- Keep the offline path working: training and prediction on synthetic data must need no API key and no download.
- If you change the model, keep the held-out accuracy honestly measured, not hardcoded.

## Reporting a problem

Open an issue and include:

- The exact command you ran
- The input file, if it is about one
- What happened, and what you expected instead

Do not include API keys, private documents, or personal details.
