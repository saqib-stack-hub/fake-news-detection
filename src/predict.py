from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from data_utils import normalize_text


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "artifacts" / "best_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fake news from a headline and article body.")
    parser.add_argument("--title", required=True, help="News headline")
    parser.add_argument("--text", required=True, help="News article body")
    parser.add_argument("--subject", default="general", help="Optional category")
    args = parser.parse_args()

    artifact_bundle = joblib.load(MODEL_PATH)
    model = artifact_bundle["model"]

    combined_text = normalize_text(f"{args.title} {args.subject} {args.text}")
    prediction = int(model.predict([combined_text])[0])

    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        score = float(model.predict_proba([combined_text])[0][1])
    else:
        raw_score = float(model.decision_function([combined_text])[0])
        score = 1.0 / (1.0 + pow(2.718281828, -raw_score))

    label = "fake" if prediction == 1 else "true"
    print(f"prediction: {label}")
    print(f"fake_probability: {score:.4f}")


if __name__ == "__main__":
    main()
