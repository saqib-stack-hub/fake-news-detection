from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_utils import load_dataset, split_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("model", LogisticRegression(max_iter=2000, n_jobs=1)),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("model", LinearSVC(C=1.0)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("model", MultinomialNB(alpha=0.3)),
            ]
        ),
        "sgd_classifier": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("model", SGDClassifier(loss="log_loss", random_state=42)),
            ]
        ),
    }


def get_score_signal(model: Pipeline, texts: pd.Series) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def evaluate_model(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(texts)
    scores = get_score_signal(model, texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, scores),
    }


def extract_top_terms(model: Pipeline, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        empty = pd.DataFrame(columns=["term", "weight"])
        return empty, empty

    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = estimator.coef_[0]
    fake_indices = np.argsort(coefficients)[-top_n:][::-1]
    true_indices = np.argsort(coefficients)[:top_n]

    fake_terms = pd.DataFrame(
        {"term": feature_names[fake_indices], "weight": coefficients[fake_indices]}
    )
    true_terms = pd.DataFrame(
        {"term": feature_names[true_indices], "weight": coefficients[true_indices]}
    )
    return fake_terms, true_terms


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    dataset = load_dataset(PROJECT_DIR)
    bundle = split_dataset(dataset)

    dataset_overview = {
        "rows": int(len(dataset)),
        "fake_rows": int(dataset["label"].sum()),
        "true_rows": int((dataset["label"] == 0).sum()),
        "avg_word_count": round(float(dataset["word_count"].mean()), 2),
        "median_word_count": round(float(dataset["word_count"].median()), 2),
        "subjects": dataset["subject"].value_counts().head(10).to_dict(),
    }
    dataset.groupby("label")[["word_count", "char_count"]].agg(["mean", "median"]).to_csv(
        REPORTS_DIR / "dataset_profile.csv"
    )

    comparison_rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in build_models().items():
        pipeline.fit(bundle.train_texts, bundle.train_labels)
        metrics = evaluate_model(pipeline, bundle.val_texts, bundle.val_labels)
        comparison_rows.append({"model": model_name, **metrics})
        trained_models[model_name] = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["f1", "roc_auc", "accuracy"], ascending=False
    )
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = str(comparison_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]

    test_predictions = best_model.predict(bundle.test_texts)
    test_scores = get_score_signal(best_model, bundle.test_texts)

    test_metrics = {
        "accuracy": accuracy_score(bundle.test_labels, test_predictions),
        "precision": precision_score(bundle.test_labels, test_predictions),
        "recall": recall_score(bundle.test_labels, test_predictions),
        "f1": f1_score(bundle.test_labels, test_predictions),
        "roc_auc": roc_auc_score(bundle.test_labels, test_scores),
    }

    summary = {
        "best_model": best_model_name,
        "validation_leaderboard": comparison_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "dataset_overview": dataset_overview,
    }

    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    report = classification_report(
        bundle.test_labels,
        test_predictions,
        target_names=["true", "fake"],
        digits=4,
    )
    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    confusion = confusion_matrix(bundle.test_labels, test_predictions)
    pd.DataFrame(
        confusion,
        index=["actual_true", "actual_fake"],
        columns=["pred_true", "pred_fake"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")

    fake_terms, true_terms = extract_top_terms(best_model)
    fake_terms.to_csv(REPORTS_DIR / "top_fake_terms.csv", index=False)
    true_terms.to_csv(REPORTS_DIR / "top_true_terms.csv", index=False)

    artifact_bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "label_mapping": {"0": "true", "1": "fake"},
    }
    joblib.dump(artifact_bundle, ARTIFACTS_DIR / "best_model.joblib")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
