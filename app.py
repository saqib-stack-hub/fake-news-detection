from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from src.data_utils import normalize_text


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "best_model.joblib"
REPORT_PATH = PROJECT_DIR / "reports" / "summary.json"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("Fake News Detection System")
st.caption("Advanced NLP pipeline using TF-IDF feature engineering and multiple model comparison.")

left, right = st.columns([2, 1])

with left:
    title = st.text_input("Headline")
    subject = st.text_input("Subject", value="politics")
    article = st.text_area("Article Text", height=280)

    if st.button("Analyze News"):
        if not title.strip() or not article.strip():
            st.warning("Please enter both headline and article text.")
        else:
            artifact_bundle = load_artifacts()
            model = artifact_bundle["model"]
            combined_text = normalize_text(f"{title} {subject} {article}")
            prediction = int(model.predict([combined_text])[0])

            estimator = model.named_steps["model"]
            if hasattr(estimator, "predict_proba"):
                fake_probability = float(model.predict_proba([combined_text])[0][1])
            else:
                raw_score = float(model.decision_function([combined_text])[0])
                fake_probability = 1.0 / (1.0 + pow(2.718281828, -raw_score))

            if prediction == 1:
                st.error(f"Prediction: FAKE NEWS ({fake_probability:.2%} confidence)")
            else:
                st.success(f"Prediction: TRUE NEWS ({1 - fake_probability:.2%} confidence)")

            st.progress(min(max(fake_probability, 0.0), 1.0), text="Fake news likelihood")

with right:
    st.subheader("Project Features")
    st.markdown(
        """
        - Clean train/validation/test workflow
        - Multiple classical NLP models compared
        - Saved artifact for deployment
        - Reusable CLI prediction script
        - Explainability via top weighted terms
        """
    )

    if REPORT_PATH.exists():
        st.subheader("Training Summary")
        st.json(json.loads(REPORT_PATH.read_text(encoding="utf-8")))
