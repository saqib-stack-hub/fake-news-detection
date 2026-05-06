# Fake News Detection Project

This project upgrades the basic dataset folder into an advanced end-to-end data science project for fake news detection. It uses NLP preprocessing, TF-IDF feature engineering, model comparison, evaluation reports, saved artifacts, and a deployable prediction interface.

## Project Highlights

- Clean modular code inside `src/`
- Training pipeline with train/validation/test split
- Comparison of `LogisticRegression`, `LinearSVC`, `MultinomialNB`, and `SGDClassifier`
- Evaluation outputs including F1 score, ROC-AUC, confusion matrix, and classification report
- Explainability files with top words associated with fake and true news
- CLI prediction script for new articles
- Streamlit app for demo/deployment

## Project Structure

```text
fake news detection project/
|-- Fake.csv
|-- True.csv
|-- app.py
|-- requirements.txt
|-- README.md
|-- artifacts/
|-- reports/
|-- notebooks/
`-- src/
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

This command will:

- load and merge `Fake.csv` and `True.csv`
- clean and engineer text features
- train multiple models
- select the best validation model
- evaluate it on the test set
- save outputs in `artifacts/` and `reports/`

## Predict on New Input

```bash
python src/predict.py --title "Breaking headline" --subject "politics" --text "Full article content goes here"
```

## Run the App

```bash
streamlit run app.py
```

## Output Files

- `artifacts/best_model.joblib`: trained production artifact
- `reports/model_comparison.csv`: validation leaderboard
- `reports/summary.json`: project summary and metrics
- `reports/dataset_profile.csv`: label-wise text statistics
- `reports/classification_report.txt`: full per-class evaluation
- `reports/confusion_matrix.csv`: error distribution
- `reports/top_fake_terms.csv`: strongest fake-news indicators
- `reports/top_true_terms.csv`: strongest true-news indicators

## Suggested Viva / Presentation Points

- Why TF-IDF works well for classical NLP classification
- Why F1 score is more useful than only accuracy
- Difference between validation metrics and final test metrics
- How explainability helps build trust in ML predictions
- How this project can be extended with transformers like BERT in a future version
