# Fake News Detection Project

This project detects whether a news article is fake or real using Natural Language Processing and classical machine learning. It includes a complete Jupyter notebook workflow, a modular training pipeline, saved model artifacts, evaluation reports, a CLI predictor, and a Streamlit web app.

## Project Overview

- Dataset size: `44,898` news articles
- Classes: `Fake` and `True`
- Main notebook: `fake_news_detection_complete.ipynb`
- Best model: `Linear SVC`
- Test accuracy: `99.91%`
- Test F1-score: `99.91%`
- Test ROC-AUC: `0.99999`

## What This Project Covers

- Data loading from `Fake.csv` and `True.csv`
- Data cleaning for null values, duplicates, and blank text
- Feature engineering using combined text, word count, character count, title length, and keyword flags
- Exploratory Data Analysis with multiple graphs
- TF-IDF vectorization with unigram and bigram features
- Model comparison across multiple classifiers
- Final model evaluation with confusion matrix and classification report
- Prediction on custom news articles
- Deployment-ready Streamlit app

## Models Compared

- `LinearSVC`
- `LogisticRegression`
- `SGDClassifier`
- `MultinomialNB`

The project selects the best model based on validation performance, then evaluates it on the test set.

## Results

### Best Validation Model

- Model: `linear_svc`
- Validation accuracy: `99.88%`
- Validation F1-score: `99.89%`
- Validation ROC-AUC: `0.99999`

### Final Test Metrics

- Accuracy: `99.91%`
- Precision: `99.94%`
- Recall: `99.89%`
- F1-score: `99.91%`
- ROC-AUC: `0.99999`

## Project Structure

```text
fake news detection project/
|-- fake_news_detection_complete.ipynb
|-- main.ipynb
|-- Fake.csv
|-- True.csv
|-- app.py
|-- requirements.txt
|-- README.md
|-- artifacts/
|   `-- best_model.joblib
|-- reports/
|   |-- summary.json
|   |-- model_comparison.csv
|   |-- classification_report.txt
|   |-- confusion_matrix.csv
|   |-- dataset_profile.csv
|   |-- top_fake_terms.csv
|   `-- top_true_terms.csv
|-- notebooks/
`-- src/
    |-- data_utils.py
    |-- predict.py
    `-- train.py
```

## Notebook Workflow

The notebook `fake_news_detection_complete.ipynb` walks through the full project step by step:

1. Import libraries
2. Load both datasets
3. Merge fake and true news
4. Visualize class distribution
5. Check null values
6. Clean dataset
7. Create text-based features
8. Perform EDA on word counts, title length, and categories
9. Train TF-IDF + ML models
10. Compare model performance
11. Analyze confusion matrix and class metrics
12. Show top fake and real terms
13. Predict custom news with fake/real percentages
14. Build a final summary dashboard

## Installation

```bash
python -m pip install -r requirements.txt
```

## How to Run

### 1. Train the model

```bash
python src/train.py
```

This will:

- preprocess the dataset
- split it into train, validation, and test sets
- train multiple models
- save the best model in `artifacts/`
- generate reports in `reports/`

### 2. Predict from the command line

```bash
python src/predict.py --title "Breaking headline" --subject "politics" --text "Full news article goes here"
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- NLP with TF-IDF

## Key Learning Outcomes

- Real-world text preprocessing
- NLP feature extraction
- Classical machine learning model comparison
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Turning a notebook project into a reusable ML application

## Future Improvements

- Add deep learning models such as LSTM or BERT
- Improve text preprocessing with stemming or lemmatization
- Add model explainability dashboards
- Deploy the app online
- Add API support for predictions

## Author

**Saqib Ali**  
BS Computer Science Student  
Interested in Web Development, AI, and Data Science
