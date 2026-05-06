from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetBundle:
    train_texts: pd.Series
    val_texts: pd.Series
    test_texts: pd.Series
    train_labels: pd.Series
    val_labels: pd.Series
    test_labels: pd.Series
    full_dataframe: pd.DataFrame


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _prepare_frame(df: pd.DataFrame, label: int) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["label"] = label
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["title"] = cleaned["title"].fillna("")
    cleaned["text"] = cleaned["text"].fillna("")
    cleaned["subject"] = cleaned["subject"].fillna("unknown")
    cleaned["content"] = (
        cleaned["title"].map(normalize_text)
        + " "
        + cleaned["subject"].map(normalize_text)
        + " "
        + cleaned["text"].map(normalize_text)
    )
    cleaned["word_count"] = cleaned["content"].str.split().str.len()
    cleaned["char_count"] = cleaned["content"].str.len()
    return cleaned


def load_dataset(project_dir: Path) -> pd.DataFrame:
    fake_df = pd.read_csv(project_dir / "Fake.csv")
    true_df = pd.read_csv(project_dir / "True.csv")

    combined = pd.concat(
        [
            _prepare_frame(fake_df, label=1),
            _prepare_frame(true_df, label=0),
        ],
        ignore_index=True,
    )
    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return combined


def split_dataset(df: pd.DataFrame) -> DatasetBundle:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )
    return DatasetBundle(
        train_texts=train_df["content"],
        val_texts=val_df["content"],
        test_texts=test_df["content"],
        train_labels=train_df["label"],
        val_labels=val_df["label"],
        test_labels=test_df["label"],
        full_dataframe=df,
    )
