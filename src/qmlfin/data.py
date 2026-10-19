"""Loaders for the three benchmark datasets. The positive class (1) is always the minority class."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "german_credit": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    "bank_churn": "https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling",
    "ulb_fraud": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    cat_cols: list
    num_cols: list


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download it from {SOURCES[name]} and place it in {path.parent}/.")
    return path


def load_german_credit(data_dir) -> Dataset:
    path = _require(Path(data_dir) / "german.data", "german_credit")
    cols = [f"A{i}" for i in range(1, 21)] + ["target"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    num_cols = ["A2", "A5", "A8", "A11", "A13", "A16", "A18"]
    cat_cols = [c for c in cols[:-1] if c not in num_cols]
    y = (df["target"] == 2).astype(int).to_numpy()  # 2 = bad credit
    return Dataset("german_credit", df[cols[:-1]], y, cat_cols, num_cols)


def load_bank_churn(data_dir) -> Dataset:
    path = _require(Path(data_dir) / "Churn_Modelling.csv", "bank_churn")
    df = pd.read_csv(path).drop(columns=["RowNumber", "CustomerId", "Surname"])
    cat_cols = ["Geography", "Gender"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["Exited"]]
    y = df["Exited"].astype(int).to_numpy()
    return Dataset("bank_churn", df[cat_cols + num_cols], y, cat_cols, num_cols)


def load_ulb_fraud(data_dir) -> Dataset:
    path = _require(Path(data_dir) / "creditcard.csv", "ulb_fraud")
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if c != "Class"]
    y = df["Class"].astype(int).to_numpy()
    return Dataset("ulb_fraud", df[num_cols], y, [], num_cols)


LOADERS = {
    "german_credit": load_german_credit,
    "bank_churn": load_bank_churn,
    "ulb_fraud": load_ulb_fraud,
}


def load(name: str, data_dir) -> Dataset:
    return LOADERS[name](data_dir)
