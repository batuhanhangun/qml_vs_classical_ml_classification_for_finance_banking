"""Preprocessing pipelines. Always fit on the training fold only, then apply to the test fold."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class AngleScaler(BaseEstimator, TransformerMixin):
    """Min-max scaling to a rotation-angle range, clipped so test values stay inside it."""

    def __init__(self, low=0.0, high=np.pi):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler(feature_range=(self.low, self.high)).fit(X)
        return self

    def transform(self, X):
        return np.clip(self.scaler_.transform(X), self.low, self.high)


def make_preprocessor(cat_cols, num_cols, n_components=None, angle_range=None, seed=42):
    """Encoding and standardisation, optionally followed by PCA to n_components and angle scaling.

    n_components=None keeps the full feature set (classical full-dimensional regime).
    angle_range=(low, high) prepares features for quantum encoding.
    """
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
            ("num", "passthrough", list(num_cols)),
        ]
    )
    steps = [("encode", encode), ("standardise", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    if angle_range is not None:
        steps.append(("angle", AngleScaler(*angle_range)))
    return Pipeline(steps)
