"""Classical baselines and their hyperparameter search spaces."""

from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from qmlfin.models.torch_mlp import MLPClassifier


def build(name, seed=0, n_jobs=1):
    """Untuned estimator with class weighting for the imbalanced target."""
    if name == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=5000)
    if name == "svm_linear":
        return LinearSVC(class_weight="balanced", max_iter=20000, random_state=seed)
    if name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs)
    if name == "xgboost":
        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=n_jobs, eval_metric="logloss")
    if name == "lightgbm":
        return LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs, verbose=-1)
    if name == "mlp":
        return MLPClassifier(seed=seed)
    raise ValueError(f"unknown model {name}")


SEARCH_SPACES = {
    "lr": {"C": loguniform(1e-3, 1e2)},
    "svm_linear": {"C": loguniform(1e-3, 1e2)},
    "svm_rbf": {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-4, 1e1)},
    "rf": {
        "n_estimators": [200, 500],
        "max_depth": [None, 4, 8, 16],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
    "xgboost": {
        "n_estimators": randint(100, 600),
        "max_depth": randint(2, 9),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 11),
        "reg_lambda": loguniform(1e-3, 1e1),
    },
    "lightgbm": {
        "n_estimators": randint(100, 600),
        "num_leaves": randint(8, 65),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),
        "subsample_freq": [1],
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 51),
    },
    "mlp": {
        "hidden": [(16,), (32,), (64,), (32, 16)],
        "lr": loguniform(1e-4, 1e-2),
        "weight_decay": loguniform(1e-6, 1e-2),
    },
}

# XGBoost has no class_weight; the positive-class weight is set from the training labels at fit time.
NEEDS_SCALE_POS_WEIGHT = {"xgboost"}

STOCHASTIC = {"mlp"}  # run with several seeds per fold
