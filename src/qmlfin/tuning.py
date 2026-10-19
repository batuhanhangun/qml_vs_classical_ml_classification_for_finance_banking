"""Random-search tuning with the same budget for every model, inside the training fold only."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from qmlfin.metrics import select_threshold


def scores(pipe, X):
    """Continuous score where larger means more likely positive."""
    if hasattr(pipe[-1], "decision_function"):
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, 1]


def fit(pipe, X, y):
    if "scale_pos_weight" in pipe[-1].get_params():
        pipe.set_params(clf__scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    return pipe.fit(X, y)


def tune(pipe, space, X, y, sample_weight, n_configs, inner_folds, seed):
    """Return the best parameters by inner-CV PR-AUC and an F1-optimal threshold from its out-of-fold scores."""
    configs = list(ParameterSampler(space, n_configs, random_state=seed))
    inner = list(StratifiedKFold(inner_folds, shuffle=True, random_state=seed).split(X, y))
    best = (-np.inf, None, None)
    for params in configs:
        oof = np.zeros(len(y))
        for tr, va in inner:
            model = fit(clone(pipe).set_params(**{f"clf__{k}": v for k, v in params.items()}), X.iloc[tr], y[tr])
            oof[va] = scores(model, X.iloc[va])
        ap = average_precision_score(y, oof, sample_weight=sample_weight)
        if ap > best[0]:
            best = (ap, params, oof)
    ap, params, oof = best
    return params, select_threshold(y, oof, sample_weight), ap
