"""Outer cross-validation folds, training-fold undersampling, and learning-curve subsets."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def outer_folds(y, n_splits=5, seed=42):
    """Stratified folds as a list of (train_idx, test_idx)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def undersample_train(train_idx, y, neg_per_pos, seed=42):
    """Keep every positive in the training fold and a random set of negatives, neg_per_pos per positive."""
    rng = np.random.default_rng(seed)
    pos = train_idx[y[train_idx] == 1]
    neg = train_idx[y[train_idx] == 0]
    n_neg = min(len(neg), int(neg_per_pos * len(pos)))
    keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    return np.sort(keep)


def learning_curve_subsets(train_idx, y, sizes, seed=42):
    """Nested stratified subsets of a training fold: each smaller subset is contained in the next larger one."""
    rng = np.random.default_rng(seed)
    order = {}
    for cls in (0, 1):
        idx = train_idx[y[train_idx] == cls]
        order[cls] = rng.permutation(idx)
    pos_rate = (y[train_idx] == 1).mean()
    subsets = {}
    for n in sorted(sizes):
        if n > len(train_idx):
            continue
        n_pos = int(round(n * pos_rate))
        subsets[n] = np.sort(np.concatenate([order[1][:n_pos], order[0][: n - n_pos]]))
    return subsets
