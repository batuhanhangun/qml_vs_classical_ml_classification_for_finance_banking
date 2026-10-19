"""Run models on outer folds and append one JSON line per (model, regime, fold, seed).

Example:
    python -m qmlfin.experiment --dataset german_credit --models lr rf --regimes full 4 --folds 0 1
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.pipeline import Pipeline

from qmlfin import data, splits
from qmlfin.metrics import evaluate
from qmlfin.models import registry
from qmlfin.preprocess import make_preprocessor
from qmlfin.tuning import fit, scores, tune


_TUNED = {}


@lru_cache(maxsize=None)
def _load(name, data_dir, drop_duplicates):
    opts = {"drop_duplicates": drop_duplicates} if name == "ulb_fraud" else {}
    return data.load(name, data_dir, **opts)


def n_parameters(clf):
    if hasattr(clf, "n_parameters"):
        return clf.n_parameters()
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + np.size(clf.intercept_))
    if hasattr(clf, "support_vectors_"):
        return int(clf.support_vectors_.size + clf.dual_coef_.size + 1)
    if hasattr(clf, "estimators_"):
        return int(sum(t.tree_.node_count for t in clf.estimators_))
    if hasattr(clf, "get_booster"):
        return int(len(clf.get_booster().trees_to_dataframe()))
    if hasattr(clf, "booster_"):
        return int(clf.booster_.trees_to_dataframe().shape[0])
    return None


def run_fold(cfg, dataset, model, regime, fold, seed, train_size=None):
    dcfg = cfg["datasets"][dataset]
    ds = _load(dataset, cfg["data_dir"], dcfg.get("drop_duplicates", False))
    tr, te = splits.outer_folds(ds.y, cfg["cv"]["outer_folds"], cfg["seed"])[fold]

    weight = np.ones(len(tr))
    if "train_neg_per_pos" in dcfg:
        n_neg_all = (ds.y[tr] == 0).sum()
        tr = splits.undersample_train(tr, ds.y, dcfg["train_neg_per_pos"], cfg["seed"] + fold)
        weight = np.where(ds.y[tr] == 0, n_neg_all / (ds.y[tr] == 0).sum(), 1.0)
    if train_size is not None:
        sub = splits.learning_curve_subsets(tr, ds.y, [train_size], cfg["seed"] + fold)[train_size]
        weight = weight[np.isin(tr, sub)]
        tr = sub

    n = None if regime == "full" else int(regime)
    kind = registry.MODELS[model]["input"]
    if n is None and kind != "any":
        raise ValueError(f"{model} needs a qubit count as its regime")
    if kind == "full":  # trainable encoder sees the full standardised feature set
        pre = make_preprocessor(ds.cat_cols, ds.num_cols, seed=cfg["seed"])
    else:
        pre = make_preprocessor(ds.cat_cols, ds.num_cols, n_components=n,
                                angle_range=None if n is None else cfg["angle_range"], seed=cfg["seed"])
    clf = registry.build(model, seed, n, cfg.get("n_jobs", 1))
    torch_opts = {k: v for k, v in cfg.get("torch", {}).items() if k in clf.get_params()}
    if "device" in clf.get_params():
        torch_opts["device"] = cfg.get("torch", {}).get("device", "cpu")
    clf.set_params(**torch_opts)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, ytr, Xte, yte = ds.X.iloc[tr], ds.y[tr], ds.X.iloc[te], ds.y[te]
    key = (dataset, model, regime, fold, train_size)
    if key not in _TUNED:  # tune once per fold; seeds only vary the final training run
        _TUNED[key] = tune(pipe, registry.SEARCH_SPACES[model], Xtr, ytr, weight,
                           cfg["tuning"]["n_configs"], cfg["cv"]["inner_folds"], cfg["seed"] + fold)
    params, threshold, inner_ap = _TUNED[key]
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})

    t0 = time.perf_counter()
    fit(pipe, Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = scores(pipe, Xte)
    predict_ms = 1000 * (time.perf_counter() - t0) / len(te)

    extra = pipe[-1].resources() if hasattr(pipe[-1], "resources") else {}
    if model == "qsvc":  # kernel entries a device would estimate: training Gram matrix and test-vs-support
        n_sv = len(pipe[-1].sv_states_)
        extra["kernel_evaluations"] = len(tr) * (len(tr) - 1) // 2 + len(te) * n_sv

    return {
        "dataset": dataset, "model": model, "regime": regime, "fold": fold, "seed": seed,
        "train_size": len(tr), "test_size": len(te),
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "inner_pr_auc": inner_ap, "threshold": float(threshold),
        **evaluate(yte, s, threshold),
        "fit_time_s": fit_time, "predict_ms_per_sample": predict_ms,
        "n_parameters": n_parameters(pipe[-1]), **extra,
    }


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--regimes", nargs="+", default=["full"], help="'full' or qubit counts, e.g. full 4 6 8")
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--train-size", type=int, help="learning-curve training subset size")
    p.add_argument("--out")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    folds = args.folds if args.folds is not None else range(cfg["cv"]["outer_folds"])
    out = Path(args.out or Path(cfg["results_dir"]) / "raw" / f"{args.dataset}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        n_seeds = cfg["cv"]["seeds_per_fold"] if registry.MODELS[model]["stochastic"] else 1
        for regime in args.regimes:
            for fold in folds:
                for seed in range(n_seeds):
                    row = run_fold(cfg, args.dataset, model, regime, fold, seed, args.train_size)
                    with open(out, "a") as f:
                        f.write(json.dumps(row, default=_json_default) + "\n")
                    print(f"{args.dataset} {model} {regime} fold {fold} seed {seed}: "
                          f"ROC-AUC {row['roc_auc']:.3f} PR-AUC {row['pr_auc']:.3f} F1 {row['f1']:.3f}")


if __name__ == "__main__":
    main()
