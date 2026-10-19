"""Aggregate per-fold results into tables and run the corrected resampled t-test.

    python -m qmlfin.report --config configs/default.yaml

Writes CSV and LaTeX tables to <results_dir>/tables/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from qmlfin.models import registry

METRICS = ["roc_auc", "pr_auc", "f1", "recall", "specificity", "g_mean"]
FRAUD_METRICS = ["recall_at_fpr_0.001", "recall_at_fpr_0.01"]
RESOURCES = ["n_qubits", "circuit_depth", "two_qubit_gates", "n_parameters", "fit_time_s", "kernel_evaluations"]
KEY = ["dataset", "model", "regime", "train_size_req"]
QUANTUM = {m for m, v in registry.MODELS.items() if v["input"] != "any"} - {"mlp_matched"}


def load(results_dir):
    rows = [json.loads(line) for f in sorted(Path(results_dir, "raw").rglob("*.jsonl")) for line in open(f)]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no results under {results_dir}/raw")
    lc = df["learning_curve"].fillna(False).astype(bool) if "learning_curve" in df else False
    df["train_size_req"] = np.where(lc, df["train_size"], -1)
    # stochastic models: average seeds within each fold first
    return df.groupby(KEY + ["fold"], as_index=False).mean(numeric_only=True)


def summarise(folds):
    cols = [c for c in METRICS + FRAUD_METRICS + RESOURCES if c in folds]
    g = folds.groupby(KEY)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    out["family"] = out["model"].map(lambda m: "quantum" if m in QUANTUM else
                                     "hybrid-matched" if m == "mlp_matched" else "classical")
    return out


def nadeau_bengio(diff, test_train_ratio):
    """Corrected resampled t-test for k-fold CV differences (Nadeau and Bengio, 2003)."""
    diff = np.asarray(diff, dtype=float)
    k = len(diff)
    var = diff.var(ddof=1) * (1 / k + test_train_ratio)
    if var == 0:
        return np.inf * np.sign(diff.mean()), 0.0 if diff.mean() != 0 else 1.0
    t = diff.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def compare(folds, summary, metric="pr_auc"):
    """Each quantum and hybrid configuration against the best classical model overall and at the same regime."""
    main = folds[folds["train_size_req"] == -1]
    sm = summary[summary["train_size_req"] == -1]
    out = []
    for dataset, d in main.groupby("dataset"):
        s = sm[sm["dataset"] == dataset]
        classical = s[s["family"] == "classical"]
        best_all = classical.loc[classical[f"{metric}_mean"].idxmax()]
        for _, row in s[s["family"] != "classical"].iterrows():
            same = classical[classical["regime"] == row["regime"]]
            for label, ref in (("best classical", best_all), ("best classical, same input size",
                                                              same.loc[same[f"{metric}_mean"].idxmax()])):
                a = d[(d["model"] == row["model"]) & (d["regime"] == row["regime"])].set_index("fold")
                b = d[(d["model"] == ref["model"]) & (d["regime"] == ref["regime"])].set_index("fold")
                diff = (a[metric] - b[metric]).dropna()
                ratio = (a["test_size"] / a["train_size"]).mean()
                t, p = nadeau_bengio(diff, ratio)
                out.append({"dataset": dataset, "model": row["model"], "regime": row["regime"],
                            "reference": label, "ref_model": ref["model"], "ref_regime": ref["regime"],
                            "metric": metric, "mean_diff": diff.mean(), "t": t, "p": p})
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_holm"] = res.groupby(["dataset", "reference"])["p"].transform(holm)
    return res


def _latex(summary, dataset, metrics):
    s = summary[(summary["dataset"] == dataset) & (summary["train_size_req"] == -1)].copy()
    order = {"classical": 0, "quantum": 1, "hybrid-matched": 2}
    s["_o"] = s["family"].map(order) + s["model"].map(lambda m: 0.5 if m == "hqnn" else 0)
    s = s.sort_values(["_o", "model", "regime"])
    best = {m: s[f"{m}_mean"].max() for m in metrics}
    head = " & ".join(["Model", "Input"] + [m.replace("_", r"\_") for m in metrics])
    lines = [r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\hline", head + r" \\", r"\hline"]
    for _, r in s.iterrows():
        cells = []
        for m in metrics:
            v = f"{r[f'{m}_mean']:.3f} $\\pm$ {r[f'{m}_std']:.3f}"
            cells.append(rf"\textbf{{{v}}}" if np.isclose(r[f"{m}_mean"], best[m]) else v)
        regime = "full" if r["regime"] == "full" else f"{r['regime']}"
        lines.append(" & ".join([r["model"].replace("_", r"\_"), regime] + cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def learning_curve(folds):
    """PR-AUC against training size for the learning-curve models, adding the full training fold from the main grid."""
    lc = folds[folds["train_size_req"] != -1]
    if lc.empty:
        return pd.DataFrame()
    pairs = lc[["dataset", "model", "regime"]].drop_duplicates()
    full = folds[folds["train_size_req"] == -1].merge(pairs, on=["dataset", "model", "regime"])
    both = pd.concat([lc, full])
    return both.groupby(["dataset", "model", "regime", "train_size"])[["pr_auc", "roc_auc"]].agg(["mean", "std"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(p.parse_args().config))
    out = Path(cfg["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    folds = load(cfg["results_dir"])
    summary = summarise(folds)
    summary.to_csv(out / "summary.csv", index=False)
    for dataset in summary["dataset"].unique():
        metrics = METRICS + (FRAUD_METRICS if dataset == "ulb_fraud" else [])
        (out / f"{dataset}.tex").write_text(_latex(summary, dataset, metrics))
    for metric in ("pr_auc", "roc_auc"):
        compare(folds, summary, metric).to_csv(out / f"tests_{metric}.csv", index=False)
    lc = learning_curve(folds)
    if not lc.empty:
        lc.to_csv(out / "learning_curve.csv")
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
