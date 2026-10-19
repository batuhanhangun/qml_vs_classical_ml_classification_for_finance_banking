"""Print a summary of each dataset and its folds: python -m qmlfin.check_data --config configs/default.yaml"""

import argparse

import numpy as np
import yaml

from qmlfin import data, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(parser.parse_args().config))

    for name, dcfg in cfg["datasets"].items():
        ds = data.load(name, cfg["data_dir"])
        print(f"\n{name}: {len(ds.y)} samples, {ds.X.shape[1]} features, {ds.y.sum()} positives ({ds.y.mean():.2%})")
        for k, (tr, te) in enumerate(splits.outer_folds(ds.y, cfg["cv"]["outer_folds"], cfg["seed"])):
            if "train_neg_per_pos" in dcfg:
                tr = splits.undersample_train(tr, ds.y, dcfg["train_neg_per_pos"], cfg["seed"] + k)
            print(f"  fold {k}: train {len(tr)} ({ds.y[tr].mean():.2%} pos), test {len(te)} ({ds.y[te].mean():.2%} pos)")
        if "learning_curve_sizes" in dcfg:
            tr, _ = splits.outer_folds(ds.y, cfg["cv"]["outer_folds"], cfg["seed"])[0]
            subs = splits.learning_curve_subsets(tr, ds.y, dcfg["learning_curve_sizes"], cfg["seed"])
            print("  learning-curve subsets (fold 0):", {n: f"{ds.y[i].mean():.2%}" for n, i in subs.items()})


if __name__ == "__main__":
    main()
