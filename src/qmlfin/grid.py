"""The full experiment grid as a numbered list of jobs, one result file per job.

    python -m qmlfin.grid --list                      # number of jobs per group
    python -m qmlfin.grid --group sklearn --task-id 7 # run one job (Slurm array task)
    python -m qmlfin.grid --group all --run-all       # run everything sequentially

Finished jobs are skipped, so an interrupted run can be resumed.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from qmlfin.experiment import _json_default, run_fold
from qmlfin.models import registry

TORCH_MODELS = {"qsvc", "vqc", "hqnn", "mlp", "mlp_matched"}  # benefit from a GPU


def jobs(cfg, group="all"):
    g = cfg["grid"]
    qubits = [str(n) for n in cfg["qubits"]]
    out = []
    for dataset in cfg["datasets"]:
        for model in g["classical"] + g["quantum"] + g["hybrid"]:
            regimes = ["full"] + qubits if registry.MODELS[model]["input"] == "any" else qubits
            for regime in regimes:
                for fold in range(cfg["cv"]["outer_folds"]):
                    out.append({"dataset": dataset, "model": model, "regime": regime, "fold": fold, "train_size": None})
    lc = g["learning_curve"]
    for size in cfg["datasets"][lc["dataset"]]["learning_curve_sizes"]:
        for model in lc["models"]:
            regime = "full" if registry.MODELS[model]["input"] == "any" else str(lc["qubits"])
            for fold in range(cfg["cv"]["outer_folds"]):
                out.append({"dataset": lc["dataset"], "model": model, "regime": regime, "fold": fold, "train_size": size})
    if group == "torch":
        out = [j for j in out if j["model"] in TORCH_MODELS]
    elif group == "sklearn":
        out = [j for j in out if j["model"] not in TORCH_MODELS]
    return out


def job_path(cfg, job):
    name = f"{job['model']}_{job['regime']}_f{job['fold']}"
    if job["train_size"] is not None:
        name += f"_n{job['train_size']}"
    return Path(cfg["results_dir"]) / "raw" / job["dataset"] / f"{name}.jsonl"


def run_job(cfg, job):
    path = job_path(cfg, job)
    if path.exists():
        return False
    n_seeds = cfg["cv"]["seeds_per_fold"] if registry.MODELS[job["model"]]["stochastic"] else 1
    rows = []
    for seed in range(n_seeds):
        row = run_fold(cfg, job["dataset"], job["model"], job["regime"], job["fold"], seed, job["train_size"])
        row["learning_curve"] = job["train_size"] is not None
        rows.append(json.dumps(row, default=_json_default))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + "\n")
    os.replace(tmp, path)  # a result file only appears once the whole job has finished
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--group", choices=["all", "sklearn", "torch"], default="all")
    p.add_argument("--list", action="store_true")
    p.add_argument("--task-id", type=int)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--device", help="override torch.device, e.g. cuda")
    p.add_argument("--n-jobs", type=int, help="override n_jobs for tree ensembles")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.device:
        cfg.setdefault("torch", {})["device"] = args.device
    if args.n_jobs:
        cfg["n_jobs"] = args.n_jobs

    if args.list:
        for group in ("sklearn", "torch", "all"):
            todo = [j for j in jobs(cfg, group) if not job_path(cfg, j).exists()]
            print(f"{group}: {len(jobs(cfg, group))} jobs, {len(todo)} not finished")
        return
    selected = jobs(cfg, args.group)
    if args.task_id is not None:
        selected = [selected[args.task_id]]
    elif not args.run_all:
        p.error("give --list, --task-id, or --run-all")
    for job in selected:
        print(job, "done" if run_job(cfg, job) else "already finished", flush=True)


if __name__ == "__main__":
    main()
