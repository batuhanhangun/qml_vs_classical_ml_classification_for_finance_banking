import numpy as np
import pytest
import yaml

from qmlfin import grid, report

from test_classical import cfg  # noqa: F401  (fixture)
from test_data_pipeline import raw_dir  # noqa: F401  (fixture)


def test_grid_size():
    c = yaml.safe_load(open("configs/default.yaml"))
    all_jobs = grid.jobs(c)
    # per dataset: 7 classical x 4 inputs x 5 folds + 4 quantum/hybrid x 3 qubit counts x 5 folds
    assert len(all_jobs) == 3 * (7 * 4 * 5 + 4 * 3 * 5) + 4 * 8 * 5
    assert len(grid.jobs(c, "sklearn")) + len(grid.jobs(c, "torch")) == len(all_jobs)
    assert len({str(grid.job_path(c, j)) for j in all_jobs}) == len(all_jobs)  # one file per job


def test_nadeau_bengio_known_value():
    t, p = report.nadeau_bengio([1, 2, 3, 4, 5], 0.25)
    assert t == pytest.approx(3 / np.sqrt(2.5 * (0.2 + 0.25)))
    assert 0 < p < 0.06


def test_holm():
    np.testing.assert_allclose(report.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_jobs_resume_and_report(cfg, tmp_path):  # noqa: F811
    cfg["results_dir"] = str(tmp_path / "results")
    cfg["torch"]["epochs"] = 2
    for model, regime in [("lr", "full"), ("lr", "4"), ("rf", "4"), ("qsvc", "4"), ("vqc", "4")]:
        for fold in range(5):
            job = {"dataset": "german_credit", "model": model, "regime": regime, "fold": fold, "train_size": None}
            assert grid.run_job(cfg, job)
    assert not grid.run_job(cfg, job)  # finished jobs are skipped

    folds = report.load(cfg["results_dir"])
    assert len(folds) == 25  # VQC seeds averaged within each fold
    summary = report.summarise(folds)
    tests = report.compare(folds, summary)
    assert set(tests["model"]) == {"qsvc", "vqc"}
    assert set(tests["reference"]) == {"best classical", "best classical, same input size"}
    assert tests["p_holm"].between(0, 1).all()
    assert r"\textbf" in report._latex(summary, "german_credit", report.METRICS)
