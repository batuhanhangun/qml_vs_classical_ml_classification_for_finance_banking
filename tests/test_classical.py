import numpy as np
import pytest
import yaml

from qmlfin import experiment
from qmlfin.metrics import evaluate, select_threshold
from qmlfin.models import classical

from test_data_pipeline import raw_dir  # noqa: F401  (fixture)


def test_metrics_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    m = evaluate(y, y.astype(float), 0.5)
    assert m["roc_auc"] == 1 and m["f1"] == 1 and m["g_mean"] == 1 and m["recall_at_fpr_0.001"] == 1
    s = np.random.default_rng(0).random(100)
    assert 0.2 < evaluate(y, s, 0.5)["roc_auc"] < 0.8


def test_weighted_threshold_is_stricter():
    """Up-weighting negatives (natural-rate estimate) should not lower the chosen threshold."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 400 + [1] * 100)
    s = rng.normal(y * 1.5, 1.0)
    t_plain = select_threshold(y, s)
    t_weighted = select_threshold(y, s, np.where(y == 0, 50.0, 1.0))
    assert t_weighted >= t_plain


@pytest.fixture
def cfg(raw_dir):  # noqa: F811
    c = yaml.safe_load(open("configs/default.yaml"))
    c["data_dir"] = str(raw_dir)
    c["tuning"]["n_configs"] = 2
    experiment._load.cache_clear()
    return c


@pytest.mark.parametrize("model", sorted(classical.SEARCH_SPACES))
def test_every_model_runs(cfg, model):
    row = experiment.run_fold(cfg, "german_credit", model, "4", fold=0, seed=0)
    assert 0 <= row["roc_auc"] <= 1 and row["n_parameters"]


def test_ulb_fold_trains_undersampled_and_tests_natural(cfg):
    row = experiment.run_fold(cfg, "ulb_fraud", "lr", "full", fold=0, seed=0)
    assert row["train_size"] == 200  # 40 training frauds + 160 legitimate (1:4)
    assert row["test_size"] == 1000


def test_learning_curve_train_size(cfg):
    row = experiment.run_fold(cfg, "bank_churn", "lr", "full", fold=0, seed=0, train_size=100)
    assert row["train_size"] == 100
