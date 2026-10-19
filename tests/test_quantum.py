import numpy as np
import pennylane as qml
import pytest
import torch

from qmlfin import experiment
from qmlfin.models.quantum import QSVC, HybridQNN, MatchedMLP
from qmlfin.quantum import reference, sim

from test_classical import cfg  # noqa: F401  (fixture)
from test_data_pipeline import raw_dir  # noqa: F401  (fixture)


def _pl_state(circuit, *args, n):
    @qml.qnode(qml.device("default.qubit", wires=n))
    def f():
        circuit(*args)
        return qml.state()

    return f()


@pytest.mark.parametrize("n,reps", [(3, 1), (4, 2), (6, 2)])
def test_zz_map_matches_pennylane(n, reps):
    x = np.random.default_rng(n).random(n) * np.pi
    ours = sim.zz_feature_map(torch.tensor(x[None]), reps)[0].numpy()
    assert abs(np.vdot(_pl_state(reference.zz_feature_map, x, reps, n=n), ours)) ** 2 == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("n,layers", [(2, 1), (4, 3), (6, 2)])
def test_variational_matches_pennylane(n, layers):
    rng = np.random.default_rng(n)
    x, w = rng.random(n) * np.pi, rng.random((layers, n, 3)) * 2 * np.pi
    ours = sim.strongly_entangling(sim.angle_embedding_y(torch.tensor(x[None])), torch.tensor(w), n)[0].numpy()
    assert abs(np.vdot(_pl_state(reference.variational, x, w, n=n), ours)) ** 2 == pytest.approx(1, abs=1e-10)
    # <Z_i> readout
    @qml.qnode(qml.device("default.qubit", wires=n))
    def z():
        reference.variational(x, w)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    np.testing.assert_allclose(sim.expval_z(torch.tensor(ours[None]), n)[0].numpy(), z(), atol=1e-10)


def test_qsvc_kernel_is_a_valid_gram_matrix():
    X = np.random.default_rng(0).random((20, 4)) * np.pi
    m = QSVC(bandwidth=0.5)
    S = m._states(X)
    K = m._kernel(S, S)
    np.testing.assert_allclose(np.diag(K), 1, atol=1e-12)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_qsvc_decision_matches_full_kernel_svc():
    rng = np.random.default_rng(1)
    X, y = rng.random((60, 4)) * np.pi, rng.integers(0, 2, 60)
    m = QSVC(bandwidth=0.3, C=5).fit(X, y)
    Xt = rng.random((10, 4)) * np.pi
    K_full = m._kernel(m._states(Xt), m._states(X))
    np.testing.assert_allclose(m.decision_function(Xt), m.svc_.decision_function(K_full), atol=1e-8)


@pytest.mark.parametrize("n,layers", [(4, 1), (6, 2), (8, 3)])
def test_matched_mlp_parameter_count(n, layers):
    X, y = np.random.default_rng(0).random((32, 13)), np.array([0, 1] * 16)
    h = HybridQNN(n_qubits=n, n_layers=layers, epochs=1).fit(X, y).n_parameters()
    m = MatchedMLP(n_qubits=n, n_layers=layers, epochs=1).fit(X, y).n_parameters()
    assert abs(h - m) <= n + 2  # within one hidden unit


@pytest.mark.parametrize("model", ["qsvc", "vqc", "hqnn", "mlp_matched"])
def test_quantum_models_run(cfg, model):  # noqa: F811
    cfg["torch"]["epochs"] = 2
    row = experiment.run_fold(cfg, "german_credit", model, "4", fold=0, seed=0)
    assert 0 <= row["roc_auc"] <= 1 and row["n_parameters"]
    if model != "mlp_matched":
        assert row["n_qubits"] == 4 and row["two_qubit_gates"] > 0
    if model == "qsvc":
        assert row["kernel_evaluations"] > 160 * 159 // 2


def test_quantum_model_rejects_full_regime(cfg):  # noqa: F811
    with pytest.raises(ValueError):
        experiment.run_fold(cfg, "german_credit", "vqc", "full", fold=0, seed=0)
