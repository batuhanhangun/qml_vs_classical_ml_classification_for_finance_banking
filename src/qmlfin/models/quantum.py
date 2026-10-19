"""Quantum kernel, variational, and hybrid models, plus a parameter-matched classical counterpart."""

import numpy as np
import torch
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from torch import nn

from qmlfin.models.torch_mlp import TorchBinaryClassifier
from qmlfin.quantum import reference, sim


class QSVC(BaseEstimator, ClassifierMixin):
    """SVM on the exact fidelity kernel |<phi(x)|phi(x')>|^2 of a ZZ feature map with bandwidth scaling."""

    def __init__(self, reps=1, bandwidth=1.0, C=1.0, device="cpu", chunk=4096):
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C
        self.device = device
        self.chunk = chunk

    def _states(self, X):
        X = torch.as_tensor(np.asarray(X) * self.bandwidth, dtype=torch.float64, device=self.device)
        return torch.cat([sim.zz_feature_map(X[i : i + self.chunk], self.reps) for i in range(0, len(X), self.chunk)])

    def _kernel(self, A, B):
        return np.concatenate([(A[i : i + self.chunk] @ B.conj().T).abs().pow(2).cpu().numpy()
                               for i in range(0, len(A), self.chunk)])

    def fit(self, X, y):
        S = self._states(X)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(self._kernel(S, S), y)
        self.classes_ = self.svc_.classes_
        self.sv_states_ = S[self.svc_.support_]
        self.n_qubits_ = S.shape[1].bit_length() - 1
        return self

    def decision_function(self, X):
        K = self._kernel(self._states(X), self.sv_states_)
        return K @ self.svc_.dual_coef_[0] + self.svc_.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def n_parameters(self):
        return int(self.svc_.dual_coef_.size + 1)

    def resources(self):
        x = np.zeros(self.n_qubits_)
        return reference.resources(reference.zz_feature_map, x, self.reps, n_qubits=self.n_qubits_)


class _Variational(nn.Module):
    """Angle embedding of n angles, StronglyEntanglingLayers, and <Z_i> readout on every qubit."""

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n = n_qubits
        self.weights = nn.Parameter(2 * np.pi * torch.rand(n_layers, n_qubits, 3))

    def forward(self, angles):
        state = sim.strongly_entangling(sim.angle_embedding_y(angles), self.weights, self.n)
        return sim.expval_z(state, self.n)


class VQC(TorchBinaryClassifier):
    """Variational quantum classifier on reduced features: logit = a <Z_0> + b."""

    def __init__(self, n_layers=2, lr=1e-2, weight_decay=0.0, epochs=30, batch_size=128, seed=0, device="cpu"):
        super().__init__(lr, weight_decay, epochs, batch_size, seed, device)
        self.n_layers = n_layers

    def build_module(self, n_features):
        self.n_qubits_ = n_features
        circuit = _Variational(n_features, self.n_layers)

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.circuit = circuit
                self.out = nn.Linear(1, 1)

            def forward(self, x):
                return self.out(self.circuit(x)[:, :1])

        return Head()

    def resources(self):
        w = np.zeros((self.n_layers, self.n_qubits_, 3))
        return reference.resources(reference.variational, np.zeros(self.n_qubits_), w, n_qubits=self.n_qubits_)


def _encoder(n_features, n_qubits, hidden=16):
    """Trainable classical compression to n_qubits angles in (-pi, pi)."""
    return nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_qubits), nn.Tanh())


class HybridQNN(TorchBinaryClassifier):
    """Dressed QNN: classical encoder, variational circuit, linear head on all <Z_i>."""

    def __init__(self, n_qubits=4, n_layers=2, lr=1e-3, weight_decay=0.0, epochs=30, batch_size=128, seed=0,
                 device="cpu"):
        super().__init__(lr, weight_decay, epochs, batch_size, seed, device)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def build_module(self, n_features):
        enc, circ, head = _encoder(n_features, self.n_qubits), _Variational(self.n_qubits, self.n_layers), \
            nn.Linear(self.n_qubits, 1)
        return nn.Sequential(enc, _Scale(np.pi), circ, head)

    def resources(self):
        w = np.zeros((self.n_layers, self.n_qubits, 3))
        return reference.resources(reference.variational, np.zeros(self.n_qubits), w, n_qubits=self.n_qubits)


class MatchedMLP(TorchBinaryClassifier):
    """Classical counterpart of HybridQNN: same encoder, with the circuit and head replaced by a
    hidden layer whose width is chosen so the parameter count matches."""

    def __init__(self, n_qubits=4, n_layers=2, lr=1e-3, weight_decay=0.0, epochs=30, batch_size=128, seed=0,
                 device="cpu"):
        super().__init__(lr, weight_decay, epochs, batch_size, seed, device)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def hidden_width(self):
        # circuit + head: 3 n L + (n + 1); replacement: k (n + 2) + 1
        return max(1, round((3 * self.n_qubits * self.n_layers + self.n_qubits) / (self.n_qubits + 2)))

    def build_module(self, n_features):
        k = self.hidden_width()
        return nn.Sequential(_encoder(n_features, self.n_qubits), nn.Linear(self.n_qubits, k), nn.ReLU(),
                             nn.Linear(k, 1))


class _Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return self.factor * x


_TORCH_SPACE = {"lr": loguniform(1e-4, 1e-1), "weight_decay": loguniform(1e-6, 1e-2), "n_layers": [1, 2, 3]}

SEARCH_SPACES = {
    "qsvc": {"reps": [1, 2], "bandwidth": loguniform(1e-2, 1e0), "C": loguniform(1e-2, 1e3)},
    "vqc": dict(_TORCH_SPACE),
    "hqnn": dict(_TORCH_SPACE),
    "mlp_matched": dict(_TORCH_SPACE),
}


def build(name, seed=0, n_qubits=None):
    if name == "qsvc":
        return QSVC()
    if name == "vqc":
        return VQC(seed=seed)
    if name == "hqnn":
        return HybridQNN(n_qubits=n_qubits, seed=seed)
    if name == "mlp_matched":
        return MatchedMLP(n_qubits=n_qubits, seed=seed)
    raise ValueError(f"unknown model {name}")
