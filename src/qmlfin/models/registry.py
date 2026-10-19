"""All models with their search spaces and the input each one receives.

input "any":     classical model, runs on the full feature set or on the reduced n-dimensional features
input "reduced": quantum model on the same PCA features scaled to angles (n = number of qubits)
input "full":    trainable encoder from the full feature set to n qubits (hybrid QNN and its matched MLP)
"""

from qmlfin.models import classical, quantum

MODELS = {
    **{name: {"input": "any", "stochastic": name in classical.STOCHASTIC} for name in classical.SEARCH_SPACES},
    "qsvc": {"input": "reduced", "stochastic": False},
    "vqc": {"input": "reduced", "stochastic": True},
    "hqnn": {"input": "full", "stochastic": True},
    "mlp_matched": {"input": "full", "stochastic": True},
}

SEARCH_SPACES = {**classical.SEARCH_SPACES, **quantum.SEARCH_SPACES}


def build(name, seed=0, n_qubits=None, n_jobs=1):
    if name in classical.SEARCH_SPACES:
        return classical.build(name, seed, n_jobs)
    return quantum.build(name, seed, n_qubits)
