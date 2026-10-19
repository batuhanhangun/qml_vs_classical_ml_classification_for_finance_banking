"""Exact batched statevector simulation in PyTorch for the circuits used in this study.

Qubit 0 is the most significant bit, as in PennyLane. Outputs are checked against PennyLane in the tests.
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def _bits(n):
    idx = np.arange(2**n)
    return (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1  # (2^n, n)


@lru_cache(maxsize=None)
def _cnot_perm(n, c, t):
    b = _bits(n)
    idx = np.arange(2**n)
    return torch.as_tensor(np.where(b[:, c] == 1, idx ^ (1 << (n - 1 - t)), idx))


def apply_1q(state, U, q, n):
    """Apply a 2x2 gate (shared or batched (B, 2, 2)) to qubit q of a (B, 2^n) state."""
    B = state.shape[0]
    s = state.reshape(B, 2**q, 2, 2 ** (n - q - 1))
    s = torch.einsum("ij,bljr->blir", U, s) if U.dim() == 2 else torch.einsum("bij,bljr->blir", U, s)
    return s.reshape(B, 2**n)


def apply_cnot(state, c, t, n):
    return state[:, _cnot_perm(n, c, t).to(state.device)]


def angle_embedding_y(x):
    """Product state RY(x_i)|0> for each qubit; x has shape (B, n)."""
    c, s = torch.cos(x / 2), torch.sin(x / 2)
    state = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    for i in range(x.shape[1]):
        state = torch.stack([state * c[:, i : i + 1], state * s[:, i : i + 1]], dim=-1).reshape(x.shape[0], -1)
    return state.to(torch.complex128 if x.dtype == torch.float64 else torch.complex64)


def rot_matrices(w):
    """PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi) for each row of w (n, 3); returns (n, 2, 2)."""
    phi, theta, omega = w[:, 0], w[:, 1], w[:, 2]
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    a, d = (phi + omega) / 2, (phi - omega) / 2
    row0 = torch.stack([torch.exp(-1j * a) * c, -torch.exp(1j * d) * s], dim=-1)
    row1 = torch.stack([torch.exp(-1j * d) * s, torch.exp(1j * a) * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


@lru_cache(maxsize=None)
def _ring_perm(n, r):
    """Combined permutation of the CNOT ring i -> (i + r) mod n, applied in order i = 0..n-1."""
    perm = torch.arange(2**n)
    for i in range(n):
        perm = perm[_cnot_perm(n, i, (i + r) % n)]
    return perm


def strongly_entangling(state, weights, n):
    """PennyLane StronglyEntanglingLayers with default ranges; weights has shape (L, n, 3).

    Each layer's single-qubit rotations are combined into one 2^n x 2^n Kronecker product,
    and its CNOT ring into one basis permutation."""
    for l in range(weights.shape[0]):
        rots = rot_matrices(weights[l]).to(state.dtype)
        U = rots[0]
        for i in range(1, n):
            U = torch.kron(U, rots[i])
        state = state @ U.T
        if n > 1:
            state = state[:, _ring_perm(n, l % (n - 1) + 1).to(state.device)]
    return state


def expval_z(state, n):
    """<Z_i> for every qubit, shape (B, n)."""
    probs = state.abs() ** 2
    z = torch.as_tensor(1 - 2 * _bits(n), dtype=probs.dtype, device=probs.device)
    return probs @ z


def zz_feature_map(x, reps):
    """ZZFeatureMap (Havlicek et al. 2019) with full entanglement: reps of H, then the diagonal
    exp(i [sum_i x_i Z_i + sum_{i<j} (pi - x_i)(pi - x_j) Z_i Z_j]) up to a global phase,
    implemented as P(2 x_i) and CNOT-P(2 (pi - x_i)(pi - x_j))-CNOT as in Qiskit."""
    B, n = x.shape
    bits = torch.as_tensor(_bits(n), dtype=x.dtype, device=x.device)  # (2^n, n)
    phase = 2 * x @ bits.T                                             # single-qubit P(2 x_i)
    for i in range(n):
        for j in range(i + 1, n):
            parity = (bits[:, i] + bits[:, j]) % 2
            phase = phase + 2 * ((np.pi - x[:, i]) * (np.pi - x[:, j]))[:, None] * parity[None, :]
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    h = torch.tensor([[1, 1], [1, -1]], dtype=cdtype, device=x.device) / np.sqrt(2)
    state = torch.zeros(B, 2**n, dtype=cdtype, device=x.device)
    state[:, 0] = 1
    for _ in range(reps):
        for q in range(n):
            state = apply_1q(state, h, q, n)
        state = state * torch.exp(1j * phase.to(cdtype))
    return state
