"""Scikit-learn compatible wrapper for PyTorch binary classifiers, trained with class-weighted BCE."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class TorchBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Subclasses implement build_module(n_features) returning a module that outputs one logit per sample."""

    def __init__(self, lr=1e-3, weight_decay=0.0, epochs=100, batch_size=64, seed=0, device="cpu"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def build_module(self, n_features):
        raise NotImplementedError

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        self.classes_ = np.array([0, 1])
        self.module_ = self.build_module(X.shape[1]).to(self.device)
        pos_weight = (y == 0).sum() / (y == 1).sum().clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.module_.train()
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(y)), max(1, len(y) // self.batch_size)):
                opt.zero_grad()
                loss = loss_fn(self.module_(X[batch]).squeeze(-1), y[batch])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def decision_function(self, X):
        self.module_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.module_(X).squeeze(-1).cpu().numpy()

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def n_parameters(self):
        return sum(p.numel() for p in self.module_.parameters() if p.requires_grad)


class MLPClassifier(TorchBinaryClassifier):
    def __init__(self, hidden=(32,), lr=1e-3, weight_decay=0.0, epochs=100, batch_size=64, seed=0, device="cpu"):
        super().__init__(lr, weight_decay, epochs, batch_size, seed, device)
        self.hidden = hidden

    def build_module(self, n_features):
        layers, width = [], n_features
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)
