"""(d) A simple MLP: game situation -> target defenders.

Metrica: ball + attackers (positions, velocities) -> 11 defenders.
World Cup: ball + attackers + defenders outside the line (with visibility and
presence flags) -> the defensive line in 5 common slots, masked.

Training details:
- Inputs standardized per feature with train statistics.
- Targets centred per slot/axis and divided by ONE shared scale for x and y, so
  the network works on small numbers; the loss un-does this and measures the
  Euclidean distance in metres over the existing targets (mask). Loss = metric.
- Adam, mini-batches, early stopping on the validation error (best weights kept).
- Fixed seed, CPU only.

Two variants (group = defending team for Metrica, back-line size for World Cup):
- per group: one network per group;
- unified: one network for all, the group as a one-hot input.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from horse_ml.ghosting.baselines import GhostModel
from horse_ml.ghosting.dataset import GhostData


@dataclass
class MLPConfig:
    # Strong regularization: with one game the samples are highly correlated and a
    # lightly regularized network overfits after 1-2 epochs (chosen on validation).
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 400
    patience: int = 20
    seed: int = 0


@dataclass
class TrainHistory:
    best_epoch: int = 0
    best_val_m: float = float("inf")
    train_m: list[float] = field(default_factory=list)
    val_m: list[float] = field(default_factory=list)
    val_source: str = "validation"


def build_net(n_in: int, n_out: int, config: MLPConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for _ in range(config.layers):
        layers += [nn.Linear(width, config.hidden), nn.ReLU(), nn.Dropout(config.dropout)]
        width = config.hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def distance_loss(pred_m: torch.Tensor, true_m: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean Euclidean distance in metres over existing targets. (n, K, 2) tensors.

    A tiny epsilon keeps the gradient finite when a prediction is exact.
    """
    dist = torch.sqrt(((pred_m - true_m) ** 2).sum(dim=-1) + 1e-8)
    if mask is None:
        return dist.mean()
    return (dist * mask).sum() / mask.sum().clamp(min=1.0)


class _Normalizer:
    """Train-set statistics for inputs and targets (targets: existing ones only)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.x_mean = X.mean(axis=0)
        self.x_std = np.maximum(X.std(axis=0), 1e-6)
        m = np.ones(Y.shape[:2], dtype=bool) if mask is None else mask
        w = m[..., None].astype(float)
        self.y_mean = (Y * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1.0)  # (K, 2)
        self.y_scale = float(np.sqrt((((Y - self.y_mean) ** 2) * w).sum() / max(w.sum(), 1.0)))
        self.k = Y.shape[1]

    def x(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)

    def to_metres(self, out: torch.Tensor) -> torch.Tensor:
        """Network output (n, 2K) -> positions in metres (n, K, 2)."""
        y_mean = torch.as_tensor(self.y_mean, dtype=torch.float32)
        return out.view(-1, getattr(self, "k", 11), 2) * self.y_scale + y_mean


class _Trained:
    """One trained network with its normalizer."""

    def __init__(self, net: nn.Module, norm: _Normalizer, history: TrainHistory) -> None:
        self.net = net
        self.norm = norm
        self.history = history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.norm.to_metres(self.net(self.norm.x(X))).numpy()


def train_net(
    X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, config: MLPConfig,
    mask_train: np.ndarray | None = None, mask_val: np.ndarray | None = None,
) -> _Trained:
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    norm = _Normalizer(X_train, Y_train, mask_train)
    xt, yt = norm.x(X_train), torch.as_tensor(Y_train, dtype=torch.float32)
    xv, yv = norm.x(X_val), torch.as_tensor(Y_val, dtype=torch.float32)
    mt = None if mask_train is None else torch.as_tensor(mask_train, dtype=torch.float32)
    mv = None if mask_val is None else torch.as_tensor(mask_val, dtype=torch.float32)

    net = build_net(xt.shape[1], Y_train.shape[1] * 2, config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = TrainHistory()
    best_state = copy.deepcopy(net.state_dict())
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        net.train()
        order = torch.randperm(len(xt), generator=generator)
        total = 0.0
        for start in range(0, len(xt), config.batch_size):
            idx = order[start : start + config.batch_size]
            loss = distance_loss(norm.to_metres(net(xt[idx])), yt[idx], None if mt is None else mt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.train_m.append(total / len(xt))

        net.eval()
        with torch.no_grad():
            val_m = distance_loss(norm.to_metres(net(xv)), yv, mv).item()
        history.val_m.append(val_m)

        if val_m < history.best_val_m - 1e-4:
            history.best_val_m, history.best_epoch = val_m, epoch
            best_state = copy.deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    net.load_state_dict(best_state)
    return _Trained(net, norm, history)


class MLPGhost(GhostModel):
    def __init__(self, per_team: bool, config: MLPConfig | None = None) -> None:
        self.per_team = per_team  # True = one network per group
        self.config = config or MLPConfig()
        self.name = "mlp_per_team" if per_team else "mlp_unified"
        self.teams: list = []  # the groups (defending teams for Metrica, line sizes for World Cup)
        self.trained: dict = {}

    def _inputs(self, data: GhostData) -> np.ndarray:
        if self.per_team:
            return data.X
        one_hot = (data.group[:, None] == np.array(self.teams, dtype=object)[None, :]).astype(float)
        return np.hstack([data.X, one_hot])

    @staticmethod
    def _masks(data: GhostData) -> np.ndarray | None:
        return data.mask if data.schema.masked else None

    def fit(self, train: GhostData, val: GhostData | None = None) -> MLPGhost:
        if val is None:
            raise ValueError("The MLP needs a validation set for early stopping")
        self.teams = sorted(set(train.group), key=str)
        if self.per_team:
            for g in self.teams:
                tr, va = train.subset(train.group == g), val.subset(val.group == g)
                source = "validation"
                if len(va) == 0:
                    # group missing from validation: hold out the last 15% of its train
                    # samples (time order) for early stopping only
                    cut = int(0.85 * len(tr))
                    tr, va = tr.subset(np.arange(len(tr)) < cut), tr.subset(np.arange(len(tr)) >= cut)
                    source = "last 15% of train (group absent from validation)"
                self.trained[g] = train_net(tr.X, tr.Y, va.X, va.Y, self.config, self._masks(tr), self._masks(va))
                self.trained[g].history.val_source = source
        else:
            self.trained["all"] = train_net(
                self._inputs(train), train.Y, self._inputs(val), val.Y, self.config,
                self._masks(train), self._masks(val),
            )
        return self

    def predict(self, data: GhostData) -> np.ndarray:
        if not self.per_team:
            return self.trained["all"].predict(self._inputs(data))
        out = np.empty(data.Y.shape)
        group = data.group
        unknown = set(group) - set(self.trained)
        if unknown:
            raise ValueError(f"No model for group(s) {unknown}")
        for g, model in self.trained.items():
            rows = group == g
            if rows.any():
                out[rows] = model.predict(data.X[rows])
        return out

    def describe(self) -> dict:
        return {
            "name": self.name,
            **asdict(self.config),
            "best_epoch": {str(k): m.history.best_epoch for k, m in self.trained.items()},
            "early_stopping_on": {str(k): getattr(m.history, "val_source", "validation")
                                  for k, m in self.trained.items()},
        }
