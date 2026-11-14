"""Baseline ghosting models, from the simplest.

All models share one interface: ``fit(train, val=None)`` then ``predict(data) -> (n, K, 2)``
(only the MLP uses ``val``, for early stopping).
Every baseline is fitted separately per group (``schema.group_col``): per defending
team for Metrica (slot k is a different role in each formation), per back-line size
(3, 4 or 5) for the World Cup (one model for all nations).
Targets that do not exist (unused slots of a short line) are ignored via the mask.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from horse_ml.ghosting.dataset import GhostData, feature_columns


class GhostModel:
    """Base class of every ghosting model."""

    name = "model"

    def fit(self, train: GhostData, val: GhostData | None = None) -> GhostModel:  # pragma: no cover
        raise NotImplementedError

    def predict(self, data: GhostData) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


def _check_groups(groups: np.ndarray, known: set) -> None:
    unknown = set(groups) - known
    if unknown:
        raise ValueError(f"No model for group(s) {unknown}")


class MeanShape(GhostModel):
    """(a) Average position of each slot per group. Ignores the ball."""

    name = "mean_shape"

    def __init__(self) -> None:
        self.shapes: dict = {}

    def fit(self, train: GhostData, val: GhostData | None = None) -> MeanShape:
        group = train.group
        self.shapes = {}
        for g in sorted(set(group), key=str):
            rows = group == g
            w = train.mask[rows][..., None].astype(float)
            count = np.maximum(w.sum(axis=0), 1.0)
            self.shapes[g] = (train.Y[rows] * w).sum(axis=0) / count
        return self

    def predict(self, data: GhostData) -> np.ndarray:
        _check_groups(data.group, set(self.shapes))
        return np.stack([self.shapes[g] for g in data.group])


class _PerGroupSklearn(GhostModel):
    """Fits one scikit-learn regressor per group on chosen input columns."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.models: dict = {}

    def make(self) -> Pipeline:  # pragma: no cover - interface
        raise NotImplementedError

    def fit(self, train: GhostData, val: GhostData | None = None) -> _PerGroupSklearn:
        group = train.group
        X = train.columns(self.columns)
        Y = train.Y.reshape(len(train), -1)
        self.models = {g: self.make().fit(X[group == g], Y[group == g]) for g in sorted(set(group), key=str)}
        return self

    def predict(self, data: GhostData) -> np.ndarray:
        group = data.group
        _check_groups(group, set(self.models))
        X = data.columns(self.columns)
        k = data.Y.shape[1]
        out = np.empty((len(data), k * 2))
        for g, model in self.models.items():
            rows = group == g
            if rows.any():
                out[rows] = model.predict(X[rows])
        return out.reshape(len(data), k, 2)


class BallRidge(_PerGroupSklearn):
    """(b) Shape that follows the ball: ridge regression on the ball position.

    Quadratic terms (x², y², x·y) let the block bend, e.g. squeeze towards the
    ball side and drop deeper when the ball is close to goal.
    """

    name = "ball_ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        super().__init__(["ball_x", "ball_y"])

    def make(self) -> Pipeline:
        return make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=self.alpha)
        )

    def describe(self) -> dict:
        return {"name": self.name, "alpha": self.alpha}


class SituationKNN(_PerGroupSklearn):
    """(c) Average targets of the k most similar train situations (ball + other players).

    Inputs are standardized so that every coordinate weighs the same in the
    distance. Players are compared slot by slot (same role order everywhere).
    """

    name = "situation_knn"

    def __init__(self, k: int = 10, velocity: bool = False, columns: list[str] | None = None) -> None:
        self.k = k
        self.velocity = velocity
        super().__init__(columns if columns is not None else feature_columns(velocity=velocity))

    def make(self) -> Pipeline:
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=self.k))

    def describe(self) -> dict:
        return {"name": self.name, "k": self.k, "velocity": self.velocity}
