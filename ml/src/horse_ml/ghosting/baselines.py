"""Baseline ghosting models, from the simplest.

All models share one interface: ``fit(train, val=None)`` then ``predict(data) -> (n, 11, 2)``
(only the MLP uses ``val``, for early stopping).
Every baseline is fitted separately per defending team: slot k is a different
role in each formation, so pooling teams would mix different players.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from horse_ml.ghosting.dataset import GhostData, feature_columns
from horse_ml.preprocessing.roles import N_SLOTS


class GhostModel:
    """Base class: per-team fitting and prediction routing."""

    name = "model"

    def fit(self, train: GhostData, val: GhostData | None = None) -> GhostModel:  # pragma: no cover
        raise NotImplementedError

    def predict(self, data: GhostData) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class MeanShape(GhostModel):
    """(a) Average position of each slot for the defending team. Ignores the ball."""

    name = "mean_shape"

    def __init__(self) -> None:
        self.shapes: dict[str, np.ndarray] = {}

    def fit(self, train: GhostData, val: GhostData | None = None) -> MeanShape:
        team = train.team
        self.shapes = {t: train.Y[team == t].mean(axis=0) for t in sorted(set(team))}
        return self

    def predict(self, data: GhostData) -> np.ndarray:
        return np.stack([self.shapes[t] for t in data.team])


class _PerTeamSklearn(GhostModel):
    """Fits one scikit-learn regressor per defending team on chosen input columns."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.models: dict[str, Pipeline] = {}

    def make(self) -> Pipeline:  # pragma: no cover - interface
        raise NotImplementedError

    def fit(self, train: GhostData, val: GhostData | None = None) -> _PerTeamSklearn:
        team = train.team
        X = train.columns(self.columns)
        Y = train.Y.reshape(len(train), -1)
        self.models = {t: self.make().fit(X[team == t], Y[team == t]) for t in sorted(set(team))}
        return self

    def predict(self, data: GhostData) -> np.ndarray:
        team = data.team
        X = data.columns(self.columns)
        out = np.empty((len(data), N_SLOTS * 2))
        for t, model in self.models.items():
            rows = team == t
            if rows.any():
                out[rows] = model.predict(X[rows])
        unknown = ~np.isin(team, list(self.models))
        if unknown.any():
            raise ValueError(f"No model for defending team(s) {set(team[unknown])}")
        return out.reshape(len(data), N_SLOTS, 2)


class BallRidge(_PerTeamSklearn):
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


class SituationKNN(_PerTeamSklearn):
    """(c) Average defenders of the k most similar train situations (ball + attackers).

    Inputs are standardized so that every coordinate weighs the same in the
    distance. Attackers are compared slot by slot (same role order everywhere).
    """

    name = "situation_knn"

    def __init__(self, k: int = 10, velocity: bool = False) -> None:
        self.k = k
        self.velocity = velocity
        super().__init__(feature_columns(velocity=velocity))

    def make(self) -> Pipeline:
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=self.k))

    def describe(self) -> dict:
        return {"name": self.name, "k": self.k, "velocity": self.velocity}
