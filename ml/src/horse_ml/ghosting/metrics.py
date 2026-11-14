"""Ghosting error: distance in metres between predicted and real defenders.

Per sample, the error is the mean over the 11 defenders of the Euclidean distance
between predicted and real position. Reports aggregate it over samples (mean and
median), per slot, per defending team, and on "settled" situations only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from horse_ml.ghosting.dataset import GhostData


def defender_distances(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(n, 11) distance in metres per defender. ``pred``, ``true``: (n, 11, 2)."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)


def sample_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(n,) mean distance over the 11 defenders."""
    return defender_distances(pred, true).mean(axis=1)


@dataclass
class ErrorReport:
    n: int
    mean: float
    median: float
    settled_n: int
    settled_mean: float
    per_team: dict[str, float]
    per_slot: list[float]  # mean distance per slot, over all samples
    per_team_slot: dict[str, list[float]]  # slot k is a different role for each team

    def as_dict(self) -> dict:
        return asdict(self)


def report(pred: np.ndarray, data: GhostData) -> ErrorReport:
    dist = defender_distances(pred, data.Y)
    errors = dist.mean(axis=1)
    settled = data.settled
    team = data.team
    return ErrorReport(
        n=len(errors),
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        settled_n=int(settled.sum()),
        settled_mean=float(errors[settled].mean()) if settled.any() else float("nan"),
        per_team={str(t): float(errors[team == t].mean()) for t in sorted(set(team))},
        per_slot=dist.mean(axis=0).round(4).tolist(),
        per_team_slot={
            str(t): dist[team == t].mean(axis=0).round(4).tolist() for t in sorted(set(team))
        },
    )
