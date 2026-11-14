"""Ghosting error: distance in metres between predicted and real defenders.

Per sample, the error is the mean over the target defenders (11 for Metrica, the
3-5 players of the defensive line for the World Cup) of the Euclidean distance
between predicted and real position. Reports aggregate it over samples (mean and
median), per slot, per defending team, on "settled" situations only and, for the
World Cup, on visible targets only (defenders tracked on camera, not estimated).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from horse_ml.ghosting.dataset import GhostData


def defender_distances(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(n, K) distance in metres per defender. ``pred``, ``true``: (n, K, 2)."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    count = mask.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (values * mask).sum(axis=axis) / count, np.nan)


def sample_errors(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(n,) mean distance over the target defenders of each sample."""
    dist = defender_distances(pred, true)
    if mask is None:
        return dist.mean(axis=1)
    return masked_mean(dist, mask, axis=1)


@dataclass
class ErrorReport:
    n: int
    mean: float
    median: float
    settled_n: int
    settled_mean: float
    visible_n: int  # samples with at least one visible target
    visible_mean: float  # error on visible targets only
    per_team: dict[str, float]
    per_slot: list[float]  # mean distance per slot, over samples where the slot is used
    per_team_slot: dict[str, list[float]]
    per_group: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _nan_to_none(values: np.ndarray) -> list:
    return [None if not np.isfinite(v) else round(float(v), 4) for v in values]


def report(pred: np.ndarray, data: GhostData) -> ErrorReport:
    dist = defender_distances(pred, data.Y)
    errors = masked_mean(dist, data.mask, axis=1)
    vis_errors = masked_mean(dist, data.visible, axis=1)
    has_vis = np.isfinite(vis_errors)
    settled = data.settled
    team, group = data.team, data.group
    return ErrorReport(
        n=len(errors),
        mean=float(np.nanmean(errors)),
        median=float(np.nanmedian(errors)),
        settled_n=int(settled.sum()),
        settled_mean=float(np.nanmean(errors[settled])) if settled.any() else float("nan"),
        visible_n=int(has_vis.sum()),
        visible_mean=float(vis_errors[has_vis].mean()) if has_vis.any() else float("nan"),
        per_team={str(t): float(np.nanmean(errors[team == t])) for t in sorted(set(team))},
        per_slot=_nan_to_none(masked_mean(dist, data.mask, axis=0)),
        per_team_slot={
            str(t): _nan_to_none(masked_mean(dist[team == t], data.mask[team == t], axis=0))
            for t in sorted(set(team))
        },
        per_group={str(g): float(np.nanmean(errors[group == g])) for g in sorted(set(group), key=str)},
    )
