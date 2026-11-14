"""Velocities from tracking positions with a Savitzky-Golay filter.

Differentiating raw positions amplifies noise: a 3 cm jitter between two frames
at 25 Hz becomes 0.75 m/s of fake speed. Savitzky-Golay fits a low-order
polynomial to a sliding window and differentiates the polynomial, i.e. it is the
velocity of the *smoothed* trajectory, computed in one step.

Default window: 7 frames (0.28 s at 25 Hz), polynomial order 2.
- Long enough to average out frame-to-frame jitter.
- Shorter than a typical change of direction (0.3-0.5 s), so cuts and sprints
  are not flattened.
The build script prints speed percentiles for several windows to back this up.

The filter only runs on contiguous runs of valid (non-NaN) positions inside one
period; runs shorter than the window get NaN velocity. Speeds above a physical
limit are set to NaN (not clipped: a clipped value would be invented data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DEFAULT_WINDOW = 7
DEFAULT_POLYORDER = 2
MAX_PLAYER_SPEED = 12.0  # m/s, faster than any sprinter
MAX_BALL_SPEED = 40.0  # m/s, ~145 km/h, above the hardest shots


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs (end exclusive) of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(valid, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def savgol_velocity(
    x: np.ndarray,
    y: np.ndarray,
    fps: float,
    window: int = DEFAULT_WINDOW,
    polyorder: int = DEFAULT_POLYORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """vx, vy (units per second) of one entity in one period."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)
    for start, end in valid_runs(~(np.isnan(x) | np.isnan(y))):
        if end - start < window:
            continue
        vx[start:end] = savgol_filter(x[start:end], window, polyorder, deriv=1, delta=1 / fps)
        vy[start:end] = savgol_filter(y[start:end], window, polyorder, deriv=1, delta=1 / fps)
    return vx, vy


def add_velocities(
    df: pd.DataFrame,
    entities: list[str],
    fps: float,
    *,
    window: int = DEFAULT_WINDOW,
    polyorder: int = DEFAULT_POLYORDER,
    max_player_speed: float = MAX_PLAYER_SPEED,
    max_ball_speed: float = MAX_BALL_SPEED,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add ``<entity>_vx, _vy, _speed`` columns. ``df`` must be sorted by period, frame.

    Returns the new frame and the number of frames invalidated per entity kind.
    """
    period_ids = df["period_id"].to_numpy()
    new: dict[str, np.ndarray] = {}
    invalidated = {"player": 0, "ball": 0}
    for entity in entities:
        x = df[f"{entity}_x"].to_numpy(dtype=float)
        y = df[f"{entity}_y"].to_numpy(dtype=float)
        vx = np.full_like(x, np.nan)
        vy = np.full_like(y, np.nan)
        for period in np.unique(period_ids):
            idx = np.flatnonzero(period_ids == period)
            vx[idx], vy[idx] = savgol_velocity(x[idx], y[idx], fps, window, polyorder)
        speed = np.hypot(vx, vy)

        kind = "ball" if entity == "ball" else "player"
        limit = max_ball_speed if kind == "ball" else max_player_speed
        too_fast = speed > limit  # NaN compares False
        invalidated[kind] += int(too_fast.sum())
        vx[too_fast] = vy[too_fast] = speed[too_fast] = np.nan

        new[f"{entity}_vx"], new[f"{entity}_vy"], new[f"{entity}_speed"] = vx, vy, speed
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1), invalidated
