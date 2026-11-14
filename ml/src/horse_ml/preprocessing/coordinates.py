"""Coordinate conversion and attacking-direction normalization.

Target system (same as kloppy's "secondspectrum"): metres, origin at the centre
spot, x along the pitch length in [-L/2, L/2], y across in [-W/2, W/2], y up.

Why not ``dataset.transform(to_coordinate_system="secondspectrum")``? It gives the
same numbers (a test checks this) but transforms frame objects one by one: ~4
extra minutes per game. The conversion is an affine map, so we apply it to whole
numpy arrays instead.

Two normalization steps, in this order:
1. ``home_attacks_right`` sign per period: undo the half-time side swap, so the
   home team always attacks to the right. Positions stay continuous within a
   period, so velocities can be computed safely afterwards.
2. ``flip`` per frame (after velocities): rotate the pitch 180 degrees for frames
   where the team in possession is the away team, so the attacking team always
   attacks left-to-right. With the origin at the centre this is just a sign
   change of x, y, vx and vy.
"""

from __future__ import annotations

import numpy as np
from kloppy.domain import Orientation


def metrica_to_metres(
    x: np.ndarray | float, y: np.ndarray | float, pitch_length: float, pitch_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Metrica [0, 1] (origin top-left, y down) -> metres (origin centre, y up)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x - 0.5) * pitch_length, (0.5 - y) * pitch_width


def home_attacks_right(orientation: Orientation, period_id: int) -> bool:
    """Whether the home team attacks towards +x in ``period_id`` (regular time only)."""
    if period_id not in (1, 2):
        raise ValueError(f"Attacking direction not defined for period {period_id}")
    if orientation == Orientation.STATIC_HOME_AWAY:
        return True
    if orientation == Orientation.STATIC_AWAY_HOME:
        return False
    if orientation == Orientation.HOME_AWAY:
        return period_id == 1
    if orientation == Orientation.AWAY_HOME:
        return period_id == 2
    raise ValueError(f"Cannot derive the attacking direction from orientation {orientation}")


def period_signs(period_ids: np.ndarray, orientation: Orientation) -> np.ndarray:
    """+1 for frames where home already attacks right, -1 where the pitch must be rotated."""
    period_ids = np.asarray(period_ids)
    signs = np.ones(len(period_ids))
    for period in np.unique(period_ids):
        if not home_attacks_right(orientation, int(period)):
            signs[period_ids == period] = -1.0
    return signs


def flip(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rotate the pitch 180 degrees on rows where ``mask`` is True.

    ``values`` holds x, y, vx or vy (any of them: all change sign), shape
    (n_rows,) or (n_rows, n_columns). Returns a new array.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    sign = np.where(mask, -1.0, 1.0)
    return values * (sign[:, None] if values.ndim == 2 else sign)
