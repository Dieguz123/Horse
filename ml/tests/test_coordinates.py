"""Coordinate conversion and attacking-direction normalization."""

from __future__ import annotations

import numpy as np
import pytest
from kloppy.domain import Orientation

from conftest import requires_data
from horse_ml.preprocessing.coordinates import flip, home_attacks_right, metrica_to_metres, period_signs

L, W = 105.0, 68.0


@pytest.mark.parametrize(
    ("metrica", "metres"),
    [
        ((0.5, 0.5), (0.0, 0.0)),  # centre spot
        ((0.0, 0.0), (-52.5, 34.0)),  # top-left corner: y up -> positive y
        ((1.0, 1.0), (52.5, -34.0)),  # bottom-right corner
        ((0.0, 0.5), (-52.5, 0.0)),  # centre of the left goal line
        ((0.75, 0.25), (26.25, 17.0)),
    ],
)
def test_metrica_to_metres_landmarks(metrica, metres) -> None:
    x, y = metrica_to_metres(*metrica, L, W)
    assert (float(x), float(y)) == pytest.approx(metres)


def test_half_time_swap_is_undone() -> None:
    # AWAY_HOME: away attacks right in period 1, home attacks right in period 2
    assert not home_attacks_right(Orientation.AWAY_HOME, 1)
    assert home_attacks_right(Orientation.AWAY_HOME, 2)
    signs = period_signs(np.array([1, 1, 2, 2]), Orientation.AWAY_HOME)
    np.testing.assert_array_equal(signs, [-1, -1, 1, 1])

    # A home player standing near the goal he attacks in period 2 (right side)
    # appears near the left goal in period 1 raw data; both end up on the right.
    x_p1, _ = metrica_to_metres(0.1, 0.5, L, W)
    x_p2, _ = metrica_to_metres(0.9, 0.5, L, W)
    assert x_p1 * signs[0] == pytest.approx(42.0)
    assert x_p2 * signs[2] == pytest.approx(42.0)


def test_flip_rotates_only_masked_rows() -> None:
    # frame 0: home attacking (keep); frame 1: away attacking (rotate 180 degrees)
    xy = np.array([[30.0, -10.0], [-30.0, 10.0]])
    out = flip(xy, np.array([False, True]))
    np.testing.assert_array_equal(out, [[30.0, -10.0], [30.0, -10.0]])
    # 1-D input (e.g. a velocity column) and the original stays untouched
    np.testing.assert_array_equal(flip(np.array([2.0, 2.0]), np.array([True, False])), [-2.0, 2.0])
    np.testing.assert_array_equal(xy[1], [-30.0, 10.0])


def test_unknown_orientation_raises() -> None:
    with pytest.raises(ValueError):
        home_attacks_right(Orientation.BALL_OWNING_TEAM, 1)
    with pytest.raises(ValueError):
        home_attacks_right(Orientation.HOME_AWAY, 3)


@requires_data
def test_matches_kloppy_transform() -> None:
    """Our vectorized conversion equals kloppy's (slow) transform on real frames."""
    from horse_ml.data.loading import load_tracking
    from horse_ml.preprocessing.frames import build_frame_table

    tracking = load_tracking(limit=200)
    table = build_frame_table("test", tracking)
    reference = tracking.transform(
        to_coordinate_system="secondspectrum", to_orientation="STATIC_HOME_AWAY"
    ).to_df()

    pid = tracking.metadata.teams[0].players[0].player_id
    for col in (f"{pid}_x", f"{pid}_y", "ball_x", "ball_y"):
        np.testing.assert_allclose(table.df[col], reference[col], atol=1e-6, equal_nan=True)
