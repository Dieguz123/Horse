"""Possession from events and formation from role templates."""

from __future__ import annotations

import numpy as np

from horse_ml.preprocessing.possession import EventMark, ball_in_play, possession_per_frame
from horse_ml.preprocessing.roles import formation_from_template


def test_possession_timeline() -> None:
    marks = [
        EventMark(1, 1.0, "A", "on_ball"),  # A kicks off
        EventMark(1, 3.0, "A", "on_ball"),  # A keeps the ball
        EventMark(1, 5.0, "B", "on_ball"),  # B wins it
        EventMark(1, 7.0, "B", "stoppage"),  # ball out
        EventMark(1, 9.0, "B", "on_ball"),  # B restarts (same team: no change)
    ]
    t = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 10.0])
    out = possession_per_frame(marks, np.ones(len(t), dtype=int), t)

    owner = out["possession_team_id"]
    assert owner.isna().tolist() == [True, False, False, False, False, False]
    assert owner.iloc[1:].tolist() == ["A", "A", "B", "B", "B"]
    assert list(out["event_alive"]) == [False, True, True, True, False, True]
    np.testing.assert_allclose(out["seconds_since_possession_change"], [np.nan, 1.0, 3.0, 1.0, 3.0, 5.0])


def test_ball_in_play_needs_known_ball_inside_pitch() -> None:
    alive = np.array([True, True, True, True, False])
    bx = np.array([0.0, np.nan, 53.0, 54.0, 0.0])
    by = np.zeros(5)
    np.testing.assert_array_equal(ball_in_play(alive, bx, by, 105, 68), [True, False, True, False, False])


def test_formation_from_template() -> None:
    # Metrica template depths of Sample Game 3
    team_a = [0.02, 0.11, 0.11, 0.11, 0.11, 0.29, 0.21, 0.29, 0.21, 0.29, 0.39]
    team_b = [0.02, 0.11, 0.11, 0.11, 0.11, 0.25, 0.25, 0.25, 0.25, 0.39, 0.39]
    assert formation_from_template(team_a) == "4-2-3-1"
    assert formation_from_template(team_b) == "4-4-2"
