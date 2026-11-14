"""Velocities from positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_ml.preprocessing.kinematics import add_velocities, savgol_velocity, valid_runs

FPS = 25.0


def straight_run(speed: float, angle_deg: float, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / FPS
    a = np.deg2rad(angle_deg)
    return 10 + speed * np.cos(a) * t, -5 + speed * np.sin(a) * t


def test_straight_line_at_known_speed() -> None:
    x, y = straight_run(speed=7.0, angle_deg=30)
    vx, vy = savgol_velocity(x, y, FPS)
    np.testing.assert_allclose(np.hypot(vx, vy), 7.0, atol=1e-9)
    np.testing.assert_allclose(vx, 7.0 * np.cos(np.deg2rad(30)), atol=1e-9)
    np.testing.assert_allclose(vy, 7.0 * np.sin(np.deg2rad(30)), atol=1e-9)


def test_noise_is_smoothed() -> None:
    rng = np.random.default_rng(0)
    x, y = straight_run(speed=5.0, angle_deg=0, n=500)
    x_noisy = x + rng.normal(0, 0.03, len(x))  # 3 cm tracking jitter
    naive = np.diff(x_noisy) * FPS
    vx, _ = savgol_velocity(x_noisy, y, FPS)
    assert np.abs(vx - 5.0).mean() < 0.5 * np.abs(naive - 5.0).mean()


def test_gaps_and_short_runs_give_nan() -> None:
    x, y = straight_run(speed=5.0, angle_deg=0, n=30)
    x[10] = np.nan  # splits into runs of 10 and 19 frames
    x[25:] = np.nan  # second run: 14 frames
    assert valid_runs(~np.isnan(x)) == [(0, 10), (11, 25)]
    vx, _ = savgol_velocity(x, y, FPS, window=11)
    assert np.isnan(vx[10]) and np.isnan(vx[25:]).all()
    assert np.isnan(vx[:10]).all()  # run of 10 frames < window of 11
    np.testing.assert_allclose(vx[11:25], 5.0, atol=1e-9)


def test_impossible_speed_is_invalidated_and_periods_are_separate() -> None:
    x_fast, y_fast = straight_run(speed=15.0, angle_deg=0, n=20)  # above 12 m/s
    x_ok, y_ok = straight_run(speed=6.0, angle_deg=90, n=20)
    df = pd.DataFrame(
        {
            "period_id": [1] * 20 + [2] * 20,
            "p1_x": np.concatenate([x_ok, x_fast]),
            "p1_y": np.concatenate([y_ok, y_fast]),
        }
    )
    out, invalidated = add_velocities(df, ["p1"], FPS)
    np.testing.assert_allclose(out["p1_speed"][:20], 6.0, atol=1e-9)
    assert out["p1_speed"][20:].isna().all()
    assert invalidated == {"player": 20, "ball": 0}
    # the jump between the periods did not leak into the velocities
    assert out["p1_vx"][:20].abs().max() == pytest.approx(0.0, abs=1e-9)
