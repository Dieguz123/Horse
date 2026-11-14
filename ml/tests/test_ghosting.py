"""Ghosting metric, baselines and train/val/test separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ghost_fixtures import synthetic_samples
from horse_ml.ghosting.baselines import BallRidge, MeanShape
from horse_ml.ghosting.dataset import from_frame
from horse_ml.ghosting.metrics import defender_distances, report, sample_errors
from horse_ml.ghosting.mlp import MLPConfig, MLPGhost
from horse_ml.ghosting.train import fit_baselines, train_and_validation

# ------------------------------------------------------------------------ metric


def test_metric_hand_made_cases() -> None:
    true = np.zeros((2, 11, 2))
    pred = np.zeros((2, 11, 2))
    pred[1, 0] = [3.0, 4.0]  # one defender 5 m off in the second sample
    assert sample_errors(pred, true).tolist() == pytest.approx([0.0, 5.0 / 11])

    pred_all = np.zeros((1, 11, 2)) + [3.0, 4.0]  # every defender 5 m off
    assert sample_errors(pred_all, true[:1]) == pytest.approx([5.0])
    assert defender_distances(pred_all, true[:1]).shape == (1, 11)


def test_report_breakdowns() -> None:
    frame = synthetic_samples(n=4)
    frame["seconds_since_possession_change"] = [1.0, 5.0, 10.0, 2.0]
    data = from_frame(frame)
    pred = data.Y.copy()
    pred[0] += [3.0, 4.0]  # sample 0 (Team A, not settled): 5 m error everywhere
    r = report(pred, data)
    assert r.mean == pytest.approx(5.0 / 4)
    assert r.settled_n == 2 and r.settled_mean == pytest.approx(0.0)
    assert r.per_team == pytest.approx({"Team A": 2.5, "Team B": 0.0})
    assert r.per_slot == pytest.approx([5.0 / 4] * 11)
    assert r.per_team_slot["Team A"] == pytest.approx([2.5] * 11)


# --------------------------------------------------------------------- baselines


def test_mean_shape_is_the_per_team_average() -> None:
    data = from_frame(synthetic_samples())
    model = MeanShape().fit(data)
    pred = model.predict(data)
    for team in ("Team A", "Team B"):
        rows = data.team == team
        np.testing.assert_allclose(pred[rows], np.broadcast_to(data.Y[rows].mean(axis=0), pred[rows].shape))


def test_ridge_recovers_a_shape_that_follows_the_ball() -> None:
    data = from_frame(synthetic_samples())
    pred = BallRidge(alpha=1e-6).fit(data).predict(data)
    assert report(pred, data).mean < 1e-3  # targets are exactly linear in the ball


# ------------------------------------------------------------- no data leakage


@pytest.fixture
def poisoned_parquet(tmp_path: Path) -> Path:
    """Test targets set to absurd values: any leak into training would drag
    predictions far outside the pitch."""
    frame = synthetic_samples()
    target_cols = [c for c in frame.columns if c.startswith("def_")]
    frame.loc[frame["split"] == "test", target_cols] = 1e6
    path = tmp_path / "samples.parquet"
    frame.to_parquet(path)
    return path


def test_training_sees_no_test_rows_and_splits_are_disjoint(poisoned_parquet: Path) -> None:
    train, val = train_and_validation(poisoned_parquet)
    assert set(train.frame["split"]) == {"train"}
    assert set(val.frame["split"]) == {"val"}
    assert not set(train.frame["sample_id"]) & set(val.frame["sample_id"])
    assert np.abs(train.Y).max() < 1e3 and np.abs(val.Y).max() < 1e3  # no poisoned test rows


def test_models_are_fitted_on_train_only(poisoned_parquet: Path) -> None:
    train, val = train_and_validation(poisoned_parquet)
    # poison validation targets too: they may only be used to *choose* settings
    val.Y[:] = 1e6
    models = [m for m, _ in fit_baselines(train, val)]
    models.append(MLPGhost(per_team=False, config=MLPConfig(max_epochs=3, patience=3)).fit(train, val))
    for model in models:
        pred = model.predict(train)
        assert np.abs(pred).max() < 200, model.name  # would be ~1e6 if val/test leaked
