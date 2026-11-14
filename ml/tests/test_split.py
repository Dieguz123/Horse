"""Alternating time-block split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_ml.preprocessing.split import assign_splits

GAP = 10.0
STEP = 0.2  # 5 Hz


def make_samples() -> pd.DataFrame:
    # two games x two periods, 45 minutes each; defending team changes every 40 s
    t = np.arange(0, 45 * 60, STEP)
    team = np.where((t // 40) % 2 == 0, "A", "B")
    parts = [
        pd.DataFrame({"game_id": g, "period": p, "timestamp_s": t, "defending_team": team})
        for g in ("game_a", "game_b")
        for p in (1, 2)
    ]
    samples = pd.concat(parts, ignore_index=True)
    samples["sample_id"] = samples.index.astype(str)
    return samples


def test_no_sample_in_two_splits_and_ids_disjoint() -> None:
    samples = make_samples()
    split, _ = assign_splits(samples, gap_s=GAP, trials=200)
    ids = {name: set(samples.loc[split == name, "sample_id"]) for name in ("train", "val", "test")}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert split.isna().mean() < 0.05  # only the boundary gaps are discarded


def test_gap_between_different_splits() -> None:
    samples = make_samples()
    samples["split"], _ = assign_splits(samples, gap_s=GAP, trials=200)
    for _, group in samples.dropna(subset=["split"]).groupby(["game_id", "period"]):
        group = group.sort_values("timestamp_s")
        changes = group["split"].to_numpy()[1:] != group["split"].to_numpy()[:-1]
        jumps = np.diff(group["timestamp_s"].to_numpy())[changes]
        assert (jumps >= GAP - 1e-9).all()


def test_blocks_alternate_and_fractions_respected() -> None:
    samples = make_samples()
    split, blocks = assign_splits(samples, gap_s=GAP, trials=200)
    shares = split.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.03
    assert abs(shares["val"] - 0.15) < 0.03
    assert abs(shares["test"] - 0.15) < 0.03
    # val and test are spread over several blocks, not one chunk at the end
    assert sum(b.split == "val" for b in blocks) >= 3
    assert sum(b.split == "test" for b in blocks) >= 3
    # every block is ~3 minutes
    lengths = [b.end_s - b.start_s for b in blocks]
    assert max(lengths) < 2 * 180


def test_defending_team_is_balanced() -> None:
    samples = make_samples()
    samples["split"], _ = assign_splits(samples, gap_s=GAP, trials=500)
    overall = (samples["defending_team"] == "A").mean()
    for name in ("val", "test"):
        share = (samples.loc[samples["split"] == name, "defending_team"] == "A").mean()
        assert abs(share - overall) < 0.05


def test_split_is_deterministic() -> None:
    samples = make_samples()
    a, _ = assign_splits(samples, trials=100, seed=3)
    b, _ = assign_splits(samples, trials=100, seed=3)
    pd.testing.assert_series_equal(a, b)
