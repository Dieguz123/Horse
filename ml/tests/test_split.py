"""Contiguous time-block split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_ml.preprocessing.split import assign_splits

GAP = 10.0


def make_samples() -> pd.DataFrame:
    # two games x two periods, a sample every 0.2 s (5 Hz) for 45 minutes
    t = np.arange(0, 45 * 60, 0.2)
    parts = [
        pd.DataFrame({"game_id": g, "period": p, "timestamp_s": t})
        for g in ("game_a", "game_b")
        for p in (1, 2)
    ]
    return pd.concat(parts, ignore_index=True)


def test_blocks_are_ordered_and_do_not_overlap() -> None:
    samples = make_samples()
    split, blocks = assign_splits(samples, gap_s=GAP)
    samples["split"] = split

    for (game, period), group in samples.groupby(["game_id", "period"]):
        kept = group.dropna(subset=["split"])
        ranges = kept.groupby("split")["timestamp_s"].agg(["min", "max"])
        # every split present, in time order train < val < test
        assert list(ranges.sort_values("min").index) == ["train", "val", "test"]
        # no overlap and at least GAP seconds between consecutive blocks
        assert ranges.loc["val", "min"] - ranges.loc["train", "max"] >= GAP - 0.2 + 1e-9
        assert ranges.loc["test", "min"] - ranges.loc["val", "max"] >= GAP - 0.2 + 1e-9
        assert len([b for b in blocks if b.game_id == game and b.period == period]) == 3


def test_each_sample_in_exactly_one_split_and_fractions_respected() -> None:
    samples = make_samples()
    split, _ = assign_splits(samples, gap_s=GAP)
    assert split.notna().mean() > 0.98  # only the gaps are discarded
    shares = split.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.01
    assert abs(shares["val"] - 0.15) < 0.01
    assert abs(shares["test"] - 0.15) < 0.01
    # identical timestamps of different games/periods never mix splits within a group
    samples["split"] = split
    assert samples.dropna().groupby(["game_id", "period", "timestamp_s"])["split"].nunique().max() == 1


def test_no_sample_id_in_two_splits() -> None:
    samples = make_samples()
    samples["sample_id"] = samples.index.astype(str)
    split, _ = assign_splits(samples, gap_s=GAP)
    ids = {name: set(samples.loc[split == name, "sample_id"]) for name in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
