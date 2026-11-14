"""Split by whole games: a game is entirely in train, validation or test.

The assignment lives in ``reference/pff_splits.csv`` (versioned) so it stays stable
when games are added.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from horse_ml.paths import REFERENCE_DIR

SPLITS_FILE = REFERENCE_DIR / "pff_splits.csv"
VALID = ("train", "val", "test")


def load_split_map(path: Path = SPLITS_FILE) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        mapping = {row["game_id"]: row["split"] for row in csv.DictReader(f)}
    bad = {g: s for g, s in mapping.items() if s not in VALID}
    if bad:
        raise ValueError(f"Unknown split names in {path}: {bad}")
    return mapping


def assign_game_splits(samples: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    games = samples["game_id"].astype(str)
    missing = sorted(set(games) - set(mapping))
    if missing:
        raise ValueError(f"Games without a split in pff_splits.csv: {missing}")
    return games.map(mapping)
