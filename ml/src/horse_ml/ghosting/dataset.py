"""Processed samples as numpy arrays for the ghosting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from horse_ml.paths import PROCESSED_DIR
from horse_ml.preprocessing.roles import N_SLOTS

SAMPLES_PATH = PROCESSED_DIR / "samples.parquet"
SETTLED_AFTER_S = 3.0  # "settled" = more than 3 s since the last possession change
SPLITS = ("train", "val", "test")

BALL_COLUMNS = ["ball_x", "ball_y", "ball_vx", "ball_vy"]
ATTACKER_COLUMNS = [f"att_{k}_{f}" for k in range(N_SLOTS) for f in ("x", "y", "vx", "vy")]
FEATURE_COLUMNS = BALL_COLUMNS + ATTACKER_COLUMNS  # 4 + 44 = 48
TARGET_COLUMNS = [f"def_{k}_{f}" for k in range(N_SLOTS) for f in ("x", "y")]


def feature_columns(*, velocity: bool = True) -> list[str]:
    if velocity:
        return FEATURE_COLUMNS
    return [c for c in FEATURE_COLUMNS if c.endswith(("_x", "_y"))]


@dataclass
class GhostData:
    """A set of samples. ``Y`` has shape (n, 11, 2): defenders' (x, y) per slot."""

    frame: pd.DataFrame  # the original rows (context columns, ids, ...)
    X: np.ndarray  # (n, 48) ball + attackers, positions and velocities
    Y: np.ndarray  # (n, 11, 2)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def team(self) -> np.ndarray:
        return self.frame["defending_team"].to_numpy(dtype=object)

    @property
    def settled(self) -> np.ndarray:
        return self.frame["seconds_since_possession_change"].to_numpy() > SETTLED_AFTER_S

    def columns(self, names: list[str]) -> np.ndarray:
        return self.frame[names].to_numpy(dtype=float)

    def subset(self, mask: np.ndarray) -> GhostData:
        mask = np.asarray(mask, dtype=bool)
        return GhostData(self.frame[mask].reset_index(drop=True), self.X[mask], self.Y[mask])

    def split(self, name: str) -> GhostData:
        return self.subset(self.frame["split"].to_numpy() == name)


def from_frame(frame: pd.DataFrame) -> GhostData:
    frame = frame.reset_index(drop=True)
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    Y = frame[TARGET_COLUMNS].to_numpy(dtype=float).reshape(len(frame), N_SLOTS, 2)
    return GhostData(frame, X, Y)


def load(path: Path = SAMPLES_PATH) -> GhostData:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python -m horse_ml.preprocessing.build` first."
        )
    return from_frame(pd.read_parquet(path))


def load_splits(path: Path = SAMPLES_PATH) -> dict[str, GhostData]:
    data = load(path)
    return {name: data.split(name) for name in SPLITS}
