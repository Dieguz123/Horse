"""Processed samples as numpy arrays for the ghosting models.

Two datasets share the same models through a ``Schema``:
- ``METRICA``: 11 defenders, all always present, models grouped by defending team;
- ``WORLDCUP``: the defensive line in 5 common slots with a presence mask (3, 4 or
  5 used), models grouped by back-line size, plus a visibility flag per target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from horse_ml.paths import PROCESSED_DIR, PROCESSED_PFF_DIR
from horse_ml.preprocessing.roles import N_SLOTS

SAMPLES_PATH = PROCESSED_DIR / "samples.parquet"
SETTLED_AFTER_S = 3.0  # "settled" = more than 3 s since the last possession change
SPLITS = ("train", "val", "test")

BALL_COLUMNS = ["ball_x", "ball_y", "ball_vx", "ball_vy"]
ATTACKER_COLUMNS = [f"att_{k}_{f}" for k in range(N_SLOTS) for f in ("x", "y", "vx", "vy")]
FEATURE_COLUMNS = BALL_COLUMNS + ATTACKER_COLUMNS  # 4 + 44 = 48
TARGET_COLUMNS = [f"def_{k}_{f}" for k in range(N_SLOTS) for f in ("x", "y")]

WC_PLAYER_FIELDS = ("x", "y", "vx", "vy", "visible", "present")
WC_FEATURE_COLUMNS = (
    ["ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_visible"]
    + [f"att_{k}_{f}" for k in range(11) for f in WC_PLAYER_FIELDS]
    + [f"dof_{k}_{f}" for k in range(8) for f in WC_PLAYER_FIELDS]
)  # 6 + 66 + 48 = 120


@dataclass(frozen=True)
class Schema:
    name: str
    samples_path: Path
    feature_columns: tuple[str, ...]
    target_prefix: str
    n_targets: int
    group_col: str  # models are fitted per group / get the group as one-hot input
    masked: bool  # targets have a presence mask and a visibility flag

    def target_columns(self) -> list[str]:
        return [f"{self.target_prefix}_{k}_{f}" for k in range(self.n_targets) for f in ("x", "y")]

    def position_columns(self) -> list[str]:
        """Feature columns without velocities (for the kNN)."""
        return [c for c in self.feature_columns if not c.endswith(("_vx", "_vy"))]


METRICA = Schema("metrica", SAMPLES_PATH, tuple(FEATURE_COLUMNS), "def", N_SLOTS, "defending_team", False)
WORLDCUP = Schema("worldcup", PROCESSED_PFF_DIR / "samples.parquet", tuple(WC_FEATURE_COLUMNS),
                  "back", 5, "back_line_size", True)
SCHEMAS = {s.name: s for s in (METRICA, WORLDCUP)}


def feature_columns(*, velocity: bool = True) -> list[str]:
    """Metrica feature columns (kept for backward compatibility)."""
    if velocity:
        return FEATURE_COLUMNS
    return [c for c in FEATURE_COLUMNS if c.endswith(("_x", "_y"))]


@dataclass
class GhostData:
    """A set of samples. ``Y``: (n, K, 2) target positions, ``mask``: (n, K) which
    targets exist, ``visible``: (n, K) which targets were tracked on camera."""

    frame: pd.DataFrame
    X: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    visible: np.ndarray
    schema: Schema = METRICA

    def __len__(self) -> int:
        return len(self.X)

    @property
    def team(self) -> np.ndarray:
        return self.frame["defending_team"].to_numpy(dtype=object)

    @property
    def group(self) -> np.ndarray:
        return self.frame[self.schema.group_col].to_numpy(dtype=object)

    @property
    def settled(self) -> np.ndarray:
        return self.frame["seconds_since_possession_change"].to_numpy() > SETTLED_AFTER_S

    def columns(self, names: list[str]) -> np.ndarray:
        return self.frame[names].to_numpy(dtype=float)

    def subset(self, mask: np.ndarray) -> GhostData:
        m = np.asarray(mask, dtype=bool)
        return GhostData(self.frame[m].reset_index(drop=True), self.X[m], self.Y[m],
                         self.mask[m], self.visible[m], self.schema)

    def split(self, name: str) -> GhostData:
        return self.subset(self.frame["split"].to_numpy() == name)


def from_frame(frame: pd.DataFrame, schema: Schema = METRICA) -> GhostData:
    frame = frame.reset_index(drop=True)
    n, k = len(frame), schema.n_targets
    X = frame[list(schema.feature_columns)].to_numpy(dtype=float)
    Y = frame[schema.target_columns()].to_numpy(dtype=float).reshape(n, k, 2)
    if schema.masked:
        mask = np.stack([frame[f"{schema.target_prefix}_{s}_present"].to_numpy() == 1 for s in range(k)], axis=1)
        visible = np.stack([frame[f"{schema.target_prefix}_{s}_visible"].to_numpy() == 1 for s in range(k)], axis=1)
        visible &= mask
    else:
        mask = np.ones((n, k), dtype=bool)
        visible = np.ones((n, k), dtype=bool)
    Y = np.where(mask[..., None], Y, 0.0)  # unused slots: 0, ignored through the mask
    return GhostData(frame, X, Y, mask, visible, schema)


def load(path: Path | None = None, schema: Schema = METRICA) -> GhostData:
    path = path or schema.samples_path
    if not path.exists():
        builder = "preprocessing.build" if schema is METRICA else "worldcup.build"
        raise FileNotFoundError(f"{path} not found. Run `uv run python -m horse_ml.{builder}` first.")
    return from_frame(pd.read_parquet(path), schema)


def load_splits(path: Path | None = None, schema: Schema = METRICA) -> dict[str, GhostData]:
    data = load(path, schema)
    return {name: data.split(name) for name in SPLITS}
