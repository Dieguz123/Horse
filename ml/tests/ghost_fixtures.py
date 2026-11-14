"""Synthetic processed samples for ghosting tests (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_ml.ghosting.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from horse_ml.preprocessing.roles import N_SLOTS


def synthetic_samples(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Defenders = team shape + 0.3 * ball position; two teams; 70/15/15 split in time order."""
    rng = np.random.default_rng(seed)
    team = np.where(np.arange(n) % 2 == 0, "Team A", "Team B")
    features = rng.normal(0, 10, size=(n, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    shape = {
        "Team A": rng.uniform(-40, 10, size=(N_SLOTS, 2)),
        "Team B": rng.uniform(-40, 10, size=(N_SLOTS, 2)),
    }
    ball = frame[["ball_x", "ball_y"]].to_numpy()
    targets = np.stack([shape[t] for t in team]) + 0.3 * ball[:, None, :]
    frame[TARGET_COLUMNS] = targets.reshape(n, -1)
    frame["defending_team"] = team
    frame["seconds_since_possession_change"] = rng.uniform(0, 20, n)
    frame["sample_id"] = [f"s{i}" for i in range(n)]
    cut1, cut2 = int(0.7 * n), int(0.85 * n)
    frame["split"] = ["train"] * cut1 + ["val"] * (cut2 - cut1) + ["test"] * (n - cut2)
    return frame
