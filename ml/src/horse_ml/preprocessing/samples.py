"""Defensive samples: ball + 11 attackers (input) -> 11 defenders (target).

One row per sample (wide format). Every coordinate is in metres with the team in
possession attacking left-to-right (+x). Slot k always holds the same role of a
team (see ``roles.py``); the role labels are stored as columns too, because the
two teams play different formations, so slot k of team A and slot k of team B
are not the same role.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from horse_ml.preprocessing.coordinates import flip
from horse_ml.preprocessing.frames import FrameTable
from horse_ml.preprocessing.roles import N_SLOTS, TeamSlots

FIELDS = ("x", "y", "vx", "vy")
DEFAULT_SAMPLE_EVERY = 5  # 25 Hz / 5 = 5 Hz


@dataclass
class SampleStats:
    frames: int = 0
    in_play: int = 0
    candidates: int = 0  # in play, possession known, on the 5 Hz grid
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


def _team_matrices(df: pd.DataFrame, slots: TeamSlots) -> dict[str, np.ndarray]:
    return {
        f: df[[f"{pid}_{f}" for pid in slots.player_ids]].to_numpy(dtype=float) for f in FIELDS
    }


def _gather(matrix: np.ndarray, rows: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """(len(rows), 11) values of the player in each slot; NaN for empty slots."""
    idx = assignment[rows]
    values = matrix[rows[:, None], np.where(idx >= 0, idx, 0)]
    return np.where(idx >= 0, values, np.nan)


def build_samples(
    table: FrameTable,
    possession: pd.DataFrame,
    in_play: np.ndarray,
    team_slots: dict[str, TeamSlots],
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> tuple[pd.DataFrame, SampleStats]:
    df = table.df
    owner = possession["possession_team_id"].to_numpy(dtype=object)
    on_grid = df["frame_id"].to_numpy() % sample_every == 0
    candidates = in_play & pd.notna(owner) & on_grid
    stats = SampleStats(frames=len(df), in_play=int(in_play.sum()), candidates=int(candidates.sum()))

    matrices = {tid: _team_matrices(df, s) for tid, s in team_slots.items()}
    home_id = table.home.team_id
    parts = []
    for att_id, att_slots in team_slots.items():
        def_id = table.other(att_id).team_id
        def_slots = team_slots[def_id]
        rows = np.flatnonzero(candidates & (owner == att_id))
        if len(rows) == 0:
            continue
        # FrameTable has home attacking right; rotate when the away team attacks
        rotate = np.full(len(rows), att_id != home_id)

        cols: dict[str, object] = {
            "game_id": table.game_id,
            "frame_id": df["frame_id"].to_numpy()[rows],
            "period": df["period_id"].to_numpy()[rows],
            "timestamp_s": df["t"].to_numpy()[rows],
            "attacking_team": att_slots.team_name,
            "defending_team": def_slots.team_name,
            "attacking_formation": att_slots.formation,
            "defending_formation": def_slots.formation,
            "seconds_since_possession_change": possession[
                "seconds_since_possession_change"
            ].to_numpy()[rows],
        }
        for f in FIELDS:
            cols[f"ball_{f}"] = flip(df[f"ball_{f}"].to_numpy(dtype=float)[rows], rotate)
        for prefix, slots in (("att", att_slots), ("def", def_slots)):
            values = {f: flip(_gather(matrices[slots.team_id][f], rows, slots.assignment), rotate) for f in FIELDS}
            idx = slots.assignment[rows]
            ids = np.array(slots.player_ids + [None], dtype=object)[np.where(idx >= 0, idx, -1)]
            for k in range(N_SLOTS):
                for f in FIELDS:
                    cols[f"{prefix}_{k}_{f}"] = values[f][:, k]
                cols[f"{prefix}_{k}_player_id"] = ids[:, k]
                cols[f"{prefix}_{k}_role"] = slots.slots[k].role
        parts.append(pd.DataFrame(cols))

    samples = pd.concat(parts, ignore_index=True).sort_values(
        ["game_id", "period", "frame_id"], ignore_index=True
    )

    # drop incomplete rows, recording why
    pos_cols = [c for c in samples.columns if c.endswith(("_x", "_y")) and not c.startswith("ball")]
    vel_cols = [c for c in samples.columns if c.endswith(("_vx", "_vy")) and not c.startswith("ball")]
    missing_player = samples[pos_cols].isna().any(axis=1)
    bad_player_vel = ~missing_player & samples[vel_cols].isna().any(axis=1)
    bad_ball_vel = ~missing_player & ~bad_player_vel & samples[["ball_vx", "ball_vy"]].isna().any(axis=1)
    stats.dropped = {
        "missing_player": int(missing_player.sum()),
        "invalid_player_velocity": int(bad_player_vel.sum()),
        "invalid_ball_velocity": int(bad_ball_vel.sum()),
    }
    samples = samples[~(missing_player | bad_player_vel | bad_ball_vel)].reset_index(drop=True)
    stats.kept = len(samples)
    samples.insert(1, "sample_id", [f"{table.game_id}_{fid}" for fid in samples["frame_id"]])
    return samples, stats
