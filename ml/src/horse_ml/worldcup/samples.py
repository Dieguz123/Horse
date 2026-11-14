"""Defensive-line samples (one row per instant), team in possession attacking +x.

Groups of players per sample:
- att_{k}  k=0..10: the 11 players of the team in possession (input);
- dof_{k}  k=0..7:  defending team outside the line, goalkeeper first (input);
- back_{s} s=0..4:  the defensive line in the common slots (TARGET, masked).

Every player has x, y, vx, vy, visible (tracked on camera, else estimated by
PFF), present (1 if there is a player in the slot), player_id and role.
Absent players get zeros in the inputs and present = 0, so the model can ignore
them. An instant is dropped only if a player of the defensive line is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from horse_ml.worldcup.backline import N_ATTACK_SLOTS, N_BACK_SLOTS, N_OTHER_SLOTS, SLOT_NAMES, SegmentRoles
from horse_ml.worldcup.frames import WcFrames
from horse_ml.worldcup.possession import WcPossession

SAMPLE_EVERY = 6  # 29.97 fps / 6 ≈ 5 Hz
NUMERIC = ("x", "y", "vx", "vy", "visible", "present")


@dataclass
class SampleStats:
    frames: int = 0
    in_play: int = 0
    in_play_all_estimated: int = 0
    candidates: int = 0
    dropped_backline_missing: int = 0
    velocities_filled_with_zero: int = 0
    kept: int = 0
    back_line_sizes: dict[int, int] = field(default_factory=dict)


def in_play_mask(frames: WcFrames, poss: WcPossession) -> np.ndarray:
    """Ball alive per events, possession known, ball position known, not a frame with
    every player estimated (replays, close-ups)."""
    return (
        poss.alive
        & ~np.isnan(poss.home_ball)
        & np.isfinite(frames.ball[:, 0])
        & ~frames.all_estimated
    )


def segment_index(roles: list[SegmentRoles], n: int) -> np.ndarray:
    idx = np.full(n, -1)
    for i, r in enumerate(roles):
        idx[r.start_row:r.end_row] = i
    return idx


def _gather(frames: WcFrames, rows: np.ndarray, cols: list[int], sign: float) -> dict[str, np.ndarray]:
    cols_arr = np.array(cols)
    valid = cols_arr >= 0
    safe = np.where(valid, cols_arr, 0)
    out = {}
    x = frames.x[np.ix_(rows, safe)]
    present = valid[None, :] & np.isfinite(x)
    for name, src in (("x", frames.x), ("y", frames.y), ("vx", frames.vx), ("vy", frames.vy)):
        values = src[np.ix_(rows, safe)] * sign
        out[name] = np.where(present, values, np.nan)
    out["visible"] = np.where(present, frames.visible[np.ix_(rows, safe)], False).astype(np.int8)
    out["present"] = present.astype(np.int8)
    return out


def build_samples(
    frames: WcFrames,
    poss: WcPossession,
    roles: dict[str, list[SegmentRoles]],
    *,
    sample_every: int = SAMPLE_EVERY,
    row_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, SampleStats]:
    n = len(frames.t)
    in_play = in_play_mask(frames, poss)
    stats = SampleStats(
        frames=n,
        in_play=int(in_play.sum()),
        in_play_all_estimated=int((poss.alive & ~np.isnan(poss.home_ball) & frames.all_estimated).sum()),
    )
    cand = in_play & (frames.frame_num % sample_every == 0)
    if row_mask is not None:
        cand &= row_mask
    stats.candidates = int(cand.sum())
    att_side = poss.attacking_side()
    seg_idx = {s: segment_index(roles[s], n) for s in ("home", "away")}
    players = frames.players
    parts = []

    for att in ("home", "away"):
        d = "away" if att == "home" else "home"
        rows_all = np.flatnonzero(cand & (att_side == att) & (seg_idx[d] >= 0) & (seg_idx[att] >= 0))
        sign = 1.0 if att == "home" else -1.0  # frames have home attacking right
        keys = np.stack([seg_idx[d][rows_all], seg_idx[att][rows_all]], axis=1)
        for dseg_i, aseg_i in np.unique(keys, axis=0):
            rows = rows_all[(keys[:, 0] == dseg_i) & (keys[:, 1] == aseg_i)]
            dseg, aseg = roles[d][dseg_i], roles[att][aseg_i]
            cols: dict[str, object] = {
                "game_id": frames.game_id,
                "match": f"{frames.meta.home_name} - {frames.meta.away_name}",
                "stage": frames.meta.stage,
                "period": frames.period[rows],
                "frame_num": frames.frame_num[rows],
                "timestamp_s": frames.t[rows],
                "attacking_team": frames.team_name(att),
                "defending_team": frames.team_name(d),
                "segment_start_s": dseg.start_s,
                "back_line_size": dseg.size,
                "backline_method": dseg.method,
                "defending_formation_labels": dseg.formation_labels,
                "seconds_since_possession_change": poss.seconds_since_change[rows],
                "setpiece": poss.setpiece[rows],
                "sequence": poss.sequence[rows],
                "ball_x": frames.ball[rows, 0] * sign,
                "ball_y": frames.ball[rows, 1] * sign,
                "ball_z": frames.ball[rows, 2],
                "ball_vx": frames.ball_v[rows, 0] * sign,
                "ball_vy": frames.ball_v[rows, 1] * sign,
                "ball_visible": frames.ball_visible[rows].astype(np.int8),
            }
            for prefix, slots in (("att", aseg.attack), ("dof", dseg.others), ("back", dseg.back)):
                g = _gather(frames, rows, slots, sign)
                for k, p in enumerate(slots):
                    for f in NUMERIC:
                        cols[f"{prefix}_{k}_{f}"] = g[f][:, k]
                    cols[f"{prefix}_{k}_player_id"] = players[p].player_id if p >= 0 else None
                    cols[f"{prefix}_{k}_role"] = (
                        SLOT_NAMES[k] if prefix == "back" and p >= 0 else (players[p].role if p >= 0 else None)
                    )
                    if prefix == "back":
                        cols[f"{prefix}_{k}_label"] = players[p].role if p >= 0 else None
            parts.append(pd.DataFrame(cols))

    if not parts:
        return pd.DataFrame(), stats
    samples = pd.concat(parts, ignore_index=True).sort_values(["period", "frame_num"], ignore_index=True)

    # drop instants where a player of the line is missing (decision 4)
    used = np.stack([samples[f"back_{s}_player_id"].notna().to_numpy() for s in range(N_BACK_SLOTS)], axis=1)
    present = np.stack([samples[f"back_{s}_present"].to_numpy() == 1 for s in range(N_BACK_SLOTS)], axis=1)
    missing = (used & ~present).any(axis=1)
    stats.dropped_backline_missing = int(missing.sum())
    samples = samples[~missing].reset_index(drop=True)

    # inputs: absent players -> 0; missing velocities of present players -> 0 (counted)
    groups = [("att", N_ATTACK_SLOTS), ("dof", N_OTHER_SLOTS), ("back", N_BACK_SLOTS)]
    filled = 0
    for prefix, n_slots in groups:
        for k in range(n_slots):
            pres = samples[f"{prefix}_{k}_present"].to_numpy() == 1
            for f in ("vx", "vy"):
                col = f"{prefix}_{k}_{f}"
                bad = pres & samples[col].isna().to_numpy()
                filled += int(bad.sum())
                samples.loc[bad, col] = 0.0
            if prefix != "back":
                for f in ("x", "y", "vx", "vy"):
                    samples.loc[~pres, f"{prefix}_{k}_{f}"] = 0.0
    for f in ("ball_vx", "ball_vy"):
        bad = samples[f].isna()
        filled += int(bad.sum())
        samples.loc[bad, f] = 0.0
    samples["ball_z"] = samples["ball_z"].fillna(0.0)
    stats.velocities_filled_with_zero = filled
    stats.kept = len(samples)
    stats.back_line_sizes = {int(k): int(v) for k, v in samples["back_line_size"].value_counts().items()}
    samples.insert(1, "sample_id", [f"{frames.game_id}_{f}" for f in samples["frame_num"]])
    return samples, stats
