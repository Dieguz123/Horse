"""Build the World Cup defensive-line dataset, game by game.

    PFF tracking + events + roster (per game)
      -> frames: smoothed positions, home attacking right, Savitzky-Golay velocities
      -> possession + ball in play from the events attached to the frames
      -> formation segments -> defensive line (labels, checked with positions, overrides)
      -> samples at ~5 Hz, team in possession attacking +x
    all games -> split by whole games -> Parquet + metadata.json

Usage (from ml/):
    uv run python -m horse_ml.worldcup.build                 # every game with local files
    uv run python -m horse_ml.worldcup.build --games 3821 3839
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from horse_ml.data.pff import game_files_present, load_matches, read_events, read_metadata, read_roster, read_tracking
from horse_ml.paths import PFF_DIR, PROCESSED_PFF_DIR
from horse_ml.preprocessing.build import git_commit
from horse_ml.worldcup.backline import (
    LAYOUT,
    N_ATTACK_SLOTS,
    N_BACK_SLOTS,
    N_OTHER_SLOTS,
    SLOT_NAMES,
    SegmentRoles,
    load_overrides,
    segment_roles,
)
from horse_ml.worldcup.frames import DEFAULT_POLYORDER, DEFAULT_WINDOW, WcFrames, build_frames, contiguous_blocks, velocities
from horse_ml.worldcup.lineups import MIN_SEGMENT_S, team_segments
from horse_ml.worldcup.possession import WcPossession, possession
from horse_ml.worldcup.samples import SAMPLE_EVERY, SampleStats, build_samples, in_play_mask
from horse_ml.worldcup.split import assign_game_splits, load_split_map

SAMPLES_FILE = "samples.parquet"
METADATA_FILE = "metadata.json"
COMPARE_WINDOWS = (5, 9, 15)
AUTO_CORRECTED = ("positions", "labels, lateral by position")
PERCENTILES = (50, 90, 99, 99.9)


@dataclass
class PreparedGame:
    frames: WcFrames
    poss: WcPossession
    roles: dict[str, list[SegmentRoles]]
    in_play: np.ndarray


def _mean_positions(frames: WcFrames, rows: np.ndarray, cols: list[int], sign: float) -> dict[int, tuple[float, float]]:
    out = {}
    if len(rows) == 0:
        return out
    for c in cols:
        x = frames.x[rows, c]
        y = frames.y[rows, c]
        ok = np.isfinite(x)
        if ok.sum() > 0:
            out[c] = (float(x[ok].mean() * sign), float(y[ok].mean() * sign))
    return out


def prepare_game(game_id: str, root: Path = PFF_DIR, overrides: list[dict] | None = None) -> PreparedGame:
    meta = read_metadata(game_id, root)
    roster = read_roster(game_id, root)
    tracking = read_tracking(game_id, meta, roster, root)
    frames = build_frames(game_id, tracking, meta)
    poss = possession(frames, read_events(game_id, root))
    in_play = in_play_mask(frames, poss)
    att_side = poss.attacking_side()
    overrides = load_overrides() if overrides is None else overrides

    roles: dict[str, list[SegmentRoles]] = {}
    for side in ("home", "away"):
        team_cols = np.flatnonzero(frames.side == side)
        present = np.isfinite(frames.x[:, team_cols])
        sign = 1.0 if side == "home" else -1.0
        side_roles = []
        for seg in team_segments(present, frames.period, side, frames.fps):
            rows = np.arange(seg.start_row, seg.end_row)
            on_pitch = [int(team_cols[i]) for i in seg.players]
            defending = rows[in_play[rows] & (att_side[rows] != side)]
            attacking = rows[in_play[rows] & (att_side[rows] == side)]
            side_roles.append(segment_roles(
                game_id=game_id, side=side, team=frames.team_name(side), period=seg.period,
                start_row=seg.start_row, end_row=seg.end_row,
                start_s=frames.t[seg.start_row], end_s=frames.t[seg.end_row - 1],
                on_pitch=on_pitch, players=frames.players,
                defending_xy=_mean_positions(frames, defending, on_pitch, sign),
                attacking_xy=_mean_positions(frames, attacking, on_pitch, sign),
                defending_frames=len(defending), overrides=overrides,
            ))
        roles[side] = side_roles
    return PreparedGame(frames, poss, roles, in_play)


def window_samples(prepared: PreparedGame, period: int, start_s: float, end_s: float) -> pd.DataFrame:
    """Samples at the full frame rate for one time window (for animations)."""
    f = prepared.frames
    mask = (f.period == period) & (f.t >= start_s) & (f.t <= end_s)
    samples, _ = build_samples(f, prepared.poss, prepared.roles, sample_every=1, row_mask=mask)
    return samples


def speed_report(frames: WcFrames, in_play: np.ndarray) -> dict[str, list[float]]:
    """Player speed percentiles in play for several Savitzky-Golay windows."""
    blocks = contiguous_blocks(frames.period, frames.frame_num)
    rows = np.flatnonzero(in_play)[::3]  # every 3rd in-play frame is plenty
    report = {}
    for window in COMPARE_WINDOWS:
        vx, vy, _ = velocities(frames.x, frames.y, blocks, frames.fps, window, DEFAULT_POLYORDER, max_speed=np.inf)
        speed = np.hypot(vx[rows], vy[rows]).ravel()
        speed = speed[np.isfinite(speed)]
        report[f"window_{window}"] = [*np.percentile(speed, PERCENTILES).round(2).tolist(), round(float(speed.max()), 2)]
    return report


def process_game(game_id: str, overrides: list[dict]) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    prepared = prepare_game(game_id, overrides=overrides)
    samples, stats = build_samples(prepared.frames, prepared.poss, prepared.roles)
    f = prepared.frames
    info = {
        "game_id": game_id,
        "match": f"{f.meta.home_name} - {f.meta.away_name}",
        "stage": f.meta.stage,
        "stats": asdict(stats),
        "duplicates_skipped": "see tracking reader",
        "speeds_invalidated": f.invalidated,
        "speed_percentiles": speed_report(f, prepared.in_play),
        "target_visible_share": float(np.mean([
            samples.loc[samples[f"back_{s}_present"] == 1, f"back_{s}_visible"].mean()
            for s in range(N_BACK_SLOTS) if (samples[f"back_{s}_present"] == 1).any()
        ])) if len(samples) else None,
        "segments": [r.summary(f.players) for side in ("home", "away") for r in prepared.roles[side]],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"[{game_id}] {info['match']}: {len(samples):,} samples ({info['seconds']:.0f} s)")
    return samples, info


COLUMN_DOCS = {
    "game_id / match / stage": "PFF game id, 'home - away', tournament stage",
    "period / frame_num / timestamp_s": "period, PFF frame number (29.97 fps), seconds since period start",
    "split": "train / val / test, by whole game (reference/pff_splits.csv)",
    "attacking_team / defending_team": "team in possession / team without the ball",
    "segment_start_s": "start of the defending team's formation segment (match key for overrides)",
    "back_line_size": "3, 4 or 5 players in the defending line",
    "backline_method": "labels | positions | override | labels (no positions)",
    "defending_formation_labels": "D-M-F counts from roster labels (rough, informative only)",
    "seconds_since_possession_change": "seconds since the attacking team won the ball",
    "setpiece / sequence": "set piece that started the current game event; PFF possession sequence",
    "ball_{x,y,z,vx,vy,visible}": "ball (smoothed), metres and m/s; visible = tracked on camera",
    "att_{k}_*": "attacking team, 11 slots ordered by role (input)",
    "dof_{k}_*": "defending team outside the line, GK first, 8 slots (input)",
    "back_{s}_*": "defensive line in common slots " + ", ".join(f"{i}={n}" for i, n in enumerate(SLOT_NAMES))
                  + " (TARGET: x, y; masked by present)",
    "*_{x,y,vx,vy}": "metres / m/s, team in possession attacks towards +x; 0 when absent (inputs)",
    "*_visible / *_present": "1 if tracked on camera (else PFF estimate) / 1 if a player is in the slot",
    "*_player_id / *_role": "PFF player id; roster role (back slots: common slot name)",
    "back_{s}_label": "roster label of the line player (e.g. MCB, RWB)",
}


def write_outputs(samples: pd.DataFrame, infos: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SAMPLES_FILE
    samples.to_parquet(path, index=False)
    metadata = {
        "description": "World Cup 2022 (PFF FC) defensive-line ghosting samples",
        "source": "Gradient Sports / PFF FC World Cup 2022 data, unofficial Google Drive mirror, no known terms",
        "created_with_commit": git_commit(),
        "frame_rate_hz": 29.97,
        "sample_every_frames": SAMPLE_EVERY,
        "velocity": {"method": "Savitzky-Golay on Kalman-smoothed positions", "window_frames": DEFAULT_WINDOW,
                     "polyorder": DEFAULT_POLYORDER},
        "segments": {"min_segment_s": MIN_SEGMENT_S},
        "back_line": {"slots": list(SLOT_NAMES), "layouts": {str(k): list(v) for k, v in LAYOUT.items()}},
        "slots": {"att": N_ATTACK_SLOTS, "dof": N_OTHER_SLOTS, "back": N_BACK_SLOTS},
        "split_counts": samples["split"].value_counts().to_dict(),
        "games": infos,
        "columns": COLUMN_DOCS,
    }
    meta_path = out_dir / METADATA_FILE
    meta_path.write_text(json.dumps(metadata, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path, meta_path


def print_summary(samples: pd.DataFrame, infos: list[dict]) -> None:
    print()
    for info in infos:
        s = info["stats"]
        print(f"=== {info['game_id']} {info['match']} ({info['stage']}) ===")
        print(f"frames {s['frames']:,} | in play {s['in_play'] / s['frames']:.1%} | "
              f"in play but every player estimated (excluded) {s['in_play_all_estimated']:,}")
        print(f"candidates (~5 Hz) {s['candidates']:,} | dropped, line player missing {s['dropped_backline_missing']:,} "
              f"| kept {s['kept']:,} | velocities filled with 0: {s['velocities_filled_with_zero']:,}")
        print(f"back line sizes {s['back_line_sizes']} | target visible share {info['target_visible_share']:.1%} "
              f"| speeds set to NaN {info['speeds_invalidated']}")
        for seg in info["segments"]:
            flag = "  <-- CORRECTED AUTOMATICALLY" if seg["method"] in AUTO_CORRECTED else (
                "  <-- check: no positions" if "no positions" in seg["method"] else "")
            names = ", ".join(n.split()[-1] if n else "-" for n in seg["back_names"])
            print(f"  {seg['team']:<12} P{seg['period']} {seg['start_s']:>6.0f}-{seg['end_s']:<6.0f}s "
                  f"{seg['formation_labels']:<6} line {seg['size']} [{names}] via {seg['method']}{flag}")
            if seg["reason"]:
                print(f"      reason: {seg['reason']}")
        print("  player speed percentiles in play (m/s) p50/p90/p99/p99.9/max:")
        for w, v in info["speed_percentiles"].items():
            print(f"    {w:<10} {v}" + ("  <- used" if w == f"window_{DEFAULT_WINDOW}" else ""))
        print()
    corrected = [(i["game_id"], s) for i in infos for s in i["segments"] if s["method"] in AUTO_CORRECTED]
    print(f"=== Segments corrected automatically: {len(corrected)} "
          f"(review them; fix with reference/backline_overrides.json) ===")
    for game_id, s in corrected:
        names = ", ".join(n.split()[-1] if n else "-" for n in s["back_names"])
        print(f"  game {game_id} | team {s['team']} | period {s['period']} | start_s {s['start_s']} | "
              f"{s['method']} -> [{names}] | ids {[i for i in s['back_ids'] if i]}")
    print()
    print("=== Samples per split ===")
    print(pd.crosstab([samples["split"], samples["game_id"]], samples["defending_team"], margins=True).to_string())
    print("\nBack line size per defending team:")
    print(pd.crosstab(samples["defending_team"], samples["back_line_size"]).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the World Cup defensive-line dataset.")
    parser.add_argument("--games", nargs="+", help="PFF game ids (default: every game with local files)")
    parser.add_argument("--out", type=Path, default=PROCESSED_PFF_DIR)
    args = parser.parse_args()

    games = args.games or [m["game_id"] for m in load_matches() if game_files_present(m["game_id"])]
    if not games:
        raise SystemExit(f"No PFF game files found in {PFF_DIR}")
    overrides = load_overrides()
    split_map = load_split_map()
    parts, infos = [], []
    for game_id in games:
        samples, info = process_game(str(game_id), overrides)
        parts.append(samples)
        infos.append(info)
    samples = pd.concat(parts, ignore_index=True)
    samples.insert(samples.columns.get_loc("timestamp_s") + 1, "split", assign_game_splits(samples, split_map))
    path, meta_path = write_outputs(samples, infos, args.out)
    print_summary(samples, infos)
    print(f"\nSaved {path} ({path.stat().st_size / 1e6:.1f} MB, {len(samples):,} rows, {samples.shape[1]} columns)")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
