"""Build the ghosting training dataset.

    raw tracking + events (per game)
      -> frame table in metres, home attacking right in both periods
      -> Savitzky-Golay velocities
      -> possession + ball in play from events
      -> role slots per team
      -> defensive samples at 5 Hz, attacking team attacking left-to-right
    all games -> contiguous time-block split -> Parquet + metadata.json

Usage (from ml/):
    uv run python -m horse_ml.preprocessing.build
    uv run python -m horse_ml.preprocessing.build --games metrica_3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from horse_ml.data.games import GAMES, GameSource
from horse_ml.paths import ML_ROOT, PROCESSED_DIR
from horse_ml.preprocessing.frames import FrameTable, build_frame_table
from horse_ml.preprocessing.kinematics import (
    DEFAULT_POLYORDER,
    DEFAULT_WINDOW,
    MAX_BALL_SPEED,
    MAX_PLAYER_SPEED,
    add_velocities,
    savgol_velocity,
)
from horse_ml.preprocessing.possession import (
    OUT_OF_BOUNDS_MARGIN,
    ball_in_play,
    marks_from_events,
    possession_per_frame,
)
from horse_ml.preprocessing.roles import N_SLOTS, TeamSlots, assign_slots
from horse_ml.preprocessing.samples import DEFAULT_SAMPLE_EVERY, SampleStats, build_samples
from horse_ml.preprocessing.split import (
    DEFAULT_BLOCK_S,
    DEFAULT_FRACTIONS,
    DEFAULT_GAP_S,
    DEFAULT_SEED,
    assign_splits,
    blocks_as_dicts,
)

SAMPLES_FILE = "samples.parquet"
METADATA_FILE = "metadata.json"
PERCENTILES = (50, 90, 95, 99, 99.9)
COMPARE_WINDOWS = (5, 7, 11)


@dataclass
class GameResult:
    samples: pd.DataFrame
    stats: SampleStats
    team_slots: dict[str, TeamSlots]
    invalidated: dict[str, int]
    speed_report: dict[str, dict[str, list[float]]]
    slot_check: dict[str, list[dict]]
    in_play_share: float
    possession_known_share: float


def team_positions(table: FrameTable, team_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(n_frames, n_players) x, y of a team, rotated so the team attacks towards +x."""
    team = table.team(team_id)
    cols = [p.player_id for p in team.players]
    x = table.df[[f"{c}_x" for c in cols]].to_numpy(dtype=float)
    y = table.df[[f"{c}_y" for c in cols]].to_numpy(dtype=float)
    sign = 1.0 if team.ground.value == "home" else -1.0
    return x * sign, y * sign


def speed_percentiles(table: FrameTable, in_play: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Speed percentiles (before invalidation) for several Savitzky-Golay windows."""
    df, periods = table.df, table.df["period_id"].to_numpy()
    report: dict[str, dict[str, list[float]]] = {}
    for window in COMPARE_WINDOWS:
        speeds: dict[str, list[np.ndarray]] = {"player": [], "ball": []}
        for entity in table.entities:
            x = df[f"{entity}_x"].to_numpy(dtype=float)
            y = df[f"{entity}_y"].to_numpy(dtype=float)
            speed = np.full(len(df), np.nan)
            for period in np.unique(periods):
                idx = np.flatnonzero(periods == period)
                vx, vy = savgol_velocity(x[idx], y[idx], table.fps, window, DEFAULT_POLYORDER)
                speed[idx] = np.hypot(vx, vy)
            values = speed[in_play]
            speeds["ball" if entity == "ball" else "player"].append(values[~np.isnan(values)])
        report[f"window_{window}"] = {
            kind: [*np.percentile(np.concatenate(v), PERCENTILES).round(2).tolist(),
                   round(float(np.concatenate(v).max()), 2)]
            for kind, v in speeds.items()
        }
    return report


def slot_check(table: FrameTable, slots: TeamSlots, x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Average real position per slot, to compare with the nominal role."""
    rows = np.arange(len(x))
    out = []
    for k, slot in enumerate(slots.slots):
        idx = slots.assignment[:, k]
        ok = idx >= 0
        xs = x[rows[ok], idx[ok]]
        ys = y[rows[ok], idx[ok]]
        out.append(
            {
                "slot": k,
                "role": slot.role,
                "mean_x": round(float(np.nanmean(xs)), 1),
                "mean_y": round(float(np.nanmean(ys)), 1),
            }
        )
    return out


@dataclass
class PreparedGame:
    """Everything needed to cut samples from one game, at any sampling rate."""

    table: FrameTable
    possession: pd.DataFrame
    in_play: np.ndarray
    team_slots: dict[str, TeamSlots]
    invalidated: dict[str, int]


def prepare_game(source: GameSource) -> PreparedGame:
    t0 = time.perf_counter()
    print(f"[{source.game_id}] loading tracking + events...")
    tracking = source.load_tracking()
    events = source.load_events()

    table = build_frame_table(source.game_id, tracking)
    print(f"[{source.game_id}] {len(table.df):,} frames in metres ({time.perf_counter() - t0:.0f} s)")

    table.df, invalidated = add_velocities(
        table.df, table.entities, table.fps, window=DEFAULT_WINDOW, polyorder=DEFAULT_POLYORDER
    )

    df = table.df
    possession = possession_per_frame(
        marks_from_events(events), df["period_id"].to_numpy(), df["t"].to_numpy()
    )
    in_play = ball_in_play(
        possession["event_alive"].to_numpy(),
        df["ball_x"].to_numpy(),
        df["ball_y"].to_numpy(),
        table.pitch_length,
        table.pitch_width,
    )
    team_slots = {
        team.team_id: assign_slots(team, *team_positions(table, team.team_id)) for team in table.teams
    }
    return PreparedGame(table, possession, in_play, team_slots, invalidated)


def window_samples(prepared: PreparedGame, period: int, start_s: float, end_s: float) -> pd.DataFrame:
    """Samples at the full frame rate (25 Hz) for one time window, e.g. for animations.

    Same features as the training samples; frames that are not in play or are
    incomplete are simply missing from the result.
    """
    df = prepared.table.df
    window = (df["period_id"].to_numpy() == period) & df["t"].between(start_s, end_s).to_numpy()
    samples, _ = build_samples(
        prepared.table, prepared.possession, prepared.in_play & window, prepared.team_slots, sample_every=1
    )
    return samples


def process_game(source: GameSource) -> GameResult:
    t0 = time.perf_counter()
    prepared = prepare_game(source)
    table, in_play = prepared.table, prepared.in_play

    checks = {
        table.team(tid).name: slot_check(table, slots, *team_positions(table, tid))
        for tid, slots in prepared.team_slots.items()
    }
    samples, stats = build_samples(table, prepared.possession, in_play, prepared.team_slots)
    report = speed_percentiles(table, in_play)
    print(f"[{source.game_id}] {len(samples):,} samples ({time.perf_counter() - t0:.0f} s)")

    return GameResult(
        samples=samples,
        stats=stats,
        team_slots=prepared.team_slots,
        invalidated=prepared.invalidated,
        speed_report=report,
        slot_check=checks,
        in_play_share=float(in_play.mean()),
        possession_known_share=float(prepared.possession["possession_team_id"].notna().mean()),
    )


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ML_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


COLUMN_DOCS = {
    "game_id": "game identifier (see horse_ml.data.games)",
    "sample_id": "<game_id>_<frame_id>, unique",
    "frame_id": "tracking frame number (25 Hz)",
    "period": "1 = first half, 2 = second half",
    "timestamp_s": "seconds since the start of the period",
    "split": "train / val / test (contiguous time blocks)",
    "attacking_team / defending_team": "team names; attacking = team in possession",
    "attacking_formation / defending_formation": "formation from metadata roles, e.g. 4-2-3-1",
    "seconds_since_possession_change": "seconds since the attacking team won the ball",
    "ball_{x,y,vx,vy}": "ball position (m) and velocity (m/s)",
    "att_{k}_{x,y,vx,vy}": "attacker in slot k: position (m) and velocity (m/s) - model INPUT",
    "def_{k}_{x,y}": "defender in slot k: position (m) - model TARGET",
    "def_{k}_{vx,vy}": "defender velocity (m/s) - analysis only, not a target",
    "{att,def}_{k}_player_id": "player in slot k (changes after substitutions)",
    "{att,def}_{k}_role": "nominal role of slot k for that team",
}


def write_outputs(
    samples: pd.DataFrame, results: dict[str, GameResult], blocks: list[dict], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / SAMPLES_FILE
    samples.to_parquet(samples_path, index=False)

    metadata = {
        "description": "Ghosting samples: ball + 11 attackers (input) -> 11 defenders (target)",
        "created_with_commit": git_commit(),
        "games": {gid: GAMES[gid].description for gid in results},
        "frame_rate_hz": 25,
        "sample_rate_hz": 25 / DEFAULT_SAMPLE_EVERY,
        "coordinates": {
            "unit": "metres",
            "origin": "centre spot",
            "x": "along the pitch, [-52.5, 52.5]; team in possession attacks towards +x",
            "y": "across the pitch, [-34, 34], pointing up (left side when attacking +x)",
        },
        "velocity": {
            "method": "Savitzky-Golay first derivative on contiguous valid runs per period",
            "window_frames": DEFAULT_WINDOW,
            "polyorder": DEFAULT_POLYORDER,
            "max_player_speed_mps": MAX_PLAYER_SPEED,
            "max_ball_speed_mps": MAX_BALL_SPEED,
        },
        "filters": {
            "ball_in_play": "alive per events, ball position known, ball within "
            f"{OUT_OF_BOUNDS_MARGIN} m of the pitch lines",
            "possession": "from last on-ball event (pass, carry, recovery, shot)",
            "complete": "all 22 players and the ball with valid position and velocity",
            "transitions": "not filtered, see seconds_since_possession_change",
        },
        "split": {
            "method": "alternating contiguous time blocks per game and period, assignment "
            "balanced on sample counts and defending team (no target information)",
            "fractions": dict(DEFAULT_FRACTIONS),
            "block_s": DEFAULT_BLOCK_S,
            "gap_s": DEFAULT_GAP_S,
            "seed": DEFAULT_SEED,
            "blocks": blocks,
            "counts": samples["split"].value_counts().to_dict(),
        },
        "teams": {
            gid: {
                s.team_name: {
                    "team_id": s.team_id,
                    "formation": s.formation,
                    "slots": [{"slot": sl.index, "role": sl.role} for sl in s.slots],
                    "lineup_segments": s.segments,
                    "average_position_fallback": s.fallback,
                }
                for s in r.team_slots.values()
            }
            for gid, r in results.items()
        },
        "stats": {gid: asdict(r.stats) for gid, r in results.items()},
        "n_slots": N_SLOTS,
        "columns": COLUMN_DOCS,
    }
    metadata_path = out_dir / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return samples_path, metadata_path


def print_summary(samples: pd.DataFrame, results: dict[str, GameResult], blocks: list[dict]) -> None:
    print()
    for gid, r in results.items():
        s = r.stats
        print(f"=== {gid} ===")
        print(f"Frames              : {s.frames:,}")
        print(f"Possession known    : {r.possession_known_share:.1%} of frames")
        print(f"Ball in play        : {r.in_play_share:.1%} of frames")
        print(f"Candidates (5 Hz)   : {s.candidates:,}")
        print(f"Dropped             : {s.dropped}")
        print(f"Kept                : {s.kept:,}")
        print(f"Speeds set to NaN   : {r.invalidated} frame-entity values above the limits")
        print()
        for team in r.team_slots.values():
            print(
                f"{team.team_name} ({team.formation}): {team.segments} lineup segments, "
                f"{len(team.fallback)} player(s) placed by average position"
            )
            for fb in team.fallback:
                print(f"  fallback: {fb['player_id']} ({fb['role']}) -> slot {fb['slot']} ({fb['slot_role']})")
            for row in r.slot_check[team.team_name]:
                print(f"  slot {row['slot']:>2} {row['role']:<40} mean x {row['mean_x']:>6} y {row['mean_y']:>6}")
        print()
        print("Speed percentiles in play (m/s), Savitzky-Golay polyorder 2:")
        header = " ".join(f"p{p:<5}" for p in PERCENTILES) + " max"
        print(f"  {'':18} {header}")
        for window, kinds in r.speed_report.items():
            for kind, values in kinds.items():
                row = " ".join(f"{v:<6.2f}" for v in values)
                marker = " <- used" if window == f"window_{DEFAULT_WINDOW}" else ""
                print(f"  {window:<9} {kind:<8} {row}{marker}")
        print()

    print("=== Samples per split and defending team ===")
    order = ["train", "val", "test", "total"]
    counts = pd.crosstab(samples["split"], samples["defending_team"], margins=True, margins_name="total")
    counts = counts.reindex(order)
    shares = counts.drop(columns="total").div(counts["total"], axis=0).mul(100).round(1)
    shares.columns = [f"{c} %" for c in shares.columns]
    split_share = (counts["total"] / counts.loc["total", "total"] * 100).round(1).rename("of all %")
    print(pd.concat([counts, shares, split_share], axis=1).to_string())
    n_blocks = pd.Series([b["split"] for b in blocks]).value_counts().reindex(order[:-1])
    print("Blocks per split: " + ", ".join(f"{k} {v}" for k, v in n_blocks.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ghosting training dataset.")
    parser.add_argument("--games", nargs="+", default=list(GAMES), choices=list(GAMES))
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    results = {gid: process_game(GAMES[gid]) for gid in args.games}
    samples = pd.concat([r.samples for r in results.values()], ignore_index=True)
    split, blocks = assign_splits(samples)
    samples.insert(samples.columns.get_loc("timestamp_s") + 1, "split", split)
    samples = samples[samples["split"].notna()].reset_index(drop=True)

    block_dicts = blocks_as_dicts(blocks)
    samples_path, metadata_path = write_outputs(samples, results, block_dicts, args.out)
    print_summary(samples, results, block_dicts)
    print(f"\nSaved {samples_path} ({samples_path.stat().st_size / 1e6:.1f} MB, {len(samples):,} rows)")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
