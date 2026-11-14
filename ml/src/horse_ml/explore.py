"""Print a short summary of the Metrica tracking + event data.

Usage (from ml/):
    uv run python -m horse_ml.explore
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
from kloppy.domain import EventDataset, TrackingDataset

from horse_ml.data.loading import load_events, load_tracking


def player_columns(dataset: TrackingDataset, team_ground: str, axis: str = "x") -> list[str]:
    """Column names of ``dataset.to_df()`` holding one axis for a team's players."""
    team = next(t for t in dataset.metadata.teams if t.ground.value == team_ground)
    return [f"{p.player_id}_{axis}" for p in team.players]


def players_on_pitch(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Number of players with valid coordinates in each frame."""
    present = [c for c in columns if c in df.columns]
    return df[present].notna().sum(axis=1)


def summarize(tracking: TrackingDataset, events: EventDataset) -> str:
    meta = tracking.metadata
    df = tracking.to_df()
    dims = meta.pitch_dimensions
    lines: list[str] = []
    add = lines.append

    add("=== Tracking data ===")
    add(f"Provider            : {meta.provider.value}")
    add(f"Frame rate          : {meta.frame_rate} Hz ({1000 / meta.frame_rate:.0f} ms per frame)")
    add(f"Frames              : {len(tracking):,}")
    for period in meta.periods:
        minutes = (period.end_timestamp - period.start_timestamp).total_seconds() / 60
        add(f"  Period {period.id}          : {minutes:.1f} min")
    add(f"Coordinate system   : {type(meta.coordinate_system).__name__}")
    add(f"  x range           : [{dims.x_dim.min}, {dims.x_dim.max}]  (pitch length {dims.pitch_length} m)")
    add(f"  y range           : [{dims.y_dim.min}, {dims.y_dim.max}]  (pitch width {dims.pitch_width} m)")
    add(f"  origin / y-axis   : {meta.coordinate_system.origin.value} / {meta.coordinate_system.vertical_orientation.value}")
    add(f"Orientation         : {meta.orientation.value}")

    add("")
    add("=== Teams ===")
    for team in meta.teams:
        cols = player_columns(tracking, team.ground.value)
        on_pitch = players_on_pitch(df, cols)
        add(
            f"{team.ground.value:<5} {team.name:<7}: {len(team.players)} players in squad, "
            f"on pitch per frame: median {on_pitch.median():.0f}, "
            f"min {on_pitch.min()}, max {on_pitch.max()}"
        )

    add("")
    add("=== Ball ===")
    ball_known = df["ball_x"].notna()
    add(f"Ball position known : {ball_known.mean():.1%} of frames")
    add(f"Ball z (height)     : {'available' if df['ball_z'].notna().any() else 'not available'}")
    states = df["ball_state"].dropna()
    if len(states):
        add(f"Ball state          : {dict(Counter(s.value for s in states))}")
    else:
        add("Ball state          : not provided in tracking (only in events)")
    xs = df["ball_x"].dropna()
    ys = df["ball_y"].dropna()
    add(f"Ball x / y observed : x [{xs.min():.2f}, {xs.max():.2f}], y [{ys.min():.2f}, {ys.max():.2f}]")

    add("")
    add("=== Events ===")
    add(f"Events              : {len(events):,}")
    counts = Counter(e.event_type.value for e in events.events)
    for name, n in counts.most_common(8):
        add(f"  {name:<18}: {n:,}")

    return "\n".join(lines)


def main() -> None:
    print("Loading tracking data (takes ~1 minute for the full game)...")
    tracking = load_tracking()
    events = load_events()
    print(summarize(tracking, events))


if __name__ == "__main__":
    main()
