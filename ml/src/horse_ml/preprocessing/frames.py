"""Frame table: one row per tracking frame, positions in metres, home attacking right."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kloppy.domain import MetricaCoordinateSystem, Player, Team, TrackingDataset

from horse_ml.preprocessing.coordinates import metrica_to_metres, period_signs


@dataclass
class FrameTable:
    """Tracking data of one game as a wide DataFrame.

    Columns: ``period_id, frame_id, t`` (seconds since period start), then
    ``ball_x, ball_y`` and ``<player_id>_x, <player_id>_y`` for every player in
    both squads (NaN when not on the pitch). Velocity columns are added later.
    """

    game_id: str
    df: pd.DataFrame
    teams: list[Team]
    fps: float
    pitch_length: float
    pitch_width: float

    @property
    def home(self) -> Team:
        return next(t for t in self.teams if t.ground.value == "home")

    def team(self, team_id: str) -> Team:
        return next(t for t in self.teams if t.team_id == team_id)

    def other(self, team_id: str) -> Team:
        return next(t for t in self.teams if t.team_id != team_id)

    @property
    def players(self) -> list[Player]:
        return [p for t in self.teams for p in t.players]

    @property
    def entities(self) -> list[str]:
        return ["ball"] + [p.player_id for p in self.players]


def build_frame_table(game_id: str, tracking: TrackingDataset) -> FrameTable:
    meta = tracking.metadata
    if not isinstance(meta.coordinate_system, MetricaCoordinateSystem):
        raise ValueError("Expected tracking loaded in Metrica coordinates (coordinates='metrica')")
    dims = meta.pitch_dimensions
    length, width = float(dims.pitch_length), float(dims.pitch_width)

    raw = tracking.to_df()
    players = [p for t in meta.teams for p in t.players]
    entities = ["ball"] + [p.player_id for p in players]
    # reindex: players that never appear in the tracking get all-NaN columns
    xs = raw.reindex(columns=[f"{e}_x" for e in entities]).to_numpy(dtype=float)
    ys = raw.reindex(columns=[f"{e}_y" for e in entities]).to_numpy(dtype=float)

    xm, ym = metrica_to_metres(xs, ys, length, width)
    period_ids = raw["period_id"].to_numpy(dtype=int)
    signs = period_signs(period_ids, meta.orientation)[:, None]
    xm, ym = xm * signs, ym * signs

    data: dict[str, np.ndarray] = {
        "period_id": period_ids,
        "frame_id": raw["frame_id"].to_numpy(dtype=int),
        "t": raw["timestamp"].dt.total_seconds().to_numpy(),
    }
    for i, entity in enumerate(entities):
        data[f"{entity}_x"] = xm[:, i]
        data[f"{entity}_y"] = ym[:, i]
    df = pd.DataFrame(data).sort_values(["period_id", "frame_id"], ignore_index=True)

    return FrameTable(
        game_id=game_id,
        df=df,
        teams=list(meta.teams),
        fps=float(meta.frame_rate),
        pitch_length=length,
        pitch_width=width,
    )
