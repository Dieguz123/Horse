"""Shared drawing primitives: pitch, team colors, one tracking frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kloppy.domain import Frame, Ground, Metadata
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from mplsoccer import Pitch

# Dark grass-like surface; team colors validated for color-vision deficiency
# (protan/deutan/tritan) against this surface.
PITCH_COLOR = "#22312b"
LINE_COLOR = "#c7d5cc"
TEXT_COLOR = "#f0efec"
TEAM_COLORS = {Ground.HOME: "#d95926", Ground.AWAY: "#3987e5"}  # orange / blue
BALL_COLOR = "#ffffff"
BALL_EDGE = "#000000"

PLAYER_SIZE = 260
BALL_SIZE = 90


@dataclass(frozen=True)
class PlayerDot:
    x: float
    y: float
    jersey: str
    ground: Ground


def make_pitch(metadata: Metadata) -> Pitch:
    """An mplsoccer pitch matching Metrica's 0-1 coordinates (origin top-left)."""
    dims = metadata.pitch_dimensions
    return Pitch(
        pitch_type="metricasports",
        pitch_length=dims.pitch_length,
        pitch_width=dims.pitch_width,
        pitch_color=PITCH_COLOR,
        line_color=LINE_COLOR,
        linewidth=1.5,
        goal_type="box",
    )


def _valid(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def player_dots(frame: Frame) -> list[PlayerDot]:
    """Players with valid coordinates in ``frame``."""
    dots = []
    for player, data in frame.players_data.items():
        c = data.coordinates
        if c is None or not (_valid(c.x) and _valid(c.y)):
            continue
        dots.append(PlayerDot(c.x, c.y, str(player.jersey_no), player.team.ground))
    return dots


def draw_frame(ax: Axes, pitch: Pitch, frame: Frame, metadata: Metadata) -> list[Artist]:
    """Draw players (with jersey numbers) and ball. Returns the created artists
    so an animation can remove them before drawing the next frame."""
    artists: list[Artist] = []
    dots = player_dots(frame)

    for team in metadata.teams:
        team_dots = [d for d in dots if d.ground == team.ground]
        if not team_dots:
            continue
        artists.append(
            pitch.scatter(
                [d.x for d in team_dots],
                [d.y for d in team_dots],
                ax=ax,
                s=PLAYER_SIZE,
                color=TEAM_COLORS[team.ground],
                edgecolors=PITCH_COLOR,  # surface ring separates overlapping players
                linewidth=2,
                zorder=3,
                label=f"{team.name} ({team.ground.value})",
            )
        )
        for d in team_dots:
            artists.append(
                pitch.annotate(
                    d.jersey,
                    (d.x, d.y),
                    ax=ax,
                    color=TEXT_COLOR,
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                )
            )

    ball = frame.ball_coordinates
    if ball is not None and _valid(ball.x) and _valid(ball.y):
        artists.append(
            pitch.scatter(
                ball.x,
                ball.y,
                ax=ax,
                s=BALL_SIZE,
                color=BALL_COLOR,
                edgecolors=BALL_EDGE,
                linewidth=1.5,
                zorder=5,
                label="Ball",
            )
        )
    return artists


def add_legend(ax: Axes) -> None:
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        labelcolor=TEXT_COLOR,
        fontsize=10,
        markerscale=0.7,
    )
    legend.set_zorder(10)
