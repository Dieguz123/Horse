"""Team in possession and dead-ball state per tracking frame, derived from events.

The tracking data has no ball owner. Rules:
- On-ball events (pass, carry, recovery, shot) give possession to the event's team,
  from the event start until the next event. The ball is alive.
- Stoppages (ball out, foul) make the ball dead until the next on-ball event
  (the restart), possession stays with the last team.
- Challenges, cards and other generic events do not change anything: in a duel
  the event team is the challenger, not necessarily the owner.
- Before the first event of a period: possession unknown, ball dead.

``seconds_since_possession_change`` counts from the event where the current team
gained the ball. A restart by the same team (e.g. own throw-in) is not a change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kloppy.domain import EventDataset

ON_BALL_TYPES = frozenset({"PASS", "CARRY", "RECOVERY", "SHOT"})
STOPPAGE_TYPES = frozenset({"BALL_OUT", "FOUL_COMMITTED"})
OUT_OF_BOUNDS_MARGIN = 1.0  # metres beyond the lines before the ball counts as out


@dataclass(frozen=True)
class EventMark:
    """The only event information possession needs, independent of kloppy."""

    period_id: int
    t: float  # seconds since period start
    team_id: str | None
    kind: str  # "on_ball" or "stoppage"


def marks_from_events(events: EventDataset) -> list[EventMark]:
    marks = []
    for event in events.events:
        name = event.event_type.value
        if name in ON_BALL_TYPES:
            kind = "on_ball"
        elif name in STOPPAGE_TYPES:
            kind = "stoppage"
        else:
            continue
        marks.append(
            EventMark(
                period_id=event.period.id,
                t=event.timestamp.total_seconds(),
                team_id=event.team.team_id if event.team else None,
                kind=kind,
            )
        )
    return marks


def possession_per_frame(
    marks: list[EventMark], period_ids: np.ndarray, t: np.ndarray
) -> pd.DataFrame:
    """Columns: possession_team_id (None if unknown), event_alive, seconds_since_possession_change."""
    period_ids = np.asarray(period_ids)
    t = np.asarray(t, dtype=float)
    n = len(t)
    owner_out = np.full(n, None, dtype=object)
    alive_out = np.zeros(n, dtype=bool)
    since_out = np.full(n, np.nan)

    for period in np.unique(period_ids):
        # stable sort; on equal times a stoppage comes after the on-ball event
        period_marks = sorted(
            (m for m in marks if m.period_id == period),
            key=lambda m: (m.t, m.kind == "stoppage"),
        )
        if not period_marks:
            continue
        times = np.array([m.t for m in period_marks])
        owners, alive, starts = [], [], []
        owner, is_alive, start = None, False, np.nan
        for m in period_marks:
            if m.kind == "on_ball":
                if m.team_id != owner:
                    owner, start = m.team_id, m.t
                is_alive = True
            else:
                is_alive = False
            owners.append(owner)
            alive.append(is_alive)
            starts.append(start)

        rows = np.flatnonzero(period_ids == period)
        idx = np.searchsorted(times, t[rows], side="right") - 1
        known = idx >= 0
        rows, idx = rows[known], idx[known]
        owner_out[rows] = np.array(owners, dtype=object)[idx]
        alive_out[rows] = np.array(alive)[idx]
        since_out[rows] = t[rows] - np.array(starts)[idx]

    return pd.DataFrame(
        {
            "possession_team_id": owner_out,
            "event_alive": alive_out,
            "seconds_since_possession_change": since_out,
        }
    )


def ball_in_play(
    event_alive: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    margin: float = OUT_OF_BOUNDS_MARGIN,
) -> np.ndarray:
    """Alive per events AND ball position known AND ball inside the pitch (+ margin)."""
    known = ~(np.isnan(ball_x) | np.isnan(ball_y))
    with np.errstate(invalid="ignore"):
        inside = (np.abs(ball_x) <= pitch_length / 2 + margin) & (
            np.abs(ball_y) <= pitch_width / 2 + margin
        )
    return np.asarray(event_alive, dtype=bool) & known & inside
