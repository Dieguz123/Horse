"""Ball state and team in possession per frame, from the events attached to the tracking.

Frames carrying a ``game_event`` give the team in possession of the current
sequence (``home_ball``) and the event type. Both are carried forward:
- kick-offs and on-the-ball events make the ball alive;
- ball out, end of half and missing video make it dead;
- substitutions and "G" (ball hits the post and stays in play) change nothing.
The set-piece type (open play, throw-in, corner, ...) and the possession sequence
come from the event file, linked by ``game_event_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from horse_ml.worldcup.frames import WcFrames

ALIVE_EVENTS = frozenset({"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"})
DEAD_EVENTS = frozenset({"OUT", "END", "VID"})
SETPIECE_NAMES = {"O": "open_play", "K": "kick_off", "T": "throw_in", "G": "goal_kick",
                  "C": "corner", "F": "free_kick", "P": "penalty"}


@dataclass
class WcPossession:
    home_ball: np.ndarray  # (n,) 1.0 home, 0.0 away, NaN unknown
    alive: np.ndarray  # (n,) bool
    seconds_since_change: np.ndarray  # (n,)
    setpiece: np.ndarray  # (n,) object, name of the set piece that started the game event
    sequence: np.ndarray  # (n,) float

    def attacking_side(self) -> np.ndarray:
        out = np.full(len(self.home_ball), None, dtype=object)
        out[self.home_ball == 1.0] = "home"
        out[self.home_ball == 0.0] = "away"
        return out


def event_info(events: list[dict]) -> dict[int, tuple[str | None, float | None]]:
    """game_event_id -> (set-piece name, possession sequence)."""
    info = {}
    for e in events:
        ge = e.get("gameEvents") or {}
        gid = e.get("gameEventId")
        if gid is None:
            continue
        info[int(gid)] = (SETPIECE_NAMES.get(ge.get("setpieceType")), e.get("sequence"))
    return info


def possession(frames: WcFrames, events: list[dict]) -> WcPossession:
    n = len(frames.t)
    info = event_info(events)
    home_ball = np.full(n, np.nan)
    alive = np.zeros(n, dtype=bool)
    since = np.full(n, np.nan)
    setpiece = np.full(n, None, dtype=object)
    sequence = np.full(n, np.nan)

    owner, is_alive, start, sp, seq = np.nan, False, np.nan, None, np.nan
    prev_period = None
    for i in range(n):
        if frames.period[i] != prev_period:  # reset at every period
            owner, is_alive, start, sp, seq = np.nan, False, np.nan, None, np.nan
            prev_period = frames.period[i]
        ge = frames.game_events.get(i)
        if ge is not None:
            kind = ge.get("game_event_type")
            if kind in ALIVE_EVENTS:
                is_alive = True
            elif kind in DEAD_EVENTS:
                is_alive = False
            hb = ge.get("home_ball")
            if hb is not None:
                new_owner = 1.0 if hb else 0.0
                if new_owner != owner:
                    owner, start = new_owner, frames.t[i]
            gid = ge.get("game_event_id")
            if gid is not None and int(gid) in info:
                sp, seq = info[int(gid)]
        home_ball[i], alive[i], setpiece[i], sequence[i] = owner, is_alive, sp, seq if seq is not None else np.nan
        since[i] = frames.t[i] - start
    return WcPossession(home_ball, alive, since, setpiece, sequence)
