"""PFF tracking as a frame table: home team attacking right, smoothed positions, velocities.

PFF coordinates are already metres with the origin at the centre spot and y up,
the system we use everywhere, so the only conversion is the attacking direction:
each period is rotated (x, y -> -x, -y) when the home team attacks left, which
undoes the half-time swap. The per-frame possession flip happens later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from horse_ml.data.pff import PffMetadata, PffTracking, RosterPlayer
from horse_ml.preprocessing.kinematics import MAX_BALL_SPEED, MAX_PLAYER_SPEED, savgol_velocity

DEFAULT_WINDOW = 9  # frames at 29.97 fps = 0.30 s (≈ the 7 frames at 25 Hz used for Metrica)
DEFAULT_POLYORDER = 2


def home_attacks_right(meta: PffMetadata, period: int) -> bool | None:
    """Direction of the home team; None when unknown (extra time without metadata)."""
    if period in (1, 2):
        start_right = meta.home_start_left  # starting on the left = attacking right
        return start_right if period == 1 else not start_right
    if period in (3, 4) and meta.home_start_left_extra_time is not None:
        start_right = bool(meta.home_start_left_extra_time)
        return start_right if period == 3 else not start_right
    return None


@dataclass
class WcFrames:
    game_id: str
    meta: PffMetadata
    players: list[RosterPlayer]
    side: np.ndarray  # (P,) "home" / "away"
    frame_num: np.ndarray
    period: np.ndarray
    t: np.ndarray
    ball: np.ndarray  # (n, 3)
    ball_visible: np.ndarray
    x: np.ndarray  # (n, P) home attacking right
    y: np.ndarray
    visible: np.ndarray  # (n, P)
    vx: np.ndarray
    vy: np.ndarray
    ball_v: np.ndarray  # (n, 2)
    game_events: dict[int, dict]
    invalidated: dict[str, int]
    dropped_rows: int = 0

    @property
    def fps(self) -> float:
        return self.meta.fps

    def team_name(self, side: str) -> str:
        return self.meta.home_name if side == "home" else self.meta.away_name

    @property
    def all_estimated(self) -> np.ndarray:
        """True where no player at all is visible (replays, close-ups, other cameras)."""
        return ~self.visible.any(axis=1)


def contiguous_blocks(period: np.ndarray, frame_num: np.ndarray) -> np.ndarray:
    """Block id per row: a new block at every period change or gap in frame numbers."""
    breaks = np.concatenate(([True], (np.diff(period) != 0) | (np.diff(frame_num) != 1)))
    return np.cumsum(breaks) - 1


def velocities(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, fps: float,
               window: int = DEFAULT_WINDOW, polyorder: int = DEFAULT_POLYORDER,
               max_speed: float = MAX_PLAYER_SPEED) -> tuple[np.ndarray, np.ndarray, int]:
    """Savitzky-Golay velocities per column inside contiguous blocks; too fast -> NaN."""
    vx = np.full_like(x, np.nan, dtype=float)
    vy = np.full_like(y, np.nan, dtype=float)
    starts = np.flatnonzero(np.concatenate(([True], np.diff(blocks) != 0)))
    ends = np.concatenate((starts[1:], [len(blocks)]))
    for s, e in zip(starts, ends, strict=True):
        for j in range(x.shape[1]):
            vx[s:e, j], vy[s:e, j] = savgol_velocity(x[s:e, j], y[s:e, j], fps, window, polyorder)
    too_fast = np.hypot(vx, vy) > max_speed
    vx[too_fast] = vy[too_fast] = np.nan
    return vx, vy, int(too_fast.sum())


def build_frames(game_id: str, tracking: PffTracking, meta: PffMetadata,
                 window: int = DEFAULT_WINDOW) -> WcFrames:
    signs = np.array([
        {True: 1.0, False: -1.0, None: np.nan}[home_attacks_right(meta, int(p))] for p in tracking.period
    ])
    keep = ~np.isnan(signs)  # drop periods with unknown direction
    s = signs[keep][:, None]
    period, frame_num = tracking.period[keep], tracking.frame_num[keep]
    x, y = tracking.x[keep] * s, tracking.y[keep] * s
    ball = tracking.ball[keep].copy()
    ball[:, :2] *= s
    blocks = contiguous_blocks(period, frame_num)

    vx, vy, bad_players = velocities(x, y, blocks, meta.fps, window)
    bvx, bvy, bad_ball = velocities(ball[:, :1], ball[:, 1:2], blocks, meta.fps, window, max_speed=MAX_BALL_SPEED)
    rows = np.flatnonzero(keep)
    remap = {int(r): i for i, r in enumerate(rows)}
    side = np.array(["home" if p.team_id == meta.home_id else "away" for p in tracking.players])

    return WcFrames(
        game_id=game_id, meta=meta, players=tracking.players, side=side,
        frame_num=frame_num, period=period, t=tracking.t[keep],
        ball=ball, ball_visible=tracking.ball_visible[keep],
        x=x, y=y, visible=tracking.visible[keep], vx=vx, vy=vy,
        ball_v=np.column_stack([bvx[:, 0], bvy[:, 0]]),
        game_events={remap[r]: ge for r, ge in tracking.game_events.items() if r in remap},
        invalidated={"player": bad_players, "ball": bad_ball},
        dropped_rows=int((~keep).sum()),
    )
