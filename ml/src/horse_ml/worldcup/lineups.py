"""Formation segments: stretches of a period with the same players on the pitch.

A new segment starts at every period and every lasting change of the players on
the pitch (substitution, red card). Very short lineups (players briefly tracked
during a substitution) are merged into the neighbouring segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SEGMENT_S = 30.0


@dataclass(frozen=True)
class Segment:
    side: str
    period: int
    start_row: int  # inclusive
    end_row: int  # exclusive
    players: tuple[int, ...]  # player column indices on the pitch


def _runs(keys: list[tuple], period: np.ndarray) -> list[tuple[int, int, tuple]]:
    runs = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start] or period[i] != period[start]:
            runs.append((start, i, keys[start]))
            start = i
    return runs


def team_segments(present: np.ndarray, period: np.ndarray, side: str, fps: float,
                  min_s: float = MIN_SEGMENT_S) -> list[Segment]:
    """``present``: (n, P_team) bool for one team; returns segments with team-local columns."""
    keys = [tuple(np.flatnonzero(row)) for row in present]
    min_rows = int(min_s * fps)
    merged: list[list] = []  # [start, end, period, {lineup: rows}]
    for start, end, lineup in _runs(keys, period):
        p = int(period[start])
        short = end - start < min_rows
        if merged and merged[-1][2] == p and (short or merged[-1][1] - merged[-1][0] < min_rows):
            last = merged[-1]
            last[1] = end
            last[3][lineup] = last[3].get(lineup, 0) + end - start
        else:
            merged.append([start, end, p, {lineup: end - start}])
    segments = []
    for start, end, p, lineups in merged:
        main = max(lineups, key=lineups.get)  # the lineup that lasts longest
        segments.append(Segment(side, p, start, end, tuple(int(i) for i in main)))
    return segments
