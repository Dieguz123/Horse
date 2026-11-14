"""Fixed player order ("slots") per team, so slot k means the same role in every sample.

Why: the model outputs 11 (x, y) pairs. If slot 0 is the goalkeeper in one sample
and a winger in the next, the loss compares the wrong players and the model can
only learn a meaningless average. Sorting players by position at each frame is
worse: two players swap order between nearly identical frames.

Primary method - roles from the metadata. Each player has a nominal role and a
template position in the team's formation (x towards the opponent goal, y from
the left touchline). Slots are the 11 starters sorted by depth, then right to
left: GK, defenders, midfielders, forwards. A substitute takes the slot whose
role matches his own.

The lineup is processed in "segments" (stretches of frames with the same players
on the pitch). In each segment:
1. Continuity: a player keeps the slot he already had (also after a short
   tracking dropout).
2. Role: a player entering the pitch takes the free slot with his nominal role.
3. Fallback - average position: players still without a slot (role unknown or
   already taken, e.g. a substitute whose metadata role duplicates a teammate's)
   are matched to the remaining free slots by comparing their average position in
   the segment with the slots' template positions (Hungarian algorithm on
   standardized coordinates).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from kloppy.domain import Player, Team
from scipy.optimize import linear_sum_assignment

N_SLOTS = 11


@dataclass(frozen=True)
class Slot:
    index: int
    role: str
    template_x: float  # 0 = own goal line, larger = closer to the opponent goal
    template_y: float  # 0 = left touchline, 1 = right touchline (attacking direction)


@dataclass
class TeamSlots:
    team_id: str
    team_name: str
    formation: str
    slots: list[Slot]
    player_ids: list[str]  # the team's squad; assignment indexes into this list
    assignment: np.ndarray  # (n_frames, 11) player index per slot, -1 if empty
    segments: int
    fallback: list[dict]  # players placed by average position: player_id, role, slot

    def roles(self) -> list[str]:
        return [s.role for s in self.slots]


def player_template(player: Player) -> tuple[str, float, float]:
    attrs = player.attributes or {}
    try:
        return str(attrs["position_type"]), float(attrs["position_x"]), float(attrs["position_y"])
    except KeyError as exc:
        raise ValueError(f"Player {player.player_id} has no role template in metadata") from exc


def formation_from_template(xs: list[float]) -> str:
    """'4-2-3-1' style string: outfield players grouped by template depth, GK excluded."""
    depths = sorted(xs)[1:]  # the deepest player is the goalkeeper
    counts = Counter(round(x, 2) for x in depths)
    return "-".join(str(counts[d]) for d in sorted(counts))


def slot_template(starters: list[Player]) -> list[Slot]:
    templates = sorted(
        (player_template(p) for p in starters),
        key=lambda r: (r[1], -r[2]),  # depth, then right (y=1) to left (y=0)
    )
    return [Slot(i, role, x, y) for i, (role, x, y) in enumerate(templates)]


def _segments(present: np.ndarray) -> list[tuple[int, int]]:
    """Ranges of frames with an identical set of on-pitch players."""
    changes = np.flatnonzero(np.any(present[1:] != present[:-1], axis=1)) + 1
    bounds = np.concatenate(([0], changes, [len(present)]))
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _standardize(points: np.ndarray) -> np.ndarray:
    std = points.std(axis=0)
    return (points - points.mean(axis=0)) / np.where(std > 0, std, 1.0)


def assign_slots(team: Team, x: np.ndarray, y: np.ndarray) -> TeamSlots:
    """Assign players to slots for every frame.

    ``x``, ``y``: (n_frames, n_players) positions of ``team.players`` in the
    team's own attacking direction (attacking towards +x, y up), NaN when off
    the pitch.
    """
    players = list(team.players)
    present = ~(np.isnan(x) | np.isnan(y))
    full = np.flatnonzero(present.sum(axis=1) == N_SLOTS)
    if len(full) == 0:
        raise ValueError(f"{team.name}: no frame with {N_SLOTS} players on the pitch")
    starters = [players[i] for i in np.flatnonzero(present[full[0]])]
    slots = slot_template(starters)
    formation = formation_from_template([s.template_x for s in slots])
    role_to_slot = {s.role: s.index for s in slots}
    roles = [player_template(p)[0] for p in players]

    # template positions in the same frame as x/y: depth -> +x, right (y=1) -> -y
    template_xy = np.array([[s.template_x, -s.template_y] for s in slots])

    template_std = _standardize(template_xy)

    assignment = np.full((len(present), N_SLOTS), -1, dtype=int)
    segments = _segments(present)
    remembered: dict[int, int] = {}  # player index -> slot, kept across segments
    fallback: list[dict] = []
    for start, end in segments:
        on = np.flatnonzero(present[start])
        if len(on) == 0:
            continue
        current: dict[int, int] = {}
        # 1. continuity
        for p in on:
            if p in remembered and remembered[p] not in current.values():
                current[p] = remembered[p]
        # 2. role of newcomers
        unresolved = []
        for p in on:
            if p in current:
                continue
            slot = role_to_slot.get(roles[p])
            if slot is not None and slot not in current.values():
                current[p] = slot
            else:
                unresolved.append(p)
        # 3. average position, only against the free slots
        if unresolved:
            free = [s for s in range(N_SLOTS) if s not in current.values()]
            mean_xy = np.column_stack(
                [np.nanmean(x[start:end, on], axis=0), np.nanmean(y[start:end, on], axis=0)]
            )
            mean_std = _standardize(mean_xy)  # standardized w.r.t. all on-pitch players
            rows = [int(np.flatnonzero(on == p)[0]) for p in unresolved]
            cost = ((mean_std[rows][:, None, :] - template_std[free][None, :, :]) ** 2).sum(axis=2)
            player_rows, slot_cols = linear_sum_assignment(cost)
            for r, c in zip(player_rows, slot_cols, strict=True):
                p, slot = unresolved[r], free[c]
                current[p] = slot
                fallback.append(
                    {"player_id": players[p].player_id, "role": roles[p],
                     "slot": slot, "slot_role": slots[slot].role, "from_frame": int(start)}
                )
        for p, slot in current.items():
            assignment[start:end, slot] = p
        remembered.update(current)

    return TeamSlots(
        team_id=team.team_id,
        team_name=team.name,
        formation=formation,
        slots=slots,
        player_ids=[p.player_id for p in players],
        assignment=assignment,
        segments=len(segments),
        fallback=fallback,
    )
