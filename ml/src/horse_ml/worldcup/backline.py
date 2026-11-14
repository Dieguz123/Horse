"""Common roles and the defensive line of each formation segment.

Target of the model = the defensive line of the team without the ball: 3, 4 or 5
players. Who belongs to the line is decided per formation segment (never per
frame), so a full back caught upfield in a counter-attack is still a target.

Hybrid rule (decided with the project lead):
1. Roster labels decide: players labelled RB/RWB/RCB/CB/MCB/LCB/LB/LWB form the line.
2. The labels are only checked against the average defending positions of the
   segment, and corrected when they are inconsistent:
   - two defenders with the same lateral label (e.g. two RBs after a change of
     system, as Japan in the second half against Germany): the labelled players
     stay the line, only their right-to-left order comes from positions
     (method "labels, lateral by position");
   - fewer than 3 or more than 5 labelled defenders, or an outfield player who is
     not labelled as a defender defends clearly deeper than the line (median
     depth of the labelled defenders minus 2 m): the line is rebuilt from
     positions (method "positions"): outfield players sorted by average depth,
     the 3, 4 or 5 deepest, cut where the depth gap to the next player is largest.
3. Manual overrides (reference/backline_overrides.json) win over both.
The order inside the line (right to left) always comes from the average lateral
position, because lateral labels can be wrong (e.g. Costa Rica's RCB/MCB).

Common slots for every nation: 0 right (RB/RWB), 1 right-centre, 2 centre,
3 left-centre, 4 left (LB/LWB). A back three uses slots 1-3, a back four 0,1,3,4,
a back five all of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from horse_ml.data.pff import RosterPlayer
from horse_ml.paths import REFERENCE_DIR

SLOT_NAMES = ("right", "right_centre", "centre", "left_centre", "left")
N_BACK_SLOTS = 5
LAYOUT = {3: (1, 2, 3), 4: (0, 1, 3, 4), 5: (0, 1, 2, 3, 4)}
BACK_RANK = {"RB": 0, "RWB": 0, "RCB": 1, "CB": 2, "MCB": 2, "LCB": 3, "LB": 4, "LWB": 4}
MIDFIELD = {"DM", "CM", "AM"}
FORWARDS = {"RW", "LW", "CF"}
ROLE_ORDER = ["GK", "RWB", "RB", "RCB", "MCB", "CB", "LCB", "LB", "LWB", "DM", "CM", "AM", "RW", "LW", "CF"]
N_OTHER_SLOTS = 8  # goalkeeper + up to 7 outfield players outside the line
N_ATTACK_SLOTS = 11

DEPTH_MARGIN_M = 2.0
MIN_DEFENDING_FRAMES = 300  # ~10 s at 30 fps of in-play defending to trust averages
OVERRIDES_FILE = REFERENCE_DIR / "backline_overrides.json"
OVERRIDE_TOLERANCE_S = 15.0


@dataclass
class SegmentRoles:
    game_id: str
    side: str
    team: str
    period: int
    start_row: int
    end_row: int
    start_s: float
    end_s: float
    back: list[int]  # 5 slots, player column or -1
    others: list[int]  # 8 slots: GK first, then outfield outside the line, -1 padded
    attack: list[int]  # 11 slots: order used when this team attacks
    size: int
    method: str  # "labels", "positions", "override", "labels (no positions)"
    reason: str
    formation_labels: str
    defending_frames: int
    mean_xy: dict[int, tuple[float, float]] = field(default_factory=dict)

    def summary(self, players: list[RosterPlayer]) -> dict:
        d = asdict(self)
        d.pop("mean_xy")
        d["back_names"] = [players[i].name if i >= 0 else None for i in self.back]
        d["back_ids"] = [players[i].player_id if i >= 0 else None for i in self.back]
        return d


def formation_from_labels(roles: list[str]) -> str:
    d = sum(r in BACK_RANK for r in roles)
    m = sum(r in MIDFIELD for r in roles)
    f = sum(r in FORWARDS for r in roles)
    return f"{d}-{m}-{f}"


def check_labels(back_roles: list[str]) -> tuple[str, str]:
    """('ok' | 'count' | 'duplicate', reason)."""
    if not 3 <= len(back_roles) <= 5:
        return "count", f"{len(back_roles)} players labelled as defenders"
    ranks = [BACK_RANK[r] for r in back_roles]
    if len(set(ranks)) != len(ranks):
        return "duplicate", "duplicate lateral labels " + "/".join(sorted(back_roles))
    return "ok", ""


def positional_line(depth: dict[int, float]) -> list[int]:
    """The 3, 4 or 5 deepest players, cut at the largest depth gap."""
    order = sorted(depth, key=depth.get)
    xs = [depth[p] for p in order]
    best_k = max((k for k in (3, 4, 5) if k < len(xs)), key=lambda k: xs[k] - xs[k - 1])
    return order[:best_k]


def assign_lateral(line: list[int], mean_xy: dict[int, tuple[float, float]],
                   players: list[RosterPlayer]) -> list[int]:
    """Place the line into the common slots, right (most negative y) to left."""
    def key(p: int) -> float:
        if p in mean_xy and np.isfinite(mean_xy[p][1]):
            return mean_xy[p][1]
        return (BACK_RANK.get(players[p].role, 2) - 2) * 10.0  # label fallback
    ordered = sorted(line, key=key)
    slots = [-1] * N_BACK_SLOTS
    for slot, p in zip(LAYOUT[len(ordered)], ordered, strict=True):
        slots[slot] = p
    return slots


def _role_key(p: int, players: list[RosterPlayer], mean_xy: dict[int, tuple[float, float]]) -> tuple:
    role = players[p].role
    y = mean_xy.get(p, (0.0, 0.0))[1]
    return (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), y if np.isfinite(y) else 0.0)


def load_overrides(path: Path = OVERRIDES_FILE) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("overrides", [])


def find_override(overrides: list[dict], game_id: str, team: str, period: int, start_s: float) -> dict | None:
    for o in overrides:
        if (str(o["game_id"]) == str(game_id) and o["team"] == team and int(o["period"]) == period
                and abs(float(o["start_s"]) - start_s) <= OVERRIDE_TOLERANCE_S):
            return o
    return None


def segment_roles(
    *,
    game_id: str,
    side: str,
    team: str,
    period: int,
    start_row: int,
    end_row: int,
    start_s: float,
    end_s: float,
    on_pitch: list[int],
    players: list[RosterPlayer],
    defending_xy: dict[int, tuple[float, float]],
    attacking_xy: dict[int, tuple[float, float]],
    defending_frames: int,
    overrides: list[dict],
) -> SegmentRoles:
    """Decide the line, the other defenders and the attacking order for one segment.

    ``defending_xy``/``attacking_xy``: average positions per player column in the
    team's own attacking direction (+x), while defending / attacking in play.
    """
    goalkeepers = [p for p in on_pitch if players[p].role == "GK"]
    if not goalkeepers and defending_xy:
        goalkeepers = [min(on_pitch, key=lambda p: defending_xy.get(p, (0.0, 0.0))[0])]
    gk = goalkeepers[0] if goalkeepers else None
    outfield = [p for p in on_pitch if p != gk]
    roles = [players[p].role for p in outfield]
    labelled = [p for p in outfield if players[p].role in BACK_RANK]
    has_positions = defending_frames >= MIN_DEFENDING_FRAMES and all(p in defending_xy for p in outfield)

    status, reason = check_labels([players[p].role for p in labelled])
    if status != "count" and has_positions:
        depth_line = float(np.median([defending_xy[p][0] for p in labelled]))
        deeper = [p for p in outfield if p not in labelled and defending_xy[p][0] < depth_line - DEPTH_MARGIN_M]
        if deeper:
            status = "depth"
            reason = "; ".join(
                f"{players[p].name} ({players[p].role}) defends at x={defending_xy[p][0]:.1f}, "
                f"deeper than the line median x={depth_line:.1f}" for p in deeper
            )

    override = find_override(overrides, game_id, team, period, start_s)
    if override is not None:
        by_id = {players[p].player_id: p for p in on_pitch}
        line = [by_id[str(pid)] for pid in override["back_line"]]
        slots = [-1] * N_BACK_SLOTS
        for slot, p in zip(LAYOUT[len(line)], line, strict=True):
            slots[slot] = p
        method, reason = "override", override.get("reason", "manual override")
    elif status == "ok":
        slots = assign_lateral(labelled, defending_xy if has_positions else {}, players)
        method, reason = ("labels", "") if has_positions else ("labels (no positions)", "")
    elif status == "duplicate" and has_positions:
        slots = assign_lateral(labelled, defending_xy, players)
        method = "labels, lateral by position"
    elif has_positions:
        line = positional_line({p: defending_xy[p][0] for p in outfield})
        slots = assign_lateral(line, defending_xy, players)
        method = "positions"
    else:  # inconsistent labels and no reliable positions: best effort with labels
        line = sorted(labelled, key=lambda p: BACK_RANK[players[p].role])[:5]
        while len(line) < 3 and len(line) < len(outfield):
            line.append(next(p for p in outfield if p not in line))
        slots = assign_lateral(line, {}, players)
        method = "labels (no positions)"
        reason = (reason + "; " if reason else "") + "not enough defending frames to check"

    in_line = {p for p in slots if p >= 0}
    rest = sorted((p for p in outfield if p not in in_line), key=lambda p: _role_key(p, players, defending_xy))
    others = ([gk] if gk is not None else []) + rest
    others = (others + [-1] * N_OTHER_SLOTS)[:N_OTHER_SLOTS]
    attack = sorted(on_pitch, key=lambda p: _role_key(p, players, attacking_xy))
    attack = (attack + [-1] * N_ATTACK_SLOTS)[:N_ATTACK_SLOTS]

    return SegmentRoles(
        game_id=game_id, side=side, team=team, period=period, start_row=start_row, end_row=end_row,
        start_s=round(float(start_s), 1), end_s=round(float(end_s), 1),
        back=slots, others=others, attack=attack, size=len(in_line), method=method, reason=reason,
        formation_labels=formation_from_labels(roles), defending_frames=int(defending_frames),
        mean_xy=defending_xy,
    )
