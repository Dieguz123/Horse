"""Readers for PFF FC (Gradient Sports) World Cup 2022 data.

Files per game (in ``ml/data/raw/pff/``, never committed):
    Metadata/<id>.json            teams, fps, pitch size, homeTeamStartLeft
    Rosters/<id>.json             players with jersey, role (positionGroupType), started
    Event Data/<id>.json          events (latest version, June 2025 update)
    Tracking Data/<id>.jsonl.bz2  one JSON frame per line, 29.97 fps

Tracking facts that shape this reader (see the specification PDF):
- coordinates in metres, origin at the centre spot, x along the length, y up;
- every player has a raw and a Kalman-smoothed position; we use the smoothed one
  (raw positions jump: a centre back reaches 111 m/s);
- players off camera have ``visibility == "ESTIMATED"``: positions inferred from
  context. We keep them and carry a visibility flag;
- frames with several possession events are duplicated: we keep the first copy;
- ``game_event`` is attached to some frames and carries ``home_ball`` (team in
  possession) and the event type (on-the-ball, ball out, end of half, ...).

kloppy has a PFF tracking loader, but it drops ``visibility``, maps the roles
MCB/RWB/LWB to Unknown, keeps duplicated frames and has no event reader, so we
read the files directly.

CLI (from ml/): build the match table from the Metadata zip or folder
    uv run python -m horse_ml.data.pff index <Metadata.zip | Metadata folder>
"""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from horse_ml.paths import PFF_DIR, REFERENCE_DIR

MATCHES_FILE = REFERENCE_DIR / "pff_matches.csv"
STAGES = {1: "Group 1", 2: "Group 2", 3: "Group 3", 4: "Round of 16", 5: "Quarter-final",
          6: "Semi-final", 7: "Third place", 8: "Final"}


@dataclass(frozen=True)
class PffMetadata:
    game_id: str
    date: str
    week: int
    home_id: str
    home_name: str
    away_id: str
    away_name: str
    fps: float
    pitch_length: float
    pitch_width: float
    home_start_left: bool
    home_start_left_extra_time: bool | None

    @property
    def stage(self) -> str:
        return STAGES.get(self.week, f"week {self.week}")


@dataclass(frozen=True)
class RosterPlayer:
    player_id: str
    name: str
    team_id: str
    jersey: str
    role: str  # PFF positionGroupType: GK, RB, RCB, MCB, LCB, LB, RWB, LWB, DM, CM, AM, RW, LW, CF
    started: bool


@dataclass
class PffTracking:
    """Tracking of one game as arrays, one row per unique frame.

    Players are the columns of ``x``/``y``/``visible``, in the order of ``players``.
    Positions are NaN when the player is not in the frame.
    """

    frame_num: np.ndarray  # (n,)
    period: np.ndarray  # (n,)
    t: np.ndarray  # seconds since period start
    ball: np.ndarray  # (n, 3) smoothed x, y, z (NaN if no ball)
    ball_visible: np.ndarray  # (n,) bool
    players: list[RosterPlayer]
    x: np.ndarray  # (n, P) smoothed
    y: np.ndarray
    visible: np.ndarray  # (n, P) bool
    game_events: dict[int, dict] = field(default_factory=dict)  # row -> game_event dict
    duplicates: int = 0
    unknown_jerseys: int = 0


def _first(obj):
    return obj[0] if isinstance(obj, list) else obj


def parse_metadata(raw: dict | list) -> PffMetadata:
    m = _first(raw)
    pitch = (m.get("stadium") or {}).get("pitches") or [{}]
    return PffMetadata(
        game_id=str(m["id"]),
        date=str(m.get("date", ""))[:10],
        week=int(m.get("week") or 0),
        home_id=str(m["homeTeam"]["id"]),
        home_name=m["homeTeam"]["name"],
        away_id=str(m["awayTeam"]["id"]),
        away_name=m["awayTeam"]["name"],
        fps=float(m["fps"]),
        pitch_length=float(pitch[0].get("length") or 105.0),
        pitch_width=float(pitch[0].get("width") or 68.0),
        home_start_left=bool(m["homeTeamStartLeft"]),
        home_start_left_extra_time=m.get("homeTeamStartLeftExtraTime"),
    )


def read_metadata(game_id: str, root: Path = PFF_DIR) -> PffMetadata:
    return parse_metadata(json.loads((root / "Metadata" / f"{game_id}.json").read_text(encoding="utf-8")))


def read_roster(game_id: str, root: Path = PFF_DIR) -> list[RosterPlayer]:
    raw = json.loads((root / "Rosters" / f"{game_id}.json").read_text(encoding="utf-8"))
    return [
        RosterPlayer(
            player_id=str(r["player"]["id"]),
            name=r["player"]["nickname"],
            team_id=str(r["team"]["id"]),
            jersey=str(r["shirtNumber"]),
            role=r["positionGroupType"],
            started=bool(r["started"]),
        )
        for r in raw
    ]


def read_events(game_id: str, root: Path = PFF_DIR) -> list[dict]:
    """Latest event version (folder root of ``Event Data``)."""
    return json.loads((root / "Event Data" / f"{game_id}.json").read_text(encoding="utf-8"))


def game_files_present(game_id: str, root: Path = PFF_DIR) -> bool:
    return all(
        p.exists()
        for p in (
            root / "Metadata" / f"{game_id}.json",
            root / "Rosters" / f"{game_id}.json",
            root / "Event Data" / f"{game_id}.json",
            root / "Tracking Data" / f"{game_id}.jsonl.bz2",
        )
    )


def read_tracking(game_id: str, meta: PffMetadata, roster: list[RosterPlayer],
                  root: Path = PFF_DIR, limit: int | None = None) -> PffTracking:
    """Stream the compressed JSON lines into arrays (smoothed positions + visibility)."""
    side_team = {"home": meta.home_id, "away": meta.away_id}
    index = {(p.team_id, p.jersey): i for i, p in enumerate(roster)}
    n_players = len(roster)

    frame_num, period, t, ball, ball_vis = [], [], [], [], []
    xs, ys, vis = [], [], []
    game_events: dict[int, dict] = {}
    seen: set[int] = set()
    duplicates = unknown = 0
    with bz2.open(root / "Tracking Data" / f"{game_id}.jsonl.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            frame = json.loads(line)
            num = frame["frameNum"]
            if num in seen:
                duplicates += 1
                continue
            seen.add(num)
            if frame.get("period") is None:
                continue
            row = len(frame_num)
            frame_num.append(num)
            period.append(frame["period"])
            t.append(frame.get("periodElapsedTime") or 0.0)

            bs = frame.get("ballsSmoothed")
            raw_balls = frame.get("balls") or []
            if isinstance(bs, dict) and bs.get("x") is not None and raw_balls:
                ball.append((bs["x"], bs["y"], bs.get("z") if bs.get("z") is not None else np.nan))
            else:
                ball.append((np.nan, np.nan, np.nan))
            ball_vis.append(bool(raw_balls) and raw_balls[0].get("visibility") == "VISIBLE")

            px = np.full(n_players, np.nan, dtype=np.float32)
            py = np.full(n_players, np.nan, dtype=np.float32)
            pv = np.zeros(n_players, dtype=bool)
            for side in ("home", "away"):
                for p in frame.get(f"{side}PlayersSmoothed") or []:
                    i = index.get((side_team[side], str(p.get("jerseyNum"))))
                    if i is None or p.get("x") is None:
                        unknown += i is None
                        continue
                    px[i], py[i] = p["x"], p["y"]
                    pv[i] = p.get("visibility") == "VISIBLE"
            xs.append(px)
            ys.append(py)
            vis.append(pv)
            if frame.get("game_event"):
                game_events[row] = {**frame["game_event"], "game_event_id": frame.get("game_event_id")}
            if limit is not None and row + 1 >= limit:
                break

    return PffTracking(
        frame_num=np.array(frame_num),
        period=np.array(period, dtype=int),
        t=np.array(t, dtype=float),
        ball=np.array(ball, dtype=float).reshape(-1, 3),
        ball_visible=np.array(ball_vis, dtype=bool),
        players=roster,
        x=np.vstack(xs).astype(float),
        y=np.vstack(ys).astype(float),
        visible=np.vstack(vis),
        game_events=game_events,
        duplicates=duplicates,
        unknown_jerseys=unknown,
    )


# --------------------------------------------------------------------------- match table


def match_rows(metadata_files: list[tuple[str, bytes]]) -> list[dict]:
    rows = []
    for _, content in metadata_files:
        m = parse_metadata(json.loads(content))
        rows.append({
            "game_id": m.game_id, "date": m.date, "stage": m.stage, "week": m.week,
            "home": m.home_name, "away": m.away_name, "home_id": m.home_id, "away_id": m.away_id,
            "fps": m.fps, "pitch_length": m.pitch_length, "pitch_width": m.pitch_width,
            "home_start_left": m.home_start_left,
        })
    return sorted(rows, key=lambda r: (r["date"], int(r["game_id"])))


def read_metadata_source(source: Path) -> list[tuple[str, bytes]]:
    if source.is_dir():
        return [(p.name, p.read_bytes()) for p in sorted(source.glob("*.json"))]
    with zipfile.ZipFile(source) as z:
        return [(n, z.read(n)) for n in z.namelist() if n.endswith(".json")]


def load_matches(path: Path = MATCHES_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="PFF helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    index = sub.add_parser("index", help="build reference/pff_matches.csv from the Metadata files")
    index.add_argument("source", type=Path, help="Metadata zip or folder")
    index.add_argument("--out", type=Path, default=MATCHES_FILE)
    args = parser.parse_args()

    rows = match_rows(read_metadata_source(args.source))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} games -> {args.out}")


if __name__ == "__main__":
    main()
