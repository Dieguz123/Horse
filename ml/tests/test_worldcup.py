"""World Cup pipeline: common roles, variable line, missing players, split by game."""

from __future__ import annotations

import bz2
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_ml.data.pff import PffMetadata, RosterPlayer, read_tracking
from horse_ml.ghosting.metrics import sample_errors
from horse_ml.worldcup.backline import LAYOUT, assign_lateral, check_labels, positional_line, segment_roles
from horse_ml.worldcup.frames import WcFrames, contiguous_blocks, home_attacks_right
from horse_ml.worldcup.lineups import team_segments
from horse_ml.worldcup.possession import WcPossession
from horse_ml.worldcup.samples import build_samples
from horse_ml.worldcup.split import assign_game_splits

META = PffMetadata("1", "2022-11-23", 1, "H", "Home", "A", "Away", 30.0, 105.0, 68.0, True, None)


def player(i: int, role: str, team: str = "H") -> RosterPlayer:
    return RosterPlayer(str(i), f"P{i}", team, str(i), role, True)


# ------------------------------------------------------------------ common roles


def test_label_checks() -> None:
    assert check_labels(["RB", "RCB", "LCB", "LB"])[0] == "ok"
    assert check_labels(["RWB", "RCB", "MCB", "LCB", "LWB"])[0] == "ok"
    assert check_labels(["RCB", "LCB"])[0] == "count"
    assert check_labels(["RB", "RB", "RCB", "LCB", "LB"])[0] == "duplicate"


@pytest.mark.parametrize("size", [3, 4, 5])
def test_line_fills_common_slots_right_to_left(size: int) -> None:
    players = [player(i, "RCB") for i in range(size)]
    # right side = negative y when attacking +x
    mean_xy = {i: (-30.0, 20.0 - 10.0 * i) for i in range(size)}  # player 0 left ... last right
    slots = assign_lateral(list(range(size)), mean_xy, players)
    used = [s for s, p in enumerate(slots) if p >= 0]
    assert tuple(used) == LAYOUT[size]
    assert slots[used[0]] == size - 1  # rightmost player in the rightmost used slot
    assert slots[used[-1]] == 0


def test_positional_line_cuts_at_largest_gap() -> None:
    depth = {0: -30.0, 1: -29.0, 2: -28.5, 3: -27.0, 4: -15.0, 5: -14.0, 6: -5.0, 7: -4.0, 8: 0.0, 9: 5.0}
    assert sorted(positional_line(depth)) == [0, 1, 2, 3]


def _roles(players, on_pitch, defending_xy, overrides=()):
    return segment_roles(
        game_id="1", side="home", team="Home", period=1, start_row=0, end_row=100, start_s=0.0, end_s=100.0,
        on_pitch=on_pitch, players=players, defending_xy=defending_xy, attacking_xy=defending_xy,
        defending_frames=10_000, overrides=list(overrides),
    )


def _team(roles: list[str], xy: list[tuple[float, float]]):
    players = [player(i, r) for i, r in enumerate(roles)]
    return players, list(range(len(players))), {i: p for i, p in enumerate(xy)}


def test_labels_decide_and_lateral_order_comes_from_positions() -> None:
    # Costa Rica case: RCB plays in the centre, MCB on the right
    players, on, xy = _team(
        ["GK", "RWB", "RCB", "MCB", "LCB", "LWB", "CM", "CM", "RW", "LW", "CF"],
        [(-45, 0), (-24, -18), (-27, 0), (-27, -8), (-27, 9), (-24, 16), (-18, 4), (-18, -5), (-9, -9), (-16, 5), (-10, 4)],
    )
    r = _roles(players, on, xy)
    assert r.method == "labels" and r.size == 5
    assert [players[p].role for p in r.back] == ["RWB", "MCB", "RCB", "LCB", "LWB"]


def test_duplicate_labels_keep_members_but_order_by_position() -> None:
    # Japan 2nd half: two "RB", one of them plays left centre-back in a back five
    players, on, xy = _team(
        ["GK", "RB", "RB", "RCB", "LCB", "LB", "CM", "CM", "RW", "AM", "CF"],
        [(-41, 0), (-6, -20), (-18, 5), (-18, -11), (-20, -1), (-10, 17), (-10, -4), (2, -1), (-5, -15), (-3, 7), (18, 0)],
    )
    r = _roles(players, on, xy)
    assert r.method == "labels, lateral by position" and r.size == 5
    assert [p for p in r.back] == [1, 3, 4, 2, 5]  # right to left by average y


def test_non_defender_deeper_than_line_triggers_positions() -> None:
    players, on, xy = _team(
        ["GK", "RB", "RCB", "LCB", "LB", "CM", "CM", "AM", "RW", "LW", "CF"],
        [(-38, 0), (-6, -15), (-12, -3), (-12, 8), (-4, 18), (-14, 1), (-1, 3), (8, -3), (9, -12), (10, 12), (15, 0)],
    )
    r = _roles(players, on, xy)
    assert r.method == "positions"
    assert 5 in [p for p in r.back if p >= 0]  # the deep CM joins the line
    assert "deeper than the line" in r.reason


def test_override_wins() -> None:
    players, on, xy = _team(
        ["GK", "RB", "RCB", "LCB", "LB", "CM", "CM", "AM", "RW", "LW", "CF"],
        [(-38, 0), (-10, -15), (-12, -3), (-12, 8), (-10, 18), (-5, 1), (-1, 3), (8, -3), (9, -12), (10, 12), (15, 0)],
    )
    override = {"game_id": "1", "team": "Home", "period": 1, "start_s": 5.0, "back_line": ["2", "3", "6"]}
    r = _roles(players, on, xy, [override])
    assert r.method == "override" and r.size == 3
    assert [p for p in r.back if p >= 0] == [2, 3, 6]


# ---------------------------------------------------- frames, segments, direction


def test_direction_per_period() -> None:
    assert home_attacks_right(META, 1) is True and home_attacks_right(META, 2) is False
    assert home_attacks_right(META, 3) is None  # extra time without metadata


def test_contiguous_blocks_break_at_period_and_gaps() -> None:
    blocks = contiguous_blocks(np.array([1, 1, 1, 2, 2, 2]), np.array([10, 11, 13, 14, 15, 16]))
    assert blocks.tolist() == [0, 0, 1, 2, 2, 2]


def test_short_lineups_are_merged() -> None:
    fps = 10.0
    present = np.zeros((1000, 3), dtype=bool)
    present[:, :2] = True
    present[400:405, 2] = True  # a player tracked for half a second (substitution noise)
    present[700:, 1] = False  # substitution: player 1 off, player 2 on
    present[700:, 2] = True
    segments = team_segments(present, np.ones(1000, dtype=int), "home", fps, min_s=30.0)
    assert [(s.start_row, s.end_row, s.players) for s in segments] == [(0, 700, (0, 1)), (700, 1000, (0, 2))]


def test_tracking_reader_dedupes_and_keeps_visibility(tmp_path: Path) -> None:
    roster = [player(7, "CF"), player(8, "GK", "A")]
    frames = [
        {"frameNum": 1, "period": 1, "periodElapsedTime": 0.0, "balls": [{"visibility": "VISIBLE"}],
         "ballsSmoothed": {"x": 1.0, "y": 2.0, "z": 0.1},
         "homePlayersSmoothed": [{"jerseyNum": "7", "visibility": "VISIBLE", "x": 3.0, "y": 4.0}],
         "awayPlayersSmoothed": [{"jerseyNum": "8", "visibility": "ESTIMATED", "x": -40.0, "y": 0.0},
                                 {"jerseyNum": "99", "visibility": "VISIBLE", "x": 0.0, "y": 0.0}],
         "game_event": {"game_event_type": "OTB", "home_ball": True}, "game_event_id": 5},
        {"frameNum": 1, "period": 1, "periodElapsedTime": 0.0},  # duplicate (2nd possession event)
        {"frameNum": 2, "period": 1, "periodElapsedTime": 0.033, "balls": [], "ballsSmoothed": None,
         "homePlayersSmoothed": [], "awayPlayersSmoothed": []},
    ]
    (tmp_path / "Tracking Data").mkdir()
    with bz2.open(tmp_path / "Tracking Data" / "1.jsonl.bz2", "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(fr) for fr in frames))
    tr = read_tracking("1", META, roster, root=tmp_path)
    assert tr.duplicates == 1 and tr.unknown_jerseys == 1
    assert tr.x.shape == (2, 2)
    assert tr.visible[0].tolist() == [True, False]
    assert np.isnan(tr.x[1]).all() and np.isnan(tr.ball[1]).all()
    assert tr.game_events[0]["game_event_id"] == 5


# ------------------------------------------------- variable line, missing players


def _synthetic_game(n: int = 60) -> tuple[WcFrames, WcPossession, dict]:
    """Home (attacking) 11 players in columns 0-10; away (defending, columns 11-21) plays
    a back three. The away CF (column 21, id "30") is absent in the first half of the
    frames; the away MCB (column 13, line player) is missing in the last 6 frames."""
    home = [player(i, r) for i, r in enumerate(["GK", "RB", "RCB", "LCB", "LB", "CM", "CM", "AM", "RW", "LW", "CF"])]
    away = [player(20 + i, r, "A") for i, r in enumerate(["GK", "RCB", "MCB", "LCB", "RW", "LW", "CM", "CM", "AM", "CF", "CF"])]
    players = home + away
    P = len(players)
    rng = np.random.default_rng(0)
    x = rng.uniform(-40, 40, (n, P))
    y = rng.uniform(-30, 30, (n, P))
    x[: n // 2, 21] = np.nan
    x[n - 6 :, 13] = np.nan
    y[np.isnan(x)] = np.nan
    side = np.array(["home"] * 11 + ["away"] * 11)
    frames = WcFrames(
        game_id="1", meta=META, players=players, side=side,
        frame_num=np.arange(n) * 1, period=np.ones(n, dtype=int), t=np.arange(n) / 30.0,
        ball=np.column_stack([np.full(n, 10.0), np.full(n, 5.0), np.zeros(n)]), ball_visible=np.ones(n, dtype=bool),
        x=x, y=y, visible=np.ones((n, P), dtype=bool), vx=np.zeros((n, P)), vy=np.zeros((n, P)),
        ball_v=np.zeros((n, 2)), game_events={}, invalidated={},
    )
    poss = WcPossession(home_ball=np.ones(n), alive=np.ones(n, dtype=bool), seconds_since_change=np.full(n, 5.0),
                        setpiece=np.full(n, "open_play", dtype=object), sequence=np.ones(n))
    xy = {i: (float(np.nanmean(x[:, i])), float(np.nanmean(y[:, i]))) for i in range(P)}
    common = dict(game_id="1", period=1, start_row=0, end_row=n, start_s=0.0, end_s=n / 30.0,
                  players=players, defending_frames=10_000, overrides=[])
    away_xy = {i: (-30.0 if players[i].role in ("RCB", "MCB", "LCB") else 0.0, float(-i)) for i in range(11, 22)}
    roles = {
        "home": [segment_roles(side="home", team="Home", on_pitch=list(range(11)), defending_xy=xy, attacking_xy=xy, **common)],
        "away": [segment_roles(side="away", team="Away", on_pitch=list(range(11, 22)), defending_xy=away_xy,
                               attacking_xy=away_xy, **common)],
    }
    return frames, poss, roles


def test_samples_with_variable_line_and_missing_players() -> None:
    frames, poss, roles = _synthetic_game()
    samples, stats = build_samples(frames, poss, roles, sample_every=1)
    assert roles["away"][0].size == 3  # back three (labels RCB, MCB, LCB)
    # instants where a line player is missing are dropped; the others stay
    assert stats.dropped_backline_missing == 6 and len(samples) == 60 - 6
    # unused wide slots of a back three: no player, not present
    assert samples["back_0_player_id"].isna().all() and (samples["back_0_present"] == 0).all()
    assert (samples[["back_1_present", "back_2_present", "back_3_present"]] == 1).all().all()
    # an absent input player gets present = 0 and zeros, and the instant is kept
    cf_slot = next(k for k in range(8) if samples[f"dof_{k}_player_id"].iloc[0] == "30")
    first_half = samples["frame_num"] < 30
    assert (samples.loc[first_half, f"dof_{cf_slot}_present"] == 0).all()
    assert (samples.loc[first_half, f"dof_{cf_slot}_x"] == 0).all()
    assert (samples.loc[~first_half, f"dof_{cf_slot}_present"] == 1).all()


def test_masked_metric_ignores_unused_slots() -> None:
    true = np.zeros((1, 5, 2))
    pred = np.zeros((1, 5, 2))
    pred[0, 0] = [100.0, 0.0]  # unused slot: huge error that must be ignored
    pred[0, 2] = [3.0, 4.0]  # used slot: 5 m off
    mask = np.array([[False, True, True, True, False]])
    assert sample_errors(pred, true, mask) == pytest.approx([5.0 / 3])


# ----------------------------------------------------------- real data (optional)

from horse_ml.paths import PFF_DIR  # noqa: E402

REAL_GAME = "3821"
requires_pff = pytest.mark.skipif(
    not (PFF_DIR / "Tracking Data" / f"{REAL_GAME}.jsonl.bz2").exists(), reason="PFF data not available locally"
)


@requires_pff
def test_reader_matches_kloppy_smoothed_positions(tmp_path: Path) -> None:
    """Our reader and kloppy's PFF loader give the same smoothed positions."""
    from kloppy import pff

    from horse_ml.data.pff import read_metadata, read_roster

    for sub in ("Metadata", "Rosters", "Tracking Data"):
        (tmp_path / sub).mkdir()
    for sub in ("Metadata", "Rosters"):
        (tmp_path / sub / f"{REAL_GAME}.json").write_bytes((PFF_DIR / sub / f"{REAL_GAME}.json").read_bytes())
    with bz2.open(PFF_DIR / "Tracking Data" / f"{REAL_GAME}.jsonl.bz2", "rt", encoding="utf-8") as src:
        head = [next(src) for _ in range(300)]
    with bz2.open(tmp_path / "Tracking Data" / f"{REAL_GAME}.jsonl.bz2", "wt", encoding="utf-8") as dst:
        dst.writelines(head)
    plain = tmp_path / "tracking.jsonl"
    plain.write_text("".join(head), encoding="utf-8")

    meta, roster = read_metadata(REAL_GAME, tmp_path), read_roster(REAL_GAME, tmp_path)
    ours = read_tracking(REAL_GAME, meta, roster, root=tmp_path)
    theirs = pff.load_tracking(
        meta_data=tmp_path / "Metadata" / f"{REAL_GAME}.json",
        roster_meta_data=tmp_path / "Rosters" / f"{REAL_GAME}.json",
        raw_data=plain,
        coordinates="pff",
    )
    first = theirs.frames[0]
    for p, data in first.players_data.items():
        col = next(i for i, r in enumerate(roster) if r.player_id == str(p.player_id))
        assert ours.x[0, col] == pytest.approx(data.coordinates.x, abs=1e-3)
        assert ours.y[0, col] == pytest.approx(data.coordinates.y, abs=1e-3)


# ------------------------------------------------------------------ split by game


def test_split_by_whole_games() -> None:
    samples = pd.DataFrame({"game_id": ["1", "1", "2", "3", "3"]})
    split = assign_game_splits(samples, {"1": "train", "2": "val", "3": "test"})
    assert samples.assign(split=split).groupby("game_id")["split"].nunique().max() == 1
    with pytest.raises(ValueError, match="without a split"):
        assign_game_splits(pd.DataFrame({"game_id": ["4"]}), {"1": "train"})
