"""Tests for downloading and loading Metrica data.

Unit tests run offline. Tests marked ``requires_data`` use the real files and
are skipped automatically if they have not been downloaded.
"""

from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import requires_data
from horse_ml.data import download
from horse_ml.data.loading import game_files, load_events, load_tracking
from horse_ml.data.selection import frame_window, frames_needed

N_FRAMES = 250  # 10 seconds at 25 Hz


# --------------------------------------------------------------------------- unit


def test_missing_files_raise_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="horse_ml.data.download"):
        load_tracking(tmp_path)
    with pytest.raises(FileNotFoundError, match="horse_ml.data.download"):
        load_events(tmp_path)


def test_game_files_point_inside_data_dir(tmp_path: Path) -> None:
    files = game_files(tmp_path)
    assert files.tracking == tmp_path / "Sample_Game_3_tracking.txt"
    assert len(files.missing()) == 3


def test_download_writes_file_and_skips_existing(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float):
        calls.append(url)
        return io.BytesIO(b"payload")

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "sub" / "file.txt"

    assert download.download_file("https://example.test/file.txt", target) is True
    assert target.read_bytes() == b"payload"
    assert not target.with_suffix(".txt.part").exists()

    assert download.download_file("https://example.test/file.txt", target) is False
    assert len(calls) == 1  # second call did not hit the network


# -------------------------------------------------------------------- real data


@pytest.fixture(scope="module")
def tracking():
    return load_tracking(limit=N_FRAMES)


@requires_data
def test_tracking_basic_properties(tracking) -> None:
    meta = tracking.metadata
    assert len(tracking) == N_FRAMES
    assert meta.frame_rate == 25
    assert [t.ground.value for t in meta.teams] == ["home", "away"]

    for frame in tracking.frames:
        # 11 vs 11 on the pitch, all coordinates in Metrica's normalised [0, 1] range
        # (tracking can slightly exceed the lines, so allow a small margin)
        coords = [d.coordinates for d in frame.players_data.values() if d.coordinates is not None]
        valid = [c for c in coords if c.x == c.x]  # drop NaN
        assert len(valid) == 22
        assert all(-0.1 <= c.x <= 1.1 and -0.1 <= c.y <= 1.1 for c in valid)


@requires_data
def test_events_load_and_share_teams_with_tracking(tracking) -> None:
    events = load_events()
    assert len(events) > 1000
    assert {t.team_id for t in events.metadata.teams} == {t.team_id for t in tracking.metadata.teams}
    assert any(e.event_type.value == "SHOT" for e in events.events)


@requires_data
def test_frame_window_matches_frame_rate(tracking) -> None:
    window = frame_window(tracking, 1, timedelta(seconds=2), timedelta(seconds=4))
    assert len(window) == 51  # 2 s at 25 Hz, both ends inclusive
    assert frames_needed(tracking.metadata, 1, timedelta(seconds=4), margin_s=0) == 101
