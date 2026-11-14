"""Load Metrica Sports tracking and event data with kloppy.

All loaders read from local files downloaded by ``horse_ml.data.download``;
nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kloppy import metrica
from kloppy.domain import EventDataset, TrackingDataset

from horse_ml.data.download import FILES, local_path
from horse_ml.paths import METRICA_DIR

# Metrica's native coordinate system: x and y normalised to [0, 1],
# origin in the top-left corner, y pointing down.
DEFAULT_COORDINATES = "metrica"


@dataclass(frozen=True)
class GameFiles:
    """Local paths of the files that make up one Metrica game."""

    metadata: Path
    tracking: Path
    events: Path

    def missing(self) -> list[Path]:
        return [p for p in (self.metadata, self.tracking, self.events) if not p.exists()]


def game_files(data_dir: Path = METRICA_DIR) -> GameFiles:
    """Return the expected local paths for the configured Metrica game."""
    return GameFiles(
        metadata=local_path(FILES["metadata"], data_dir),
        tracking=local_path(FILES["tracking"], data_dir),
        events=local_path(FILES["events"], data_dir),
    )


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing Metrica data file(s):\n  {listed}\n"
            "Run `uv run python -m horse_ml.data.download` from the ml/ folder first."
        )


def load_tracking(
    data_dir: Path = METRICA_DIR,
    *,
    limit: int | None = None,
    sample_rate: float | None = None,
    coordinates: str = DEFAULT_COORDINATES,
) -> TrackingDataset:
    """Load the tracking data (positions of all players + ball, every frame).

    Args:
        data_dir: folder containing the downloaded files.
        limit: load only the first ``limit`` frames (handy for tests).
        sample_rate: keep only a fraction of frames, e.g. 1/5 -> 5 Hz.
        coordinates: kloppy coordinate system name (default: Metrica's own).
    """
    files = game_files(data_dir)
    _require(files.metadata, files.tracking)
    return metrica.load_tracking_epts(
        meta_data=files.metadata,
        raw_data=files.tracking,
        limit=limit,
        sample_rate=sample_rate,
        coordinates=coordinates,
    )


def load_events(
    data_dir: Path = METRICA_DIR,
    *,
    coordinates: str = DEFAULT_COORDINATES,
) -> EventDataset:
    """Load the event data (passes, shots, ...) of the same game."""
    files = game_files(data_dir)
    _require(files.metadata, files.events)
    return metrica.load_event(
        event_data=files.events,
        meta_data=files.metadata,
        coordinates=coordinates,
    )
