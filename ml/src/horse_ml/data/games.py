"""Registry of the games the preprocessing pipeline can process.

To add a game: download its files, then register a ``GameSource`` whose loaders
return kloppy datasets in Metrica coordinates (every kloppy loader accepts
``coordinates="metrica"``) with player roles in ``player.attributes``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kloppy.domain import EventDataset, TrackingDataset

from horse_ml.data.loading import load_events, load_tracking


@dataclass(frozen=True)
class GameSource:
    game_id: str
    description: str
    load_tracking: Callable[[], TrackingDataset]
    load_events: Callable[[], EventDataset]


GAMES: dict[str, GameSource] = {
    "metrica_3": GameSource(
        game_id="metrica_3",
        description="Metrica Sports Sample Game 3 (EPTS tracking + JSON events)",
        load_tracking=load_tracking,
        load_events=load_events,
    ),
}
