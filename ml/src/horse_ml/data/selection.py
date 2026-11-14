"""Helpers to pick interesting moments and slice tracking frames by time."""

from __future__ import annotations

from datetime import timedelta

from kloppy.domain import Event, EventDataset, Frame, Metadata, TrackingDataset


def find_event(events: EventDataset, event_type: str, result: str | None = None) -> Event:
    """First event of ``event_type`` (e.g. "SHOT"), optionally with a given result (e.g. "GOAL")."""
    for event in events.events:
        if event.event_type.value != event_type:
            continue
        if result is not None and (event.result is None or event.result.value != result):
            continue
        return event
    raise LookupError(f"No {event_type} event with result={result!r} found")


def frames_needed(metadata: Metadata, period_id: int, end: timedelta, margin_s: float = 2.0) -> int:
    """How many frames to load (from the start of the game) to reach ``end`` in ``period_id``.

    Lets callers pass ``limit=`` to the loader instead of parsing the whole match.
    """
    elapsed = timedelta(0)
    for period in metadata.periods:
        if period.id == period_id:
            break
        elapsed += period.end_timestamp - period.start_timestamp
    total_seconds = (elapsed + end).total_seconds() + margin_s
    return int(total_seconds * metadata.frame_rate) + 1


def frame_window(
    tracking: TrackingDataset, period_id: int, start: timedelta, end: timedelta
) -> list[Frame]:
    """Frames of ``period_id`` whose timestamp (relative to period start) is in [start, end]."""
    return [
        frame
        for frame in tracking.frames
        if frame.period.id == period_id and start <= frame.timestamp <= end
    ]


def format_clock(t: timedelta) -> str:
    """mm:ss.ss game clock within a period."""
    seconds = t.total_seconds()
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
