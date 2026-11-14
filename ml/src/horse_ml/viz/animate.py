"""Animate ~10 seconds of play and save as GIF.

Default clip: the 8 seconds before the first goal and 2 seconds after it.

Usage (from ml/):
    uv run python -m horse_ml.viz.animate
    uv run python -m horse_ml.viz.animate --period 1 --start 300 --duration 10
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from horse_ml.data.loading import load_events, load_tracking
from horse_ml.data.selection import find_event, format_clock, frame_window, frames_needed
from horse_ml.paths import OUTPUTS_DIR
from horse_ml.viz.pitch import PITCH_COLOR, TEXT_COLOR, add_legend, draw_frame, make_pitch


def animate(
    period_id: int,
    start: timedelta,
    end: timedelta,
    output: Path,
    *,
    step: int = 1,
    title: str = "Metrica Sample Game 3",
) -> Path:
    """Render frames in [start, end] of ``period_id`` to a GIF.

    ``step`` keeps every n-th frame (step=2 -> 12.5 fps, half the file size);
    playback speed stays real-time either way.
    """
    events = load_events()
    tracking = load_tracking(limit=frames_needed(events.metadata, period_id, end))
    frames = frame_window(tracking, period_id, start, end)[::step]
    if not frames:
        raise LookupError("No frames in the requested window")

    meta = tracking.metadata
    pitch = make_pitch(meta)
    fig, ax = pitch.draw(figsize=(9, 6.2))
    fig.set_facecolor(PITCH_COLOR)
    title_text = ax.set_title("", color=TEXT_COLOR, fontsize=12)

    current: list[Artist] = []

    def update(i: int) -> list[Artist]:
        for artist in current:
            artist.remove()
        current.clear()
        frame = frames[i]
        current.extend(draw_frame(ax, pitch, frame, meta))
        if i == 0:
            add_legend(ax)
        title_text.set_text(f"{title} · Period {period_id} · {format_clock(frame.timestamp)}")
        return [*current, title_text]

    fps = meta.frame_rate / step
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": PITCH_COLOR})
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a window of tracking data as GIF.")
    parser.add_argument("--period", type=int, help="period id (default: around the first goal)")
    parser.add_argument("--start", type=float, help="window start, seconds since period start")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds (default 10)")
    parser.add_argument("--step", type=int, default=2, help="keep every n-th frame (default 2)")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "animation.gif")
    args = parser.parse_args()

    if args.period is None or args.start is None:
        goal = find_event(load_events(), "SHOT", result="GOAL")
        period_id = goal.period.id
        start = goal.timestamp - timedelta(seconds=args.duration - 2)
        title = f"Build-up to goal by {goal.team.name}"
    else:
        period_id, start = args.period, timedelta(seconds=args.start)
        title = "Metrica Sample Game 3"
    end = start + timedelta(seconds=args.duration)

    path = animate(period_id, start, end, args.output, step=args.step, title=title)
    print(f"Saved {path} ({path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
