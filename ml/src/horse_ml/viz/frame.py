"""Plot a single tracking frame on a pitch and save it as PNG.

Default moment: the first goal of the game (Team B, period 1, 18:20.96).

Usage (from ml/):
    uv run python -m horse_ml.viz.frame
    uv run python -m horse_ml.viz.frame --period 2 --time 600   # 10:00 into period 2
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no GUI window
import matplotlib.pyplot as plt

from horse_ml.data.loading import load_events, load_tracking
from horse_ml.data.selection import find_event, format_clock, frame_window, frames_needed
from horse_ml.paths import OUTPUTS_DIR
from horse_ml.viz.pitch import PITCH_COLOR, TEXT_COLOR, add_legend, draw_frame, make_pitch


def plot_frame(period_id: int, t: timedelta, output: Path, title_suffix: str = "") -> Path:
    events = load_events()
    tracking = load_tracking(limit=frames_needed(events.metadata, period_id, t))
    window = frame_window(tracking, period_id, t - timedelta(milliseconds=20), t + timedelta(milliseconds=20))
    if not window:
        raise LookupError(f"No frame at period {period_id}, {format_clock(t)}")
    frame = window[0]

    pitch = make_pitch(tracking.metadata)
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.set_facecolor(PITCH_COLOR)
    draw_frame(ax, pitch, frame, tracking.metadata)
    add_legend(ax)
    ax.set_title(
        f"Metrica Sample Game 3 · Period {period_id} · {format_clock(frame.timestamp)}{title_suffix}",
        color=TEXT_COLOR,
        fontsize=13,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=PITCH_COLOR)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one tracking frame.")
    parser.add_argument("--period", type=int, help="period id (default: first goal)")
    parser.add_argument("--time", type=float, help="seconds since period start")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "frame.png")
    args = parser.parse_args()

    if args.period is None or args.time is None:
        goal = find_event(load_events(), "SHOT", result="GOAL")
        period_id, t, suffix = goal.period.id, goal.timestamp, f" · Goal by {goal.team.name}"
    else:
        period_id, t, suffix = args.period, timedelta(seconds=args.time), ""

    path = plot_frame(period_id, t, args.output, suffix)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
