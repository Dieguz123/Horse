"""Sanity-check plot: one processed sample on the pitch with velocity arrows.

Arrows show where each player (and the ball) would be in 1 second at the current
velocity. The attacking team always attacks left-to-right.

Usage (from ml/):
    uv run python -m horse_ml.viz.sample                    # a random test sample
    uv run python -m horse_ml.viz.sample --sample-id metrica_3_27510
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from horse_ml.paths import OUTPUTS_DIR, PROCESSED_DIR
from horse_ml.preprocessing.build import SAMPLES_FILE
from horse_ml.preprocessing.roles import N_SLOTS
from horse_ml.viz.pitch import BALL_COLOR, BALL_EDGE, PITCH_COLOR, TEXT_COLOR, make_metre_pitch

# Same two validated hues as the frame plots, here encoding attack / defence.
ATTACK_COLOR = "#d95926"
DEFENCE_COLOR = "#3987e5"
# Small, semi-transparent ball so it never hides the player on it.
BALL_SIZE = 45
BALL_ALPHA = 0.75


def plot_sample(row: pd.Series, output: Path, pitch_length: float = 105, pitch_width: float = 68) -> Path:
    pitch = make_metre_pitch(pitch_length, pitch_width)
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.set_facecolor(PITCH_COLOR)

    for prefix, color, label in (
        ("att", ATTACK_COLOR, f"{row['attacking_team']} attacking ({row['attacking_formation']})"),
        ("def", DEFENCE_COLOR, f"{row['defending_team']} defending ({row['defending_formation']}) = target"),
    ):
        xs = [row[f"{prefix}_{k}_x"] for k in range(N_SLOTS)]
        ys = [row[f"{prefix}_{k}_y"] for k in range(N_SLOTS)]
        vxs = [row[f"{prefix}_{k}_vx"] for k in range(N_SLOTS)]
        vys = [row[f"{prefix}_{k}_vy"] for k in range(N_SLOTS)]
        pitch.arrows(xs, ys, [x + vx for x, vx in zip(xs, vxs)], [y + vy for y, vy in zip(ys, vys)],
                     ax=ax, color=color, width=1.5, headwidth=4, headlength=4, alpha=0.9, zorder=2)
        pitch.scatter(xs, ys, ax=ax, s=260, color=color, edgecolors=PITCH_COLOR, linewidth=2,
                      zorder=3, label=label)
        for k, (x, y) in enumerate(zip(xs, ys)):
            pitch.annotate(str(k), (x, y), ax=ax, color=TEXT_COLOR, fontsize=7, fontweight="bold",
                           ha="center", va="center", zorder=4)

    bx, by = row["ball_x"], row["ball_y"]
    pitch.arrows(bx, by, bx + row["ball_vx"], by + row["ball_vy"], ax=ax, color=BALL_COLOR,
                 width=1.5, headwidth=4, headlength=4, zorder=5)
    pitch.scatter(bx, by, ax=ax, s=BALL_SIZE, color=BALL_COLOR, edgecolors=BALL_EDGE, linewidth=1,
                  alpha=BALL_ALPHA, zorder=6, label="Ball")

    ax.annotate("attack →", xy=(0.5, 1.0), xycoords="axes fraction", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
              labelcolor=TEXT_COLOR, fontsize=9, markerscale=0.7)
    minutes, seconds = divmod(row["timestamp_s"], 60)
    ax.set_title(
        f"{row['sample_id']} · {row['split']} · P{row['period']} {int(minutes):02d}:{seconds:05.2f} · "
        f"{row['seconds_since_possession_change']:.1f} s since possession change\n"
        "numbers = slot · arrows = position in 1 s",
        color=TEXT_COLOR, fontsize=11, pad=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=PITCH_COLOR)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one processed sample.")
    parser.add_argument("--sample-id", help="default: a random test sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / SAMPLES_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "sample.png")
    args = parser.parse_args()

    samples = pd.read_parquet(args.input)
    if args.sample_id:
        row = samples.loc[samples["sample_id"] == args.sample_id].iloc[0]
    else:
        row = samples[samples["split"] == "test"].sample(1, random_state=args.seed).iloc[0]
    print(f"Saved {plot_sample(row, args.output)} ({row['sample_id']})")


if __name__ == "__main__":
    main()
