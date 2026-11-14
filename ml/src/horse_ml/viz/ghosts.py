"""The ghosts: real defenders (filled) vs predicted defenders (dashed circles).

Uses the best model on VALIDATION and looks for moments in the validation set
only (the test set stays untouched):
- high error: the 10-second window with the highest mean error among settled
  situations (> 3 s since the last possession change) with the ball in play for
  the whole window, skipping windows with signs of tracking errors;
- low error: a settled validation sample with a low error, for comparison.

The high-error window is rebuilt at 25 Hz with the phase 2 pipeline, so the GIF
is smooth (training samples are only 5 Hz).

Usage (from ml/):
    uv run python -m horse_ml.viz.ghosts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from mplsoccer import Pitch

from horse_ml.data.games import GAMES
from horse_ml.ghosting.baselines import GhostModel
from horse_ml.ghosting.dataset import SETTLED_AFTER_S, GhostData, from_frame, load
from horse_ml.ghosting.metrics import defender_distances
from horse_ml.ghosting.train import REGISTRY_FILE
from horse_ml.paths import MODELS_DIR, OUTPUTS_DIR
from horse_ml.preprocessing.build import prepare_game, window_samples
from horse_ml.preprocessing.roles import N_SLOTS
from horse_ml.viz.pitch import BALL_COLOR, BALL_EDGE, PITCH_COLOR, TEXT_COLOR, make_metre_pitch
from horse_ml.viz.sample import ATTACK_COLOR, BALL_ALPHA, BALL_SIZE, DEFENCE_COLOR

WINDOW_S = 10.0
SAMPLE_STEP = 5  # frames between 5 Hz samples
MAX_CANDIDATES = 10

# tracking-error checks on the 25 Hz window
MAX_JUMP_M = 0.6  # between two frames at 25 Hz = 15 m/s
MIN_TEAMMATE_GAP_M = 0.25  # two teammates on the same spot = identity swap / glitch
MAX_OUTSIDE_M = 3.0  # players this far beyond the lines


@dataclass
class Window:
    game_id: str
    period: int
    start_s: float
    end_s: float
    mean_error: float


def best_model(models_dir: Path) -> tuple[GhostModel, str]:
    registry = json.loads((models_dir / REGISTRY_FILE).read_text(encoding="utf-8"))
    file = registry["best_by_val"]
    return joblib.load(models_dir / file), file


def candidate_windows(val: GhostData, errors: np.ndarray) -> list[Window]:
    """Non-overlapping 10 s windows of consecutive, settled, in-play 5 Hz samples,
    sorted by mean error (highest first)."""
    frame = val.frame.assign(error=errors, settled=val.settled)
    n_window = int(WINDOW_S * 5) + 1
    windows: list[Window] = []
    for (game_id, period), group in frame.groupby(["game_id", "period"]):
        group = group.sort_values("frame_id")
        fid = group["frame_id"].to_numpy()
        # consecutive 5 Hz samples = ball in play and data complete in between
        breaks = np.flatnonzero(np.diff(fid) != SAMPLE_STEP) + 1
        for run in np.split(np.arange(len(group)), breaks):
            run_rows = group.iloc[run]
            ok = run_rows["settled"].to_numpy()
            err = run_rows["error"].to_numpy()
            t = run_rows["timestamp_s"].to_numpy()
            for i in range(len(run_rows) - n_window + 1):
                if ok[i : i + n_window].all():
                    windows.append(
                        Window(str(game_id), int(period), float(t[i]), float(t[i + n_window - 1]),
                               float(err[i : i + n_window].mean()))
                    )
    windows.sort(key=lambda w: -w.mean_error)
    chosen: list[Window] = []
    for w in windows:
        if all(w.game_id != c.game_id or w.period != c.period or w.end_s < c.start_s or w.start_s > c.end_s
               for c in chosen):
            chosen.append(w)
    return chosen


def tracking_problems(samples: pd.DataFrame, n_expected: int) -> list[str]:
    """Reasons to distrust a 25 Hz window (empty list = looks clean)."""
    problems = []
    if len(samples) < n_expected:
        problems.append(f"only {len(samples)}/{n_expected} frames in play and complete")
    if samples["attacking_team"].nunique() > 1:
        problems.append("possession changes inside the window")
    for prefix in ("att", "def"):
        x = samples[[f"{prefix}_{k}_x" for k in range(N_SLOTS)]].to_numpy()
        y = samples[[f"{prefix}_{k}_y" for k in range(N_SLOTS)]].to_numpy()
        jump = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
        if len(jump) and jump.max() > MAX_JUMP_M:
            problems.append(f"{prefix}: position jump of {jump.max():.1f} m between frames")
        if (np.abs(x) > 52.5 + MAX_OUTSIDE_M).any() or (np.abs(y) > 34 + MAX_OUTSIDE_M).any():
            problems.append(f"{prefix}: player far outside the pitch")
        gaps = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        gaps[:, np.arange(N_SLOTS), np.arange(N_SLOTS)] = np.inf
        if gaps.min() < MIN_TEAMMATE_GAP_M:
            problems.append(f"{prefix}: two teammates {gaps.min():.2f} m apart")
    return problems


def draw_ghosts(ax: Axes, pitch: Pitch, row: pd.Series, ghosts: np.ndarray) -> list[Artist]:
    """Attackers, real defenders (filled), ghosts (dashed), linking lines, ball."""
    artists: list[Artist] = []
    att = np.array([[row[f"att_{k}_x"], row[f"att_{k}_y"]] for k in range(N_SLOTS)])
    real = np.array([[row[f"def_{k}_x"], row[f"def_{k}_y"]] for k in range(N_SLOTS)])
    artists.append(pitch.lines(real[:, 0], real[:, 1], ghosts[:, 0], ghosts[:, 1], ax=ax,
                               color=DEFENCE_COLOR, lw=1.5, alpha=0.8, zorder=2))
    artists.append(pitch.scatter(ghosts[:, 0], ghosts[:, 1], ax=ax, s=260, facecolors="none",
                                 edgecolors=DEFENCE_COLOR, linewidth=1.8, linestyle="--", zorder=3,
                                 label="Ghost defender (predicted)"))
    artists.append(pitch.scatter(att[:, 0], att[:, 1], ax=ax, s=220, color=ATTACK_COLOR,
                                 edgecolors=PITCH_COLOR, linewidth=2, zorder=4,
                                 label=f"{row['attacking_team']} attacking"))
    artists.append(pitch.scatter(real[:, 0], real[:, 1], ax=ax, s=260, color=DEFENCE_COLOR,
                                 edgecolors=PITCH_COLOR, linewidth=2, zorder=4,
                                 label=f"{row['defending_team']} defending (real)"))
    for k in range(N_SLOTS):
        artists.append(pitch.annotate(str(k), (real[k, 0], real[k, 1]), ax=ax, color=TEXT_COLOR,
                                      fontsize=7, fontweight="bold", ha="center", va="center", zorder=5))
    artists.append(pitch.scatter(row["ball_x"], row["ball_y"], ax=ax, s=BALL_SIZE, color=BALL_COLOR,
                                 edgecolors=BALL_EDGE, linewidth=1, alpha=BALL_ALPHA, zorder=6, label="Ball"))
    return artists


def _title(row: pd.Series, error: float, model_file: str) -> str:
    minutes, seconds = divmod(row["timestamp_s"], 60)
    return (
        f"Ghost error {error:.1f} m (mean of 11 defenders) Â· P{row['period']} "
        f"{int(minutes):02d}:{seconds:05.2f} Â· {row['seconds_since_possession_change']:.0f} s in possession\n"
        f"attack â†’ Â· model: {model_file.removesuffix('.joblib')} Â· lines join each defender to its ghost"
    )


def _figure() -> tuple[plt.Figure, Axes, Pitch]:
    pitch = make_metre_pitch()
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.set_facecolor(PITCH_COLOR)
    return fig, ax, pitch


def _legend(ax: Axes) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
              labelcolor=TEXT_COLOR, fontsize=9, markerscale=0.7)


def plot_moment(row: pd.Series, ghosts: np.ndarray, error: float, model_file: str, output: Path) -> Path:
    fig, ax, pitch = _figure()
    draw_ghosts(ax, pitch, row, ghosts)
    _legend(ax)
    ax.set_title(_title(row, error, model_file), color=TEXT_COLOR, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=PITCH_COLOR)
    plt.close(fig)
    return output


def animate_ghosts(samples: pd.DataFrame, ghosts: np.ndarray, errors: np.ndarray, model_file: str,
                   output: Path, step: int = 2) -> Path:
    fig, ax, pitch = _figure()
    title = ax.set_title("", color=TEXT_COLOR, fontsize=11)
    current: list[Artist] = []
    rows = list(range(0, len(samples), step))

    def update(i: int) -> list[Artist]:
        for artist in current:
            artist.remove()
        current.clear()
        r = rows[i]
        current.extend(draw_ghosts(ax, pitch, samples.iloc[r], ghosts[r]))
        if i == 0:
            _legend(ax)
        title.set_text(_title(samples.iloc[r], errors[r], model_file))
        return current

    fps = 25 / step
    anim = FuncAnimation(fig, update, frames=len(rows), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": PITCH_COLOR})
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ghost defenders (validation set only).")
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--low-percentile", type=float, default=5.0)
    args = parser.parse_args()

    model, model_file = best_model(args.models)
    val = load().split("val")
    pred = model.predict(val)
    errors = defender_distances(pred, val.Y).mean(axis=1)
    print(f"model {model_file}: validation error {errors.mean():.2f} m")

    # low error: a settled validation sample at a low percentile of the error
    settled = np.flatnonzero(val.settled)
    target = np.percentile(errors[settled], args.low_percentile)
    low = settled[np.argmin(np.abs(errors[settled] - target))]
    path = plot_moment(val.frame.iloc[low], pred[low], errors[low], model_file, args.out / "ghost_low.png")
    print(f"low error  ({errors[low]:.1f} m, p{args.low_percentile:.0f} of settled): {path}")

    # high error: highest-error settled window that passes the tracking checks at 25 Hz
    windows = candidate_windows(val, errors)[:MAX_CANDIDATES]
    prepared = {}
    for w in windows:
        if w.game_id not in prepared:
            prepared[w.game_id] = prepare_game(GAMES[w.game_id])
        samples = window_samples(prepared[w.game_id], w.period, w.start_s, w.end_s)
        problems = tracking_problems(samples, n_expected=int(round((w.end_s - w.start_s) * 25)) + 1)
        label = f"P{w.period} {w.start_s:.0f}-{w.end_s:.0f} s, mean error {w.mean_error:.1f} m"
        if problems:
            print(f"skip {label}: {'; '.join(problems)}")
            continue
        print(f"chosen {label}")
        data = from_frame(samples)
        window_pred = model.predict(data)
        window_err = defender_distances(window_pred, data.Y).mean(axis=1)
        worst = int(np.argmax(window_err))
        print("high error:", plot_moment(samples.iloc[worst], window_pred[worst], window_err[worst],
                                        model_file, args.out / "ghost_high.png"))
        print("animation :", animate_ghosts(samples, window_pred, window_err, model_file,
                                            args.out / "ghost_high.gif"))
        per_slot = defender_distances(window_pred, data.Y).mean(axis=0)
        roles = [samples.iloc[0][f"def_{k}_role"] for k in range(N_SLOTS)]
        print(f"defending: {samples.iloc[0]['defending_team']} - error per slot over the window:")
        for k in np.argsort(-per_slot):
            print(f"  slot {k:>2} {roles[k]:<42} {per_slot[k]:.1f} m")
        break
    else:
        print("No clean high-error window found among the candidates")


if __name__ == "__main__":
    main()
