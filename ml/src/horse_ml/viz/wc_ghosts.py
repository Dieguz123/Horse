"""World Cup ghosts: the defensive line only, real (filled) vs predicted (dashed).

Uses the best World Cup model on VALIDATION and moments from the validation games
only. Players of the defending team outside the line are drawn hollow (inputs, not
predicted); estimated positions (off camera) are drawn semi-transparent.

Usage (from ml/):
    uv run python -m horse_ml.viz.wc_ghosts
"""

from __future__ import annotations

import argparse
import json
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

from horse_ml.ghosting.dataset import WORLDCUP, from_frame, load
from horse_ml.ghosting.metrics import masked_mean, defender_distances
from horse_ml.ghosting.train import REGISTRY_FILE, models_dir
from horse_ml.paths import OUTPUTS_DIR
from horse_ml.viz.pitch import BALL_COLOR, BALL_EDGE, PITCH_COLOR, TEXT_COLOR, make_metre_pitch
from horse_ml.viz.sample import ATTACK_COLOR, BALL_ALPHA, BALL_SIZE, DEFENCE_COLOR
from horse_ml.worldcup.build import prepare_game, window_samples

WINDOW_S = 10.0
SAMPLE_STEP = 6  # frames between ~5 Hz samples at 29.97 fps
MAX_CANDIDATES = 10
MAX_JUMP_M = 0.5  # between two frames at 29.97 fps = 15 m/s
MIN_TEAMMATE_GAP_M = 0.25
ESTIMATED_ALPHA = 0.45


def _group(row: pd.Series, prefix: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.array([[row[f"{prefix}_{k}_x"], row[f"{prefix}_{k}_y"]] for k in range(n)], dtype=float)
    present = np.array([row[f"{prefix}_{k}_present"] == 1 for k in range(n)])
    visible = np.array([row[f"{prefix}_{k}_visible"] == 1 for k in range(n)])
    return xy, present, visible


def _scatter(pitch: Pitch, ax: Axes, xy: np.ndarray, vis: np.ndarray, **kw) -> list[Artist]:
    out = []
    for flag, alpha in ((True, 1.0), (False, ESTIMATED_ALPHA)):
        sel = vis == flag
        if sel.any():
            label = kw.pop("label", None) if flag or not (vis == True).any() else None  # noqa: E712
            out.append(pitch.scatter(xy[sel, 0], xy[sel, 1], ax=ax, alpha=alpha, label=label, **kw))
    return out


def draw_line_ghosts(ax: Axes, pitch: Pitch, row: pd.Series, ghosts: np.ndarray) -> list[Artist]:
    artists: list[Artist] = []
    att, att_p, att_v = _group(row, "att", 11)
    dof, dof_p, dof_v = _group(row, "dof", 8)
    back, back_p, back_v = _group(row, "back", 5)
    g = ghosts[back_p]
    artists.append(pitch.lines(back[back_p, 0], back[back_p, 1], g[:, 0], g[:, 1], ax=ax,
                               color=DEFENCE_COLOR, lw=1.8, alpha=0.9, zorder=2))
    artists.append(pitch.scatter(g[:, 0], g[:, 1], ax=ax, s=260, facecolors="none", edgecolors=DEFENCE_COLOR,
                                 linewidth=1.8, linestyle="--", zorder=3, label="Ghost (predicted line)"))
    artists += _scatter(pitch, ax, att[att_p], att_v[att_p], s=200, color=ATTACK_COLOR, edgecolors=PITCH_COLOR,
                        linewidth=2, zorder=4, label=f"{row['attacking_team']} attacking")
    artists += _scatter(pitch, ax, dof[dof_p], dof_v[dof_p], s=200, facecolors="none", edgecolors=DEFENCE_COLOR,
                        linewidth=2, zorder=4, label=f"{row['defending_team']}, rest of the team")
    artists += _scatter(pitch, ax, back[back_p], back_v[back_p], s=260, color=DEFENCE_COLOR, edgecolors=PITCH_COLOR,
                        linewidth=2, zorder=5, label=f"{row['defending_team']} defensive line (real)")
    artists.append(pitch.scatter(row["ball_x"], row["ball_y"], ax=ax, s=BALL_SIZE, color=BALL_COLOR,
                                 edgecolors=BALL_EDGE, linewidth=1, alpha=BALL_ALPHA, zorder=6, label="Ball"))
    return artists


def _title(row: pd.Series, error: float, model_file: str) -> str:
    minutes, seconds = divmod(row["timestamp_s"], 60)
    return (
        f"{row['match']} · {row['defending_team']} defending (line of {row['back_line_size']}) · "
        f"line ghost error {error:.1f} m\nP{row['period']} {int(minutes):02d}:{seconds:04.1f} · "
        f"{row['seconds_since_possession_change']:.0f} s in possession · attack → · "
        f"faded = position estimated off camera · {model_file.removesuffix('.joblib')}"
    )


def _figure() -> tuple[plt.Figure, Axes, Pitch]:
    pitch = make_metre_pitch()
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.set_facecolor(PITCH_COLOR)
    return fig, ax, pitch


def _legend(ax: Axes) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
              labelcolor=TEXT_COLOR, fontsize=9, markerscale=0.7)


def plot_moment(row: pd.Series, ghosts: np.ndarray, error: float, model_file: str, output: Path) -> Path:
    fig, ax, pitch = _figure()
    draw_line_ghosts(ax, pitch, row, ghosts)
    _legend(ax)
    ax.set_title(_title(row, error, model_file), color=TEXT_COLOR, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=PITCH_COLOR)
    plt.close(fig)
    return output


def animate(samples: pd.DataFrame, ghosts: np.ndarray, errors: np.ndarray, model_file: str,
            output: Path, step: int = 3) -> Path:
    fig, ax, pitch = _figure()
    title = ax.set_title("", color=TEXT_COLOR, fontsize=10)
    current: list[Artist] = []
    rows = list(range(0, len(samples), step))

    def update(i: int) -> list[Artist]:
        for artist in current:
            artist.remove()
        current.clear()
        r = rows[i]
        current.extend(draw_line_ghosts(ax, pitch, samples.iloc[r], ghosts[r]))
        if i == 0:
            _legend(ax)
        title.set_text(_title(samples.iloc[r], errors[r], model_file))
        return current

    fps = 29.97 / step
    anim = FuncAnimation(fig, update, frames=len(rows), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": PITCH_COLOR})
    plt.close(fig)
    return output


def candidate_windows(frame: pd.DataFrame, errors: np.ndarray, settled: np.ndarray) -> list[tuple]:
    """Non-overlapping windows of consecutive, settled ~5 Hz samples, highest error first."""
    frame = frame.assign(error=errors, settled=settled)
    n_window = int(round(WINDOW_S * 29.97 / SAMPLE_STEP)) + 1
    found = []
    for (game, period), g in frame.groupby(["game_id", "period"]):
        g = g.sort_values("frame_num")
        fn = g["frame_num"].to_numpy()
        breaks = np.flatnonzero(np.diff(fn) != SAMPLE_STEP) + 1
        for run in np.split(np.arange(len(g)), breaks):
            r = g.iloc[run]
            ok, err, t = r["settled"].to_numpy(), r["error"].to_numpy(), r["timestamp_s"].to_numpy()
            same_team = r["defending_team"].to_numpy()
            for i in range(len(r) - n_window + 1):
                if ok[i:i + n_window].all() and len(set(same_team[i:i + n_window])) == 1:
                    found.append((float(err[i:i + n_window].mean()), str(game), int(period),
                                  float(t[i]), float(t[i + n_window - 1])))
    found.sort(reverse=True)
    chosen = []
    for w in found:
        if all(w[1] != c[1] or w[2] != c[2] or w[4] < c[3] or w[3] > c[4] for c in chosen):
            chosen.append(w)
    return chosen


def tracking_problems(samples: pd.DataFrame, n_expected: int) -> list[str]:
    problems = []
    if len(samples) < n_expected:
        problems.append(f"only {len(samples)}/{n_expected} frames in play and complete")
    for prefix, n in (("att", 11), ("dof", 8), ("back", 5)):
        x = samples[[f"{prefix}_{k}_x" for k in range(n)]].to_numpy(dtype=float)
        y = samples[[f"{prefix}_{k}_y" for k in range(n)]].to_numpy(dtype=float)
        p = samples[[f"{prefix}_{k}_present" for k in range(n)]].to_numpy() == 1
        both = p[1:] & p[:-1]
        jump = np.where(both, np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)), 0.0)
        if jump.size and jump.max() > MAX_JUMP_M:
            problems.append(f"{prefix}: jump of {jump.max():.1f} m between frames")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup line ghosts (validation games only).")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--low-percentile", type=float, default=5.0)
    args = parser.parse_args()

    models = models_dir(WORLDCUP)
    model_file = json.loads((models / REGISTRY_FILE).read_text(encoding="utf-8"))["best_by_val"]
    model = joblib.load(models / model_file)
    val = load(schema=WORLDCUP).split("val")
    pred = model.predict(val)
    errors = masked_mean(defender_distances(pred, val.Y), val.mask, axis=1)
    print(f"model {model_file}: validation error {np.nanmean(errors):.2f} m")

    settled = np.flatnonzero(val.settled)
    target = np.percentile(errors[settled], args.low_percentile)
    low = settled[np.argmin(np.abs(errors[settled] - target))]
    print("low error :", plot_moment(val.frame.iloc[low], pred[low], errors[low], model_file,
                                     args.out / "wc_ghost_low.png"))

    prepared = {}
    for mean_err, game, period, start, end in candidate_windows(val.frame, errors, val.settled)[:MAX_CANDIDATES]:
        if game not in prepared:
            prepared[game] = prepare_game(game)
        samples = window_samples(prepared[game], period, start, end)
        problems = tracking_problems(samples, int(round((end - start) * 29.97)) + 1)
        label = f"game {game} P{period} {start:.0f}-{end:.0f} s, mean error {mean_err:.1f} m"
        if problems:
            print(f"skip {label}: {'; '.join(problems)}")
            continue
        print(f"chosen {label}")
        data = from_frame(samples, WORLDCUP)
        wp = model.predict(data)
        we = masked_mean(defender_distances(wp, data.Y), data.mask, axis=1)
        worst = int(np.nanargmax(we))
        print("high error:", plot_moment(samples.iloc[worst], wp[worst], we[worst], model_file,
                                         args.out / "wc_ghost_high.png"))
        print("animation :", animate(samples, wp, we, model_file, args.out / "wc_ghost_high.gif"))
        break
    else:
        print("No clean high-error window found among the candidates")


if __name__ == "__main__":
    main()
