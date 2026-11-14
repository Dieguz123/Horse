"""Train every ghosting model, choosing hyperparameters on the VALIDATION set only.

The test set is never loaded here: it is used once, by ``ghosting.evaluate``.

Usage (from ml/):
    uv run python -m horse_ml.ghosting.train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from horse_ml.ghosting.baselines import BallRidge, GhostModel, MeanShape, SituationKNN
from horse_ml.ghosting.dataset import SAMPLES_PATH, GhostData, load
from horse_ml.ghosting.metrics import report
from horse_ml.ghosting.mlp import MLPConfig, MLPGhost
from horse_ml.paths import MODELS_DIR

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
KNN_KS = (1, 5, 10, 20, 50, 100, 200, 500)
MLP_SEEDS = (0, 1, 2)
REGISTRY_FILE = "registry.json"


def val_error(model: GhostModel, val: GhostData) -> float:
    return report(model.predict(val), val).mean


def train_and_validation(path: Path = SAMPLES_PATH) -> tuple[GhostData, GhostData]:
    """Only the train and validation samples: test rows are dropped right away."""
    data = load(path)
    return data.split("train"), data.split("val")


def fit_baselines(train: GhostData, val: GhostData) -> list[tuple[GhostModel, dict]]:
    """Fit baselines; for tunable ones keep the best setting on validation."""
    out: list[tuple[GhostModel, dict]] = []

    mean_shape = MeanShape().fit(train)
    out.append((mean_shape, {}))

    ridge_grid = {}
    for alpha in RIDGE_ALPHAS:
        ridge_grid[alpha] = val_error(BallRidge(alpha).fit(train), val)
    best_alpha = min(ridge_grid, key=ridge_grid.get)
    out.append((BallRidge(best_alpha).fit(train), {"val_grid": {str(k): v for k, v in ridge_grid.items()}}))

    knn_grid = {}
    smallest_team = min(int((train.team == t).sum()) for t in set(train.team))
    for velocity in (False, True):
        for k in (k for k in KNN_KS if k <= smallest_team):
            knn_grid[(k, velocity)] = val_error(SituationKNN(k, velocity).fit(train), val)
    best_k, best_vel = min(knn_grid, key=knn_grid.get)
    out.append(
        (
            SituationKNN(best_k, best_vel).fit(train),
            {"val_grid": {f"k={k},velocity={v}": e for (k, v), e in knn_grid.items()}},
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ghosting models (validation only).")
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MLP_SEEDS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train, val = train_and_validation()
    print(f"train {len(train):,} samples, val {len(val):,} samples")

    registry: list[dict] = []

    def save(model: GhostModel, file_name: str, extra: dict) -> None:
        joblib.dump(model, args.out / file_name)
        entry = {
            "model": model.name,
            "file": file_name,
            "params": model.describe(),
            "val_mean_m": val_error(model, val),
            **extra,
        }
        registry.append(entry)
        print(f"  {file_name:<28} val {entry['val_mean_m']:.2f} m  {model.describe()}")

    t0 = time.perf_counter()
    for model, extra in fit_baselines(train, val):
        save(model, f"{model.name}.joblib", extra)

    for per_team in (True, False):
        for seed in args.seeds:
            model = MLPGhost(per_team, MLPConfig(seed=seed)).fit(train, val)
            save(model, f"{model.name}_seed{seed}.joblib", {"seed": seed})
    print(f"trained in {time.perf_counter() - t0:.0f} s")

    best = min(registry, key=lambda e: e["val_mean_m"])
    (args.out / REGISTRY_FILE).write_text(
        json.dumps({"models": registry, "best_by_val": best["file"]}, indent=2), encoding="utf-8"
    )
    print(f"best on validation: {best['file']} ({best['val_mean_m']:.2f} m)")


if __name__ == "__main__":
    main()
