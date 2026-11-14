"""Train every ghosting model, choosing hyperparameters on the VALIDATION set only.

The test set is never loaded here: it is used once, by ``ghosting.evaluate``.

Usage (from ml/):
    uv run python -m horse_ml.ghosting.train                      # Metrica (phase 3)
    uv run python -m horse_ml.ghosting.train --dataset worldcup   # PFF World Cup (phase 4)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from horse_ml.ghosting.baselines import BallRidge, GhostModel, MeanShape, SituationKNN
from horse_ml.ghosting.dataset import METRICA, SCHEMAS, GhostData, Schema, load
from horse_ml.ghosting.metrics import report
from horse_ml.ghosting.mlp import MLPConfig, MLPGhost
from horse_ml.paths import MODELS_DIR

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
KNN_KS = (1, 5, 10, 20, 50, 100, 200, 500)
MLP_SEEDS = (0, 1, 2)
REGISTRY_FILE = "registry.json"


def models_dir(schema: Schema) -> Path:
    """Metrica models stay in models/ (phase 3), other datasets in models/<name>/."""
    return MODELS_DIR if schema is METRICA else MODELS_DIR / schema.name


def val_error(model: GhostModel, val: GhostData) -> float:
    return report(model.predict(val), val).mean


def train_and_validation(path: Path | None = None, schema: Schema = METRICA) -> tuple[GhostData, GhostData]:
    """Only the train and validation samples: test rows are dropped right away."""
    data = load(path, schema)
    return data.split("train"), data.split("val")


def fit_baselines(train: GhostData, val: GhostData) -> list[tuple[GhostModel, dict]]:
    """Fit baselines; for tunable ones keep the best setting on validation."""
    out: list[tuple[GhostModel, dict]] = []
    schema = train.schema

    mean_shape = MeanShape().fit(train)
    out.append((mean_shape, {}))

    ridge_grid = {}
    for alpha in RIDGE_ALPHAS:
        ridge_grid[alpha] = val_error(BallRidge(alpha).fit(train), val)
    best_alpha = min(ridge_grid, key=ridge_grid.get)
    out.append((BallRidge(best_alpha).fit(train), {"val_grid": {str(k): v for k, v in ridge_grid.items()}}))

    def knn(k: int, velocity: bool) -> SituationKNN:
        columns = list(schema.feature_columns) if velocity else schema.position_columns()
        return SituationKNN(k, velocity, columns)

    knn_grid = {}
    smallest_group = min(int((train.group == g).sum()) for g in set(train.group))
    for velocity in (False, True):
        for k in (k for k in KNN_KS if k <= smallest_group):
            knn_grid[(k, velocity)] = val_error(knn(k, velocity).fit(train), val)
    best_k, best_vel = min(knn_grid, key=knn_grid.get)
    out.append((knn(best_k, best_vel).fit(train),
                {"val_grid": {f"k={k},velocity={v}": e for (k, v), e in knn_grid.items()}}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ghosting models (validation only).")
    parser.add_argument("--dataset", choices=list(SCHEMAS), default="metrica")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MLP_SEEDS))
    args = parser.parse_args()
    schema = SCHEMAS[args.dataset]
    out_dir = args.out or models_dir(schema)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = train_and_validation(schema=schema)
    print(f"[{schema.name}] train {len(train):,} samples, val {len(val):,} samples")

    registry: list[dict] = []

    def save(model: GhostModel, file_name: str, extra: dict) -> None:
        joblib.dump(model, out_dir / file_name)
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
    (out_dir / REGISTRY_FILE).write_text(
        json.dumps({"dataset": schema.name, "models": registry, "best_by_val": best["file"]}, indent=2),
        encoding="utf-8",
    )
    print(f"best on validation: {best['file']} ({best['val_mean_m']:.2f} m)")


if __name__ == "__main__":
    main()
