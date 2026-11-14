"""ONE-TIME evaluation on the test set, after all choices were made on validation.

Loads the models saved by ``ghosting.train`` and reports the error in metres on
train (to see overfitting), validation and test. MLP rows are aggregated over the
training seeds as mean ± standard deviation. The "Δ test" column compares each
model with the best one (chosen on VALIDATION); differences smaller than twice
the seed-to-seed standard deviation are marked with "≈" (not meaningful).

Usage (from ml/):
    uv run python -m horse_ml.ghosting.evaluate
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from horse_ml.ghosting.dataset import load
from horse_ml.ghosting.metrics import ErrorReport, report
from horse_ml.ghosting.train import REGISTRY_FILE
from horse_ml.paths import MODELS_DIR, PROCESSED_DIR

RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"
LABELS = {
    "mean_shape": "(a) Mean shape",
    "ball_ridge": "(b) Ridge on ball position",
    "situation_knn": "(c) kNN on situation",
    "mlp_per_team": "(d1) MLP, one per team",
    "mlp_unified": "(d2) MLP, unified + team input",
}


def _fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}"


def evaluate(models_dir: Path) -> dict:
    registry = json.loads((models_dir / REGISTRY_FILE).read_text(encoding="utf-8"))
    data = load()
    splits = {name: data.split(name) for name in ("train", "val", "test")}

    per_file: dict[str, dict[str, ErrorReport]] = {}
    for entry in registry["models"]:
        model = joblib.load(models_dir / entry["file"])
        per_file[entry["file"]] = {
            name: report(model.predict(split), split) for name, split in splits.items()
        }

    # group seeds of the same model
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in registry["models"]:
        groups[entry["model"]].append(entry["file"])

    teams = sorted(per_file[registry["models"][0]["file"]]["test"].per_team)
    columns = {
        "train": lambda r: r["train"].mean,
        "val": lambda r: r["val"].mean,
        "val_settled": lambda r: r["val"].settled_mean,
        "test": lambda r: r["test"].mean,
        "test_settled": lambda r: r["test"].settled_mean,
        **{f"test_{t}": (lambda r, t=t: r["test"].per_team[t]) for t in teams},
    }
    rows = {
        name: {col: [fn(per_file[f]) for f in files] for col, fn in columns.items()}
        for name, files in groups.items()
    }

    best_name = next(e["model"] for e in registry["models"] if e["file"] == registry["best_by_val"])
    seed_std = max(
        (np.std(r["test"], ddof=1) for r in rows.values() if len(r["test"]) > 1), default=0.0
    )
    best_test = float(np.mean(rows[best_name]["test"]))
    for name, r in rows.items():
        delta = float(np.mean(r["test"])) - best_test
        r["delta_test"] = delta
        r["within_seed_noise"] = bool(name != best_name and abs(delta) < 2 * seed_std)

    best_file = registry["best_by_val"]
    return {
        "best_by_val": best_name,
        "best_file": best_file,
        "seed_std_test": float(seed_std),
        "settled_after_s": 3.0,
        "n": {name: len(s) for name, s in splits.items()},
        "rows": rows,
        "best_test_report": per_file[best_file]["test"].as_dict(),
        "teams": teams,
    }


def to_markdown(results: dict) -> str:
    teams = results["teams"]
    header = (
        ["Model", "Train", "Val", "Val settled", "Test", "Test settled"]
        + [f"Test, {t} defending" for t in teams]
        + ["Δ test vs best"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, r in results["rows"].items():
        if name == results["best_by_val"]:
            delta = "best (on val)"
        else:
            delta = f"{r['delta_test']:+.2f}" + (" ≈" if r["within_seed_noise"] else "")
        cells = (
            [LABELS.get(name, name)]
            + [_fmt(r[c]) for c in ("train", "val", "val_settled", "test", "test_settled")]
            + [_fmt(r[f"test_{t}"]) for t in teams]
            + [delta]
        )
        lines.append("| " + " | ".join(cells) + " |")
    n = results["n"]
    lines.append("")
    lines.append(
        f"Mean distance in metres between predicted and real defenders (average of the 11). "
        f"Samples: train {n['train']:,}, val {n['val']:,}, test {n['test']:,}. "
        f"Settled = more than {results['settled_after_s']:.0f} s since the last possession change. "
        f"MLP rows: mean ± std over seeds. ≈ = difference smaller than 2 × the seed-to-seed "
        f"std on test ({results['seed_std_test']:.2f} m), i.e. not meaningful."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on the test set.")
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    results = evaluate(args.models)
    markdown = to_markdown(results)
    (args.models / RESULTS_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.models / RESULTS_MD).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    print(f"\nBest model on validation: {results['best_file']} - test error per slot (m)")
    print(per_slot_table(results))
    print(f"\nSaved {args.models / RESULTS_MD}")


def per_slot_table(results: dict) -> str:
    meta = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    per_team_slot = results["best_test_report"]["per_team_slot"]
    lines = []
    for game_teams in meta["teams"].values():
        for team, info in game_teams.items():
            if team not in per_team_slot:
                continue
            lines.append(f"  {team} defending ({info['formation']}):")
            for slot, err in zip(info["slots"], per_team_slot[team], strict=True):
                lines.append(f"    {slot['slot']:>2} {slot['role']:<42} {err:5.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
