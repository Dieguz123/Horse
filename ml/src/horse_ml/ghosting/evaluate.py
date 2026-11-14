"""ONE-TIME evaluation on the test set, after all choices were made on validation.

Loads the models saved by ``ghosting.train`` and reports the error in metres on
train (to see overfitting), validation and test. MLP rows are aggregated over the
training seeds as mean ± standard deviation. The "Δ test" column compares each
model with the best one (chosen on VALIDATION); differences smaller than twice
the seed-to-seed standard deviation are marked with "≈" (not meaningful).
For the World Cup dataset the error is also reported on visible targets only.

Usage (from ml/):
    uv run python -m horse_ml.ghosting.evaluate
    uv run python -m horse_ml.ghosting.evaluate --dataset worldcup
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from horse_ml.ghosting.dataset import METRICA, SCHEMAS, Schema, load
from horse_ml.ghosting.metrics import ErrorReport, report
from horse_ml.ghosting.train import REGISTRY_FILE, models_dir
from horse_ml.paths import PROCESSED_DIR

RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"
LABELS = {
    "metrica": {
        "mean_shape": "(a) Mean shape",
        "ball_ridge": "(b) Ridge on ball position",
        "situation_knn": "(c) kNN on situation",
        "mlp_per_team": "(d1) MLP, one per team",
        "mlp_unified": "(d2) MLP, unified + team input",
    },
    "worldcup": {
        "mean_shape": "(a) Mean line shape",
        "ball_ridge": "(b) Ridge on ball position",
        "situation_knn": "(c) kNN on situation",
        "mlp_per_team": "(d1) MLP, one per line size",
        "mlp_unified": "(d2) MLP, unified + line size input",
    },
}


def _fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}"


def evaluate(models: Path, schema: Schema = METRICA) -> dict:
    registry = json.loads((models / REGISTRY_FILE).read_text(encoding="utf-8"))
    data = load(schema=schema)
    splits = {name: data.split(name) for name in ("train", "val", "test")}

    per_file: dict[str, dict[str, ErrorReport]] = {}
    for entry in registry["models"]:
        model = joblib.load(models / entry["file"])
        per_file[entry["file"]] = {name: report(model.predict(split), split) for name, split in splits.items()}

    groups: dict[str, list[str]] = defaultdict(list)  # seeds of the same model
    for entry in registry["models"]:
        groups[entry["model"]].append(entry["file"])

    teams = sorted(per_file[registry["models"][0]["file"]]["test"].per_team) if not schema.masked else []
    columns = {
        "train": lambda r: r["train"].mean,
        "val": lambda r: r["val"].mean,
        "val_settled": lambda r: r["val"].settled_mean,
        "test": lambda r: r["test"].mean,
        "test_settled": lambda r: r["test"].settled_mean,
        **{f"test_{t}": (lambda r, t=t: r["test"].per_team[t]) for t in teams},
    }
    if schema.masked:
        columns["val_visible"] = lambda r: r["val"].visible_mean
        columns["test_visible"] = lambda r: r["test"].visible_mean
    rows = {name: {col: [fn(per_file[f]) for f in files] for col, fn in columns.items()}
            for name, files in groups.items()}

    best_name = next(e["model"] for e in registry["models"] if e["file"] == registry["best_by_val"])
    seed_std = max((np.std(r["test"], ddof=1) for r in rows.values() if len(r["test"]) > 1), default=0.0)
    best_test = float(np.mean(rows[best_name]["test"]))
    for name, r in rows.items():
        delta = float(np.mean(r["test"])) - best_test
        r["delta_test"] = delta
        r["within_seed_noise"] = bool(name != best_name and abs(delta) < 2 * seed_std)

    best_file = registry["best_by_val"]
    return {
        "dataset": schema.name,
        "best_by_val": best_name,
        "best_file": best_file,
        "seed_std_test": float(seed_std),
        "settled_after_s": 3.0,
        "n": {name: len(s) for name, s in splits.items()},
        "rows": rows,
        "best_test_report": per_file[best_file]["test"].as_dict(),
        "teams": teams,
        "masked": schema.masked,
    }


def to_markdown(results: dict) -> str:
    teams, masked = results["teams"], results.get("masked", False)
    labels = LABELS[results.get("dataset", "metrica")]
    base = ["train", "val", "val_settled"] + (["val_visible"] if masked else []) + ["test", "test_settled"] + (
        ["test_visible"] if masked else [])
    titles = {"train": "Train", "val": "Val", "val_settled": "Val settled", "val_visible": "Val visible",
              "test": "Test", "test_settled": "Test settled", "test_visible": "Test visible"}
    header = ["Model"] + [titles[c] for c in base] + [f"Test, {t} defending" for t in teams] + ["Δ test vs best"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, r in results["rows"].items():
        delta = "best (on val)" if name == results["best_by_val"] else (
            f"{r['delta_test']:+.2f}" + (" ≈" if r["within_seed_noise"] else ""))
        cells = [labels.get(name, name)] + [_fmt(r[c]) for c in base] + [_fmt(r[f"test_{t}"]) for t in teams] + [delta]
        lines.append("| " + " | ".join(cells) + " |")
    n = results["n"]
    target = "the defensive line (3-5 players)" if masked else "the 11"
    lines += ["", (
        f"Mean distance in metres between predicted and real defenders (average over {target}). "
        f"Samples: train {n['train']:,}, val {n['val']:,}, test {n['test']:,}. "
        f"Settled = more than {results['settled_after_s']:.0f} s since the last possession change. "
        + ("Visible = only defenders tracked on camera (not estimated by PFF). " if masked else "")
        + f"MLP rows: mean ± std over seeds. ≈ = difference smaller than 2 × the seed-to-seed "
        f"std on test ({results['seed_std_test']:.2f} m), i.e. not meaningful."
    )]
    return "\n".join(lines)


def per_slot_table(results: dict) -> str:
    best = results["best_test_report"]
    if results.get("masked"):
        from horse_ml.worldcup.backline import SLOT_NAMES

        lines = ["  per slot: " + ", ".join(
            f"{name} {err:.2f}" for name, err in zip(SLOT_NAMES, best["per_slot"], strict=True) if err is not None)]
        lines.append("  per defending team: " + ", ".join(f"{t} {e:.2f}" for t, e in best["per_team"].items()))
        lines.append("  per line size: " + ", ".join(f"{g} {e:.2f}" for g, e in best["per_group"].items()))
        return "\n".join(lines)
    meta = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    per_team_slot = best["per_team_slot"]
    lines = []
    for game_teams in meta["teams"].values():
        for team, info in game_teams.items():
            if team not in per_team_slot:
                continue
            lines.append(f"  {team} defending ({info['formation']}):")
            for slot, err in zip(info["slots"], per_team_slot[team], strict=True):
                lines.append(f"    {slot['slot']:>2} {slot['role']:<42} {err:5.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on the test set.")
    parser.add_argument("--dataset", choices=list(SCHEMAS), default="metrica")
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    schema = SCHEMAS[args.dataset]
    models = args.models or models_dir(schema)

    results = evaluate(models, schema)
    markdown = to_markdown(results)
    (models / RESULTS_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    (models / RESULTS_MD).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nBest model on validation: {results['best_file']} - test error (m)")
    print(per_slot_table(results))
    print(f"\nSaved {models / RESULTS_MD}")


if __name__ == "__main__":
    main()
