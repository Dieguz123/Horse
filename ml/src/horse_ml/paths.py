"""Central place for filesystem paths used by the ML package."""

from __future__ import annotations

import os
from pathlib import Path

# src/horse_ml/paths.py -> parents[2] is the ml/ folder
ML_ROOT = Path(__file__).resolve().parents[2]

# Raw data can be relocated with the HORSE_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("HORSE_DATA_DIR", ML_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
METRICA_DIR = RAW_DIR / "metrica"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ML_ROOT / "outputs"
MODELS_DIR = ML_ROOT / "models"
