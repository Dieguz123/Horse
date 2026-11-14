from __future__ import annotations

import pytest

from horse_ml.data.loading import game_files
from horse_ml.paths import METRICA_DIR

DATA_AVAILABLE = not game_files(METRICA_DIR).missing()

requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="Metrica data not downloaded (run `uv run python -m horse_ml.data.download`)",
)
