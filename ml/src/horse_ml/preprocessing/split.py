"""Train / validation / test split by contiguous time blocks.

Consecutive frames are nearly identical, so a random split would put almost the
same situation in train and test (leakage) and give over-optimistic metrics.
Instead every period of every game is cut into contiguous blocks (70/15/15 by
time) and samples within ``gap_s / 2`` of a block boundary are discarded, so no
test sample has a near-twin in train. Splitting each period (instead of the
whole game) keeps the test set from being only the final minutes of a match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_FRACTIONS: tuple[tuple[str, float], ...] = (("train", 0.70), ("val", 0.15), ("test", 0.15))
DEFAULT_GAP_S = 10.0


@dataclass(frozen=True)
class Block:
    game_id: str
    period: int
    split: str
    start_s: float  # inclusive, seconds since period start
    end_s: float  # exclusive (inclusive for the last block)


def assign_splits(
    samples: pd.DataFrame,
    fractions: tuple[tuple[str, float], ...] = DEFAULT_FRACTIONS,
    gap_s: float = DEFAULT_GAP_S,
) -> tuple[pd.Series, list[Block]]:
    """Return the split name per sample (None = discarded in a gap) and the blocks."""
    if not np.isclose(sum(f for _, f in fractions), 1.0):
        raise ValueError("Split fractions must sum to 1")
    split = pd.Series([None] * len(samples), index=samples.index, dtype=object)
    blocks: list[Block] = []
    cumulative = np.cumsum([0.0] + [f for _, f in fractions])

    for (game_id, period), group in samples.groupby(["game_id", "period"], sort=True):
        t = group["timestamp_s"].to_numpy()
        t0, t1 = t.min(), t.max()
        edges = t0 + cumulative * (t1 - t0)
        for i, (name, _) in enumerate(fractions):
            last = i == len(fractions) - 1
            start, end = edges[i], edges[i + 1]
            blocks.append(Block(str(game_id), int(period), name, float(start), float(end)))
            in_block = (t >= start) & ((t <= end) if last else (t < end))
            # keep gap_s / 2 away from internal boundaries
            if i > 0:
                in_block &= t >= start + gap_s / 2
            if not last:
                in_block &= t < end - gap_s / 2
            split.loc[group.index[in_block]] = name
    return split, blocks


def blocks_as_dicts(blocks: list[Block]) -> list[dict]:
    return [asdict(b) for b in blocks]
