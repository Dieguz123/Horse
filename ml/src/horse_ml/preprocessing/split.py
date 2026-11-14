"""Train / validation / test split by alternating contiguous time blocks.

Consecutive frames are nearly identical, so a random split of samples would put
almost the same situation in train and test (leakage) and give over-optimistic
metrics. Instead every period of every game is cut into blocks of ~3 minutes and
whole blocks are assigned to train / val / test (~70/15/15 of the samples).
Samples within ``gap_s / 2`` of a boundary between blocks of *different* splits
are discarded, so no evaluation sample has a near-twin in train.

Block assignment: many random assignments (fixed seed) are scored and the most
balanced one is kept. The score only uses sample counts and the defending team
of each sample, never positions, so no target information leaks into the choice.
Spreading val/test blocks over the whole game (instead of the last minutes of each
half) also keeps them representative of every phase of the match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_FRACTIONS: tuple[tuple[str, float], ...] = (("train", 0.70), ("val", 0.15), ("test", 0.15))
DEFAULT_BLOCK_S = 180.0
DEFAULT_GAP_S = 10.0
DEFAULT_TRIALS = 5000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Block:
    game_id: str
    period: int
    split: str
    start_s: float  # inclusive, seconds since period start
    end_s: float  # exclusive


def make_blocks(samples: pd.DataFrame, block_s: float) -> pd.Series:
    """Block number per sample: consecutive ``block_s`` windows per game and period.

    A trailing window shorter than half a block is merged into the previous one.
    """
    block = pd.Series(-1, index=samples.index, dtype=int)
    next_id = 0
    for _, group in samples.groupby(["game_id", "period"], sort=True):
        t = group["timestamp_s"].to_numpy()
        t0, t1 = t.min(), t.max()
        n_blocks = max(1, int(round((t1 - t0) / block_s)))
        local = np.minimum(((t - t0) // block_s).astype(int), n_blocks - 1)
        block.loc[group.index] = local + next_id
        next_id += n_blocks
    return block


def _score(
    labels: np.ndarray,
    counts: np.ndarray,
    group_counts: np.ndarray,
    targets: np.ndarray,
    overall_share: np.ndarray,
) -> float:
    """Lower is better: sample fractions close to targets, group mix close to overall."""
    score = 0.0
    total = counts.sum()
    for s, target in enumerate(targets):
        in_split = labels == s
        n = counts[in_split].sum()
        if n == 0:
            return np.inf
        score += abs(n / total - target)
        share = group_counts[in_split].sum(axis=0) / n
        score += np.abs(share - overall_share).sum() * target
    return score


def assign_splits(
    samples: pd.DataFrame,
    fractions: tuple[tuple[str, float], ...] = DEFAULT_FRACTIONS,
    *,
    block_s: float = DEFAULT_BLOCK_S,
    gap_s: float = DEFAULT_GAP_S,
    balance_col: str = "defending_team",
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.Series, list[Block]]:
    """Return the split name per sample (None = discarded in a gap) and the blocks."""
    names = [n for n, _ in fractions]
    targets = np.array([f for _, f in fractions])
    if not np.isclose(targets.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1")

    block = make_blocks(samples, block_s)
    n_blocks = int(block.max()) + 1
    groups = sorted(samples[balance_col].unique())
    group_idx = samples[balance_col].map({g: i for i, g in enumerate(groups)}).to_numpy()
    counts = np.bincount(block, minlength=n_blocks)
    group_counts = np.zeros((n_blocks, len(groups)))
    np.add.at(group_counts, (block.to_numpy(), group_idx), 1)
    overall_share = group_counts.sum(axis=0) / counts.sum()

    # number of blocks per split (train gets the rest)
    sizes = [max(1, int(round(f * n_blocks))) for f in targets[1:]]
    if sum(sizes) >= n_blocks:
        raise ValueError(f"Not enough blocks ({n_blocks}) for the requested split")

    rng = np.random.default_rng(seed)
    best_labels, best_score = None, np.inf
    for _ in range(trials):
        perm = rng.permutation(n_blocks)
        labels = np.zeros(n_blocks, dtype=int)
        start = 0
        for s, size in enumerate(sizes, start=1):
            labels[perm[start : start + size]] = s
            start += size
        score = _score(labels, counts, group_counts, targets, overall_share)
        if score < best_score:
            best_labels, best_score = labels, score

    sample_label = best_labels[block.to_numpy()]
    keep = np.ones(len(samples), dtype=bool)
    blocks: list[Block] = []
    t = samples["timestamp_s"].to_numpy()
    for (game_id, period), group in samples.groupby(["game_id", "period"], sort=True):
        ids = np.unique(block.loc[group.index])
        for b in ids:
            in_b = (block == b).to_numpy()
            blocks.append(
                Block(str(game_id), int(period), names[best_labels[b]],
                      float(t[in_b].min()), float(t[in_b].max()))
            )
        # gaps only where consecutive blocks belong to different splits
        for left, right in zip(ids[:-1], ids[1:], strict=True):
            if best_labels[left] == best_labels[right]:
                continue
            boundary = t[(block == right).to_numpy()].min()
            rows = group.index[np.abs(group["timestamp_s"].to_numpy() - boundary) < gap_s / 2]
            keep[samples.index.get_indexer(rows)] = False

    split = pd.Series(np.array(names, dtype=object)[sample_label], index=samples.index, dtype=object)
    split[~keep] = None
    return split, blocks


def blocks_as_dicts(blocks: list[Block]) -> list[dict]:
    return [asdict(b) for b in blocks]
