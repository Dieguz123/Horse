# horse-ml

Python side of Horse: data loading, exploration, visualization and (later) the
ghosting models.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and every
dependency into `ml/.venv`:

```bash
cd ml
uv sync
```

## Commands

All commands run from `ml/`.

| Command | What it does |
|---|---|
| `uv run python -m horse_ml.data.download` | Downloads Metrica Sample Game 3 into `data/raw/metrica/` (~55 MB, skips existing files) |
| `uv run python -m horse_ml.explore` | Prints a summary: frame rate, coordinates, frames, players, ball, events |
| `uv run python -m horse_ml.viz.frame` | Plots one frame (default: the first goal) to `outputs/frame.png` |
| `uv run python -m horse_ml.viz.animate` | Renders 10 s of play (default: build-up to the first goal) to `outputs/animation.gif` |
| `uv run pytest` | Runs the tests (real-data tests are skipped if the data is not downloaded) |

`frame` and `animate` accept `--period` and `--time` / `--start` to pick another moment.

## Layout

```
src/horse_ml/
├── paths.py            # data/ and outputs/ locations (override with HORSE_DATA_DIR)
├── data/
│   ├── download.py     # fetches raw files from GitHub
│   ├── loading.py      # load_tracking() / load_events() via kloppy
│   └── selection.py    # find events, slice frames by time
├── explore.py          # dataset summary
└── viz/
    ├── pitch.py        # pitch, colors, draw one frame
    ├── frame.py        # single-frame PNG
    └── animate.py      # GIF animation
```

## Data notes (Metrica Sample Game 3)

- **Frame rate**: 25 Hz, 143,761 frames (~96 minutes over 2 periods).
- **Coordinates**: Metrica's own system, x and y normalized to `[0, 1]`, origin
  top-left, y pointing down. Pitch is 105 × 68 m. Positions slightly outside
  `[0, 1]` happen when the ball or a player crosses the lines.
- **Players**: 17 (home) and 18 (away) in the squads, always 11 vs 11 on the pitch.
- **Ball**: x/y known in ~68% of frames (missing when out of play or not
  tracked); no height (z) and no ball state in the tracking data.
- **Events**: 3,594 events (passes, carries, recoveries, shots...). Timestamps are
  relative to the start of each period, the same as tracking frames, so they align.
- Games 1 and 2 in the same repo use a CSV event format that kloppy does not read,
  which is why we use Game 3.
