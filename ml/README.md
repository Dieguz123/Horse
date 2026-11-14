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
| `uv run python -m horse_ml.preprocessing.build` | Builds the training samples into `data/processed/` (~75 s per game) and prints a summary |
| `uv run python -m horse_ml.viz.sample` | Plots one processed sample with velocity arrows to `outputs/sample.png` |
| `uv run pytest` | Runs the tests (real-data tests are skipped if the data is not downloaded) |

`frame` and `animate` accept `--period` and `--time` / `--start` to pick another moment;
`viz.sample` accepts `--sample-id`; `preprocessing.build` accepts `--games`.

## Layout

```
src/horse_ml/
├── paths.py            # data/ and outputs/ locations (override with HORSE_DATA_DIR)
├── data/
│   ├── download.py     # fetches raw files from GitHub
│   ├── loading.py      # load_tracking() / load_events() via kloppy
│   ├── games.py        # registry of games the pipeline processes
│   └── selection.py    # find events, slice frames by time
├── explore.py          # dataset summary
├── preprocessing/
│   ├── coordinates.py  # Metrica [0,1] -> metres, half-time and possession flips
│   ├── frames.py       # one wide table per game: positions in metres
│   ├── kinematics.py   # Savitzky-Golay velocities, impossible speeds -> NaN
│   ├── possession.py   # team in possession + ball in play from events
│   ├── roles.py        # fixed slot order per team (roles, fallback: average position)
│   ├── samples.py      # defensive samples at 5 Hz
│   ├── split.py        # train/val/test by contiguous time blocks
│   └── build.py        # runs everything, writes Parquet + metadata.json
└── viz/
    ├── pitch.py        # pitch, colors, draw one frame
    ├── frame.py        # single-frame PNG
    ├── animate.py      # GIF animation
    └── sample.py       # one processed sample with velocity arrows
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
- **Roles**: the metadata gives every player a nominal role and template position.
  Team A plays 4-2-3-1, Team B 4-4-2. Substitutes have roles too, but one (Team A
  no. 13) duplicates a teammate's role, so it is placed by average position.

## Processed dataset (phase 2)

`data/processed/samples.parquet`, one row per sample, plus `metadata.json`
(frame rate, coordinate system, split blocks, slot roles per team, column meaning).

- **Coordinates**: metres, origin at the centre spot, y up, **team in possession
  attacking towards +x**. Pitch is [-52.5, 52.5] × [-34, 34].
- **Input**: `ball_{x,y,vx,vy}` and `att_{k}_{x,y,vx,vy}` for the 11 attackers.
- **Target**: `def_{k}_{x,y}` for the 11 defenders (`def_{k}_{vx,vy}` kept for analysis).
- **Slots** `k = 0..10`: fixed role order per team (GK, back line right to left,
  midfield, forwards). Slot k is a different role for each formation: see the
  `{att,def}_{k}_role` and `*_formation` columns.
- **Context**: `game_id, period, timestamp_s, split, attacking_team, defending_team,
  seconds_since_possession_change` (transitions are kept, not filtered).
- **Sampling**: 5 Hz, only frames with the ball in play, known possession and all
  22 players + ball with valid position and velocity.
- **Velocities**: Savitzky-Golay, 7-frame window (0.28 s), polynomial order 2.
  Player speed p99 is 6.3 m/s and p99.9 is 8.0 m/s, nearly identical with 5- or
  11-frame windows; values above 12 m/s (players) or 40 m/s (ball) become NaN.
- **Split**: 70/15/15 contiguous time blocks per period, 10 s gap discarded at each
  boundary. Game 3: 13,762 train, 2,783 val, 2,660 test samples.
