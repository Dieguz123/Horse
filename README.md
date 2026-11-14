# Horse

**Learning defensive positioning from real football matches.**

Horse is a football *ghosting* tool. Given the positions of the attacking team and
the ball, a model trained on real tracking data predicts where each defender
*should* be. Comparing these "ghost" defenders with the real ones shows where a
defence was out of shape, and by how much.

## Monorepo

| Folder | Stack | Status |
|---|---|---|
| [`ml/`](ml/) | Python 3.12, uv, kloppy, pandas, mplsoccer, PyTorch, FastAPI, pytest | Phase 2 in progress |
| [`backend/`](backend/) | Spring Boot (Java 25), PostgreSQL + Flyway, Kafka, Redis, Spring AI | Skeleton only |
| [`frontend/`](frontend/) | Angular | Not started |

## Roadmap

1. **Data & visualization**: download and load tracking + event data, explore it,
   plot frames and animations. *(done)*
2. **Preprocessing**: coordinates in metres, attacking direction normalized,
   smoothed velocities, possession from events, defensive samples with a fixed
   role order, time-block train/val/test split. *(current phase)*
3. **Ghosting models**: simple baselines (average shape, nearest neighbours), then
   neural models in PyTorch (sequence models / graph neural networks), with
   evaluation metrics.
4. **Model serving**: FastAPI inference service.
5. **Backend**: Spring Boot with matches stored in PostgreSQL (Flyway migrations),
   Kafka for frame streaming, Redis caching, Spring AI for natural-language
   explanations of defensive errors.
6. **Frontend**: Angular app with an interactive pitch showing real vs ghost defenders.

## Quick start (ml)

```bash
cd ml
uv sync                                      # install Python 3.12 + dependencies
uv run python -m horse_ml.data.download      # fetch the raw data into ml/data/
uv run python -m horse_ml.explore            # print a dataset summary
uv run python -m horse_ml.viz.frame          # -> ml/outputs/frame.png
uv run python -m horse_ml.viz.animate        # -> ml/outputs/animation.gif
uv run python -m horse_ml.preprocessing.build  # -> ml/data/processed/ (training samples)
uv run python -m horse_ml.viz.sample         # -> ml/outputs/sample.png
uv run pytest
```

See [`ml/README.md`](ml/README.md) for details.

## Data and license

Horse uses the **Metrica Sports sample data**:
<https://github.com/metrica-sports/sample-data>. It contains anonymized tracking and
event data, with no player, team or competition names.

The repository has **no explicit license** (no LICENSE file; GitHub reports
`license: null`). Its README, under "Legal stuff", states only:

> Please be responsible with the use of this data.
> If you use it for anything public, please acknowledge the source.

For this reason raw data is **never committed** here: it is downloaded by a script
into `ml/data/` (git-ignored), and the source is acknowledged above. Without a
formal license, terms for redistribution or commercial use are not granted
explicitly. Check with Metrica Sports before any such use.
