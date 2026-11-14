"""Download the Metrica Sports sample data into ml/data/raw/metrica/.

Source: https://github.com/metrica-sports/sample-data
The repository has no explicit license; its README asks users to
"be responsible with the use of this data" and to acknowledge the source
for any public use. Raw files are therefore never committed to this repo.

Usage (from ml/):
    uv run python -m horse_ml.data.download
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from horse_ml.paths import METRICA_DIR

REPO_RAW_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master"

# Sample_Game_3 is the only game published in the EPTS/JSON format, which is the
# one kloppy can read for BOTH tracking and event data.
GAME_ID = 3
FILES: dict[str, str] = {
    "metadata": "data/Sample_Game_3/Sample_Game_3_metadata.xml",
    "tracking": "data/Sample_Game_3/Sample_Game_3_tracking.txt",
    "events": "data/Sample_Game_3/Sample_Game_3_events.json",
}


def file_url(repo_path: str) -> str:
    """Return the raw GitHub URL for a file inside the sample-data repo."""
    return f"{REPO_RAW_URL}/{repo_path}"


def local_path(repo_path: str, target_dir: Path = METRICA_DIR) -> Path:
    """Return where a repo file is stored locally (flat, file name only)."""
    return target_dir / Path(repo_path).name


def download_file(url: str, destination: Path, *, force: bool = False) -> bool:
    """Download ``url`` to ``destination``.

    Returns True if the file was downloaded, False if it already existed.
    The file is written to a temporary ``.part`` file first, so an interrupted
    download never leaves a truncated file behind.
    """
    if destination.exists() and not force:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)
    return True


def download_metrica(target_dir: Path = METRICA_DIR, *, force: bool = False) -> dict[str, Path]:
    """Download all files of the configured Metrica game. Returns local paths."""
    paths: dict[str, Path] = {}
    for kind, repo_path in FILES.items():
        destination = local_path(repo_path, target_dir)
        downloaded = download_file(file_url(repo_path), destination, force=force)
        status = "downloaded" if downloaded else "already present"
        size_mb = destination.stat().st_size / 1_000_000
        print(f"[{kind:8}] {status:15} {destination} ({size_mb:.1f} MB)")
        paths[kind] = destination
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Metrica Sports sample data.")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    download_metrica(force=args.force)


if __name__ == "__main__":
    main()
