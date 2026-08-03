#!/usr/bin/env python3
"""Download and verify the optional PAARBench training dataset.

Benchmark evaluation does not require this dataset. Download it to reproduce
training or develop methods that explicitly use offline trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "DATASET_MANIFEST.json"
HF_REPO_ID = "ThomasWalker1/paarbench-data"

# Updated to the immutable Hub commit after each published dataset release.
DEFAULT_REVISION = "503481e4c0c025b3504b0503d976673d844758ce"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def split_inventory(split_dir: Path, relative_root: Path) -> dict:
    """Return the count, byte size, and deterministic digest of one split tree."""
    files = sorted(path for path in split_dir.rglob("*") if path.is_file())
    tree = hashlib.sha256()
    total_bytes = 0
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        relative = path.relative_to(relative_root).as_posix()
        tree.update(f"{digest.hexdigest()}  {relative}\n".encode())
        total_bytes += path.stat().st_size
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
    }


def verify(data_dir: Path, manifest: Optional[dict] = None) -> None:
    """Fail when either downloaded split differs from the release manifest."""
    manifest = manifest or load_manifest()
    dataset_root = data_dir / manifest["root"]
    problems = []
    for split, expected in manifest["splits"].items():
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            problems.append(f"missing split: {split_dir}")
            continue
        actual = split_inventory(split_dir, dataset_root)
        for field in ("file_count", "total_bytes", "tree_sha256"):
            if actual[field] != expected[field]:
                problems.append(
                    f"{split} {field} mismatch: expected {expected[field]}, "
                    f"got {actual[field]}"
                )
    if problems:
        raise RuntimeError("dataset verification failed:\n" + "\n".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="Hub tag or full commit hash (default: benchmark-pinned revision)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="destination directory (default: <repo>/data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    print(
        f"Downloading optional training data from "
        f"{HF_REPO_ID}@{args.revision} ...",
        flush=True,
    )
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[f"{manifest['root']}/**"],
        local_dir=args.data_dir,
    )
    verify(args.data_dir, manifest)
    print(f"Verified training data in {args.data_dir / manifest['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
