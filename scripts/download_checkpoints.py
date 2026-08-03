#!/usr/bin/env python3
"""Download and verify PAARBench checkpoint artifacts from Hugging Face Hub.

Examples:
    python scripts/download_checkpoints.py settings
    python scripts/download_checkpoints.py methods
    python scripts/download_checkpoints.py all
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "CHECKPOINTS.sha256"
HF_REPO_ID = "ThomasWalker1/paarbench-checkpoints"

# Updated to the immutable Hub commit after each published artifact release.
DEFAULT_REVISION = "329f4215b0902d0eaa765a17290491ebe281a0b7"

GROUP_PREFIXES = {
    "settings": ("pushobj_shape_shift/", "pusht_visual_shift/"),
    "methods": ("pad/", "pushobj_adapters/", "pvs_adapters/"),
}


def manifest_entries(group: str) -> dict[str, str]:
    """Return ``remote path -> sha256`` for the requested artifact group."""
    prefixes = tuple(prefix for values in GROUP_PREFIXES.values() for prefix in values)
    if group != "all":
        prefixes = GROUP_PREFIXES[group]

    entries = {}
    for line in MANIFEST.read_text().splitlines():
        digest, local_path = line.split(maxsplit=1)
        local_path = local_path.strip()
        checkpoint_prefix = "checkpoints/"
        if not local_path.startswith(checkpoint_prefix):
            raise ValueError(f"unexpected checkpoint manifest path: {local_path}")
        remote_path = local_path[len(checkpoint_prefix) :]
        if remote_path.startswith(prefixes):
            entries[remote_path] = digest
    return entries


def verify(entries: dict[str, str], checkpoint_dir: Path) -> None:
    """Fail if a downloaded file is missing or differs from the benchmark manifest."""
    problems = []
    for relative_path, expected in entries.items():
        path = checkpoint_dir / relative_path
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            problems.append(f"checksum mismatch: {path}\n  expected {expected}\n  actual   {actual}")
    if problems:
        raise RuntimeError("checkpoint verification failed:\n" + "\n".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group", choices=("settings", "methods", "all"),
        help="settings=base JEPA models; methods=adapter checkpoints; all=both",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="Hub tag or full commit hash (default: benchmark-pinned revision)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
        help="destination directory (default: <repo>/checkpoints)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = manifest_entries(args.group)
    patterns = sorted(entries)
    if not patterns:
        raise SystemExit(f"manifest contains no files for group {args.group!r}")

    print(
        f"Downloading {len(patterns)} {args.group} checkpoint files from "
        f"{HF_REPO_ID}@{args.revision} ...",
        flush=True,
    )
    snapshot_download(
        repo_id=HF_REPO_ID,
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=args.checkpoint_dir,
    )
    verify(entries, args.checkpoint_dir)
    print(f"Verified {len(entries)} files in {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
