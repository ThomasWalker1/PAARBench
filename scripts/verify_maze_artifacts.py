#!/usr/bin/env python3
"""Validate and lock a staged maze artifact release.

AdaJEPA currently distributes maze assets from Google Drive rather than an immutable
artifact registry.  This script makes that mutable delivery step explicit: stage the
files, write a SHA-256 manifest, commit/publish the manifest with the result records,
and use ``--manifest`` to verify every subsequent rerun.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MEDIUM_REQUIRED = (
    "checkpoints/mediummaze_dynamics_shift/hydra.yaml",
    "checkpoints/mediummaze_dynamics_shift/checkpoints/model_latest.pth",
    "data/point_maze_medium/states.pth",
    "data/point_maze_medium/actions.pth",
    "data/point_maze_medium/seq_lengths.pth",
)
REQUIRED = (
    *MEDIUM_REQUIRED,
    "data/diverse_maze/val/maze_specs.pth",
)

REQUIRED_BY_SETTING = {
    "maze_medium": MEDIUM_REQUIRED,
    "maze_medium_low_density": MEDIUM_REQUIRED,
    "maze_medium_high_damping": MEDIUM_REQUIRED,
    "maze_diverse": REQUIRED,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        checksum, relative = line.split(maxsplit=1)
        entries[relative.strip()] = checksum
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="verify this existing SHA-256 manifest instead of creating one",
    )
    parser.add_argument(
        "--write-manifest", type=Path, default=None,
        help="write hashes for all required files to this path",
    )
    parser.add_argument(
        "--setting", action="append", choices=tuple(REQUIRED_BY_SETTING),
        help="limit validation to one or more benchmark settings (default: full suite)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    required = REQUIRED
    if args.setting:
        required = tuple(
            dict.fromkeys(
                relative
                for setting in args.setting
                for relative in REQUIRED_BY_SETTING[setting]
            )
        )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise SystemExit("missing required maze artifacts:\n  " + "\n  ".join(missing))
    if bool(args.manifest) == bool(args.write_manifest):
        raise SystemExit("give exactly one of --manifest or --write-manifest")
    if args.write_manifest:
        output = args.write_manifest
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "# AdaJEPA source revision: a29975964f966f2836a2c7e26f464367c795c333\n"
            + "".join(f"{digest(root / relative)}  {relative}\n" for relative in required)
        )
        print(f"wrote {output}")
        return 0
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    expected = parse_manifest(manifest)
    mismatches = []
    for relative in required:
        actual = digest(root / relative)
        if expected.get(relative) != actual:
            mismatches.append(
                f"{relative}: expected {expected.get(relative, '(not listed)')}, actual {actual}"
            )
    if mismatches:
        raise SystemExit("maze artifact verification failed:\n  " + "\n  ".join(mismatches))
    print(f"verified {len(required)} maze artifacts against {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
