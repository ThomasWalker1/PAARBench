#!/usr/bin/env python3
"""Materialize versionable PointMaze/DiverseMaze evaluation episodes.

The generated files are benchmark artifacts: hash and publish them before recording
results.  They deliberately replace the upstream runtime samplers.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench.maze_targets import diverse_targets, medium_hard_targets, write_targets
from paarbench.settings import get


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setting",
        required=True,
        choices=(
            "maze_medium",
            "maze_medium_low_density",
            "maze_medium_high_damping",
            "maze_diverse",
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--n-evals", type=int, default=None)
    parser.add_argument(
        "--maze-specs",
        type=Path,
        default=None,
        help="maze_specs.pth for a diverse corpus; defaults to --data-root/diverse_maze/val",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    setting = get(args.setting)
    seeds = args.seed or [setting.selection_seed, *setting.test_seeds]
    count = args.n_evals or setting.n_evals
    root = args.data_root.resolve()
    for seed in seeds:
        destination = root / setting.targets_path(setting.shapes[0], seed).relative_to("data")
        if setting.id.startswith("maze_medium"):
            targets = medium_hard_targets(seed, count)
        else:
            specs = args.maze_specs or root / "diverse_maze" / "val" / "maze_specs.pth"
            targets = diverse_targets(specs, seed=seed, n_episodes=count)
        write_targets(destination, targets)
        print(f"{sha256(destination)}  {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
