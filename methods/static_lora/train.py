#!/usr/bin/env python3
"""Train Static LoRA adapter checkpoints for PAARBench.

Thin wrapper around ``methods/hyperjepa/train.py`` with ``--static-lora``.
HyperJEPA and Static LoRA share the same offline trainer; only the adaptation
parameterization differs.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "methods" / "hyperjepa" / "train.py"


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--static-lora" not in args:
        args = ["--static-lora", *args]
    sys.argv = [str(TRAINER), *args]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
