#!/usr/bin/env python3
"""Check a contributed method without running it. No GPU, no data, no checkpoints.

This is what CI runs on a pull request, and what a contributor should run before
opening one. It catches the mistakes that would otherwise surface an hour into an
evaluation: a typo'd setting id, an adapter missing a hook, a selection rule that is
not callable, a method.yaml whose name disagrees with its directory.

    scripts/validate_method.py                 # every method
    scripts/validate_method.py my_method       # just one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench import methods


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="*", help="method directory name(s); default all")
    args = ap.parse_args()

    try:
        found = ([methods.load(n) for n in args.name] if args.name
                 else methods.discover())
    except methods.MethodError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    if not found:
        print("no methods found under methods/ (that is fine for a fresh checkout)")
        return 0

    failed = 0
    for method in found:
        problems = methods.validate(method)
        if problems:
            failed += 1
            print(f"FAIL  {method.name}")
            for p in problems:
                print(f"        - {p}")
        else:
            selection = method.selection_ref or "none (params used as-is, 0 columns)"
            print(f"ok    {method.name:20} settings={','.join(method.settings):20} "
                  f"selection={selection}")

    print(f"\n{len(found) - failed}/{len(found)} method(s) valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
