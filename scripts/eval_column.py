#!/usr/bin/env python3
"""Run a single evaluation column. A debugging tool, not the submission path.

    scripts/eval_column.py --setting pushobj --cohort selection --tag frozen
    scripts/eval_column.py --setting pushobj --cohort test100 --method adajepa \
        --param steps=10 --param lr=5e-4

Use this to check one configuration in isolation. To evaluate a submission properly --
selection rule, frozen parameters, all test cohorts, a result record -- use
``scripts/evaluate.py``, which enforces the cohort separation this script does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench import settings as settings_mod
from paarbench.runner import DEFAULT_OUT_ROOT, run_column
from paarbench.staging import DEFAULT_TMPFS, resolve_dataset_path


def parse_param(text: str):
    """``key=value``, with value parsed as YAML so numbers and bools survive."""
    import yaml

    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--param expects key=value, got {text!r}")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="pushobj")
    ap.add_argument("--cohort", default="selection",
                    help="'selection' or 'test<seed>' (or 'test' when unambiguous)")
    ap.add_argument("--tag", default=None, help="output directory name; defaults to the method")
    ap.add_argument("--method", default=None, help="method directory name; omit for frozen")
    ap.add_argument("--param", action="append", type=parse_param, default=[],
                    metavar="KEY=VALUE", help="adapter param override; repeatable")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--n-evals", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--tmpfs-root", type=Path, default=None,
                    help=f"where to stage the dataset (default {DEFAULT_TMPFS})")
    ap.add_argument("--data-path", type=Path, default=None,
                    help="use this dataset directory verbatim; skips staging")
    ap.add_argument("--no-stage", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run shapes that already have logs.json")
    args = ap.parse_args()

    setting = settings_mod.get(args.setting)
    tag = args.tag or args.method or "frozen"

    data_path = None
    if args.data_path is not None:
        if not args.data_path.is_dir():
            raise SystemExit(f"--data-path is not a directory: {args.data_path}")
        data_path = args.data_path
        print(f"[stage] using {data_path} verbatim", flush=True)
    elif not args.no_stage:
        data_path = resolve_dataset_path(setting, args.tmpfs_root)

    result = run_column(
        setting,
        args.cohort,
        tag=tag,
        method_name=args.method,
        params=dict(args.param),
        gpus=[g.strip() for g in args.gpus.split(",") if g.strip()],
        n_evals=args.n_evals,
        out_root=args.out_root,
        data_path=data_path,
        resume=not args.no_resume,
    )

    print(f"\n[column] {setting.id}/{args.cohort} tag={tag} "
          f"({result.wall_seconds / 60:.1f} min)")
    for shape, value in result.success_by_shape.items():
        print(f"  {shape:>10}: {'--' if value is None else f'{value:.3f}'}")
    if result.success is not None:
        print(f"  {'MEAN':>10}: {result.success:.4f}   (n={result.n})")
        reference = setting.frozen_success.get(
            "selection" if args.cohort == "selection" else "test")
        if reference is not None:
            print(f"  frozen reference on this cohort: {reference:.3f}")
    else:
        print("  MEAN: incomplete column")

    if result.failures:
        print(f"[fail] {result.failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
