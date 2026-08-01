#!/usr/bin/env python3
"""Evaluate a submission end to end: select, freeze, test, record.

    scripts/evaluate.py adajepa --setting pushobj --gpus 0,1,2,3
    scripts/evaluate.py --frozen --setting pushobj --gpus 0,1,2,3

The protocol this implements (docs/PLAN.md §3):

  1. Run the method's selection rule. It may only evaluate the **selection cohort**,
     and every column it runs is counted as its selection cost.
  2. Freeze whatever the rule returned.
  3. Run those frozen parameters once on each **test cohort**. That is the reported
     score.

The separation is structural, not procedural: the rule is handed a ``SelectionHarness``
that cannot address a test cohort at all. Step 3 does not consult the rule again.

Writes ``results/<method>/<setting>.json`` -- the record that populates the leaderboard
and that a pull request should include.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench import methods, settings as settings_mod
from paarbench.runner import DEFAULT_OUT_ROOT, run_column
from paarbench.selection import FixedParams, SelectionHarness
from paarbench.staging import DEFAULT_TMPFS, resolve_dataset_path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", nargs="?", help="method directory name under methods/")
    ap.add_argument("--frozen", action="store_true",
                    help="evaluate the built-in do-nothing baseline instead of a method")
    ap.add_argument("--setting", default="pushobj")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="concurrent processes per GPU; raise it for episode-isolated "
                         "methods, where each process plans a single episode")
    ap.add_argument("--n-evals", type=int, default=None)
    ap.add_argument("--selection-budget", type=int, default=None,
                    help="refuse to let the selection rule run more than N columns")
    ap.add_argument("--skip-selection", action="store_true",
                    help="use method.yaml's params as-is; records selection cost as "
                         "unknown rather than 0, since the tuning happened elsewhere")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tmpfs-root", type=Path, default=None,
                    help=f"where to stage the dataset (default {DEFAULT_TMPFS})")
    ap.add_argument("--data-path", type=Path, default=None,
                    help="use this dataset directory verbatim; skips staging")
    ap.add_argument("--no-stage", action="store_true")
    return ap.parse_args()


def resolve_data_path(setting, args):
    if args.data_path is not None:
        if not args.data_path.is_dir():
            raise SystemExit(f"--data-path is not a directory: {args.data_path}")
        print(f"[stage] using {args.data_path} verbatim", flush=True)
        return args.data_path
    if args.no_stage:
        return None
    return resolve_dataset_path(setting, args.tmpfs_root)


def main() -> int:
    args = parse_args()
    if bool(args.method) == bool(args.frozen):
        raise SystemExit("give exactly one of: a method name, or --frozen")

    setting = settings_mod.get(args.setting)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    data_path = resolve_data_path(setting, args)

    common = dict(
        gpus=gpus, n_evals=args.n_evals, out_root=args.out_root, data_path=data_path,
        per_gpu=args.per_gpu,
    )

    # ---- 1. selection -----------------------------------------------------------
    if args.frozen:
        name, display = "frozen", "Frozen base model (no adaptation)"
        params, selection_cost, selection_rule = {}, 0, "none (no hyperparameters)"
        isolation = False
    else:
        method = methods.load(args.method)
        problems = methods.validate(method)
        if problems:
            print(f"method {method.name!r} is not valid:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        if not method.supports(setting.id):
            raise SystemExit(
                f"method {method.name!r} declares settings {method.settings}, "
                f"not {setting.id!r}. Add it to method.yaml if it is supported."
            )

        name, display = method.name, method.display_name
        isolation = method.requires_episode_isolation
        if isolation:
            print(f'[isolate] {name} declares requires_episode_isolation: one process '
                  f'per episode ({setting.n_evals} per shape). A batched cohort would '
                  f'average unrelated episodes into one shared update.', flush=True)
        rule = method.load_selection_rule() or FixedParams(method.params_for(setting.id))
        if isinstance(rule, type):
            rule = rule()

        if args.skip_selection or isinstance(rule, FixedParams):
            params = method.params_for(setting.id)
            selection_cost = 0 if isinstance(rule, FixedParams) else None
            selection_rule = ("none (params used as-is)" if isinstance(rule, FixedParams)
                              else f"{method.selection_ref} (SKIPPED via --skip-selection)")
            print(f"[select] {name}: using method.yaml params {params}", flush=True)
        else:
            harness = SelectionHarness(
                setting, name, tag=name, budget=args.selection_budget,
                episode_isolation=isolation, **common
            )
            print(f"[select] {name}: running {method.selection_ref} on the "
                  f"selection cohort (seed {setting.selection_seed})", flush=True)
            chosen = rule.select(harness)
            if not isinstance(chosen, dict):
                raise SystemExit(
                    f"{method.selection_ref}.select returned {type(chosen).__name__}, "
                    f"expected a dict of parameters to freeze"
                )
            params = {**method.params_for(setting.id), **chosen}
            selection_cost = harness.columns_used
            selection_rule = method.selection_ref
            print(f"[select] {name}: froze {params} after {selection_cost} column(s)",
                  flush=True)

    # ---- 2/3. frozen parameters, run once per test cohort -----------------------
    method_name = None if args.frozen else name
    test_cohorts = [f"test{seed}" for seed in setting.test_seeds]
    results = {}
    for cohort in test_cohorts:
        result = run_column(
            setting, cohort, tag=name, method_name=method_name, params=params,
            episode_isolation=isolation, **common
        )
        results[cohort] = result.to_dict()
        score = "incomplete" if result.success is None else f"{result.success:.3f}"
        print(f"[test] {name} {setting.id}/{cohort}: {score}", flush=True)

    complete = [r for r in results.values() if r["complete"]]
    pooled = (sum(r["success"] * r["n"] for r in complete) / sum(r["n"] for r in complete)
              if complete else None)

    record = {
        "method": name,
        "display_name": display,
        "setting": setting.id,
        "params": params,
        "selection_rule": selection_rule,
        "selection_cost_columns": selection_cost,
        "episode_isolated": isolation,
        "selection_cohort_seed": setting.selection_seed,
        "test_cohorts": results,
        "success": pooled,
        "n": sum(r["n"] for r in complete),
        "complete": len(complete) == len(test_cohorts),
        "frozen_reference": setting.frozen_success.get("test"),
    }

    out = Path(args.results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{setting.id}.json").write_text(json.dumps(record, indent=2))

    print(f"\n=== {display} on {setting.id} ===")
    for cohort, r in results.items():
        s = "incomplete" if r["success"] is None else f"{r['success']:.3f}"
        print(f"  {cohort:>10}: {s}  (n={r['n']})")
    print(f"  {'pooled':>10}: {'incomplete' if pooled is None else f'{pooled:.4f}'}  "
          f"(n={record['n']})")
    print(f"  selection cost: {selection_cost} column(s)")
    if record["frozen_reference"] is not None:
        print(f"  frozen reference on test cohorts: {record['frozen_reference']:.3f}")
    print(f"  record: {out / f'{setting.id}.json'}")

    return 0 if record["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
