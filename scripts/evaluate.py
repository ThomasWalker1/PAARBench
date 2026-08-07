#!/usr/bin/env python3
"""Evaluate a submission end to end: select, freeze, test, record.

    scripts/evaluate.py adajepa --setting pushobj --gpus 0,1,2,3
    scripts/evaluate.py --frozen --setting pushobj --gpus 0,1,2,3
    scripts/evaluate.py --frozen --isolated --setting pushobj --gpus 0,1,2,3

The protocol is:

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench import methods, settings as settings_mod
from paarbench.runner import DEFAULT_OUT_ROOT, ColumnConflict, run_column
from paarbench.selection import FixedParams, SelectionHarness
from paarbench.tunable import StandardSelection

RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", nargs="?", help="method directory name under methods/")
    ap.add_argument("--frozen", action="store_true",
                    help="evaluate the built-in do-nothing baseline instead of a method")
    ap.add_argument("--isolated", action="store_true",
                    help="with --frozen: evaluate one process per episode "
                         "(Frozen Individual) instead of a batched cohort "
                         "(Frozen Batched). Required for mode-matched pairing "
                         "against episode-isolated methods.")
    ap.add_argument("--setting", default="pushobj")
    ap.add_argument("--tag", default=None,
                    help="name the test columns and the result record under this "
                         "instead of the method name. Use it to evaluate a variant "
                         "without overwriting a committed submission's evidence. "
                         "Selection columns keep the method name, so a variant "
                         "reuses the cached sweep rather than repeating it.")
    ap.add_argument("--not-a-submission", action="store_true",
                    help="record this run as an ablation. It is kept out of the "
                         "leaderboard's main table and listed separately, because a "
                         "submission is a method plus its *declared* rule and an "
                         "ablation by definition did not follow one.")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="concurrent processes per GPU; raise it for episode-isolated "
                         "methods, where each process plans a single episode")
    ap.add_argument("--no-parallel-cohorts", action="store_true",
                    help="run batched test cohorts one at a time even when spare GPUs "
                         "could run another cohort in parallel")
    ap.add_argument("--n-evals", type=int, default=None)
    ap.add_argument("--selection-budget", type=int, default=None,
                    help="refuse to let the selection rule run more than N columns")
    ap.add_argument("--skip-selection", action="store_true",
                    help="use method.yaml's params as-is; records selection cost as "
                         "unknown rather than 0, since the tuning happened elsewhere")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return ap.parse_args()


def inherit_selection(results_dir: Path, tag: str, source: str, method, setting):
    """Take the parameters this submission already froze on another setting.

    Reads the source setting's *result record* rather than re-deriving anything: that
    file is what the leaderboard reports, so inheriting from it means the shifted run
    is unambiguously the same submission and not a re-tuned cousin. Per-setting
    parameter overrides still apply on top -- weights differ per domain even when the
    selected hyperparameter does not.
    """
    record_path = results_dir / tag / f"{source}.json"
    if not record_path.is_file():
        raise SystemExit(
            f"{setting.id} inherits its selection from {source}, but there is no "
            f"record at {record_path}. Evaluate {tag} on {source} first: its frozen "
            f"parameters are the input to this run."
        )
    record = json.loads(record_path.read_text())
    if not record.get("complete", False):
        raise SystemExit(
            f"{record_path} is an incomplete run; refusing to inherit parameters from "
            f"it. A shift condition inheriting from a half-finished selection would "
            f"report a submission nobody ever made."
        )
    inherited = dict(record.get("params") or {})
    # Per-setting overrides are about *where* the method runs, not about tuning, so
    # they layer on top of the inherited choice rather than being overwritten by it.
    overrides = {k: v for k, v in method.params_for(setting.id).items()
                 if k not in method.params_for(source) or
                 method.params_for(setting.id)[k] != method.params_for(source).get(k)}
    inherited.update(overrides)
    return (inherited,
            record.get("selection_cost_columns"),
            record.get("selection_rule", "unknown rule"))


def _batched_cohort_parallelism(n_shapes: int, n_gpus: int) -> int:
    """How many batched test cohorts can run at once without sharing a GPU."""
    if n_shapes <= 0 or n_gpus <= 0:
        return 1
    return max(1, n_gpus // n_shapes)


def _run_test_cohorts(
    setting,
    test_cohorts,
    *,
    tag,
    method_name,
    params,
    isolation,
    gpus,
    parallel_cohorts,
    **common,
):
    """Run each test cohort once with frozen parameters."""
    results = {}
    n_shapes = len(setting.shapes)
    parallel = (
        not isolation
        and parallel_cohorts
        and _batched_cohort_parallelism(n_shapes, len(gpus)) > 1
        and len(test_cohorts) > 1
    )

    def run_one(cohort: str, gpu_subset: list[str]):
        try:
            return cohort, run_column(
                setting, cohort, tag=tag, method_name=method_name, params=params,
                episode_isolation=isolation, gpus=gpu_subset, **common
            )
        except ColumnConflict as exc:
            raise SystemExit(f"[refused] {exc}") from exc

    if not parallel:
        for cohort in test_cohorts:
            cohort, result = run_one(cohort, gpus)
            results[cohort] = result.to_dict()
            score = "incomplete" if result.success is None else f"{result.success:.3f}"
            print(f"[test] {tag} {setting.id}/{cohort}: {score}", flush=True)
        return results

    width = _batched_cohort_parallelism(n_shapes, len(gpus))
    print(
        f"[parallel] batched test cohorts: up to {width} at a time "
        f"({n_shapes} GPUs per cohort across {len(gpus)} slots)",
        flush=True,
    )
    for wave_start in range(0, len(test_cohorts), width):
        wave = test_cohorts[wave_start:wave_start + width]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {}
            for index, cohort in enumerate(wave):
                start = index * n_shapes
                gpu_subset = gpus[start:start + n_shapes]
                futures[pool.submit(run_one, cohort, gpu_subset)] = cohort
            for future in as_completed(futures):
                cohort, result = future.result()
                results[cohort] = result.to_dict()
                score = "incomplete" if result.success is None else f"{result.success:.3f}"
                print(f"[test] {tag} {setting.id}/{cohort}: {score}", flush=True)
    return results


def main() -> int:
    args = parse_args()
    if bool(args.method) == bool(args.frozen):
        raise SystemExit("give exactly one of: a method name, or --frozen")
    if args.isolated and not args.frozen:
        raise SystemExit("--isolated only applies with --frozen "
                         "(methods declare requires_episode_isolation instead)")

    setting = settings_mod.get(args.setting)
    # Up front, before the selection rule spends anything: a missing goal file is a
    # staging problem, and it should read as one rather than as a traceback out of the
    # first column the rule happens to launch.
    try:
        setting.require_targets()
    except settings_mod.MissingTargets as exc:
        raise SystemExit(f"[missing] {exc}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    common = dict(
        gpus=gpus, n_evals=args.n_evals, out_root=args.out_root,
        per_gpu=args.per_gpu,
    )

    # ---- 1. selection -----------------------------------------------------------
    if args.frozen:
        # Two leaderboard arms: batched and episode-isolated. The modes do not agree
        # even for NullAdapter (measured 4% outcome flips on pushobj), so isolated
        # methods must pair against Frozen (Individual) and batched methods against
        # Frozen (Batched). Tags match paarbench.metrics.ISOLATED_REFERENCE.
        isolation = bool(args.isolated)
        if isolation:
            name = "frozen_isolated"
            display = "Frozen (Individual)"
        else:
            name = "frozen"
            display = "Frozen (Batched)"
        params, selection_cost, selection_rule = {}, 0, "none (no hyperparameters)"
        # Frozen's zero is the only unqualified one on the board: it has nothing to
        # tune. Every other zero means something weaker -- see COST_BASIS below.
        cost_basis = "none"
        if isolation:
            print(f"[isolate] {name}: one process per episode "
                  f"({setting.n_evals} per shape). Mode-matched reference for "
                  f"requires_episode_isolation methods.", flush=True)
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
        if method.tunable is not None:
            rule = StandardSelection(method.tunable, setting.id)

        if setting.inherits_selection_from and not isinstance(rule, FixedParams):
            source = setting.inherits_selection_from
            params, selection_cost, source_rule = inherit_selection(
                Path(args.results_dir), args.tag or name, source, method, setting
            )
            cost_basis = "inherited"
            selection_rule = (f"inherited from {source} "
                              f"({source_rule}, {selection_cost} column(s) there)")
            print(f"[select] {name}: {setting.id} has no selectable cohort; using the "
                  f"parameters frozen on {source} -- {params}", flush=True)
        elif args.skip_selection or isinstance(rule, FixedParams):
            params = method.params_for(setting.id)
            selection_cost = 0 if isinstance(rule, FixedParams) else None
            cost_basis = "authored" if isinstance(rule, FixedParams) else "skipped"
            selection_rule = ("none (params authored, not selected)"
                              if isinstance(rule, FixedParams)
                              else f"{method.selection_ref} (SKIPPED via --skip-selection)")
            print(f"[select] {name}: using method.yaml params {params}", flush=True)
        else:
            # The selection sweep is tagged by the *method*, not by --tag: it is a
            # property of the method and its rule, and two variants that differ only
            # in which objective is read off the same columns should share them
            # rather than pay for them twice.
            harness = SelectionHarness(
                setting, name, tag=name, budget=args.selection_budget,
                episode_isolation=isolation, **common
            )
            rule_label = (StandardSelection.RULE_NAME if method.tunable is not None
                          else method.selection_ref)
            print(f"[select] {name}: running {rule_label} on the "
                  f"selection cohort (seed {setting.selection_seed})", flush=True)
            chosen = rule.select(harness)
            if not isinstance(chosen, dict):
                raise SystemExit(
                    f"{rule_label}.select returned {type(chosen).__name__}, "
                    f"expected a dict of parameters to freeze"
                )
            params = {**method.params_for(setting.id), **chosen}
            selection_cost = harness.columns_used
            cost_basis = "selected"
            selection_rule = rule_label
            print(f"[select] {name}: froze {params} after {selection_cost} column(s)",
                  flush=True)

    # ---- 2/3. frozen parameters, run once per test cohort -----------------------
    method_name = None if args.frozen else name
    tag = args.tag or name
    test_cohorts = [f"test{seed}" for seed in setting.test_seeds]
    results = _run_test_cohorts(
        setting,
        test_cohorts,
        tag=tag,
        method_name=method_name,
        params=params,
        isolation=isolation,
        gpus=gpus,
        parallel_cohorts=not args.no_parallel_cohorts,
        n_evals=common["n_evals"] if "n_evals" in common else None,
        out_root=common["out_root"],
        per_gpu=common["per_gpu"],
    )

    complete = [r for r in results.values() if r["complete"]]
    pooled = (sum(r["success"] * r["n"] for r in complete) / sum(r["n"] for r in complete)
              if complete else None)

    record = {
        # ``method`` is the *tag*: it is the key the per-episode records on disk are
        # filed under (paarbench/schema.py derives identity from the directory
        # layout), so a variant evaluated under --tag must carry that tag here or the
        # leaderboard would attach one run's success rate to another's episodes.
        "method": tag,
        "method_dir": name,
        "display_name": display,
        "setting": setting.id,
        "params": params,
        "submission": not args.not_a_submission,
        "selection_rule": selection_rule,
        "selection_cost_columns": selection_cost,
        # What the cost number means. ``0`` is not one thing: frozen has nothing to
        # tune ("none"), while a FixedParams method's hyperparameters were written
        # down by its author and never selected ("authored"). Those are different
        # claims about tuning burden and must not render identically.
        "selection_cost_basis": cost_basis,
        "episode_isolated": isolation,
        "selection_cohort_seed": setting.selection_seed,
        "test_cohorts": results,
        "success": pooled,
        "n": sum(r["n"] for r in complete),
        "complete": len(complete) == len(test_cohorts),
        "frozen_reference": setting.frozen_success.get("test"),
    }

    out = Path(args.results_dir) / tag
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
