#!/usr/bin/env python3
"""Paired per-episode comparison of two methods on the PushObj in-distribution matrix.

Every method in the matrix is scored on the *same* segments with the same
environment seeds (plan.py's cohort selection is a function of `seed`), so the
episode-level outcomes are paired and a paired test is far more powerful than
comparing seed means -- most of the variance is segment difficulty, which cancels.

Reports, pooled over shapes and seeds: both success rates, the paired delta, the
discordant pair counts, an exact two-sided McNemar p-value, and a bootstrap CI
resampling episodes.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import numpy as np

SHAPES = ["T", "L", "Z", "plus"]
SUCCESS_RE = re.compile(r"\{'success': array\(\[(.*?)\]\)", re.S)


def from_batched_log(log_path: Path) -> np.ndarray | None:
    """Per-episode success from the final evaluation printed by a batched run."""
    if not log_path.exists():
        return None
    matches = SUCCESS_RE.findall(log_path.read_text())
    if not matches:
        return None
    body = matches[-1].replace("\n", "").replace(" ", "")
    return np.array([token == "True" for token in body.split(",") if token], dtype=bool)


def from_trials(trials_dir: Path, n: int) -> np.ndarray | None:
    """Per-episode success from episode-isolated trials (trial<idx>/logs.json)."""
    if not trials_dir.is_dir():
        return None
    out = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    for trial in trials_dir.glob("trial*/logs.json"):
        idx = int(trial.parent.name.replace("trial", ""))
        final = None
        for line in trial.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "final_eval/success_rate" in item:
                final = item
        if final is None or idx >= n:
            continue
        out[idx] = float(final["final_eval/success_rate"]) > 0.5
        seen[idx] = True
    if not seen.all():
        return None
    return out


def method_vector(base: Path, method: str, seed: int, n: int) -> np.ndarray | None:
    """Concatenate a method's per-episode outcomes over shapes, in a fixed order.

    A method name may contain ``{shape}`` for layouts where the evaluated shape is
    part of the run directory (the Thread B few-shot cells), rather than only a
    subdirectory of a shared method directory.
    """
    parts = []
    for shape in SHAPES:
        run = base / method.format(shape=shape) / f"seed{seed}"
        vec = from_batched_log(run / "logs" / f"{shape}.log")
        if vec is None:
            vec = from_trials(run / f"{shape}_trials", n)
        if vec is None or len(vec) != n:
            return None
        parts.append(vec)
    return np.concatenate(parts)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("eval_outputs/indist_matrix"))
    ap.add_argument("--reference", default="frozen")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 400])
    ap.add_argument("--n-per-shape", type=int, default=50)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--shapes", nargs="+", default=None)
    ap.add_argument(
        "--reference-base",
        type=Path,
        default=None,
        help="Directory holding the reference method, when it lives outside --base.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    global SHAPES
    if args.shapes:
        SHAPES = [s.replace("+", "plus") for s in args.shapes]

    rng = np.random.default_rng(0)
    vectors: dict[str, np.ndarray] = {}
    for method in [args.reference] + args.methods:
        parts = []
        src = args.reference_base if (method == args.reference and args.reference_base) else args.base
        for seed in args.seeds:
            vec = method_vector(src, method, seed, args.n_per_shape)
            if vec is None:
                print(f"[skip] {method} seed{seed}: incomplete")
                continue
            parts.append(vec)
        if len(parts) != len(args.seeds):
            print(f"[warn] {method}: {len(parts)}/{len(args.seeds)} seeds usable")
        if parts:
            vectors[method] = np.concatenate(parts)

    if args.reference not in vectors:
        raise SystemExit(f"reference {args.reference} unavailable")
    ref = vectors[args.reference]
    print(f"\nreference = {args.reference}: {ref.mean():.3f} (n={len(ref)})\n")
    header = (
        f"{'method':<14}{'succ':>7}{'ref':>7}{'delta':>8}{'95% CI':>18}"
        f"{'b/c':>10}{'McNemar p':>11}"
    )
    print(header)
    print("-" * len(header))
    out = {}
    for method in args.methods:
        if method not in vectors:
            continue
        vec = vectors[method]
        if len(vec) != len(ref):
            print(f"[skip] {method}: length mismatch vs reference")
            continue
        delta = float(vec.mean() - ref.mean())
        b = int(np.sum(vec & ~ref))  # method wins
        c = int(np.sum(~vec & ref))  # reference wins
        p = mcnemar_exact(b, c)
        diff = vec.astype(float) - ref.astype(float)
        idx = rng.integers(0, len(diff), size=(args.boot, len(diff)))
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(
            f"{method:<14}{vec.mean():>7.3f}{ref.mean():>7.3f}{delta:>+8.3f}"
            f"{'[' + f'{lo:+.3f}, {hi:+.3f}' + ']':>18}{f'{b}/{c}':>10}{p:>11.4f}"
        )
        out[method] = {
            "success": float(vec.mean()),
            "reference": float(ref.mean()),
            "delta": delta,
            "ci95": [float(lo), float(hi)],
            "b_method_wins": b,
            "c_reference_wins": c,
            "mcnemar_p": float(p),
            "n": int(len(vec)),
        }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
