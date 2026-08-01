#!/usr/bin/env python3
"""Generate the leaderboard from the result records in ``results/``.

    scripts/leaderboard.py                      # all settings, to stdout
    scripts/leaderboard.py --setting pushobj
    scripts/leaderboard.py --out LEADERBOARD.md

Deliberately **not** collapsed to a single score. The interesting comparison is a
frontier -- a method can be worse on success rate and better on adaptation latency or
on how often it makes things worse -- and folding that into one number destroys the
thing worth seeing. Columns sort; there is no overall rank.

Two columns matter as much as the score:

  selection cost   how many evaluation columns the method's selection rule consumed.
                   A method needing a 16-cell sweep pays for it visibly here.
  vs frozen        every metric is relative to doing nothing. A method that cannot
                   beat the frozen model is reporting that, in the same table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench import settings as settings_mod

REPO_ROOT = Path(__file__).resolve().parent.parent


def binomial_se(p: float, n: int) -> float:
    if not n or p is None:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def load_records(results_dir: Path):
    records = []
    for path in sorted(results_dir.glob("*/*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"warning: {path} is not valid JSON; skipped", file=sys.stderr)
    return records


def render(records, setting_id: str) -> str:
    setting = settings_mod.SETTINGS.get(setting_id)
    rows = [r for r in records if r.get("setting") == setting_id]
    if not rows:
        return ""

    frozen = next((r for r in rows if r["method"] == "frozen"), None)
    baseline = frozen["success"] if frozen and frozen.get("success") is not None else None

    # Sort by success, but only among complete records; incomplete ones go last and
    # are labelled, never silently dropped.
    rows.sort(key=lambda r: (r.get("complete", False),
                             r["success"] if r.get("success") is not None else -1),
              reverse=True)

    out = [f"## {setting_id}", ""]
    if setting is not None:
        out.append(
            f"Test cohorts: seeds {list(setting.test_seeds)}, "
            f"{len(setting.shapes)} shapes, n={setting.n_evals} per shape per cohort. "
            f"Selection cohort: seed {setting.selection_seed} (never scored here)."
        )
        out.append("")

    detail = _metric_detail(setting_id)

    out.append("| method | success | ±1 SE | vs frozen | median dist Δ | catastrophe | "
               "compounding | adapt s/replan | n | selection cost |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r.get("success")
        n = r.get("n", 0)
        if not r.get("complete", False):
            out.append(f"| {r['display_name']} | *incomplete* | | | | | | | {n} | |")
            continue
        se = binomial_se(s, n)
        delta = "—" if baseline is None or r["method"] == "frozen" else f"{s - baseline:+.3f}"
        cost = r.get("selection_cost_columns")
        cost_s = "unknown" if cost is None else str(cost)
        d = detail.get(r["method"], {})
        # The continuous metrics come from the per-episode record on disk, the
        # success rate from the stored result record. If a re-run is in flight they
        # can disagree, and a row mixing a complete success with partial distance
        # data would be quietly wrong. Drop the continuous columns in that case.
        if d and d.get("_n", 0) != n:
            print(f"warning: {r['method']} has {d.get('_n', 0)} episodes on disk but "
                  f"n={n} in its result record; per-episode metrics suppressed "
                  f"(re-run in progress?)", file=sys.stderr)
            d = {}
        out.append(
            f"| {r['display_name']} | {s:.3f} | {se:.3f} | {delta} | "
            f"{d.get('distance', '—')} | {d.get('catastrophe', '—')} | "
            f"{d.get('compounding', '—')} | {d.get('adapt_s', '—')} | {n} | {cost_s} |"
        )

    out.append("")
    out.append(
        "**Success rate is the weakest column here.** At these n its binomial SE is "
        "around 0.02, so adjacent rows are usually not separable on it, and it is the "
        "metric the benchmark exists to argue past. The continuous columns carry far "
        "more information:"
    )
    out.append("")
    out.append(
        "- **median dist Δ** — median *paired* change in final distance-to-goal against "
        "the frozen model on the same episodes, restricted to episodes both arms fail "
        "(success is absorbing, so including successes makes this partly a success "
        "comparison). Negative is better.\n"
        "- **catastrophe** — fraction of episodes ending more than 2× further from the "
        "goal than the frozen model. How often adapting actively hurts.\n"
        "- **compounding** — slope of the paired distance gap against replan index. "
        "Positive means the correction degrades as it accumulates; ~0 means it is "
        "recomputed rather than accumulated.\n"
        "- **adapt s/replan** — median adaptation time, separated from planner time.\n"
        "- **selection cost** — evaluation columns the method's selection rule consumed."
    )
    out.append("")
    out.append(
        "Sort by whichever column matters for your use. There is deliberately no "
        "overall rank: a method can be worse on success and better on catastrophe rate "
        "and latency, and collapsing that to one number destroys the comparison."
    )
    out.append("")
    return "\n".join(out)


def _metric_detail(setting_id: str) -> dict:
    """Compute the continuous metrics from the per-episode record, if it exists.

    Degrades to an empty dict rather than failing: a result record written before the
    per-episode schema landed still produces a (thinner) leaderboard row.
    """
    try:
        from paarbench import metrics, schema

        frame = schema.load(setting=setting_id)
    except Exception as exc:  # noqa: BLE001 - the leaderboard must still render
        print(f"warning: per-episode metrics unavailable: {exc}", file=sys.stderr)
        return {}
    if frame.empty:
        return {}

    detail = {}
    for method in sorted(frame["method"].unique()):
        try:
            s = metrics.summarize(frame, method, setting_id)
        except Exception:  # noqa: BLE001
            continue
        row = {"_n": s.get("n", 0)}
        cost = s.get("cost") or {}
        if cost.get("adapt_s_per_replan") is not None:
            row["adapt_s"] = f"{cost['adapt_s_per_replan']:.3f}"
        dist = s.get("distance") or {}
        if dist.get("median_paired_delta") is not None:
            p = dist.get("wilcoxon_p")
            star = "" if p is None else ("*" if p < 0.05 else "")
            row["distance"] = f"{dist['median_paired_delta']:+.0f}{star}"
        cat = s.get("catastrophe") or {}
        if cat.get("catastrophe_rate") is not None:
            row["catastrophe"] = f"{cat['catastrophe_rate']:.1%}"
        comp = s.get("compounding") or {}
        if comp.get("slope_per_replan") is not None:
            row["compounding"] = f"{comp['slope_per_replan']:+.1f}/replan"
        detail[method] = row
    return detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--setting", default=None, help="default: every setting with records")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"no result records under {args.results_dir}. "
              f"run scripts/evaluate.py first.", file=sys.stderr)
        return 1

    ids = ([args.setting] if args.setting
           else sorted({r["setting"] for r in records}))
    body = "\n".join(filter(None, (render(records, i) for i in ids)))
    text = ("# Leaderboard\n\n"
            "Generated by `scripts/leaderboard.py` from the records in `results/`.\n"
            "Multi-objective on purpose: sort by whichever column you care about.\n\n"
            + body)

    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
