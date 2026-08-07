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

# How a selection cost should read. ``0`` alone is ambiguous: it is the honest score
# for a method with nothing to tune, and it is also what a method reports when its
# hyperparameters were simply written down. Those are different claims about tuning
# burden, and the column exists precisely to make tuning burden comparable.
COST_BASIS = {
    "none": "0",
    "authored": "0 (authored)",
    "selected": None,          # the count speaks for itself
    "skipped": "unknown",
}

# Built-in do-nothing arms. Batched and episode-isolated are separate leaderboard
# rows because the two evaluation modes are not interchangeable (see metrics).
FROZEN_METHODS = frozenset({"frozen", "frozen_isolated"})


def is_frozen(method: str) -> bool:
    return method in FROZEN_METHODS


def render_cost(record: dict) -> str:
    cost = record.get("selection_cost_columns")
    if record.get("selection_cost_basis") == "inherited":
        # Charged, not free: the columns were run, just on the source setting. Saying
        # 0 here would let a shift condition look cheaper than the thing it inherits.
        return "unknown (inherited)" if cost is None else f"{cost} (inherited)"
    basis = record.get("selection_cost_basis")
    if basis is None:
        # Written before the basis field existed. A recorded 0 from that era came
        # from FixedParams unless it is frozen, which has no rule at all.
        basis = ("none" if is_frozen(record.get("method", ""))
                 else "skipped" if cost is None
                 else "authored" if cost == 0 else "selected")
    label = COST_BASIS.get(basis)
    return label if label is not None else str(cost)


CONTINUOUS_COLUMNS = slice(4, 10)
"""Table cells computed from the per-episode record: median dist Δ through peak MB.

Indices into a rendered row: 0 method, 1 success, 2 ±1 SE, 3 vs frozen, then the six
recomputed columns, then n and selection cost -- both of which come from the result record
and are therefore always populated, so including them would mask a total loss as a partial
one.

Everything in this range comes from ``eval_outputs/**/episodes.jsonl``, which is
git-ignored -- so on a machine that has only *some* methods' raw records, exactly these
cells go blank. See ``rows_losing_columns``.
"""


def _continuous_cells(text: str) -> dict:
    """``display name -> number of populated continuous cells``, over a whole table file."""
    counts = {}
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| success |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 12:
            continue
        populated = sum(1 for cell in cells[CONTINUOUS_COLUMNS] if cell not in ("", "—"))
        counts[cells[0]] = counts.get(cells[0], 0) + populated
    return counts


def rows_losing_columns(previous: str, current: str) -> dict:
    """Methods whose continuous columns are *less* populated than before.

    The leaderboard's continuous metrics are recomputed from raw per-episode records that
    are not distributed with the repository. A contributor who has run only their own
    method and the frozen baseline therefore regenerates a table in which five columns of
    every *other* method silently become "—", and `vs frozen` shifts for the
    episode-isolated rows because their mode-matched reference is missing too. That is a
    destructive edit disguised as a rebuild, and the only current signal is one stderr
    line saying "per-episode metrics unavailable".

    Returns ``{display name: (before, after)}`` for rows that lost cells.
    """
    before, after = _continuous_cells(previous), _continuous_cells(current)
    return {name: (count, after.get(name, 0))
            for name, count in before.items() if after.get(name, 0) < count}


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


HEADER = ("| method | success | ±1 SE | vs frozen | median dist Δ [95% CI] | "
          "catastrophe [95% CI] | compounding [95% CI] | regret | "
          "adapt s/replan | peak MB | n | selection cost |")
RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def _table(rows, baseline, detail, *, isolated_baseline=None) -> list:
    out = [HEADER, RULE]
    for r in rows:
        s = r.get("success")
        n = r.get("n", 0)
        if not r.get("complete", False):
            out.append(f"| {r['display_name']} | *incomplete* | | | | | | | | | {n} | |")
            continue
        se = binomial_se(s, n)
        d = detail.get(r["method"], {})
        # An episode-isolated arm's gain is measured against frozen run in the same
        # mode, for the same reason its continuous columns are: the two modes give
        # different frozen success rates (pushobj 0.4883 isolated vs 0.4850 batched), and
        # charging that difference to the method is exactly the artifact being removed.
        ref = baseline
        if r.get("episode_isolated"):
            if isolated_baseline is not None:
                ref = isolated_baseline
            elif detail.get("frozen_isolated", {}).get("_success") is not None:
                ref = detail["frozen_isolated"]["_success"]
        delta = "—" if ref is None or is_frozen(r["method"]) else f"{s - ref:+.3f}"
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
            f"{d.get('compounding', '—')} | {d.get('regret', '—')} | "
            f"{d.get('adapt_s', '—')} | {d.get('peak_mb', '—')} | {n} | "
            f"{render_cost(r)} |"
        )
    return out


def render(records, setting_id: str) -> str:
    setting = settings_mod.SETTINGS.get(setting_id)
    rows = [r for r in records if r.get("setting") == setting_id]
    if not rows:
        return ""

    frozen = next((r for r in rows if r["method"] == "frozen"), None)
    baseline = frozen["success"] if frozen and frozen.get("success") is not None else None
    frozen_iso = next((r for r in rows if r["method"] == "frozen_isolated"), None)
    isolated_baseline = (
        frozen_iso["success"]
        if frozen_iso and frozen_iso.get("success") is not None else None
    )

    # Sort by success, but only among complete records; incomplete ones go last and
    # are labelled, never silently dropped.
    rows.sort(key=lambda r: (r.get("complete", False),
                             r["success"] if r.get("success") is not None else -1),
              reverse=True)
    # A submission is a method together with its declared selection rule. A run that
    # departed from that rule is evidence about the protocol, not an
    # entry -- so it is reported, in its own table, and never mixed into the ranking.
    submissions = [r for r in rows if r.get("submission", True)]
    ablations = [r for r in rows if not r.get("submission", True)]

    out = [f"## {setting_id}", ""]
    if setting is not None:
        # Printing "Selection cohort: seed 100" above a table whose test seed is also
        # 100 reads as a leak. A setting in that position has no cohort to select on,
        # which is exactly why it inherits its parameters from somewhere else.
        where = (f"Selection cohort: seed {setting.selection_seed} (never scored here)."
                 if setting.has_selection_cohort else
                 f"No selection cohort: every episode here is held out, so a submission "
                 f"is evaluated with the parameters it froze on "
                 f"`{setting.inherits_selection_from}`.")
        out.append(
            f"Test cohorts: seeds {list(setting.test_seeds)}, "
            f"{len(setting.shapes)} shapes, n={setting.n_evals} per shape per cohort. "
            + where
        )
        out.append("")

    detail = _metric_detail(setting_id)
    _warn_unmatched_reference(setting_id, detail, records)
    out.extend(_table(submissions, baseline, detail,
                      isolated_baseline=isolated_baseline))
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
        "- **compounding** — slope of the paired distance gap against replan index, in "
        "distance units per replan. Positive means the correction degrades as it "
        "accumulates; ~0 means it is recomputed rather than accumulated.\n"
        "- **regret** — fraction of episodes where adapting ended further from the goal "
        "than not adapting, and the 90th-percentile size of that loss. A method that can "
        "be worse than doing nothing has to show it.\n"
        "- **adapt s/replan** and **peak MB** — adaptation cost, separated from planner "
        "cost and measured around the adapter hooks only.\n"
        "- **selection cost** — evaluation columns the method's selection rule consumed. "
        "`0` means the method has no hyperparameters to tune; `0 (authored)` means it "
        "has them and they were written down rather than selected, which is a weaker "
        "claim and must not read as the same number; `unknown` means the declared rule "
        "was skipped, so the tuning happened somewhere this record cannot price.\n"
        "- **vs frozen** — absolute success change vs the mode-matched frozen arm: "
        "Frozen (Batched) for batched methods, Frozen (Individual) for "
        "episode-isolated methods."
    )
    out.append("")
    out.append(
        "Intervals are 95% percentile bootstrap over **episodes** (2000 resamples, fixed "
        "seed so a row does not move between renders). Episodes are the unit of "
        "independence: replans within an episode are a trajectory, not independent draws."
    )
    out.append("")
    out.append(
        "Sort by whichever column matters for your use. There is deliberately no "
        "overall rank: a method can be worse on success and better on catastrophe rate "
        "and latency, and collapsing that to one number destroys the comparison."
    )
    out.append("")
    if ablations:
        out.append("### Ablations (not submissions)")
        out.append("")
        out.append(
            "Held-out runs that did **not** follow a declared selection rule. A "
            "submission is a method together with the rule it declares, so these are "
            "not entries and are not ranked against the table above — they are "
            "evidence about the protocol itself. Same cohorts, same n, same metrics."
        )
        out.append("")
        out.extend(_table(ablations, baseline, detail,
                          isolated_baseline=isolated_baseline))
        out.append("")
    return "\n".join(out)


def _warn_unmatched_reference(setting_id: str, detail: dict, records) -> None:
    """Say so when a row could not be paired against its own evaluation mode.

    An episode-isolated arm scored against the *batched* frozen column carries the
    evaluation-mode difference inside its method effect, so it must never be silent.
    """
    isolated = {r["method"] for r in records
                if r.get("setting") == setting_id and r.get("episode_isolated")}
    unmatched = sorted(m for m in isolated
                       if detail.get(m, {}).get("_reference") == "frozen")
    if unmatched:
        print(f"warning: {setting_id}: {', '.join(unmatched)} are episode-isolated but "
              f"were paired against the BATCHED frozen column; no frozen_isolated "
              f"reference exists for this setting. Run "
              f"`scripts/evaluate.py --frozen --isolated --setting {setting_id}` "
              f"to remove the evaluation-mode artifact.", file=sys.stderr)


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
        row = {"_n": s.get("n", 0), "_reference": s.get("reference"),
               "_success": s.get("success")}
        cost = s.get("cost") or {}
        if cost.get("adapt_s_per_replan") is not None:
            row["adapt_s"] = f"{cost['adapt_s_per_replan']:.3f}"
        if cost.get("adapt_peak_mb") is not None:
            row["peak_mb"] = f"{cost['adapt_peak_mb']:.0f}"
        dist = s.get("distance") or {}
        if dist.get("median_paired_delta") is not None:
            row["distance"] = f"{dist['median_paired_delta']:+.0f}{_ci(dist.get('ci95'))}"
        cat = s.get("catastrophe") or {}
        if cat.get("catastrophe_rate") is not None:
            row["catastrophe"] = (f"{cat['catastrophe_rate']:.1%}"
                                  f"{_ci(cat.get('ci95'), pct=True)}")
        comp = s.get("compounding") or {}
        if comp.get("slope_per_replan") is not None:
            row["compounding"] = (f"{comp['slope_per_replan']:+.2f}"
                                  f"{_ci(comp.get('ci95'), places=2)}")
        reg = s.get("regret") or {}
        if reg.get("worse_fraction") is not None:
            row["regret"] = f"{reg['worse_fraction']:.0%} / {reg['p90_regret']:+.0f}"
        detail[method] = row
    return detail


def _ci(bounds, pct: bool = False, places: int = 0) -> str:
    """Render a bootstrap interval compactly, or nothing when it is unavailable."""
    if not bounds or bounds[0] is None or bounds[1] is None:
        return ""
    if pct:
        return f" [{bounds[0]:.0%}, {bounds[1]:.0%}]"
    return f" [{bounds[0]:+.{places}f}, {bounds[1]:+.{places}f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--setting", default=None, help="default: every setting with records")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here")
    ap.add_argument("--force", action="store_true",
                    help="write --out even if rows would lose continuous columns. Only "
                         "correct when you genuinely have every method's per-episode "
                         "records on disk and the loss is intended.")
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
        if args.out.is_file():
            lost = rows_losing_columns(args.out.read_text(), text)
            if lost and not args.force:
                print(
                    f"refusing to overwrite {args.out}: {len(lost)} row(s) would lose "
                    f"continuous columns, because their per-episode records are not on "
                    f"this machine (eval_outputs/ is git-ignored).",
                    file=sys.stderr,
                )
                for name, (before, after) in sorted(lost.items()):
                    print(f"    {name}: {before} populated cell(s) -> {after}",
                          file=sys.stderr)
                print(
                    "  This is what a contributor's regenerate looks like: it deletes "
                    "other methods' published metrics. Add your row to the existing table "
                    "instead (see CONTRIBUTING.md), or pass --force if you really do have "
                    "every method's raw records.",
                    file=sys.stderr,
                )
                return 1
            if lost:
                print(f"warning: --force given; {len(lost)} row(s) lose continuous columns",
                      file=sys.stderr)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
