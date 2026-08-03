"""The metrics, and the three traps they are designed around.

Synthetic frames, no GPU. The point of most of these is not that the arithmetic is
right but that the *conditioning* is right -- each one encodes a mistake that produced
a plausible wrong answer in the predecessor project.
"""

import numpy as np
import pandas as pd
import pytest

from paarbench import metrics
from paarbench.schema import final_replan, load


def frame(method, dists, successes=None, n_replans=3, setting="pushobj",
          cohort="test100", shape="T"):
    """One row per (episode, replan). ``dists`` is the final distance per episode."""
    successes = successes if successes is not None else [False] * len(dists)
    rows = []
    for episode, (dist, success) in enumerate(zip(dists, successes)):
        for replan in range(n_replans):
            # Distance walks linearly to its final value, so compounding slope is
            # well-defined and predictable.
            value = dist * (replan + 1) / n_replans
            rows.append({
                "method": method, "setting": setting, "cohort": cohort,
                "shape": shape, "episode": episode, "episode_local": episode,
                "replan": replan, "state_dist": value,
                "success": success and replan == n_replans - 1,
                "cumulative_success": success and replan == n_replans - 1,
                "planner_s": 1.0, "adapt_s": 0.1, "adapt_peak_mb": 100.0,
            })
    return pd.DataFrame(rows)


def paired_frame(method_dists, frozen_dists, method_success=None, frozen_success=None):
    return pd.concat([
        frame("m", method_dists, method_success),
        frame("frozen", frozen_dists, frozen_success),
    ], ignore_index=True)


# -- schema ------------------------------------------------------------------


def test_final_replan_keeps_one_row_per_episode():
    f = frame("m", [10.0, 20.0], n_replans=5)
    final = final_replan(f)
    assert len(final) == 2
    assert sorted(final["state_dist"]) == [10.0, 20.0]
    assert set(final["replan"]) == {4}


def test_schema_load_skips_a_private_frozen_selection_cohort(tmp_path):
    """The harness-owned baseline is tuning data, not a frozen test result."""
    record = ('{"episode_local": 0, "replan": 0, "state_dist": 1.0, '
              '"success": false, "cumulative_success": false}\n')
    for cohort in ("selection", "test100"):
        path = tmp_path / "frozen" / "pushobj" / cohort / "T" / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(record)

    loaded = load(tmp_path)
    assert set(loaded["cohort"]) == {"test100"}


# -- distance ----------------------------------------------------------------


def test_median_distance_is_the_median_of_paired_differences():
    """Not the difference of medians -- the pairing is the whole point."""
    f = paired_frame([10.0, 20.0, 30.0], [15.0, 18.0, 60.0])
    p = metrics.pair_with_frozen(f, "m", "pushobj")
    d = metrics.median_distance(p)
    assert d["median_paired_delta"] == pytest.approx(np.median([-5.0, 2.0, -30.0]))
    # The difference of medians would be 20 - 18 = +2, the opposite sign.
    assert d["median"] - d["frozen_median"] == pytest.approx(2.0)


def test_heavy_tail_moves_the_mean_but_not_the_median():
    """Why distance is never summarized by a mean: one episode would own it."""
    f = paired_frame([10.0, 12.0, 11.0, 10000.0], [10.0, 12.0, 11.0, 10.0])
    p = metrics.pair_with_frozen(f, "m", "pushobj")
    d = metrics.median_distance(p)
    assert d["median_paired_delta"] == pytest.approx(0.0)
    assert np.mean(p.method_values - p.frozen_values) > 2000


# -- catastrophe -------------------------------------------------------------


def test_catastrophe_counts_episodes_ending_far_worse_than_frozen():
    f = paired_frame([10.0, 25.0, 100.0, 5.0], [10.0, 10.0, 10.0, 10.0])
    p = metrics.pair_with_frozen(f, "m", "pushobj")
    c = metrics.catastrophe_rate(p)
    assert c["n_catastrophes"] == 2          # 25 > 2*10 and 100 > 2*10
    assert c["catastrophe_rate"] == pytest.approx(0.5)


def test_catastrophe_is_paired_not_absolute():
    """An episode frozen also fails badly on is not the method's catastrophe."""
    f = paired_frame([1000.0], [900.0])
    p = metrics.pair_with_frozen(f, "m", "pushobj")
    assert metrics.catastrophe_rate(p)["n_catastrophes"] == 0


# -- the traps ---------------------------------------------------------------


def test_distance_conditions_on_both_failing_not_on_the_methods_own_outcome():
    """Success is absorbing: a succeeded episode's distance is pinned at success.

    Including succeeded episodes makes a distance comparison partly a success
    comparison in disguise, which is the selection effect that made `d | failure`
    look like it rose with method quality.
    """
    # Episode 0: method succeeds (small pinned distance), frozen fails.
    # Episode 1: both fail, method is genuinely worse.
    f = paired_frame([5.0, 200.0], [300.0, 100.0],
                     method_success=[True, False], frozen_success=[False, False])
    p = metrics.pair_with_frozen(f, "m", "pushobj")

    restricted = metrics.median_distance(p, both_fail=True)
    assert restricted["n"] == 1
    assert restricted["median_paired_delta"] == pytest.approx(100.0)   # honestly worse

    unrestricted = metrics.median_distance(p, both_fail=False)
    assert unrestricted["n"] == 2
    # Including the succeeded episode flatters the method into looking net-neutral.
    assert unrestricted["median_paired_delta"] < restricted["median_paired_delta"]


def test_pairing_uses_only_episodes_both_arms_ran():
    f = pd.concat([
        frame("m", [10.0, 20.0, 30.0]),
        frame("frozen", [10.0, 20.0]),
    ], ignore_index=True)
    assert metrics.pair_with_frozen(f, "m", "pushobj").n_paired == 2


def test_missing_frozen_baseline_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="frozen"):
        metrics.pair_with_frozen(frame("m", [1.0]), "m", "pushobj")


# -- compounding -------------------------------------------------------------


def test_compounding_slope_separates_accumulating_from_flat():
    """The mechanism metric: does the gap grow with replan index, or not?"""
    growing = pd.concat([
        frame("m", [300.0], n_replans=10),        # walks 30 -> 300
        frame("frozen", [100.0], n_replans=10),   # walks 10 -> 100
    ], ignore_index=True)
    assert metrics.compounding_slope(growing, "m", "pushobj")["slope_per_replan"] > 0

    flat = pd.concat([
        frame("m", [100.0], n_replans=10),
        frame("frozen", [100.0], n_replans=10),
    ], ignore_index=True)
    assert metrics.compounding_slope(flat, "m", "pushobj")["slope_per_replan"] == pytest.approx(0.0)


# -- regret and cost ---------------------------------------------------------


def test_regret_reports_the_losing_tail_explicitly():
    f = paired_frame([50.0, 200.0, 10.0, 20.0], [100.0, 100.0, 100.0, 100.0])
    r = metrics.regret_vs_frozen(metrics.pair_with_frozen(f, "m", "pushobj"))
    assert r["worse_fraction"] == pytest.approx(0.25)
    assert r["max_regret"] == pytest.approx(100.0)


def test_cost_deduplicates_per_replan_rows_across_a_batched_cohort():
    """50 episodes share one replan's timing; counting it 50 times would be wrong."""
    f = frame("m", [10.0] * 50, n_replans=4)
    c = metrics.cost(f, "m", "pushobj")
    assert c["adapt_s_per_replan"] == pytest.approx(0.1)
    assert c["adapt_fraction"] == pytest.approx(0.1 / 1.1, rel=1e-6)


def test_summarize_covers_every_reported_metric():
    f = paired_frame([10.0, 20.0], [15.0, 25.0])
    s = metrics.summarize(f, "m", "pushobj")
    assert {"success", "distance", "catastrophe", "regret", "compounding", "cost"} <= set(s)


def test_summarize_of_frozen_has_no_self_comparison():
    f = paired_frame([10.0], [10.0])
    s = metrics.summarize(f, "frozen", "pushobj")
    assert "distance" not in s and s["success"] is not None


# --- evaluation-mode matching -------------------------------------------------
#
# Batched and episode-isolated evaluation are not interchangeable, even for the frozen
# model on identical environment seeds: measured on pushobj, no frozen episode is
# bit-identical across modes and 4% flip outcome; on pointmaze 9% flip. So an isolated
# arm paired against the batched frozen column carries the mode difference inside its
# method effect. Mode-matched, it cancels.

def _rows(method, mode, dists, n_replans=2):
    out = []
    for ep, dist in enumerate(dists):
        for replan in range(n_replans):
            out.append({
                "method": method, "setting": "s", "cohort": "c", "shape": "x",
                "episode": ep, "episode_local": ep, "replan": replan,
                "mode": mode,
                "success": False, "cumulative_success": False,
                "state_dist": dist, "visual_dist": dist, "proprio_dist": dist,
                "planner_s": 1.0, "adapt_s": 0.0, "adapt_peak_mb": 0.0,
            })
    return out


def _frame(*groups):
    import pandas as pd
    rows = []
    for g in groups:
        rows.extend(g)
    return pd.DataFrame(rows)


def test_an_isolated_arm_prefers_the_isolated_frozen_reference():
    frame = _frame(
        _rows("frozen", "batched", [10.0, 10.0]),
        _rows("frozen_isolated", "isolated", [12.0, 12.0]),
        _rows("m", "isolated", [14.0, 14.0]),
    )
    assert metrics.reference_for(frame, "m", "s") == "frozen_isolated"
    # +2 against the isolated reference, not +4 against the batched one.
    assert metrics.summarize(frame, "m", "s")["distance"]["median_paired_delta"] == 2.0


def test_a_batched_arm_uses_the_batched_frozen_reference():
    frame = _frame(
        _rows("frozen", "batched", [10.0, 10.0]),
        _rows("frozen_isolated", "isolated", [12.0, 12.0]),
        _rows("m", "batched", [14.0, 14.0]),
    )
    assert metrics.reference_for(frame, "m", "s") == "frozen"
    assert metrics.summarize(frame, "m", "s")["distance"]["median_paired_delta"] == 4.0


def test_an_isolated_arm_falls_back_when_no_isolated_reference_was_run():
    """A thinner number beats no number -- but the leaderboard must warn, and does."""
    frame = _frame(
        _rows("frozen", "batched", [10.0, 10.0]),
        _rows("m", "isolated", [14.0, 14.0]),
    )
    assert metrics.reference_for(frame, "m", "s") == "frozen"
    assert metrics.summarize(frame, "m", "s")["reference"] == "frozen"


def test_an_explicit_reference_overrides_mode_matching():
    frame = _frame(
        _rows("frozen", "batched", [10.0, 10.0]),
        _rows("frozen_isolated", "isolated", [12.0, 12.0]),
        _rows("m", "isolated", [14.0, 14.0]),
    )
    got = metrics.summarize(frame, "m", "s", frozen_method="frozen")
    assert got["reference"] == "frozen"
    assert got["distance"]["median_paired_delta"] == 4.0
