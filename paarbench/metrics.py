"""The metrics, computed against the canonical per-episode record.

Every one is measured **relative to the frozen model on the same episodes**, and
every one is *paired*: the planner is deterministic, so for a fixed episode and model
the closed loop is reproducible, which licenses comparing method and frozen
episode-by-episode instead of distribution-to-distribution. That is where the
statistical power comes from at these sample sizes.

Three traps are designed around here rather than left to the caller. They were all
paid for once already in the predecessor project.

**Never decompose by each method's own outcome.** ``distance | success`` is ~110 for
every method while ``distance | failure`` *rises* with success rate, because a better
method converts the easy near-goal failures and leaves a harder residual. It looks
like a real effect and is pure selection. Everything here conditions on a *fixed*
episode set -- frozen's outcome, or both-fail -- never on the method's own.

**Success is absorbing.** ``planning/mpc.py`` zeroes the actions of an episode that
has already succeeded, so its distance is frozen at the value it had when it
succeeded. A distance comparison across methods is therefore partly a success
comparison in disguise. ``both_fail=True`` (the default for distance metrics)
restricts to episodes where neither arm succeeded, which decouples them.

**No means or standard deviations on distance.** The metric is heavy-tailed --
distances of 1e4 occur under the frozen model, whose own maximum is ~9,000 -- so a
mean is a statement about two or three episodes and a variance is worse. Medians and
rank tests only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from paarbench.schema import episode_key, final_replan

CATASTROPHE_MULTIPLE = 2.0
"""An episode ends >2x further from the goal than the frozen model did."""


@dataclass
class Paired:
    """A method and the frozen baseline, aligned episode by episode."""

    method: str
    setting: str
    n_paired: int
    method_values: np.ndarray
    frozen_values: np.ndarray
    method_success: np.ndarray
    frozen_success: np.ndarray


def pair_with_frozen(frame: pd.DataFrame, method: str, setting: str,
                     value: str = "state_dist",
                     frozen_method: str = "frozen") -> Paired:
    """Align one method against frozen on the episodes both actually ran."""
    final = final_replan(frame)
    final = final[final["setting"] == setting]
    a = final[final["method"] == method].copy()
    b = final[final["method"] == frozen_method].copy()
    if a.empty or b.empty:
        raise ValueError(
            f"need both {method!r} and {frozen_method!r} on {setting!r}; "
            f"found {sorted(final['method'].unique())}"
        )
    a["_key"], b["_key"] = episode_key(a), episode_key(b)
    merged = a.merge(b, on="_key", suffixes=("_m", "_f"))
    return Paired(
        method=method, setting=setting, n_paired=len(merged),
        method_values=merged[f"{value}_m"].to_numpy(float),
        frozen_values=merged[f"{value}_f"].to_numpy(float),
        method_success=merged["cumulative_success_m"].to_numpy(bool),
        frozen_success=merged["cumulative_success_f"].to_numpy(bool),
    )


def _mask(paired: Paired, both_fail: bool) -> np.ndarray:
    if not both_fail:
        return np.ones(paired.n_paired, dtype=bool)
    return ~paired.method_success & ~paired.frozen_success


def success_rate(frame: pd.DataFrame, method: str, setting: str) -> dict:
    """Conventional, and low-powered at these n. Reported because it is expected."""
    final = final_replan(frame)
    rows = final[(final["method"] == method) & (final["setting"] == setting)]
    n = len(rows)
    if not n:
        return {"success": None, "n": 0, "se": None}
    p = float(rows["cumulative_success"].mean())
    return {"success": p, "n": n, "se": float(np.sqrt(max(p * (1 - p), 0.0) / n))}


def median_distance(paired: Paired, both_fail: bool = True) -> dict:
    """Median final distance, and the paired shift against frozen.

    The discriminating metric: far more power than binary success, because it uses
    where the episode actually ended rather than which side of a threshold it fell.
    """
    mask = _mask(paired, both_fail)
    m, f = paired.method_values[mask], paired.frozen_values[mask]
    if not len(m):
        return {"n": 0, "median": None, "frozen_median": None,
                "median_paired_delta": None, "wilcoxon_p": None}
    delta = m - f
    return {
        "n": int(len(m)),
        "median": float(np.median(m)),
        "frozen_median": float(np.median(f)),
        # The median of the paired differences, not the difference of medians:
        # the pairing is the whole point and the two are not the same number.
        "median_paired_delta": float(np.median(delta)),
        "wilcoxon_p": _wilcoxon_p(delta),
        "both_fail_only": both_fail,
    }


def _wilcoxon_p(delta: np.ndarray) -> Optional[float]:
    nonzero = delta[delta != 0]
    if len(nonzero) < 10:
        return None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(nonzero).pvalue)
    except Exception:
        return None


def catastrophe_rate(paired: Paired, multiple: float = CATASTROPHE_MULTIPLE,
                     both_fail: bool = True) -> dict:
    """Fraction of episodes ending >``multiple``x further from goal than frozen.

    The single most discriminative number the predecessor found -- it ranged 9% to
    89% across one hyperparameter grid whose success rates looked far more similar.
    This is the metric that answers "how often does adapting actively hurt".
    """
    mask = _mask(paired, both_fail)
    m, f = paired.method_values[mask], paired.frozen_values[mask]
    if not len(m):
        return {"n": 0, "catastrophe_rate": None, "n_catastrophes": 0}
    hit = m > multiple * f
    return {
        "n": int(len(m)),
        "catastrophe_rate": float(hit.mean()),
        "n_catastrophes": int(hit.sum()),
        "multiple": multiple,
        "both_fail_only": both_fail,
    }


def regret_vs_frozen(paired: Paired, both_fail: bool = True) -> dict:
    """Does adapting ever lose to not adapting, and by how much?

    A method that can be worse than doing nothing has to show it, so this reports the
    losing tail explicitly rather than only a central tendency.
    """
    mask = _mask(paired, both_fail)
    m, f = paired.method_values[mask], paired.frozen_values[mask]
    if not len(m):
        return {"n": 0, "worse_fraction": None}
    delta = m - f
    worse = delta > 0
    return {
        "n": int(len(m)),
        "worse_fraction": float(worse.mean()),
        "median_regret_when_worse": (float(np.median(delta[worse])) if worse.any() else 0.0),
        "p90_regret": float(np.percentile(delta, 90)),
        "max_regret": float(delta.max()),
    }


def compounding_slope(frame: pd.DataFrame, method: str, setting: str,
                      frozen_method: str = "frozen") -> dict:
    """Median paired degradation vs frozen, as a function of replan index.

    The mechanism metric. It separates an *accumulated* update from a *recomputed*
    one: the predecessor measured degradation rising monotonically to +1156 over 20
    replans for an optimizer trajectory, and flat at ~0 when the correction is
    regenerated from frozen weights each replan. Two methods can reach the same final
    score by very different routes, and this is what distinguishes them.
    """
    sub = frame[frame["setting"] == setting]
    a = sub[sub["method"] == method].copy()
    b = sub[sub["method"] == frozen_method].copy()
    if a.empty or b.empty:
        return {"by_replan": {}, "slope_per_replan": None}
    a["_key"], b["_key"] = episode_key(a), episode_key(b)
    merged = a.merge(b, on=["_key", "replan"], suffixes=("_m", "_f"))
    if merged.empty:
        return {"by_replan": {}, "slope_per_replan": None}

    merged["_delta"] = merged["state_dist_m"] - merged["state_dist_f"]
    by_replan = merged.groupby("replan")["_delta"].median()
    slope = None
    if len(by_replan) >= 2:
        slope = float(np.polyfit(by_replan.index.to_numpy(float),
                                 by_replan.to_numpy(float), 1)[0])
    return {
        "by_replan": {int(k): float(v) for k, v in by_replan.items()},
        "slope_per_replan": slope,
        "final_replan_delta": float(by_replan.iloc[-1]),
        "n_pairs": int(len(merged)),
    }


def cost(frame: pd.DataFrame, method: str, setting: str) -> dict:
    """Adaptation cost, separated from planner cost.

    Timing and memory are per replan and repeated across a batched cohort's rows, so
    deduplicate before summarizing or a 50-episode batch counts each replan 50 times.
    """
    sub = frame[(frame["method"] == method) & (frame["setting"] == setting)]
    if sub.empty:
        return {"adapt_s_per_replan": None}
    per_replan = sub.drop_duplicates(
        subset=["setting", "cohort", "shape", "episode_local", "replan"]
    )
    return {
        "adapt_s_per_replan": float(per_replan["adapt_s"].median()),
        "planner_s_per_replan": float(per_replan["planner_s"].median()),
        "adapt_fraction": float(
            per_replan["adapt_s"].sum()
            / max(per_replan["adapt_s"].sum() + per_replan["planner_s"].sum(), 1e-12)
        ),
        "adapt_peak_mb": float(per_replan["adapt_peak_mb"].max()),
    }


def summarize(frame: pd.DataFrame, method: str, setting: str,
              frozen_method: str = "frozen") -> dict:
    """Every metric for one (method, setting). The leaderboard row."""
    out = {"method": method, "setting": setting}
    out.update(success_rate(frame, method, setting))
    out["cost"] = cost(frame, method, setting)

    if method == frozen_method:
        return out
    try:
        paired = pair_with_frozen(frame, method, setting, frozen_method=frozen_method)
    except ValueError:
        return out
    out["n_paired"] = paired.n_paired
    out["distance"] = median_distance(paired)
    out["catastrophe"] = catastrophe_rate(paired)
    out["regret"] = regret_vs_frozen(paired)
    out["compounding"] = compounding_slope(frame, method, setting, frozen_method)
    return out
