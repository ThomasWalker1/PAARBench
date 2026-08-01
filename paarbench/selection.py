"""Hyperparameter-selection rules, and the boundary they cannot cross.

A submission is **an adaptation method together with its hyperparameter-selection
rule**. The rule may read only the selection cohort; it is then frozen and run once on
the held-out test cohorts, and that is what gets reported.

This is enforced structurally rather than by review. A selection rule is handed a
``SelectionHarness``, and that object has no way to evaluate a test cohort — not a
discouraged way, no way. It also counts the columns it runs, so the tuning burden
appears as a number next to the score instead of in a footnote.

Writing a rule::

    class MyRule:
        def select(self, harness):
            best, best_score = None, -1.0
            for lr in (1e-4, 5e-4):
                result = harness.run(lr=lr)
                if result.success > best_score:
                    best, best_score = {"lr": lr}, result.success
            return best

A method with no hyperparameters needs no rule at all: omit ``selection`` from
``method.yaml`` and the params in the file are used as-is, at a declared cost of zero
columns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from paarbench.runner import ColumnResult, run_column
from paarbench.settings import Setting


class CohortViolation(Exception):
    """A selection rule tried to reach a cohort it is not allowed to see."""


@runtime_checkable
class SelectionRule(Protocol):
    """Chooses a method's hyperparameters using only the selection cohort."""

    def select(self, harness: "SelectionHarness") -> Dict[str, Any]:
        """Return the parameter dict to freeze and report with.

        Anything returned here is what runs on the test cohorts. The rule may run as
        many or as few columns as it likes; the count is reported.
        """


class SelectionHarness:
    """The only thing a selection rule can do: run columns on the selection cohort.

    Deliberately does not expose the ``Setting``, the runner, or the output paths. A
    rule that wants a number has to get it by running a selection column, which is the
    behaviour the protocol is trying to make natural.
    """

    def __init__(
        self,
        setting: Setting,
        method_name: str,
        tag: str,
        *,
        gpus: Optional[List[str]] = None,
        n_evals: Optional[int] = None,
        out_root=None,
        data_path=None,
        budget: Optional[int] = None,
        episode_isolation: bool = False,
        per_gpu: int = 1,
        verbose: bool = True,
    ):
        self._setting = setting
        self._method_name = method_name
        self._tag = tag
        self._gpus = gpus
        self._n_evals = n_evals
        self._out_root = out_root
        self._data_path = data_path
        self._budget = budget
        self._episode_isolation = episode_isolation
        self._per_gpu = per_gpu
        self._verbose = verbose
        self._history: List[ColumnResult] = []

    # -- what a rule may use -------------------------------------------------

    def run(self, **params: Any) -> ColumnResult:
        """Evaluate one candidate configuration on the selection cohort."""
        if self._budget is not None and len(self._history) >= self._budget:
            raise CohortViolation(
                f"selection budget of {self._budget} column(s) exhausted for "
                f"{self._method_name!r}. Raise --selection-budget, or make the rule cheaper."
            )
        # Distinct tag per candidate so results are cached and resumable rather than
        # overwriting each other -- a grid search re-run should not redo finished cells.
        index = len(self._history)
        result = run_column(
            self._setting,
            "selection",
            f"{self._tag}/select{index:02d}",
            method_name=self._method_name,
            params=params,
            gpus=self._gpus,
            n_evals=self._n_evals,
            out_root=self._out_root,
            data_path=self._data_path,
            episode_isolation=self._episode_isolation,
            per_gpu=self._per_gpu,
            verbose=self._verbose,
        )
        self._history.append(result)
        if self._verbose:
            score = "incomplete" if result.success is None else f"{result.success:.3f}"
            print(f"[select] {self._method_name} column {index} {params} -> {score}",
                  flush=True)
        return result

    @property
    def setting_id(self) -> str:
        return self._setting.id

    @property
    def shapes(self) -> tuple:
        return self._setting.shapes

    @property
    def columns_used(self) -> int:
        """The selection cost, reported alongside the score."""
        return len(self._history)

    @property
    def history(self) -> List[ColumnResult]:
        return list(self._history)

    # -- what a rule may not use ---------------------------------------------

    def __getattr__(self, item: str):
        # Only reached for attributes that do not exist. Turn the obvious attempts
        # into an explanation rather than an AttributeError.
        if item in {"test", "run_test", "test_seeds", "evaluate_test", "setting"}:
            raise CohortViolation(
                f"a selection rule cannot access {item!r}. Selection rules may only run "
                f"columns on the selection cohort, via harness.run(**params). The test "
                f"cohorts are evaluated once, after selection is frozen."
            )
        raise AttributeError(item)


class FixedParams:
    """The trivial rule: use what is in ``method.yaml``. Costs zero columns.

    This is the honest default for a method with no hyperparameters, and it is what a
    method gets when it omits ``selection`` entirely.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})

    def select(self, harness: SelectionHarness) -> Dict[str, Any]:
        return dict(self.params)


class GridSearch:
    """Exhaustive search over a parameter grid, scored by success on the selection cohort.

    Costs one column per cell -- which is the point: a method that needs a 16-cell
    sweep to work pays for that sweep in its reported selection cost, and a method with
    no hyperparameters pays nothing.
    """

    def __init__(self, grid: Dict[str, List[Any]]):
        if not grid:
            raise ValueError("GridSearch needs a non-empty grid")
        self.grid = {k: list(v) for k, v in grid.items()}

    def cells(self) -> List[Dict[str, Any]]:
        import itertools

        keys = sorted(self.grid)
        return [dict(zip(keys, values)) for values in
                itertools.product(*(self.grid[k] for k in keys))]

    def select(self, harness: SelectionHarness) -> Dict[str, Any]:
        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        for cell in self.cells():
            result = harness.run(**cell)
            if result.success is None:
                continue
            # Strictly greater, so ties go to the first cell in a deterministic order
            # rather than to whichever finished last.
            if result.success > best_score:
                best, best_score = cell, result.success
        if best is None:
            raise RuntimeError(
                "every selection column failed; nothing to select. "
                "check the per-shape logs under eval_outputs/"
            )
        return best
