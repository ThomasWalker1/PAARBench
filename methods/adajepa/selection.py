"""AdaJEPA's hyperparameter-selection rule.

This method's score is dominated by its step size and step count. Reporting it
without the search attached would say more about the search than about the method,
so the search is the submission.

**Sizing the grid.** The total correction this method applies over an episode scales
roughly as ``lr x steps x replans``. With 20 replans, the region where the correction
is comparable to the weights it is correcting sits near ``lr x steps ~ 4e-2`` -- so
``lr ~ 4e-2`` at one step per replan, ``~4e-3`` at ten. A grid has to *bracket* that:
cells well below it under-adapt and score near frozen, cells well above it overshoot
and can end up worse than not adapting at all. A grid that only spans the low side
finds a plausible best cell while hiding the fact that the hyperparameter can reverse
the adaptation signal, which is the single most important thing to know about this
method. The axes below span roughly two orders of magnitude either side.

Cost: 16 columns on the selection cohort, and this method is episode-isolated, so each
column is ``n_evals`` processes rather than one. That is expensive and it is meant to
be visible -- it is the price of a method whose behaviour is set by its tuning.
"""

from paarbench.selection import GridSearch

STEPS = [1, 3, 5, 10]
LEARNING_RATES = [5.0e-4, 2.0e-3, 1.0e-2, 5.0e-2]


class StepSizeGrid(GridSearch):
    """Exhaustive (steps x pred_lr) search. 16 columns."""

    def __init__(self, steps=None, learning_rates=None):
        super().__init__({
            "steps": list(steps or STEPS),
            "pred_lr": list(learning_rates or LEARNING_RATES),
        })


class CoarseGrid(GridSearch):
    """A 6-cell diagonal sweep for when 16 columns is too much compute.

    Walks the ``lr x steps`` product across the interesting region rather than
    covering the plane, so it still brackets the reversal point at ~1/3 the cost.
    Report which rule was used -- these are different submissions.
    """

    def __init__(self):
        super().__init__({"steps": [1, 10], "pred_lr": [5.0e-4, 5.0e-3, 5.0e-2]})


class TunedFixed:
    """A known-good cell, for reproducing a published number cheaply.

    Declares no search, so its selection cost is honestly unknown rather than zero:
    the tuning happened, just not here. Use ``StepSizeGrid`` for a submission.
    """

    def select(self, harness):
        return {"steps": 10, "pred_lr": 5.0e-4}
