"""AdaJEPA's hyperparameter-selection rule.

The predecessor's central measurement result is that this method's score is dominated
by its step size and step count: within one 16-cell grid, closed-loop success spans
0.010 to 0.690 and median final distance-to-goal spans 12.7x. A wrong cell does not
merely waste the adaptation signal, it reverses it.

So the honest way to report this method is with the search attached. ``StepSizeGrid``
is that search, and it costs 16 columns on the selection cohort -- which is exactly
what the leaderboard's selection-cost column is for.
"""

from paarbench.selection import GridSearch

# The predecessor's grid. (steps=10, pred_lr=5e-4) is the cell it selects on PushObj.
STEPS = [1, 5, 10, 20]
LEARNING_RATES = [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3]


class StepSizeGrid(GridSearch):
    """Exhaustive (steps x pred_lr) search. 16 columns."""

    def __init__(self):
        super().__init__({"steps": STEPS, "pred_lr": LEARNING_RATES})


class TunedFixed:
    """The already-selected cell, for reproducing a published number cheaply.

    Declares its cost as unknown rather than zero: the tuning did happen, it just
    happened somewhere else. Use ``StepSizeGrid`` for a submission.
    """

    def select(self, harness):
        return {"steps": 10, "pred_lr": 5.0e-4}
