"""Selection rule for restore_tta.

The online-update configuration is held at AdaJEPA's declared selected cell. The new
hyperparameter is restoration probability p, so this small four-column search tests
whether a weak or strong leak back to pretrained weights best controls drift. It
deliberately does not include p=0: that is the existing AdaJEPA arm, not a restoration
setting.

Each candidate remains episode-isolated; selection cost is therefore four planning
columns times the selection cohort's episodes.
"""

from paarbench.selection import GridSearch

RESTORE_PROBABILITIES = [1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1]


class RestorationGrid(GridSearch):
    """Choose the restoration rate that most flattens paired compounding harm."""

    def __init__(self, restore_probabilities=None):
        super().__init__(
            {
                "restore_probability": list(
                    RESTORE_PROBABILITIES
                    if restore_probabilities is None
                    else restore_probabilities
                ),
            },
            objective="compounding_slope",
        )
