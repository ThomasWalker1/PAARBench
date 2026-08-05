"""LN-Recal's hyperparameter-selection rule.

The method has exactly one continuous knob, and it decides *how much of the method there
is*: ``ridge`` shrinks the closed-form refit toward the pretrained affine in units of the
sample count. Both ends of the axis are degenerate -- small ``ridge`` interpolates a
handful of samples per feature, large ``ridge`` is the frozen model with extra steps --
so a grid that does not bracket both would report a number about the search rather than
about the method.

**Sizing the grid.** The correction's size is directly observable without running the
closed loop: the adapter logs ``ln_recal/delta_rms_rel``, the RMS of the emitted affine
delta relative to the pretrained affine it perturbs. Measured on the pushobj *selection*
cohort at a reduced ``n_evals=3`` (12 episodes, purely to locate the axis; these columns
are not part of the declared rule and are not counted):

    fit_weight  ridge   delta_rms_rel   fit_residual_ratio
    false         0.3       0.73             0.23
    false         3.0       0.23             0.62
    false        30.0       0.029            0.95
    true         30.0       0.048            0.84
    true        300.0       0.0056           0.98
    true       3 000.0      0.0006           1.00

``fit_residual_ratio`` is the one-step latent error after the fit over the error before
it, on the transitions fitted. So the axis has a clear structure: below
``delta_rms_rel`` ~ 0.005 the method is numerically indistinguishable from frozen, and
above ~0.2 it fits its buffer well (ratio 0.23) while the closed loop deteriorates. The
grid below spans that whole range on both mechanisms -- roughly 0.002 to 0.23 -- so it
brackets the point where more adaptation stops helping instead of stopping short of it.

The second axis is what the fit may touch: the bias only (a ridge-shrunk mean prediction
residual) or bias and gain (the full 2x2 solve). With a single predicted token per frame
in this base model, both are estimated from the same handful of samples, so whether the
extra parameter buys anything or only absorbs noise is empirical.

Cost: 8 columns on the selection cohort. LN-Recal is not episode-isolated, so a column is
one process per shape rather than one per episode: the whole sweep costs about what one
of AdaJEPA's sixteen columns costs. That asymmetry is real, and the selection-cost column
cannot show it -- 8 here is not 8 there.
"""

from paarbench.selection import GridSearch

RIDGE = [3.0, 10.0, 30.0, 100.0]
FIT_WEIGHT = [False, True]


class RidgeGrid(GridSearch):
    """Exhaustive (ridge x fit_weight) search on paired distance harm. 8 columns.

    Scored on ``median_distance_delta`` rather than on success. The benchmark's own
    documentation is explicit that success rate is the weakest column it reports -- at
    200 selection episodes its binomial SE is ~0.035, which is the size of the effects
    being chosen between -- while the paired distance delta uses where each episode
    actually ended and is measured against a frozen column on the same episodes. A rule
    that picks the argmax of eight noisy binomials mostly selects the luckiest cell and
    then regresses on the test cohorts, which is a statement about the search.

    The direction of the bias this introduces is worth stating plainly: minimizing harm
    prefers a *smaller* correction, so where the evidence is weak this rule will select
    toward frozen. That is the intended behaviour for a method whose whole claim is that
    a cheap correction is a safe one -- but it means a near-frozen selected cell is a
    result about the method, not a failure of the rule. ``README.md`` tabulates every
    cell's success rate alongside its harm, so what a success-scored rule would have
    chosen instead is visible rather than hidden.
    """

    def __init__(self, ridge=None, fit_weight=None):
        super().__init__(
            {
                "ridge": list(RIDGE if ridge is None else ridge),
                "fit_weight": list(FIT_WEIGHT if fit_weight is None else fit_weight),
            },
            objective="median_distance_delta",
        )


class SuccessGrid(RidgeGrid):
    """The same 8 cells, scored on success. A different submission, not a variant.

    Kept because the two rules read the *same* cached columns, so reporting what this one
    would have chosen costs nothing beyond the first sweep. Declare it in ``method.yaml``
    instead of ``RidgeGrid`` to submit it.
    """

    def __init__(self, ridge=None, fit_weight=None):
        GridSearch.__init__(
            self,
            {
                "ridge": list(RIDGE if ridge is None else ridge),
                "fit_weight": list(FIT_WEIGHT if fit_weight is None else fit_weight),
            },
            objective="success",
        )
