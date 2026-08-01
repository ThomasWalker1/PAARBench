"""The benchmark's settings suite and its cohort split.

A *setting* is a base world model plus an episode distribution.  A *cohort* is a
disjoint set of episodes within a setting, identified by an evaluation seed.  The
split between the selection cohort and the test cohorts is declared here, by the
benchmark, and never by a submission — that separation is the whole point of the
scoring protocol in docs/PLAN.md §3, and M3 makes the harness enforce it.

A *column* is one (setting, cohort) evaluation: every shape in the setting, at one
seed, for one method configuration.  Success rate for a column is the unweighted
mean over shapes because every shape contributes the same ``n_evals``.

``frozen_success`` records the do-nothing reference measured by the predecessor
project.  It is a regression target, not an input to any metric: every metric in
§4 is computed against a frozen column re-run alongside the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Setting:
    """One base model + episode distribution, with its declared cohort split."""

    id: str
    base: str
    """Checkpoint directory, relative to ``checkpoints/``."""

    shapes: tuple[str, ...]
    selection_seed: int
    test_seeds: tuple[int, ...]
    n_evals: int
    """Episodes per shape per cohort."""

    frozen_success: dict[str, float]
    """Predecessor reference, keyed by ``"selection"`` / ``"test"``."""

    frozen_success_by_shape: dict[str, float] = field(default_factory=dict)
    """Per-shape reference on the *selection* cohort, where one is on record."""

    enabled: bool = True
    notes: str = ""

    @property
    def base_path(self) -> Path:
        return REPO_ROOT / "checkpoints" / self.base

    @property
    def cohort_n(self) -> int:
        return self.n_evals * len(self.shapes)

    def cohort_seed(self, cohort: str) -> int:
        """Resolve a cohort name to its seed, refusing anything undeclared."""
        if cohort == "selection":
            return self.selection_seed
        if cohort.startswith("test"):
            suffix = cohort[len("test") :]
            if suffix == "":
                if len(self.test_seeds) != 1:
                    raise ValueError(
                        f"setting {self.id!r} has {len(self.test_seeds)} test cohorts; "
                        f"name one explicitly, e.g. 'test{self.test_seeds[0]}'"
                    )
                return self.test_seeds[0]
            seed = int(suffix)
            if seed in self.test_seeds:
                return seed
        raise ValueError(
            f"unknown cohort {cohort!r} for setting {self.id!r}; "
            f"declared: selection={self.selection_seed}, test={list(self.test_seeds)}"
        )

    def targets_path(self, shape: str) -> Path:
        return REPO_ROOT / "data" / "pushobj_eval" / f"val_{shape}" / "plan_targets.pkl"


SETTINGS: dict[str, Setting] = {}


def _register(setting: Setting) -> Setting:
    SETTINGS[setting.id] = setting
    return setting


PUSHOBJ = _register(
    Setting(
        id="pushobj",
        base="pushobj_shape_shift",
        shapes=("T", "L", "Z", "+"),
        selection_seed=300,
        test_seeds=(100, 200, 400),
        n_evals=50,
        frozen_success={"selection": 0.490, "test": 0.485},
        frozen_success_by_shape={"T": 0.50, "L": 0.44, "Z": 0.60, "+": 0.42},
        notes="Primary setting. n=200 selection, n=600 test (3 cohorts x 200).",
    )
)

PUSHOBJ_SHIFT = _register(
    Setting(
        id="pushobj_shift",
        base="pushobj_shape_shift",
        shapes=("I", "small_tee", "square"),
        # A distribution-shift *condition* on the pushobj base, not a fourth
        # environment (docs/PLAN.md §2.3).  Held-out shapes were never trained on,
        # so there is no selection/test distinction to draw within them; seed 100
        # is the single declared cohort and it is scored as a test cohort.
        selection_seed=100,
        test_seeds=(100,),
        n_evals=50,
        frozen_success={"test": 0.293},
        notes="Held-out shapes. Report as a condition on pushobj, not as its own environment.",
    )
)

PUSHT = _register(
    Setting(
        id="pusht",
        base="pusht_visual_shift",
        shapes=("T", "L", "Z"),
        selection_seed=100,
        test_seeds=(200, 400),
        n_evals=50,
        frozen_success={"selection": 0.360},
        frozen_success_by_shape={"T": 0.640, "L": 0.260, "Z": 0.180},
        notes=(
            "Frozen headroom is very unevenly spread across shapes; always report "
            "per-shape alongside the mean. This base's released config ships an "
            "unresolved env.dataset.data_path, so dataset_data_path must be set."
        ),
    )
)

POINTMAZE = _register(
    Setting(
        id="pointmaze",
        base="pointmaze",
        shapes=(),
        selection_seed=300,
        test_seeds=(0, 1, 2),
        n_evals=50,
        frozen_success={"selection": 0.880, "test": 0.733},
        enabled=False,
        notes=(
            "DISABLED pending docs/PLAN.md §2.3. Frozen sits at 0.880 on the selection "
            "cohort, which compresses every method into the remaining 12% and shrinks the "
            "distance span to 1.4x (against pushobj's 12.7x); n=50 gives SE~0.045. "
            "Re-cohort onto the harder test distribution (frozen 0.733) or drop the setting. "
            "Also requires the optional `pointmaze` dependency extra (mujoco-py, d4rl)."
        ),
    )
)


def get(setting_id: str) -> Setting:
    try:
        setting = SETTINGS[setting_id]
    except KeyError:
        raise KeyError(
            f"unknown setting {setting_id!r}; known: {sorted(SETTINGS)}"
        ) from None
    if not setting.enabled:
        raise ValueError(f"setting {setting_id!r} is disabled: {setting.notes}")
    return setting
