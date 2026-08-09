"""The benchmark's settings suite and its cohort split.

A *setting* is a base world model plus an episode distribution.  A *cohort* is a
disjoint set of episodes within a setting, identified by an evaluation seed.  The
split between the selection cohort and the test cohorts is declared here, by the
benchmark, and never by a submission. The harness enforces that separation.

A *column* is one (setting, cohort) evaluation: every shape in the setting, at one
seed, for one method configuration.  Success rate for a column is the unweighted
mean over shapes because every shape contributes the same ``n_evals``.

``frozen_success`` records the do-nothing reference. It is a regression target, not an
input to any metric: every metric is computed against a frozen column re-run alongside
the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared across settings. Selection and test draws must use disjoint seeds; test
# cohorts are always the same three held-out draws from the segment pool.
SELECTION_SEED = 0
TEST_SEEDS = (100, 200, 300)


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

    inherits_selection_from: str = ""
    """Take the hyperparameters this method already froze on another setting.

    Some settings have no cohort to select on. A distribution-shift condition is the
    clear case: its episodes are held-out by construction, so there is nothing to
    split into a selection half and a test half, and running a selection rule there
    would be tuning on the test set. Naming a source setting says the submission is
    evaluated here *as selected elsewhere* -- which is the interesting question about
    a shift condition anyway: does the chosen hyperparameter survive the shift.
    """

    @property
    def has_selection_cohort(self) -> bool:
        """Is there a cohort a selection rule may legitimately read?

        False when the setting inherits its hyperparameters from elsewhere, or when
        the declared selection seed is also a test seed (legacy layout). For a
        held-out-shape condition there is no other seed -- it is a setting where
        selection has to come from somewhere else.
        """
        if self.inherits_selection_from:
            return False
        return self.selection_seed not in self.test_seeds

    goal_source: str = "segments"
    """The ``plan.py`` target loader used by this setting."""

    goal_horizon: int = 25
    """Environment steps from a sampled start to its planning goal."""

    target_template: str = "pushobj_eval/val_{shape}/plan_targets.pkl"
    """Path below ``data/``; may interpolate ``shape`` and cohort ``seed``."""

    planner_overrides: dict[str, object] = field(default_factory=dict)
    """Fixed, setting-owned Hydra overrides applied by the column runner."""

    dataset_path: Optional[str] = None
    """Evaluation data used to recover the base model's normalization metadata."""

    @property
    def base_path(self) -> Path:
        return Path("checkpoints") / self.base

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

    def targets_path(self, shape: str, seed: Optional[int] = None) -> Path:
        """Return the staged goal file for one shape."""
        return Path("data") / self.target_template.format(
            shape=shape, seed="" if seed is None else seed,
        )

    def missing_targets(self, root: Optional[Path] = None) -> list:
        """Goal files this setting needs that are not staged, in shape order."""
        base = Path(root) if root is not None else REPO_ROOT
        paths = []
        for shape in self.shapes:
            for seed in (self.selection_seed, *self.test_seeds):
                path = self.targets_path(shape, seed)
                if path not in paths:
                    paths.append(path)
        return [path for path in paths if not (base / path).is_file()]

    def require_targets(self, root: Optional[Path] = None) -> None:
        """Fail before launching anything if the goal files are not staged.

        These files define the setting's episodes and are **not** tracked in git, so a
        fresh checkout has none of them. Without this check each shape's ``plan.py``
        starts, loads a base model, and dies three seconds later with its own bare
        ``FileNotFoundError`` inside a per-unit log file -- so a whole column fails in a
        way whose cause is four levels down in ``eval_outputs/`` rather than on stdout.
        One message, up front, naming the files and the fix.
        """
        missing = self.missing_targets(root)
        if not missing:
            return
        listed = "\n    ".join(str(path) for path in missing)
        raise MissingTargets(
            f"setting {self.id!r} needs {len(missing)} goal file(s) that are not staged:\n"
            f"    {listed}\n"
            f"  These are not tracked in git. Fetch them with "
            f"`scripts/download_targets.py`; see docs/CHECKPOINTS.md "
            f"('Evaluation targets and training data')."
        )


class MissingTargets(Exception):
    """A setting's goal files are not staged, so none of its columns can run."""


SETTINGS: dict[str, Setting] = {}


def _register(setting: Setting) -> Setting:
    SETTINGS[setting.id] = setting
    return setting


PUSHOBJ = _register(
    Setting(
        id="pushobj",
        base="pushobj_shape_shift",
        shapes=("T", "L", "Z", "+"),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        frozen_success_by_shape={},
    )
)

PUSHOBJ_SHIFT = _register(
    Setting(
        id="pushobj_shift",
        base="pushobj_shape_shift",
        shapes=("I", "small_tee", "square"),
        # Distribution-shift condition: held-out shapes, no selection here.
        # Hyperparameters are inherited from pushobj; selection_seed is unused.
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        inherits_selection_from="pushobj",
        n_evals=50,
        frozen_success={},
    )
)

PUSHT = _register(
    Setting(
        id="pusht",
        base="pusht_visual_shift",
        shapes=("T", "L", "Z"),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        frozen_success_by_shape={},
    )
)

# Every maze perturbation shares the PointMaze Medium base. The two dynamics
# perturbations and held-out DiverseMaze layouts all inherit its selection. Goals are
# materialized per cohort seed rather than re-sampled inside workers so frozen and
# adapted columns are paired against a stable episode definition.
MAZE_MEDIUM = _register(
    Setting(
        id="maze_medium",
        base="mediummaze_dynamics_shift",
        shapes=("medium",),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        goal_source="maze_file",
        goal_horizon=50,
        target_template="maze_eval/maze_medium/seed_{seed}.pkl",
        planner_overrides={"objective.alpha": 0.0, "objective.mode": "all"},
        dataset_path="data/point_maze_medium",
    )
)

MAZE_MEDIUM_LOW_DENSITY = _register(
    Setting(
        id="maze_medium_low_density",
        base="mediummaze_dynamics_shift",
        shapes=("medium",),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        inherits_selection_from="maze_medium",
        goal_source="maze_file",
        goal_horizon=50,
        target_template="maze_eval/maze_medium/seed_{seed}.pkl",
        planner_overrides={
            "objective.alpha": 0.0,
            "objective.mode": "all",
            "env_kwargs_override.density_scale": 0.2,
        },
        dataset_path="data/point_maze_medium",
    )
)

MAZE_MEDIUM_HIGH_DAMPING = _register(
    Setting(
        id="maze_medium_high_damping",
        base="mediummaze_dynamics_shift",
        shapes=("medium",),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        inherits_selection_from="maze_medium",
        goal_source="maze_file",
        goal_horizon=50,
        target_template="maze_eval/maze_medium/seed_{seed}.pkl",
        planner_overrides={
            "objective.alpha": 0.0,
            "objective.mode": "all",
            "env_kwargs_override.damping_scale": 20.0,
        },
        dataset_path="data/point_maze_medium",
    )
)

MAZE_DIVERSE = _register(
    Setting(
        id="maze_diverse",
        base="mediummaze_dynamics_shift",
        shapes=("heldout_layouts",),
        selection_seed=SELECTION_SEED,
        test_seeds=TEST_SEEDS,
        n_evals=50,
        frozen_success={},
        inherits_selection_from="maze_medium",
        goal_source="maze_file",
        goal_horizon=50,
        target_template="maze_eval/maze_diverse/seed_{seed}.pkl",
        planner_overrides={"objective.alpha": 0.0, "objective.mode": "all"},
        dataset_path="data/point_maze_medium",
    )
)

def get(setting_id: str) -> Setting:
    try:
        setting = SETTINGS[setting_id]
    except KeyError:
        raise KeyError(
            f"unknown setting {setting_id!r}; known: {sorted(SETTINGS)}"
        ) from None
    return setting
