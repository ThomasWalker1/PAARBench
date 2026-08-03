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

    dataset_path: str = ""
    """Training dataset, when the checkpoint does not record a usable one.

    The dataset supplies normalization statistics at startup; nothing else reads it.
    Most bases record an absolute path in their ``hydra.yaml``, but some released ones
    ship an unresolved placeholder, and then every column has to be told where the data
    is. Declaring it on the setting means a caller never has to know that.
    """

    inherits_selection_from: str = ""
    """Take the hyperparameters this method already froze on another setting.

    Some settings have no cohort to select on. A distribution-shift condition is the
    clear case: its episodes are held-out by construction, so there is nothing to
    split into a selection half and a test half, and running a selection rule there
    would be tuning on the test set. Naming a source setting says the submission is
    evaluated here *as selected elsewhere* -- which is the interesting question about
    a shift condition anyway: does the chosen hyperparameter survive the shift.
    """

    model_epoch: str = "latest"
    """Which base checkpoint epoch to plan with.

    ``latest`` for the pushing bases. PointMaze's released base is selected at epoch 3,
    and ``latest`` there is a *different, later* model -- so this is not cosmetic.
    """

    goal_source: str = "segments"
    """Where the planner gets its goals.

    ``segments`` reads a per-shape target file staged under ``data/``; ``dset`` draws
    goals from the training dataset and needs no target file at all. This is the axis on
    which the pushing settings and PointMaze genuinely differ, and it was previously
    hardcoded in the column runner -- which is why no non-pushing setting could run.
    """

    cpu_only: bool = False
    """Plan on the CPU, with the GPUs left free.

    PointMaze's env is MuJoCo and its planning is CPU-bound (~100 s/episode
    single-threaded), so a PointMaze column is a *different resource pool* from a
    PushObj one and the two can run concurrently. Declaring it here rather than at the
    call site keeps that out of every driver.
    """

    needs_mujoco: bool = False
    """Source ``env.sh`` for this setting's workers.

    Without it mujoco-py never imports, PointMaze is never registered with gym, and the
    vectorized env's worker dies as a ``BrokenPipeError`` from ``env/venv.py`` -- which
    reads as an IPC bug rather than a missing shared library. Recorded here so the
    failure cannot be rediscovered.
    """

    enabled: bool = True
    notes: str = ""

    @property
    def has_selection_cohort(self) -> bool:
        """Is there a cohort a selection rule may legitimately read?

        False when the declared selection seed is also a test seed. That is not a
        misconfiguration to fix by picking another seed -- for a held-out-shape
        condition there is no other seed -- it is a setting where selection has to
        come from somewhere else.
        """
        return self.selection_seed not in self.test_seeds

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

    def targets_path(self, shape: str):
        """The per-shape goal file, or ``None`` when this setting does not use one.

        ``goal_source: dset`` settings take their goals from the dataset, so there is no
        target file to point at -- and passing a made-up one is how PointMaze failed
        before: the path was built unconditionally from ``data/pushobj_eval/``, which is
        both the wrong directory and the wrong idea for a maze.
        """
        if self.goal_source != "segments":
            return None
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
        # ...which means a selection rule run here would read the cohort it is then
        # scored on. The harness refuses to do that (see has_selection_cohort), and
        # a submission is instead evaluated with what it froze on pushobj -- the
        # same base model, the same rule, shifted shapes.
        inherits_selection_from="pushobj",
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
        frozen_success={"selection": 0.360, "test": 0.350},
        frozen_success_by_shape={"T": 0.640, "L": 0.260, "Z": 0.180},
        # This base's released hydra.yaml ships `data_path: <path>` -- the literal
        # placeholder, never substituted. Without an override the dataset call dies
        # with a bare FileNotFoundError naming only `env.dataset`. It reads the same
        # pushobj_multishape statistics as the pushobj base does.
        dataset_path="/mnt/richb/tw78/data/hyperjepa_pushobj/pushobj_multishape",
        notes=(
            "Frozen headroom is very unevenly spread across shapes (0.640/0.260/0.180); "
            "always report per-shape alongside the mean."
        ),
    )
)

POINTMAZE = _register(
    Setting(
        id="pointmaze",
        base="pointmaze/scratch_resnet_global_cos1e-1_iid_seed0",
        # One environment, no shape dimension -- but a column is still addressed by
        # variant, so the maze is a single named variant rather than an empty tuple.
        # An empty `shapes` made run_column raise "declares no shapes; nothing to run",
        # which is the correct behaviour for a misconfigured setting and was the
        # immediate blocker here.
        shapes=("umaze",),
        # Re-cohorted 2026-08-02 (docs/PLAN.md §2.3, RESULTS.md). Seed 300 is retired:
        # measured over eight candidate cohorts it is the *easiest* of them (frozen
        # 0.880 against a pooled 0.770), and a selection cohort easier than test
        # flatters every candidate. Seed 4 (0.740) sits marginally harder than the
        # pooled test cohorts (0.757), which is the safe direction.
        selection_seed=4,
        test_seeds=(0, 1, 2, 3, 5, 6),
        # Unchanged at 50: the environment seed of episode i is
        # `seed * eval_episode_total + i + 1`, so changing n_evals silently redraws
        # every cohort and voids the sweep the split was chosen from. The extra power
        # comes from pooling six test cohorts, not from longer ones.
        n_evals=50,
        frozen_success={"selection": 0.740, "test": 0.757},
        model_epoch="3",
        goal_source="dset",
        cpu_only=True,
        needs_mujoco=True,
        dataset_path="/dev/shm/tw78/point_maze",
        notes=(
            "Second task family. The least discriminating setting in the suite: 24.3% "
            "headroom against pushobj's 51.5%, and at n=300 it resolves only success "
            "effects >= +0.050 -- the predecessor's own PointMaze effect was +0.044, "
            "just under that. Report the continuous metrics here; do not read an "
            "unseparated success ordering as a result. Needs the optional `pointmaze` "
            "extra (mujoco-py, d4rl) and `.local-deps` staged for GL headers."
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
