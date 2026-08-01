"""The contribution mechanism: discovery, loading, validation, cohort enforcement.

All GPU-free. This is what CI can run on every pull request.
"""

from pathlib import Path

import pytest

from paarbench import adapter, methods
from paarbench.planner_hooks import build_adapter
from paarbench.selection import (
    CohortViolation,
    FixedParams,
    GridSearch,
    SelectionHarness,
)
from paarbench.settings import get as get_setting

FIXTURES = Path(__file__).parent / "fixtures" / "methods"


# -- discovery and loading ---------------------------------------------------


def test_discovery_finds_methods_and_skips_templates():
    found = {m.name for m in methods.discover(FIXTURES)}
    assert found == {"toy", "broken"}

    # The real methods/ tree must not expose _template as a contributable method.
    assert "_template" not in {m.name for m in methods.discover()}


def test_template_directory_exists_and_is_copyable():
    template = methods.METHODS_DIR / "_template"
    assert (template / "method.yaml").is_file()
    assert (template / "adapter.py").is_file()


def test_load_reads_the_manifest():
    m = methods.load("toy", FIXTURES)
    assert m.display_name == "Toy method (test fixture)"
    assert m.settings == ["pushobj"]
    assert m.params == {"scale": 2.0}
    assert m.selection_ref == "selection:ToyRule"


def test_unknown_method_names_the_alternatives():
    with pytest.raises(methods.MethodError) as exc:
        methods.load("does_not_exist", FIXTURES)
    assert "toy" in str(exc.value)


def test_method_directory_and_declared_name_must_agree(tmp_path):
    d = tmp_path / "mismatched"
    d.mkdir()
    (d / "method.yaml").write_text(
        "name: something_else\ndisplay_name: x\ndescription: x\n"
        "adapter: adapter:A\nsettings: [pushobj]\n"
    )
    with pytest.raises(methods.MethodError, match="must match"):
        methods.load("mismatched", tmp_path)


def test_missing_required_keys_point_at_the_template(tmp_path):
    d = tmp_path / "sparse"
    d.mkdir()
    (d / "method.yaml").write_text("name: sparse\n")
    with pytest.raises(methods.MethodError, match="_template"):
        methods.load("sparse", tmp_path)


# -- a method's code is loaded in isolation ----------------------------------


def test_adapter_class_loads_and_instantiates():
    m = methods.load("toy", FIXTURES)
    instance = m.build(wm="WM", preprocessor="PRE")
    assert instance.scale == 2.0          # from method.yaml
    assert isinstance(instance, adapter.TestTimeAdapter)


def test_overrides_beat_the_manifest():
    m = methods.load("toy", FIXTURES)
    assert m.build(wm=None, preprocessor=None, scale=9.0).scale == 9.0


def test_method_modules_do_not_collide_with_benchmark_modules(tmp_path):
    """A method may name a file settings.py without shadowing paarbench.settings."""
    d = tmp_path / "shadow"
    d.mkdir()
    (d / "method.yaml").write_text(
        "name: shadow\ndisplay_name: x\ndescription: x\n"
        "adapter: settings:ShadowAdapter\nsettings: [pushobj]\n"
    )
    (d / "settings.py").write_text(
        "class ShadowAdapter:\n"
        "    def __init__(self, wm, preprocessor): pass\n"
    )
    methods.load("shadow", tmp_path).load_adapter_class()

    import paarbench.settings as real
    assert real.SETTINGS  # still the benchmark's module, not the method's


# -- validation --------------------------------------------------------------


def test_valid_method_has_no_problems():
    assert methods.validate(methods.load("toy", FIXTURES)) == []


def test_validation_catches_bad_setting_and_missing_hooks():
    problems = methods.validate(methods.load("broken", FIXTURES))
    joined = " ".join(problems)
    assert "nonexistent_setting" in joined
    assert "on_episode_start" in joined and "on_transition" in joined


def test_shipped_methods_all_validate():
    """Every method in methods/ must be valid; this is the CI gate."""
    for m in methods.discover():
        assert methods.validate(m) == [], f"{m.name} is invalid"


# -- the planner's resolution path -------------------------------------------


@pytest.mark.parametrize("cfg", [None, {}, {"method": None}, {"method": "frozen"}])
def test_no_adapter_config_means_frozen(cfg):
    assert build_adapter(cfg, wm=None, preprocessor=None) is None


def test_planner_resolves_a_method_by_name(monkeypatch):
    monkeypatch.setattr(methods, "METHODS_DIR", FIXTURES)
    built = build_adapter({"method": "toy", "params": {"scale": 5.0}},
                          wm=None, preprocessor=None)
    assert built.scale == 5.0


def test_planner_rejects_an_adapter_missing_hooks(monkeypatch):
    monkeypatch.setattr(methods, "METHODS_DIR", FIXTURES)
    with pytest.raises(TypeError, match="on_episode_start"):
        build_adapter({"method": "broken"}, wm=None, preprocessor=None)


# -- cohort separation -------------------------------------------------------


class _RecordingHarness(SelectionHarness):
    """SelectionHarness with the expensive part stubbed out."""

    def __init__(self, setting, **kw):
        super().__init__(setting, "toy", "toy", verbose=False, **kw)
        self.seen = []

    def run(self, **params):
        if self._budget is not None and len(self.seen) >= self._budget:
            raise CohortViolation("budget exhausted")
        self.seen.append(params)
        self._history.append(None)

        class R:
            success = 0.1 * len(self.seen)
        return R()


def test_selection_harness_refuses_to_reach_a_test_cohort():
    h = _RecordingHarness(get_setting("pushobj"))
    for attempt in ("test", "run_test", "test_seeds", "evaluate_test", "setting"):
        with pytest.raises(CohortViolation, match="selection cohort"):
            getattr(h, attempt)


def test_selection_harness_counts_its_columns():
    h = _RecordingHarness(get_setting("pushobj"))
    assert h.columns_used == 0
    h.run(lr=1e-4)
    h.run(lr=5e-4)
    assert h.columns_used == 2


def test_selection_budget_is_enforced():
    h = _RecordingHarness(get_setting("pushobj"), budget=1)
    h.run(lr=1e-4)
    with pytest.raises(CohortViolation):
        h.run(lr=5e-4)


def test_grid_search_visits_every_cell_and_picks_the_best():
    h = _RecordingHarness(get_setting("pushobj"))
    rule = GridSearch({"lr": [1e-4, 5e-4], "steps": [1, 10]})
    chosen = rule.select(h)
    assert h.columns_used == 4                    # one column per cell, all declared
    assert chosen == h.seen[-1]                   # stub scores increase monotonically


def test_fixed_params_costs_nothing():
    h = _RecordingHarness(get_setting("pushobj"))
    assert FixedParams({"lr": 1e-3}).select(h) == {"lr": 1e-3}
    assert h.columns_used == 0


# -- repo-relative path resolution -------------------------------------------


def test_relative_checkpoint_paths_resolve_against_the_repo_root():
    """Hydra chdirs before the planner is built, so relative paths must be fixed up."""
    from paarbench.methods import REPO_ROOT, resolve_repo_paths

    out = resolve_repo_paths({"checkpoint_path": "pyproject.toml"})
    assert out["checkpoint_path"] == str(REPO_ROOT / "pyproject.toml")


def test_absolute_paths_are_left_alone():
    from paarbench.methods import resolve_repo_paths

    assert resolve_repo_paths({"ckpt_path": "/tmp"})["ckpt_path"] == "/tmp"


def test_non_path_values_are_never_mangled():
    """A string that only looks like a path key, or names nothing, is untouched."""
    from paarbench.methods import resolve_repo_paths

    params = {
        "update_scope": "lora_predlast_all",   # not a *_path key
        "context_path": "residual_action",     # *_path key, but names no file
        "steps": 10,
    }
    assert resolve_repo_paths(params) == params


# -- episode isolation -------------------------------------------------------


def test_isolation_is_opt_in_and_defaults_off():
    assert methods.load("toy", FIXTURES).requires_episode_isolation is False


def test_adajepa_declares_isolation_and_hyperjepa_does_not():
    """The two shipped methods differ exactly here, which is the point of the flag.

    AdaJEPA owns an optimizer trajectory over shared weights; HyperJEPA emits a
    per-episode correction from a per-episode context and batches correctly.
    """
    shipped = {m.name: m for m in methods.discover()}
    assert shipped["adajepa"].requires_episode_isolation is True
    assert shipped["hyperjepa"].requires_episode_isolation is False


def test_isolated_commands_score_one_episode_of_the_cohort():
    from paarbench.runner import build_command
    from paarbench.settings import get

    setting = get("pushobj")
    batched = " ".join(build_command(setting, 300, "T", Path("/tmp/x"), 50))
    assert "n_evals=50" in batched and "eval_episode_index" not in batched

    isolated = " ".join(build_command(setting, 300, "T", Path("/tmp/x"), 50,
                                      episode_index=7))
    assert "n_evals=1" in isolated
    assert "eval_episode_index=7" in isolated
    assert "eval_episode_total=50" in isolated


# -- resume and record hygiene -----------------------------------------------


def test_resume_keys_on_completion_not_on_the_log_file_existing(tmp_path):
    """logs.json appears on the FIRST replan, so its presence is not completion.

    Keying resume on it would skip an interrupted unit forever, leaving a truncated
    episode in the record while the column looked done.
    """
    from paarbench.runner import unit_complete

    unit = tmp_path / "T"
    unit.mkdir()
    assert not unit_complete(unit)                       # nothing written yet

    logs = unit / "logs.json"
    logs.write_text('{"step": 1, "mpc/success_rate": 0.0}\n')
    assert not unit_complete(unit)                       # started, not finished

    with logs.open("a") as fh:
        fh.write('{"final_eval/success_rate": 0.5}\n')
    assert unit_complete(unit)                           # the completion marker


def test_episode_record_is_truncated_not_appended():
    """A re-run into an existing directory must not interleave two runs' rows."""
    import inspect

    from planning.mpc import MPCPlanner

    source = inspect.getsource(MPCPlanner.plan)
    assert 'open("episodes.jsonl", "w")' in source, (
        "MPCPlanner.plan must truncate episodes.jsonl at episode start; Hydra does "
        "not clear a reused run directory and the file is appended to per replan"
    )
