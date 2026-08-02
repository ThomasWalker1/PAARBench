"""Cohort separation is the protocol's load-bearing invariant (docs/PLAN.md §3).

These tests are cheap and run without a GPU, a checkpoint, or any data. M3 extends
them to the harness level, where a selection rule that reaches for a test seed must
be *rejected by the harness*, not by review.
"""

import pytest

from paarbench import adapter, settings


def test_pushobj_cohorts_are_declared_and_disjoint():
    s = settings.get("pushobj")
    assert s.cohort_seed("selection") == 300
    assert [s.cohort_seed(f"test{n}") for n in (100, 200, 400)] == [100, 200, 400]
    assert s.selection_seed not in s.test_seeds
    assert s.cohort_n == 200


@pytest.mark.parametrize("cohort", ["test300", "train", "test999", "", "TEST100"])
def test_undeclared_cohorts_are_refused(cohort):
    """Especially 'test300': the selection seed must not be reachable as a test cohort."""
    with pytest.raises(ValueError):
        settings.get("pushobj").cohort_seed(cohort)


def test_ambiguous_bare_test_cohort_is_refused():
    """'test' is only meaningful where a setting has exactly one test cohort."""
    with pytest.raises(ValueError):
        settings.get("pushobj").cohort_seed("test")
    assert settings.get("pushobj_shift").cohort_seed("test") == 100


def test_pointmaze_is_disabled_pending_recohorting():
    """§2.3: frozen 0.880 on its selection cohort compresses every method."""
    assert "pointmaze" in settings.SETTINGS
    assert not settings.SETTINGS["pointmaze"].enabled
    with pytest.raises(ValueError):
        settings.get("pointmaze")


def test_unknown_setting_names_are_refused():
    with pytest.raises(KeyError):
        settings.get("pushobj_v2")


def test_per_shape_frozen_reference_matches_the_recorded_mean():
    s = settings.get("pushobj")
    by_shape = s.frozen_success_by_shape
    assert set(by_shape) == set(s.shapes)
    mean = sum(by_shape.values()) / len(by_shape)
    assert mean == pytest.approx(s.frozen_success["selection"], abs=1e-9)


def test_null_adapter_satisfies_the_protocol():
    assert isinstance(adapter.NullAdapter(), adapter.TestTimeAdapter)


def test_shift_condition_has_no_cohort_to_select_on():
    """Its one cohort is a test cohort, so any rule run there tunes on the test set."""
    shift = settings.get("pushobj_shift")
    assert shift.selection_seed in shift.test_seeds
    assert not shift.has_selection_cohort
    assert shift.inherits_selection_from == "pushobj"
    assert settings.get("pushobj").has_selection_cohort


def test_selection_harness_refuses_a_setting_with_no_selection_cohort():
    """Structural, not procedural: the rule never gets an object it could misuse."""
    from paarbench.selection import CohortViolation, SelectionHarness

    with pytest.raises(CohortViolation, match="also one of its test seeds"):
        SelectionHarness(settings.get("pushobj_shift"), "any_method", tag="any_method")


def test_every_setting_can_either_select_or_names_where_it_inherits_from():
    """A setting with no selection cohort and no source would silently leak."""
    for setting in settings.SETTINGS.values():
        if not setting.has_selection_cohort:
            assert setting.inherits_selection_from in settings.SETTINGS, setting.id
