"""Cohort separation is the protocol's load-bearing invariant.

These tests are cheap and run without a GPU, a checkpoint, or any data. At the harness
level, a selection rule that reaches for a test seed must
be *rejected by the harness*, not by review.
"""

import math

import pytest

from paarbench import adapter, settings


def test_pushobj_cohorts_are_declared_and_disjoint():
    s = settings.get("pushobj")
    assert s.cohort_seed("selection") == 0
    assert [s.cohort_seed(f"test{n}") for n in (100, 200, 300)] == [100, 200, 300]
    assert s.selection_seed not in s.test_seeds
    assert s.cohort_n == 200


@pytest.mark.parametrize("cohort", ["test0", "train", "test999", "", "TEST100"])
def test_undeclared_cohorts_are_refused(cohort):
    """Especially 'test0': the selection seed must not be reachable as a test cohort."""
    with pytest.raises(ValueError):
        settings.get("pushobj").cohort_seed(cohort)


def test_ambiguous_bare_test_cohort_is_refused():
    """'test' is only meaningful where a setting has exactly one test cohort."""
    with pytest.raises(ValueError):
        settings.get("pushobj").cohort_seed("test")
    with pytest.raises(ValueError):
        settings.get("pushobj_shift").cohort_seed("test")


@pytest.mark.parametrize("setting_id", sorted(settings.SETTINGS))
def test_no_selection_cohort_is_easier_than_its_test_cohorts(setting_id):
    """A selection cohort easier than test flatters every candidate it scores.

    Tolerance is one SE of the selection cohort because the pushing settings sit a few
    thousandths apart and that is draw noise, not a design error.
    """
    s = settings.SETTINGS[setting_id]
    if not s.has_selection_cohort:
        return
    sel = s.frozen_success.get("selection")
    test = s.frozen_success.get("test")
    if sel is None or test is None:
        return
    se = math.sqrt(sel * (1 - sel) / max(s.cohort_n, 1))
    assert sel - test <= se, (
        f"{setting_id}: selection cohort frozen {sel:.3f} is easier than test "
        f"{test:.3f} by {sel - test:+.3f}, more than one SE ({se:.3f})"
    )


def test_unknown_setting_names_are_refused():
    with pytest.raises(KeyError):
        settings.get("pushobj_v2")


def test_per_shape_frozen_reference_matches_the_recorded_mean():
    s = settings.get("pushobj")
    by_shape = s.frozen_success_by_shape
    if not by_shape:
        pytest.skip("frozen per-shape reference not recorded yet")
    assert set(by_shape) == set(s.shapes)
    mean = sum(by_shape.values()) / len(by_shape)
    assert mean == pytest.approx(s.frozen_success["selection"], abs=1e-9)


def test_null_adapter_satisfies_the_protocol():
    assert isinstance(adapter.NullAdapter(), adapter.TestTimeAdapter)


def test_all_settings_share_standard_test_seeds():
    for setting_id in ("pushobj", "pushobj_shift", "pusht"):
        s = settings.get(setting_id)
        assert s.test_seeds == (100, 200, 300)
    assert settings.get("pushobj").selection_seed == 0
    assert settings.get("pusht").selection_seed == 0


def test_shift_condition_has_no_cohort_to_select_on():
    """Parameters are inherited; selection runs only on pushobj."""
    shift = settings.get("pushobj_shift")
    assert shift.test_seeds == (100, 200, 300)
    assert not shift.has_selection_cohort
    assert shift.inherits_selection_from == "pushobj"
    assert settings.get("pushobj").has_selection_cohort


def test_selection_harness_refuses_a_setting_with_no_selection_cohort():
    """Structural, not procedural: the rule never gets an object it could misuse."""
    from paarbench.selection import CohortViolation, SelectionHarness

    with pytest.raises(CohortViolation, match="inherited from 'pushobj'"):
        SelectionHarness(settings.get("pushobj_shift"), "any_method", tag="any_method")


def test_every_setting_can_either_select_or_names_where_it_inherits_from():
    """A setting with no selection cohort and no source would silently leak."""
    for setting in settings.SETTINGS.values():
        if not setting.has_selection_cohort:
            assert setting.inherits_selection_from in settings.SETTINGS, setting.id
