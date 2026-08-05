"""Two guards against silent failure modes a contributor hits before doing anything wrong.

Both are about artifacts that are *not* distributed with the repository -- the goal files
that define each setting's episodes, and the per-episode records the leaderboard's
continuous columns are computed from. Neither absence is a mistake; both used to surface as
something other than an error.

GPU-free, so CI runs them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paarbench.settings import MissingTargets, get  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import leaderboard  # noqa: E402


# -- missing goal files fail before a single process is launched ---------------


def _stage(root: Path, setting, shapes=None):
    for shape in (setting.shapes if shapes is None else shapes):
        path = root / setting.targets_path(shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_missing_targets_names_every_file_and_the_fix(tmp_path):
    setting = get("pushobj")
    with pytest.raises(MissingTargets) as exc:
        setting.require_targets(tmp_path)
    message = str(exc.value)
    for shape in setting.shapes:
        assert str(setting.targets_path(shape)) in message, f"{shape} not named"
    # A message that does not say what to do next is a stack trace with better grammar.
    assert "docs/CHECKPOINTS.md" in message
    assert "not tracked in git" in message


def test_a_partially_staged_setting_names_only_what_is_missing(tmp_path):
    setting = get("pushobj")
    _stage(tmp_path, setting, shapes=setting.shapes[:2])
    missing = setting.missing_targets(tmp_path)
    assert [path.parent.name for path in missing] == [
        f"val_{shape}" for shape in setting.shapes[2:]
    ]


def test_require_targets_is_silent_once_staged(tmp_path):
    setting = get("pusht")
    _stage(tmp_path, setting)
    assert setting.missing_targets(tmp_path) == []
    setting.require_targets(tmp_path)


def test_every_setting_checks_its_targets_before_launching_a_column():
    """The check has to run in the runner, not only be available to it.

    Asserted on the source because the alternative -- calling run_column -- depends on
    whether *this* machine happens to have the goal files staged, so the test would pass
    for opposite reasons in CI and on a contributor's box.
    """
    import inspect

    from paarbench import runner

    source = inspect.getsource(runner.run_column)
    assert "require_targets()" in source, (
        "run_column must fail fast when the goal files are absent; otherwise each shape "
        "dies separately inside eval_outputs/<...>/logs/<shape>.log"
    )


# -- regenerating the leaderboard must not silently delete other rows ---------

_HEADER = (
    "| method | success | ±1 SE | vs frozen | median dist Δ [95% CI] | catastrophe "
    "[95% CI] | compounding [95% CI] | regret | adapt s/replan | peak MB | n | selection "
    "cost |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
)
_FULL = ("| Method A | 0.600 | 0.020 | +0.100 | -1 [-7, +5] | 5.3% [3%, 8%] | -0.02 "
         "[-0.08, +0.01] | 48% / +85 | 0.091 | 888 | 600 | 8 |")
_BLANKED = ("| Method A | 0.600 | 0.020 | +0.100 | — | — | — | — | — | — | 600 | 8 |")


def test_blanked_continuous_columns_are_detected():
    lost = leaderboard.rows_losing_columns(f"{_HEADER}\n{_FULL}", f"{_HEADER}\n{_BLANKED}")
    assert lost == {"Method A": (6, 0)}


def test_an_unchanged_table_loses_nothing():
    assert leaderboard.rows_losing_columns(f"{_HEADER}\n{_FULL}", f"{_HEADER}\n{_FULL}") == {}


def test_adding_a_row_is_not_a_loss():
    new_row = ("| Method B | 0.500 | 0.020 | 0.000 | -3 [-9, +2] | 1.0% [0%, 3%] | -0.04 "
               "[-0.3, +0.2] | 42% / +75 | 0.092 | 888 | 600 | 8 |")
    lost = leaderboard.rows_losing_columns(
        f"{_HEADER}\n{_FULL}", f"{_HEADER}\n{_FULL}\n{new_row}"
    )
    assert lost == {}, "a contributor adding their own row must not be blocked"


def test_a_removed_row_counts_as_a_loss():
    """Dropping a method entirely is the same harm as blanking its columns."""
    assert leaderboard.rows_losing_columns(f"{_HEADER}\n{_FULL}", _HEADER) == {"Method A": (6, 0)}


def test_the_header_row_is_not_mistaken_for_a_method():
    assert "method" not in leaderboard._continuous_cells(f"{_HEADER}\n{_FULL}")
