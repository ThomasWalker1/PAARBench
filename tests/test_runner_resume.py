"""Resume must not splice two configurations into one column.

Resume keys on a per-unit completion marker, which records that a unit *finished* and
nothing about *what it ran*. Point a second configuration at a column that is already
half-populated and the finished units are kept: the column then reports the mean of
two configurations as though it were one. That is the same class of silent error the
benchmark exists to expose in other people's evaluations, so the harness refuses it.

This nearly happened: the p=0.01 Restore TTA retest was launched at the default
out-root, on top of the committed p=0.001 evidence. It survived only because the
earlier data had been moved aside by hand first.
"""

import json

import pytest

from paarbench import runner


def write_summary(column_dir, params):
    column_dir.mkdir(parents=True, exist_ok=True)
    (column_dir / "column_summary.json").write_text(json.dumps({"params": params}))


def test_resuming_a_column_with_the_same_params_is_allowed(tmp_path):
    write_summary(tmp_path / "col", {"restore_probability": 0.001, "steps": 10})
    runner.check_resumable(tmp_path / "col", {"steps": 10, "restore_probability": 0.001})


def test_resuming_a_column_run_under_different_params_is_refused(tmp_path):
    write_summary(tmp_path / "col", {"restore_probability": 0.001, "steps": 10})
    with pytest.raises(runner.ColumnConflict) as excinfo:
        runner.check_resumable(tmp_path / "col", {"restore_probability": 0.01,
                                                  "steps": 10})
    message = str(excinfo.value)
    # The message has to name the offending key and offer the way out, or the next
    # person just deletes the directory that held the published evidence.
    assert "restore_probability: 0.001 -> 0.01" in message
    assert "--tag" in message
    assert "steps" not in message


def test_an_added_or_dropped_parameter_counts_as_a_conflict(tmp_path):
    write_summary(tmp_path / "col", {"steps": 10})
    with pytest.raises(runner.ColumnConflict):
        runner.check_resumable(tmp_path / "col", {"steps": 10, "lora_rank": 2})


def test_a_fresh_column_is_not_a_conflict(tmp_path):
    runner.check_resumable(tmp_path / "never_run", {"steps": 10})


def test_an_unreadable_summary_is_not_treated_as_evidence(tmp_path):
    """A truncated marker means we do not know, which is not the same as a conflict."""
    (tmp_path / "col").mkdir()
    (tmp_path / "col" / "column_summary.json").write_text("{not json")
    runner.check_resumable(tmp_path / "col", {"steps": 10})
