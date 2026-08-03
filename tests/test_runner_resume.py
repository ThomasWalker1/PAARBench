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


def test_result_records_use_repo_relative_output_paths():
    result = runner.ColumnResult(
        setting="pushobj", cohort="test100", seed=100, tag="toy", n_evals=1,
        success_by_shape={"T": 0.5},
        out_dir=runner.REPO_ROOT / "eval_outputs" / "toy" / "pushobj" / "test100",
    )
    assert result.to_dict()["out_dir"] == "eval_outputs/toy/pushobj/test100"


# --- concurrent launchers -----------------------------------------------------
#
# Two processes writing one column interleave their episodes.jsonl writes, and the
# result reads as a result rather than as corruption. It has happened twice: to the
# Restore TTA retest, and to a determinism check investigating that retest, where the
# interleaved data looked convincingly like planner nondeterminism.

def test_a_column_can_be_claimed_once(tmp_path):
    lock = runner._claim_column(tmp_path / "col")
    assert lock is not None and lock.is_file()
    assert lock.read_text().strip() == str(__import__("os").getpid())


def test_a_second_claim_on_the_same_column_is_refused(tmp_path):
    runner._claim_column(tmp_path / "col")
    with pytest.raises(runner.ColumnBusy) as excinfo:
        runner._claim_column(tmp_path / "col")
    assert "still running" in str(excinfo.value)


def test_a_stale_lock_says_so_rather_than_just_refusing(tmp_path):
    """A lock left by a killed launcher must be distinguishable from a live one."""
    col = tmp_path / "col"
    col.mkdir()
    # A pid that cannot exist: the kernel maximum is well below this.
    (col / ".paarbench_column_lock").write_text("999999999")
    with pytest.raises(runner.ColumnBusy) as excinfo:
        runner._claim_column(col)
    assert "stale" in str(excinfo.value)


def test_claiming_a_column_releases_it_for_the_next_run(tmp_path):
    lock = runner._claim_column(tmp_path / "col")
    lock.unlink()
    assert runner._claim_column(tmp_path / "col") is not None
