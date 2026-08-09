"""Running evaluation columns.

A **column** is one (setting, cohort, method configuration) evaluation over every shape
in the setting. It is the unit of evaluation cost, and a submission's selection rule is
priced in columns consumed.

Each shape runs as its own ``plan.py`` process, one per GPU slot, because the ported
evaluation code assumes it owns its process (it forks one environment worker per
episode and mutates global RNG state). Keeping that boundary means a contributed method
cannot break another method's run, and a method that crashes fails its own column
rather than the harness.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from paarbench.settings import Setting

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = Path("eval_outputs")


def _portable_command_path(path: Path) -> str:
    """Prefer a repository-relative path in subprocess commands and their logs."""
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class PairedSelectionMetrics:
    """Selection-cohort metrics for one candidate against frozen.

    These values are attached by :class:`SelectionHarness`, not calculated by the
    generic column runner.  Pairing needs a frozen run on the same cohort, and only
    the harness is allowed to arrange that reference run.
    """

    n_paired: int
    median_distance_delta: Optional[float]
    catastrophe_rate: Optional[float]
    compounding_slope: Optional[float]

    def to_dict(self) -> dict:
        return {
            "n_paired": self.n_paired,
            "median_distance_delta": self.median_distance_delta,
            "catastrophe_rate": self.catastrophe_rate,
            "compounding_slope": self.compounding_slope,
        }


@dataclass
class ColumnResult:
    """What one column produced.

    ``success_by_shape`` is per-shape rather than pooled because frozen headroom is
    very unevenly distributed across shapes in some settings, and a pooled mean hides
    a method that helps one shape while destroying another.
    """

    setting: str
    cohort: str
    seed: int
    tag: str
    n_evals: int
    success_by_shape: Dict[str, Optional[float]]
    out_dir: Path
    params: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    paired_metrics: Optional[PairedSelectionMetrics] = None

    @property
    def complete(self) -> bool:
        return not self.failures and all(v is not None for v in self.success_by_shape.values())

    @property
    def success(self) -> Optional[float]:
        """Unweighted mean over shapes; every shape contributes the same ``n_evals``."""
        values = [v for v in self.success_by_shape.values() if v is not None]
        if not values or not self.complete:
            return None
        return sum(values) / len(values)

    @property
    def n(self) -> int:
        return self.n_evals * len(self.success_by_shape)

    @property
    def median_distance_delta(self) -> Optional[float]:
        """Median final-distance shift against frozen on the selection cohort."""
        return (None if self.paired_metrics is None
                else self.paired_metrics.median_distance_delta)

    @property
    def catastrophe_rate(self) -> Optional[float]:
        """Paired >2x-frozen final-distance rate on the selection cohort."""
        return (None if self.paired_metrics is None
                else self.paired_metrics.catastrophe_rate)

    @property
    def compounding_slope(self) -> Optional[float]:
        """Slope of median paired degradation per replan on the selection cohort."""
        return (None if self.paired_metrics is None
                else self.paired_metrics.compounding_slope)

    def to_dict(self) -> dict:
        try:
            out_dir = str(self.out_dir.relative_to(REPO_ROOT))
        except ValueError:
            out_dir = str(self.out_dir)
        return {
            "setting": self.setting,
            "cohort": self.cohort,
            "seed": self.seed,
            "tag": self.tag,
            "n_evals": self.n_evals,
            "n": self.n,
            "params": self.params,
            "success_by_shape": self.success_by_shape,
            "success": self.success,
            "complete": self.complete,
            "failures": self.failures,
            "wall_seconds": round(self.wall_seconds, 1),
            "out_dir": out_dir,
            "paired_metrics": (None if self.paired_metrics is None
                               else self.paired_metrics.to_dict()),
        }


def read_success(logs_json: Path) -> Optional[float]:
    """Pull ``final_eval/success_rate`` out of a finished shape's logs.

    ``None`` means the unit did not finish: the key is written once, after planning
    completes, so its absence is the completion marker.
    """
    if not logs_json.is_file():
        return None
    value = None
    with logs_json.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "final_eval/success_rate" in row:
                value = row["final_eval/success_rate"]
    return value


class ColumnConflict(Exception):
    """A column on disk was produced by a different configuration than the one asked for."""


def check_resumable(column_dir: Path, params: Dict[str, Any]) -> None:
    """Refuse to resume a column that was produced under different parameters.

    Resume keys on a per-unit completion marker, which says a unit *finished* but
    nothing about *what it ran*. Point a second configuration at the same column and
    the finished units are silently kept: the column then mixes two configurations
    and reports the mean as if it were one. That is the exact failure mode the
    benchmark is built to catch elsewhere, so the harness must not commit it.

    The check is on the whole parameter dict rather than on a diff, because there is
    no principled way to decide which keys are allowed to differ -- a method that
    reads an undeclared param would slip through any allowlist.
    """
    marker = Path(column_dir) / "column_summary.json"
    if not marker.is_file():
        return
    try:
        previous = json.loads(marker.read_text()).get("params")
    except (json.JSONDecodeError, OSError):
        return  # an unreadable marker is not evidence of a conflict
    if previous is None or previous == dict(params or {}):
        return
    differing = sorted(
        set(previous) | set(params or {}),
        key=lambda k: (previous.get(k) == (params or {}).get(k), k),
    )
    shown = [f"{k}: {previous.get(k)!r} -> {(params or {}).get(k)!r}"
             for k in differing if previous.get(k) != (params or {}).get(k)]
    raise ColumnConflict(
        f"{column_dir} already holds a column run with different parameters:\n    "
        + "\n    ".join(shown)
        + "\n  Resuming would mix the two configurations into one reported mean. "
          "Run the new configuration under its own --tag or --out-root, or delete "
          "the existing column if it is not worth keeping."
    )


class ColumnBusy(Exception):
    """Another process is already writing this column."""


def _claim_column(column_dir: Path) -> Optional[Path]:
    """Take an exclusive lock on a column, or refuse.

    Two launchers pointed at one column interleave their writes into the same
    ``episodes.jsonl`` files, and the result does not look like corruption -- it looks
    like a *result*. It has now happened twice in this repo: once during a method
    retest, and once to a determinism check that was investigating the first, where the
    interleaved data read as planner nondeterminism convincingly enough to be written
    up before the replan counts gave it away.

    The lock is a file created with ``O_EXCL`` holding the owning pid, and it is removed
    on the way out. A lock left behind by a killed process is reported with its pid so
    the next person can check whether it is alive rather than guess.
    """
    column_dir.mkdir(parents=True, exist_ok=True)
    lock = column_dir / ".paarbench_column_lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner = lock.read_text().strip()
        except OSError:
            owner = "unknown"
        alive = ""
        if owner.isdigit():
            try:
                os.kill(int(owner), 0)
                alive = " (that process is still running)"
            except (ProcessLookupError, PermissionError):
                # A killed evaluate.py leaves this lock behind; reclaim it so resume
                # can continue instead of aborting the whole submission.
                lock.unlink(missing_ok=True)
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as fh:
                    fh.write(str(os.getpid()))
                return lock
        if alive:
            raise ColumnBusy(
                f"{column_dir} is locked by pid {owner}{alive}. Two processes writing one "
                f"column interleave their episodes.jsonl writes, and the result reads as a "
                f"result rather than as corruption. Wait for it, or use a different --tag."
            ) from None
        raise ColumnBusy(
            f"{column_dir} is locked by pid {owner}{alive}. Two processes writing one "
            f"column interleave their episodes.jsonl writes, and the result reads as a "
            f"result rather than as corruption. Wait for it, or use a different --tag."
        ) from None
    with os.fdopen(fd, "w") as fh:
        fh.write(str(os.getpid()))
    return lock


def unit_complete(out_dir: Path) -> bool:
    """Did this unit run to completion?

    Resume must key on *completion*, not on the existence of ``logs.json``: that file
    is created on the first replan, so an interrupted unit would otherwise look
    finished and be skipped forever, leaving a truncated episode in the record.
    ``final_eval/success_rate`` is written once, at the end, so it is the real marker.
    """
    return read_success(Path(out_dir) / "logs.json") is not None


def _worker_env(gpu: str) -> Dict[str, str]:
    # Each job forks one env worker per evaluated episode, so concurrent jobs
    # oversubscribe the box badly unless the GPU-bound torch math is held to one
    # thread per process.
    base = dict(os.environ)
    env = dict(
        base,
        WANDB_MODE=base.get("WANDB_MODE", "offline"),
        CUDA_VISIBLE_DEVICES=gpu,
        SDL_VIDEODRIVER="dummy",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    # CI and shared GPU hosts often expose CUDA for PyTorch but not the system
    # EGL/GLEW development stack that legacy mujoco-py needs.  When the optional
    # user-local runtime has been staged, build a software offscreen renderer from
    # it; torch still uses the selected CUDA device for model/planner computation.
    prefix = REPO_ROOT / ".mujoco-build"
    osmesa_lib = prefix / "usr" / "lib" / "x86_64-linux-gnu"
    if (prefix / "bin" / "patchelf").is_file() and (osmesa_lib / "libOSMesa.so").is_file():
        def prepend(value: Union[Path, str], current: str) -> str:
            return f"{value}:{current}" if current else str(value)

        env["PATH"] = prepend(prefix / "bin", env.get("PATH", ""))
        env["C_INCLUDE_PATH"] = prepend(
            prefix / "usr" / "include",
            prepend(prefix / "include", env.get("C_INCLUDE_PATH", "")),
        )
        env["LIBRARY_PATH"] = prepend(
            osmesa_lib,
            prepend(prefix / "lib", env.get("LIBRARY_PATH", "")),
        )
        env["LD_LIBRARY_PATH"] = prepend(
            osmesa_lib,
            prepend(
                prefix / "lib",
                prepend(Path.home() / ".mujoco" / "mujoco210" / "bin",
                        env.get("LD_LIBRARY_PATH", "")),
            ),
        )
        env["MUJOCO_PY_FORCE_CPU"] = "1"
    return env


def build_command(
    setting: Setting,
    seed: int,
    shape: str,
    out_dir: Path,
    n_evals: int,
    *,
    method_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    config_name: str = "eval",
    extra: Optional[List[str]] = None,
    episode_index: Optional[int] = None,
) -> List[str]:
    """The ``plan.py`` invocation for one shape, or for one episode of one shape.

    A method is selected with ``planner.adapter.method=<name>`` and configured with
    ``planner.adapter.params.<k>=<v>``. The planner resolves the name through
    ``paarbench.methods``; it has no knowledge of any specific method.

    With ``episode_index``, this scores episode ``i`` of the ``n_evals``-episode
    cohort in isolation, using that episode's own environment seed. That is what
    ``requires_episode_isolation`` methods get.
    """
    cmd = [
        ".venv/bin/python", "plan.py",
        "--config-name", config_name,
        f"ckpt_base_path={setting.base_path}",
        "model_epoch=latest",
        f"goal_source={setting.goal_source}",
        "+wandb_logging=false",
        f"hydra.run.dir={_portable_command_path(out_dir)}",
        f"seed={seed}",
        f"goal_H={setting.goal_horizon}",
        "planner.sub_planner.opt_steps=100",
        "decode_for_viz=false",
    ]
    target_path = setting.targets_path(shape, seed)
    if setting.goal_source == "segments":
        cmd.append(f"+eval_data_path={target_path}")
    else:
        cmd.append(f"+maze_target_path={target_path}")
    if setting.dataset_path:
        cmd.append(f"dataset_path={setting.dataset_path}")
    if episode_index is None:
        cmd.append(f"n_evals={n_evals}")
    else:
        cmd += [
            "n_evals=1",
            f"eval_episode_index={episode_index}",
            f"eval_episode_total={n_evals}",
        ]
    if method_name is not None:
        cmd.append(f"+planner.adapter.method={method_name}")
        for key, value in (params or {}).items():
            cmd.append(f"+planner.adapter.params.{key}={_hydra_scalar(value)}")
    for key, value in setting.planner_overrides.items():
        cmd.append(f"++{key}={_hydra_scalar(value)}")
    return cmd + list(extra or [])


def _hydra_scalar(value: Any) -> str:
    """Render a Python scalar the way Hydra's command-line parser expects it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        # Hydra reads 5e-4 as a string unless it looks unambiguously like a float.
        return repr(value)
    return str(value)


def run_column(
    setting: Setting,
    cohort: str,
    tag: str,
    *,
    method_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    gpus: Optional[List[str]] = None,
    n_evals: Optional[int] = None,
    out_root: Optional[Path] = None,
    config_name: str = "eval",
    extra: Optional[List[str]] = None,
    resume: bool = True,
    verbose: bool = True,
    episode_isolation: bool = False,
    per_gpu: int = 1,
) -> ColumnResult:
    """Run every shape of ``setting`` at ``cohort``.

    By default one process per shape, each planning the whole cohort as a batch.
    With ``episode_isolation``, one process per *episode* instead -- required by
    methods that own mutable shared state, since a batched gradient step would
    average unrelated episodes into a single correction.

    Resumable: a unit whose ``logs.json`` already exists is skipped unless
    ``resume=False``.
    """
    seed = setting.cohort_seed(cohort)
    n_evals = setting.n_evals if n_evals is None else n_evals
    gpus = gpus or ["0"]
    out_root = Path(out_root) if out_root is not None else DEFAULT_OUT_ROOT
    if not setting.shapes:
        raise ValueError(f"setting {setting.id!r} declares no shapes; nothing to run")

    # Before anything is launched: the goal files define this setting's episodes and are
    # not tracked in git, so a fresh checkout would otherwise start one process per shape
    # and have each die in its own log file with a bare FileNotFoundError.
    setting.require_targets()

    column_dir = out_root / tag / setting.id / cohort
    if resume:
        check_resumable(column_dir, params or {})
    lock = _claim_column(column_dir)
    try:
        return _run_column_locked(
            setting, cohort, tag, column_dir,
            method_name=method_name, params=params, gpus=gpus, n_evals=n_evals,
            config_name=config_name, extra=extra,
            resume=resume, verbose=verbose, episode_isolation=episode_isolation,
            per_gpu=per_gpu, seed=seed,
        )
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


def _run_column_locked(
    setting: Setting,
    cohort: str,
    tag: str,
    column_dir: Path,
    *,
    method_name, params, gpus, n_evals, config_name, extra,
    resume, verbose, episode_isolation, per_gpu, seed,
) -> ColumnResult:
    log_dir = column_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # A unit is (shape, episode_index_or_None, output dir, log name).
    units = []
    for shape in setting.shapes:
        safe = shape.replace("+", "plus")
        if episode_isolation:
            for i in range(n_evals):
                units.append((shape, i, column_dir / safe / f"ep{i:03d}", f"{safe}_ep{i:03d}"))
        else:
            units.append((shape, None, column_dir / safe, safe))
    pending = [u for u in units if not (resume and unit_complete(u[2]))]

    slot_list = [g for _ in range(max(1, per_gpu)) for g in gpus]
    if verbose:
        mode = (f"episode-isolated, {n_evals} processes/shape" if episode_isolation
                else "batched cohort")
        print(f"[column] {setting.id}/{cohort} seed={seed} tag={tag} n_evals={n_evals} "
              f"({mode}; {len(pending)}/{len(units)} units on {len(slot_list)} slots)",
              flush=True)

    slots: "queue.Queue[str]" = queue.Queue()
    for g in slot_list:
        slots.put(g)
    work: "queue.Queue" = queue.Queue()
    for item in pending:
        work.put(item)

    failures: List[str] = []
    done = {"n": 0}
    lock = threading.Lock()
    t0 = time.time()

    def worker() -> None:
        while True:
            try:
                shape, episode_index, out_dir, log_name = work.get_nowait()
            except queue.Empty:
                return
            gpu = slots.get()
            cmd = build_command(
                setting, seed, shape, out_dir, n_evals,
                method_name=method_name, params=params,
                config_name=config_name, extra=extra, episode_index=episode_index,
            )
            # Never leave rc unbound: if the launch itself raises, the unit has to
            # be reported as a failure rather than killing this worker thread with
            # an UnboundLocalError and quietly leaving the rest of the queue unrun.
            rc, launch_error = 1, None
            try:
                # Tee to disk: session interruptions lose captured stdout.
                with open(log_dir / f"{log_name}.log", "w") as fh:
                    fh.write(" ".join(cmd) + "\n\n")
                    fh.flush()
                    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=_worker_env(gpu),
                                         stdout=fh, stderr=subprocess.STDOUT)
            except OSError as exc:
                launch_error = exc
            finally:
                slots.put(gpu)
                work.task_done()
            with lock:
                done["n"] += 1
                if launch_error is not None:
                    failures.append(f"{log_name} (launch failed: {launch_error})")
                elif rc != 0:
                    failures.append(f"{log_name} (rc={rc}, see {log_dir / (log_name + '.log')})")
                # One line per shape is readable; one per episode is not. The failure
                # count rides along because a column that is failing every unit
                # otherwise looks like a column that is running unusually fast.
                if verbose and (episode_index is None or done["n"] % 25 == 0
                                or done["n"] == len(pending)):
                    bad = f", {len(failures)} failed" if failures else ""
                    print(f"[done] {setting.id}/{cohort} {done['n']}/{len(pending)}"
                          f"{bad} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in slot_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if episode_isolation:
        success_by_shape = {}
        for shape in setting.shapes:
            safe = shape.replace("+", "plus")
            scores = [read_success(column_dir / safe / f"ep{i:03d}" / "logs.json")
                      for i in range(n_evals)]
            got = [s for s in scores if s is not None]
            # Only report a shape whose every episode landed: a mean over the
            # subset that happened to finish is a different, biased quantity.
            success_by_shape[shape] = (sum(got) / len(got)
                                       if len(got) == n_evals else None)
    else:
        success_by_shape = {
            shape: read_success(column_dir / shape.replace("+", "plus") / "logs.json")
            for shape in setting.shapes
        }
    result = ColumnResult(
        setting=setting.id, cohort=cohort, seed=seed, tag=tag, n_evals=n_evals,
        success_by_shape=success_by_shape, out_dir=column_dir,
        params=dict(params or {}), failures=failures,
        wall_seconds=time.time() - t0,
    )
    (column_dir / "column_summary.json").write_text(json.dumps(result.to_dict(), indent=2))
    if verbose and failures:
        # A column whose every unit crashes finishes *faster* than a healthy one, so
        # wall-clock is a misleading progress signal and the failure has to be said
        # out loud. Three examples is enough to recognise a shared cause.
        print(f"[FAIL] {setting.id}/{cohort} tag={tag}: {len(failures)}/{len(pending)} "
              f"unit(s) failed in {result.wall_seconds / 60:.1f} min", flush=True)
        for line in failures[:3]:
            print(f"        {line}", flush=True)
        if len(failures) > 3:
            print(f"        ... and {len(failures) - 3} more; see "
                  f"{column_dir / 'column_summary.json'}", flush=True)
    return result
