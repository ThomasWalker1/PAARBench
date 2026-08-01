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
from typing import Any, Dict, List, Optional

from paarbench.settings import Setting

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "eval_outputs"


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

    def to_dict(self) -> dict:
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
            "out_dir": str(self.out_dir),
        }


def read_success(logs_json: Path) -> Optional[float]:
    """Pull ``final_eval/success_rate`` out of a finished shape's logs."""
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


def _worker_env(gpu: str) -> Dict[str, str]:
    # Each job forks one env worker per evaluated episode, so concurrent jobs
    # oversubscribe the box badly unless the GPU-bound torch math is held to one
    # thread per process. env.sh's defaults are inlined so a column does not depend
    # on the caller having sourced it.
    return dict(
        os.environ,
        DATASET_DIR=os.environ.get("DATASET_DIR", "/mnt/richb/tw78/data/datasets"),
        WANDB_MODE=os.environ.get("WANDB_MODE", "offline"),
        CUDA_VISIBLE_DEVICES=gpu,
        SDL_VIDEODRIVER="dummy",
        PYTHONUNBUFFERED="1",
        D4RL_SUPPRESS_IMPORT_ERROR="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )


def build_command(
    setting: Setting,
    seed: int,
    shape: str,
    out_dir: Path,
    n_evals: int,
    *,
    method_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    data_path: Optional[Path] = None,
    config_name: str = "eval",
    extra: Optional[List[str]] = None,
) -> List[str]:
    """The ``plan.py`` invocation for one shape.

    A method is selected with ``planner.adapter.method=<name>`` and configured with
    ``planner.adapter.params.<k>=<v>``. The planner resolves the name through
    ``paarbench.methods``; it has no knowledge of any specific method.
    """
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"), "plan.py",
        "--config-name", config_name,
        f"ckpt_base_path={setting.base_path}",
        "model_epoch=latest",
        "goal_source=segments",
        f"+eval_data_path={setting.targets_path(shape)}",
        "+wandb_logging=false",
        f"hydra.run.dir={out_dir}",
        f"seed={seed}",
        f"n_evals={n_evals}",
        "goal_H=25",
        "planner.sub_planner.opt_steps=100",
        "decode_for_viz=false",
    ]
    if data_path is not None:
        cmd.append(f"dataset_data_path={data_path}")
    if method_name is not None:
        cmd.append(f"+planner.adapter.method={method_name}")
        for key, value in (params or {}).items():
            cmd.append(f"+planner.adapter.params.{key}={_hydra_scalar(value)}")
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
    data_path: Optional[Path] = None,
    config_name: str = "eval",
    extra: Optional[List[str]] = None,
    resume: bool = True,
    verbose: bool = True,
) -> ColumnResult:
    """Run every shape of ``setting`` at ``cohort``, one shape per GPU slot.

    Resumable: a shape whose ``logs.json`` already exists is skipped unless
    ``resume=False``.
    """
    seed = setting.cohort_seed(cohort)
    n_evals = setting.n_evals if n_evals is None else n_evals
    gpus = gpus or ["0"]
    out_root = Path(out_root) if out_root is not None else DEFAULT_OUT_ROOT
    if not setting.shapes:
        raise ValueError(f"setting {setting.id!r} declares no shapes; nothing to run")

    column_dir = out_root / tag / setting.id / cohort
    log_dir = column_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for shape in setting.shapes:
        safe = shape.replace("+", "plus")
        out_dir = column_dir / safe
        if resume and (out_dir / "logs.json").exists():
            continue
        pending.append((shape, safe, out_dir))

    if verbose:
        print(f"[column] {setting.id}/{cohort} seed={seed} tag={tag} n_evals={n_evals} "
              f"({len(pending)}/{len(setting.shapes)} shapes to run on {len(gpus)} gpus)",
              flush=True)

    slots: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        slots.put(g)
    work: "queue.Queue" = queue.Queue()
    for item in pending:
        work.put(item)

    failures: List[str] = []
    lock = threading.Lock()
    t0 = time.time()

    def worker() -> None:
        while True:
            try:
                shape, safe, out_dir = work.get_nowait()
            except queue.Empty:
                return
            gpu = slots.get()
            cmd = build_command(
                setting, seed, shape, out_dir, n_evals,
                method_name=method_name, params=params, data_path=data_path,
                config_name=config_name, extra=extra,
            )
            try:
                # Tee to disk: session interruptions lose captured stdout.
                with open(log_dir / f"{safe}.log", "w") as fh:
                    fh.write(" ".join(cmd) + "\n\n")
                    fh.flush()
                    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=_worker_env(gpu),
                                         stdout=fh, stderr=subprocess.STDOUT)
            finally:
                slots.put(gpu)
                work.task_done()
            with lock:
                if rc != 0:
                    failures.append(f"{shape} (rc={rc}, see {log_dir / (safe + '.log')})")
                if verbose:
                    print(f"[done] {setting.id}/{cohort} {shape} rc={rc} gpu{gpu} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
    return result
