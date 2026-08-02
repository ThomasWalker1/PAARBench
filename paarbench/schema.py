"""The canonical evaluation record: one row per episode per replan.

The predecessor project accumulated fourteen output directories in three
incompatible layouts and a loader whose job was to paper over them. This is the
replacement: one schema, written by the planner, read by the metrics.

Written to ``episodes.jsonl`` inside each unit's Hydra run directory by
``planning/mpc.py``. The columns:

    replan              int    0-based MPC iteration
    episode_local       int    index within this process's batch
    success             bool   goal reached at this replan
    cumulative_success  bool   goal reached at this replan or any earlier one
    state_dist          float  final state distance to goal
    visual_dist         float  pixel-space distance to the goal image
    proprio_dist        float  proprioceptive distance to goal
    planner_s           float  per-replan, shared across a batched cohort
    adapt_s             float  per-replan, shared across a batched cohort
    adapt_peak_mb       float  per-replan, shared across a batched cohort

Identity beyond the episode -- method, setting, cohort, shape, global episode
index -- comes from the directory layout rather than being repeated on every row:

    eval_outputs/<tag>/<setting>/<cohort>/<shape>/episodes.jsonl            batched
    eval_outputs/<tag>/<setting>/<cohort>/<shape>/ep<NNN>/episodes.jsonl    isolated

Both layouts produce identical frames, which matters: an episode-isolated method and
a batched one have to be comparable on exactly the same footing, and the isolation
decision belongs to the method rather than to the metric.

Note ``success`` is *absorbing* in the underlying evaluation -- ``planning/mpc.py``
zeroes the actions of an episode that has already succeeded, so its distance is held
at whatever it was on the replan it succeeded. Any distance comparison across methods
has to account for that; see ``paarbench/metrics.py``, which conditions on a fixed
episode set rather than on each method's own outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODES_FILE = "episodes.jsonl"


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a partially-flushed final line on an interrupted run
    return rows


def iter_units(column_dir: Path) -> Iterator[tuple]:
    """Yield ``(shape, episode_offset, path)`` for every unit under a column."""
    for shape_dir in sorted(p for p in column_dir.iterdir() if p.is_dir()):
        if shape_dir.name == "logs":
            continue
        shape = shape_dir.name.replace("plus", "+")
        direct = shape_dir / EPISODES_FILE
        if direct.is_file():
            yield shape, 0, direct
        for ep_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir()):
            if not ep_dir.name.startswith("ep"):
                continue
            path = ep_dir / EPISODES_FILE
            if path.is_file():
                yield shape, int(ep_dir.name[2:]), path


def load_column(column_dir: Path, method: str, setting: str, cohort: str) -> pd.DataFrame:
    """Read every unit of one column into a single frame."""
    frames = []
    for shape, offset, path in iter_units(Path(column_dir)):
        rows = _read_jsonl(path)
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["shape"] = shape
        # Global episode index within the (setting, cohort, shape) cohort. For a
        # batched unit that is the local index; for an isolated one the directory
        # name carries it and the local index is always 0.
        frame["episode"] = offset + frame["episode_local"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["method"] = method
    out["setting"] = setting
    out["cohort"] = cohort
    return out


def load(out_root: Optional[Path] = None, methods=None, setting=None) -> pd.DataFrame:
    """Read every column under ``eval_outputs/`` into one frame.

    Columns produced by a selection rule live under ``<tag>/select<NN>/`` and are
    skipped: they are the method's own tuning, not results to report.  The harness
    may also cache its private frozen reference under ``frozen/<setting>/selection``;
    selection cohorts are never reportable regardless of which arm owns them.
    """
    root = Path(out_root) if out_root is not None else REPO_ROOT / "eval_outputs"
    if not root.is_dir():
        return pd.DataFrame()
    frames = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if methods is not None and method_dir.name not in methods:
            continue
        for setting_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if setting_dir.name.startswith("select"):
                continue
            if setting is not None and setting_dir.name != setting:
                continue
            for cohort_dir in sorted(p for p in setting_dir.iterdir() if p.is_dir()):
                if cohort_dir.name == "selection":
                    continue
                frame = load_column(cohort_dir, method_dir.name,
                                    setting_dir.name, cohort_dir.name)
                if not frame.empty:
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def final_replan(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per episode: its last replan.

    This is the end-of-episode outcome every headline metric is computed on.
    """
    if frame.empty:
        return frame
    keys = ["method", "setting", "cohort", "shape", "episode"]
    idx = frame.groupby(keys)["replan"].idxmax()
    return frame.loc[idx].reset_index(drop=True)


def episode_key(frame: pd.DataFrame) -> pd.Series:
    """A stable per-episode identifier for pairing methods against each other."""
    return (frame["setting"] + "/" + frame["cohort"] + "/"
            + frame["shape"] + "/" + frame["episode"].astype(str))
