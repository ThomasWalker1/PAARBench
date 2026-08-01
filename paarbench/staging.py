"""Stage training datasets onto tmpfs before a fan-out.

The datasets live on an SMB mount that goes down under a large fan-out (one eval
process per GPU slot, each reading normalization statistics at startup).  The
predecessor project worked around this with per-config ``dataset_data_path``
overrides; docs/PLAN.md §9 asks for it in the harness instead, so every driver
gets the same behaviour without remembering to pass a flag.

A stage is validated against a manifest of the source tree -- relative path, size
and mtime per file -- rather than a hash of 2 GB of tensors, because a full hash
costs more than the copy it is trying to avoid.  A stale stage is therefore
possible if someone rewrites a dataset in place while keeping every file's size
and mtime, which does not happen for these frozen artifacts.

A directory that already holds the right content but carries no manifest (staged
by hand, or copied without preserving mtimes) is *adopted* rather than re-copied,
after checking the path set and every file size.  Adoption deliberately ignores
mtime: a plain ``cp -r`` does not preserve it, and re-copying 2 GB over the slow
mount to fix a timestamp is exactly the cost this module exists to avoid.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
from pathlib import Path

DEFAULT_TMPFS = Path("/dev/shm") / getpass.getuser() / "paarbench" / "data"

_MANIFEST_NAME = ".paarbench_stage.json"


def _manifest(root: Path) -> dict:
    """A cheap fingerprint of a directory tree: relative path -> (size, mtime_ns)."""
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            entries[str(path.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return {"n_files": len(entries), "entries": entries}


def _sizes_match(want: dict, have: dict) -> bool:
    """Do two manifests agree on the path set and every file size, ignoring mtime?

    ``have`` may carry the stage's own manifest file, which the source cannot have.
    """
    a = {k: v[0] for k, v in want["entries"].items()}
    b = {k: v[0] for k, v in have["entries"].items() if k != _MANIFEST_NAME}
    return a == b


def stage_dataset(
    source: os.PathLike | str,
    tmpfs_root: os.PathLike | str | None = None,
    *,
    force: bool = False,
    adopt: bool = True,
    verbose: bool = True,
) -> Path:
    """Copy ``source`` under tmpfs and return the staged path.

    Re-staging is skipped when the existing stage's manifest already matches the
    source, so this is cheap to call unconditionally at the top of a driver.  With
    ``adopt`` (the default), a manifest-less directory whose path set and file
    sizes already match the source is claimed as the stage instead of re-copied.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dataset source is not a directory: {source}")

    root = Path(tmpfs_root) if tmpfs_root is not None else DEFAULT_TMPFS
    dest = root / source.name
    manifest_path = dest / _MANIFEST_NAME

    want = _manifest(source)
    if not force and manifest_path.is_file():
        try:
            have = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            have = None
        if have == want:
            if verbose:
                print(f"[stage] reusing {dest} ({want['n_files']} files)", flush=True)
            return dest

    if not force and adopt and dest.is_dir() and _sizes_match(want, _manifest(dest)):
        if verbose:
            print(f"[stage] adopting existing {dest} ({want['n_files']} files, "
                  f"sizes match source)", flush=True)
        manifest_path.write_text(json.dumps(want))
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[stage] copying {source} -> {dest} ({want['n_files']} files)", flush=True)
    shutil.copytree(source, dest)
    manifest_path.write_text(json.dumps(want))
    return dest


def checkpoint_dataset_path(ckpt_base_path: os.PathLike | str) -> Path | None:
    """Read the training dataset path baked into a checkpoint's ``hydra.yaml``.

    Returns ``None`` when the checkpoint does not record a usable one -- either the
    key is absent, or it holds an unsubstituted placeholder, which some released
    bases ship (``data_path: <path>``). The caller then falls back to the setting's
    declared ``dataset_path``.
    """
    hydra_yaml = Path(ckpt_base_path) / "hydra.yaml"
    if not hydra_yaml.is_file():
        return None
    # Read with omegaconf rather than yaml so that the file's own interpolations
    # do not have to be resolved -- we only want one leaf string.
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(hydra_yaml)
    path = OmegaConf.select(cfg, "env.dataset.data_path")
    if not path:
        return None
    text = str(path)
    # A placeholder looks like a path to Path() but names nothing; treat it as absent
    # rather than letting it surface later as a confusing FileNotFoundError.
    if text.startswith("<") or text in {"???", "null", "None"}:
        return None
    return Path(text)


def resolve_dataset_path(setting, tmpfs_root=None, stage: bool = True,
                         verbose: bool = True) -> Path | None:
    """Where a setting's dataset should be read from, staged if asked.

    Prefers the path recorded on the checkpoint, falls back to the one the setting
    declares. Returns ``None`` when neither is usable, which leaves
    ``dataset_data_path`` unset so the config decides.
    """
    source = checkpoint_dataset_path(setting.base_path)
    if source is None or not source.is_dir():
        declared = Path(setting.dataset_path) if setting.dataset_path else None
        if declared is not None and declared.is_dir():
            if verbose:
                print(f"[stage] {setting.id}: checkpoint records no usable dataset "
                      f"path; using the setting's declared {declared}", flush=True)
            source = declared
        else:
            if verbose:
                print(f"[stage] {setting.id}: no usable dataset path on "
                      f"{setting.base_path}/hydra.yaml and none declared on the "
                      f"setting; leaving dataset_data_path unset", flush=True)
            return None
    return stage_dataset(source, tmpfs_root, verbose=verbose) if stage else source
