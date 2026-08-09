"""Shared helpers for offline method training.

Methods that need a frozen base world model plus its training trajectories (PAD,
HyperJEPA, Static LoRA) load them through here rather than reimplementing Hydra
dataset wiring.  Evaluation continues to go through ``plan.py``; this module is
for *training* adapters and heads only.

Deliberately small: no loss loops, no checkpoint formats, no method-specific
architecture.  Those stay in ``methods/<name>/train*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
from omegaconf import OmegaConf, open_dict

from paarbench.world_model import load_model, load_train_config


@dataclass
class TrainingBundle:
    """Frozen world model, its training config, and sliced train/valid datasets."""

    wm: Any
    train_cfg: Any
    datasets: dict
    traj_datasets: dict
    ckpt_dir: Path
    model_ckpt: Path


def load_training_bundle(
    ckpt_dir,
    *,
    model_epoch: str = "latest",
    data_path: Optional[str] = None,
    n_rollout: Optional[int] = None,
    num_hist: Optional[int] = None,
    num_pred: Optional[int] = None,
    frameskip: Optional[int] = None,
    device: Optional[str] = None,
    freeze: bool = True,
) -> TrainingBundle:
    """Load a base WM and the offline trajectory pool it was trained on.

    ``data_path`` replaces the checkpoint's saved dataset root. Released maze
    configs redact that path as ``<path>``, so maze trainers must pass the staged
    local directory (e.g. ``data/point_maze_medium``).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir).resolve()
    train_cfg = load_train_config(ckpt_dir)
    # Mutate a copy so callers can override redacted / relocated data roots without
    # touching the on-disk hydra.yaml that evaluation also reads.
    train_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=False))
    with open_dict(train_cfg):
        if data_path is not None:
            train_cfg.env.dataset.data_path = str(data_path)
        if n_rollout is not None:
            train_cfg.env.dataset.n_rollout = int(n_rollout)

    hist = int(train_cfg.num_hist if num_hist is None else num_hist)
    pred = int(train_cfg.num_pred if num_pred is None else num_pred)
    skip = int(train_cfg.frameskip if frameskip is None else frameskip)
    datasets, traj_datasets = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=hist,
        num_pred=pred,
        frameskip=skip,
    )

    model_ckpt = ckpt_dir / "checkpoints" / f"model_{model_epoch}.pth"
    if not model_ckpt.is_file():
        available = sorted(p.name for p in (ckpt_dir / "checkpoints").glob("model_*.pth"))
        raise FileNotFoundError(
            f"{model_ckpt} does not exist. available: {available or '(none)'}"
        )
    wm = load_model(
        model_ckpt=model_ckpt,
        train_cfg=train_cfg,
        num_action_repeat=train_cfg.num_action_repeat,
        device=device,
    )
    if freeze:
        for param in wm.parameters():
            param.requires_grad = False
        wm.eval()
    return TrainingBundle(
        wm=wm,
        train_cfg=train_cfg,
        datasets=datasets,
        traj_datasets=traj_datasets,
        ckpt_dir=ckpt_dir,
        model_ckpt=model_ckpt,
    )
