"""Loading a base world model, without going through the evaluation entry point.

``plan.py`` builds a world model as one step of a long Hydra-driven pipeline that also
constructs a dataset, an environment, a planner and an evaluator.  Anything wanting
just the model -- a conformance test, a CI check, an interactive session -- had to run
the whole thing or reimplement it.

This is the extraction.  ``plan.py`` imports ``load_ckpt`` and ``load_model`` from here
so there is exactly one implementation; ``load_world_model`` is the convenience entry
point for everyone else.

Deliberately does **not** build the dataset.  The dataset exists only to supply
normalization statistics for the preprocessor, and reading it costs ~2 GB of tensors --
far more than a test that only needs weights should pay.  Callers needing a
preprocessor should go through ``plan.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def load_ckpt(snapshot_path, device):
    # Instantiated for its side effect: the checkpoint pickles encoder objects whose
    # class must be importable and whose torch.hub cache must be warm before unpickling.
    from models.dino import DinoV2Encoder

    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    with Path(snapshot_path).open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for key, value in payload.items():
        if key in ALL_MODEL_KEYS:
            result[key] = value.to(device) if value is not None else None
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    import hydra

    result = {}
    model_ckpt = Path(model_ckpt)
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = str(Path(__file__).resolve().parent.parent)
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint "
                "and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


def load_train_config(ckpt_base_path):
    """The training config a checkpoint was produced under.

    Load-bearing rather than documentation: it carries the model architecture, the
    dataset location and the environment settings that evaluation must match.
    """
    from omegaconf import OmegaConf

    path = Path(ckpt_base_path) / "hydra.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; a base checkpoint directory must carry the "
            f"hydra.yaml it was trained under. See docs/CHECKPOINTS.md."
        )
    return OmegaConf.load(path)


def load_world_model(ckpt_base_path, model_epoch: str = "latest",
                     device: Optional[str] = None):
    """Load a base world model in evaluation mode.

    Returns the model only. Frozen (``requires_grad=False`` throughout) and in
    ``eval()``, which is the state the planner hands to an adapter -- a method that
    wants gradients turns them on for the parameters it has chosen, exactly as it
    would under ``plan.py``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(ckpt_base_path)
    train_cfg = load_train_config(base)
    checkpoint = base / "checkpoints" / f"model_{model_epoch}.pth"
    if not checkpoint.is_file():
        available = sorted(p.name for p in (base / "checkpoints").glob("model_*.pth"))
        raise FileNotFoundError(
            f"{checkpoint} does not exist. available: {available or '(none)'}"
        )
    model = load_model(checkpoint, train_cfg, train_cfg.num_action_repeat, device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
