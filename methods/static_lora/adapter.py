"""Static LoRA -- a learned correction that ignores the episode.

The control that asks whether *conditioning* buys anything. It installs the same
rank-2 correction on the predictor's last block for every episode, every replan: no
optimizer, no context, nothing read from what the controller observed. Adaptation
in the sense of "the base model was wrong and here is a fix", but not in the sense of
"and the fix depends on this episode".

That makes it the right null for both other arms. If an amortized hypernetwork barely
beats this, its conditioning is decorative; if an online learner barely beats it, the
gradient steps are buying little beyond a better operating point. It is also the
cheapest thing on the board -- the correction is applied once per episode and never
recomputed, so its per-replan adaptation cost is essentially zero.

Written against the protocol directly rather than ported, so it doubles as a check
that a method can be written from scratch without reading the harness.
"""

from pathlib import Path

import torch

from models.hyper_lora import (
    clear_hyper_tensors,
    install_predictor_hyper_targets,
)
from paarbench.adapter import BaseWeightGuard


class StaticLoRAAdapter:
    def __init__(self, wm, preprocessor, checkpoint_path,
                 target_scope="predlast_all", rank=2, lora_scale=1.0):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.rank = int(rank)
        self.target_scope = str(target_scope)

        for param in self.wm.parameters():
            param.requires_grad = False

        self.lora_targets, self.norm_targets = install_predictor_hyper_targets(
            self.wm.predictor,
            scope=self.target_scope,
            rank=self.rank,
            scale=float(lora_scale),
        )
        self._correction = self._load(checkpoint_path)
        self._guard = BaseWeightGuard(wm)
        self._applied = False

    def _load(self, checkpoint_path):
        """Read the trained correction and check it fits the targets we installed.

        Shapes are checked eagerly. A rank or scope mismatch would otherwise surface
        as a broadcasting error deep inside a planner rollout, or -- worse -- not at
        all, if the shapes happen to line up.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"static LoRA checkpoint {path} does not exist; see docs/CHECKPOINTS.md"
            )
        payload = torch.load(path, map_location=self.device)
        correction = payload.get("static_lora")
        if not correction:
            raise ValueError(
                f"{path} carries no 'static_lora' entry. This method needs a checkpoint "
                f"trained with static_lora=true; a hypernetwork checkpoint stores a "
                f"generator instead and belongs to the hyperlora method."
            )

        expected = {t.name for t in self.lora_targets} | {t.name for t in self.norm_targets}
        missing = expected - set(correction)
        if missing:
            raise ValueError(
                f"{path} is missing correction(s) for {sorted(missing)}; it was trained "
                f"for a different target_scope than {self.target_scope!r}"
            )
        for target in self.lora_targets:
            entry = correction[target.name]
            if tuple(entry["A"].shape) != (target.spec.rank, target.spec.in_features):
                raise ValueError(
                    f"{path}: {target.name} A is {tuple(entry['A'].shape)}, expected "
                    f"{(target.spec.rank, target.spec.in_features)} -- checkpoint rank "
                    f"does not match rank={self.rank}"
                )
        return {k: {kk: (vv.to(self.device) if torch.is_tensor(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in correction.items()}

    def _apply(self, batch: int):
        """Install the correction, broadcast across the batch.

        The LoRA modules expect a per-episode correction with a leading batch
        dimension, because a conditioned method emits a different one per episode.
        This one expands the same tensors instead -- which is precisely what makes it
        the unconditioned control.
        """
        for target in self.lora_targets:
            entry = self._correction[target.name]
            target.module.set_lora(
                entry["A"].unsqueeze(0).expand(batch, -1, -1),
                entry["B"].unsqueeze(0).expand(batch, -1, -1),
            )
        for target in self.norm_targets:
            entry = self._correction[target.name]
            target.module.set_affine(
                entry["delta_weight"].unsqueeze(0).expand(batch, -1),
                entry["delta_bias"].unsqueeze(0).expand(batch, -1),
            )
        self._applied = True

    @staticmethod
    def _batch_size(obs):
        for value in (obs or {}).values():
            if hasattr(value, "shape") and len(value.shape):
                return int(value.shape[0])
        return None

    # -- TestTimeAdapter protocol ---------------------------------------------

    def on_episode_start(self, obs_0, goal):
        """Clear the previous episode's correction; defer applying to before_plan.

        Deferred because the correction has to be broadcast to the batch, and the
        batch size comes from a real observation -- which this hook is not guaranteed
        to receive.
        """
        clear_hyper_tensors(self.lora_targets, self.norm_targets)
        self._guard.restore()
        self._applied = False
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Nothing to observe: the correction does not depend on the episode."""
        return {}

    def before_plan(self, obs):
        """Apply once per episode. Re-applying every replan would be identical work."""
        if self._applied:
            return {}
        batch = self._batch_size(obs)
        if batch is None:
            return {}
        self._apply(batch)
        return {
            "static_lora/applied": 1,
            "static_lora/num_lora_modules": len(self.lora_targets),
            "static_lora/num_layer_norm_modules": len(self.norm_targets),
        }

    def metrics(self):
        return {"static_lora/applied": float(self._applied)}
