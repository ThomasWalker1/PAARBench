"""LN-Recal -- closed-form recalibration of the predictor's output LayerNorm.

The premise is that the base world model's one-step latent prediction is *miscalibrated*
rather than wrong: under a shifted shape or appearance its predictions land in the right
region of latent space with a systematic per-feature offset and gain error. If that is
true, the correction does not need gradients at all. The predictor ends in a LayerNorm,
so the last thing that happens to every predicted latent is an affine map

    y = LayerNorm(x) * w + b

and the least-squares affine that best maps the *normalized* predictor features onto the
next latents actually observed this episode has a closed form. LN-Recal solves that
2-parameter-per-feature ridge regression at every replan and installs the result as a
per-episode delta on ``w`` and ``b``.

What that buys, by construction rather than by tuning:

* **No gradients, no optimizer, no trained weights.** Nothing is backpropagated and
  nothing is loaded from disk. The method is one 2x2 solve per feature per episode.
* **Memoryless.** The fit is recomputed from the frozen affine every replan, from
  statistics that do not depend on any previously installed correction (the correction
  acts on the LayerNorm *output*; the captured pre-norm activations and the encoded
  targets are both invariant to it). There is no trajectory to drift, which is what the
  benchmark's compounding-slope metric measures.
* **Batched.** Every quantity carries a leading episode axis and the emitted delta is
  per-episode, so this method does *not* need ``requires_episode_isolation`` -- it holds
  no shared mutable state. Cohort evaluation is one process per shape.
* **Bounded by one interpretable knob.** ``ridge`` shrinks the fit toward the pretrained
  affine in units of the sample count: ``ridge -> 0`` is the unregularized refit,
  ``ridge -> inf`` is the frozen model exactly. The selected value therefore says how
  much of this recalibration the episode's own evidence actually supports.

Only the visual and proprioceptive feature blocks are fitted. The trailing action block
of the concatenated latent is overwritten from the action encoder at every rollout step
(``VWorldModel.replace_actions_from_z``), so there is no target for it and its delta is
held at zero -- the same feature range the training loss and every other method here
ignores.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from models.hyper_lora import NormTarget
from models.lora import AffineSpec, replace_layer_norm_with_hyper
from utils import move_to_device

from paarbench.adapter import BaseWeightGuard

log = logging.getLogger(__name__)

OUTPUT_NORM = "transformer.norm"
"""The predictor's final LayerNorm: the last op applied to every predicted latent."""


def _concat_time(initial, later):
    """Join a one-frame observation to a rollout slice, for numpy or torch."""
    if isinstance(initial, torch.Tensor):
        return torch.cat([initial, later], dim=1)
    if isinstance(initial, np.ndarray):
        return np.concatenate([initial, later], axis=1)
    raise TypeError(f"Unsupported observation type: {type(initial)}")


class LNRecalAdapter:
    """Ridge-regularized least-squares refit of the predictor's output LayerNorm."""

    def __init__(
        self,
        wm,
        preprocessor,
        buffer_size=20,
        ridge=1.0,
        fit_weight=True,
        min_transitions=1,
        log_prefix="ln_recal",
        **kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.buffer_size = int(buffer_size)
        self.ridge = float(ridge)
        self.fit_weight = bool(fit_weight)
        self.min_transitions = int(min_transitions)
        self.log_prefix = str(log_prefix)

        if self.buffer_size <= 0:
            raise ValueError("LN-Recal buffer_size must be positive")
        if self.min_transitions <= 0:
            raise ValueError("LN-Recal min_transitions must be positive")
        if not self.ridge > 0.0:
            # ridge == 0 is not merely aggressive, it is ill-posed: with a handful of
            # samples per feature the normal equations can be singular, and the
            # regularizer is what keeps the 2x2 determinant strictly positive.
            raise ValueError(
                "LN-Recal ridge must be > 0; it is the shrinkage toward the pretrained "
                "affine, in units of the sample count. Use a small value (1e-3) for a "
                "nearly unregularized refit."
            )
        if int(getattr(wm, "concat_dim", 1)) != 1:
            raise ValueError(
                "LN-Recal reads the predicted latent's feature blocks by slicing the "
                f"last dimension, which assumes concat_dim=1; this base model has "
                f"concat_dim={getattr(wm, 'concat_dim', None)}."
            )

        # Wrap the output LayerNorm so a per-episode affine delta can be installed.
        # The wrapper is structural and stays for the adapter's lifetime, exactly as
        # HyperJEPA's LoRA wrappers do; what a reset clears is the delta inside it.
        self.norm = replace_layer_norm_with_hyper(self.wm.predictor, OUTPUT_NORM)
        self.norm_targets = [
            NormTarget(
                name=OUTPUT_NORM,
                module=self.norm,
                spec=AffineSpec(name=OUTPUT_NORM, features=self.norm.features),
            )
        ]
        self.lora_targets = []
        self.features = int(self.norm.features)
        self.action_features = int(getattr(self.wm, "action_dim", 0))
        self.fit_features = self.features - self.action_features
        if self.fit_features <= 0:
            raise ValueError(
                f"LN-Recal found {self.features} predictor features of which "
                f"{self.action_features} are action features; nothing left to fit."
            )

        # Reading the pre-norm activations via a hook rather than re-implementing the
        # predictor's forward: the fit has to use exactly the tensor the deployed
        # LayerNorm sees, and a re-implementation would silently diverge if the
        # predictor changed.
        self._capturing = False
        self._captured = None
        self.norm.register_forward_pre_hook(self._capture_pre_norm)

        # Snapshot after installing the wrapper, so the guard covers the module names
        # the wrapper introduced. This method never writes to a base tensor, so the
        # restore is a no-op in the intended path -- it is here because the contract is
        # "undo everything you can reach", and a guard that costs one host-side copy per
        # episode is cheaper than trusting that claim.
        self._guard = BaseWeightGuard(self.wm)

        self.buffer = deque(maxlen=self.buffer_size)
        self._observed = 0
        self._fits = 0
        self._last_logs = {}

    # -- internals ------------------------------------------------------------

    def _capture_pre_norm(self, _module, args):
        if self._capturing:
            self._captured = args[0].detach()

    def _prepare_obs(self, obs):
        return move_to_device(self.preprocessor.transform_obs(obs), self.device)

    @torch.no_grad()
    def _transition_samples(self, obs_src, action, obs_tgt):
        """One (normalized feature, observed target) pair per predicted token.

        Returns ``(n, target)`` with shapes ``(N, tokens, features)`` and
        ``(N, tokens, fit_features)``. Neither depends on any installed correction:
        ``n`` is taken *before* the affine and the target is an encoder output.
        """
        z_src = self.wm.encode(obs_src, action)
        # Take the correction out for the duration of this forward pass. Two reasons,
        # and the second one is load-bearing: the samples must be invariant to whatever
        # is currently installed, and this pass batches B*steps transitions while an
        # installed per-episode delta has batch B, which HyperLayerNorm would reject.
        installed = (self.norm.affine_delta_weight, self.norm.affine_delta_bias)
        self.norm.clear_affine()
        self._capturing = True
        self._captured = None
        try:
            self.wm.predict(z_src)
        finally:
            self._capturing = False
            if installed[0] is not None:
                self.norm.set_affine(*installed)
        pre_norm = self._captured
        self._captured = None
        if pre_norm is None:
            raise RuntimeError(
                f"LN-Recal never saw {OUTPUT_NORM}: the predictor did not call its "
                "output LayerNorm, so there is nothing to recalibrate."
            )
        normalized = F.layer_norm(
            pre_norm, (self.features,), None, None, self.norm.base.eps
        )

        z_tgt = self.wm.encode_obs(obs_tgt)
        visual = z_tgt["visual"][:, -1]                       # (N, tokens, enc_dim)
        proprio = z_tgt["proprio"][:, -1].unsqueeze(1)        # (N, 1, proprio_dim)
        proprio = proprio.expand(-1, visual.shape[1], -1)
        target = torch.cat([visual, proprio], dim=-1)
        if target.shape[-1] != self.fit_features:
            raise RuntimeError(
                f"LN-Recal expected {self.fit_features} fittable features, but the "
                f"encoded target has {target.shape[-1]}."
            )
        if normalized.shape[:2] != target.shape[:2]:
            raise RuntimeError(
                f"LN-Recal token mismatch: predictor emitted "
                f"{tuple(normalized.shape[:2])}, encoder target is "
                f"{tuple(target.shape[:2])}."
            )
        return normalized, target

    @torch.no_grad()
    def _fit(self):
        """Solve the per-episode, per-feature ridge regression and install the delta."""
        normalized = torch.cat([item[0] for item in self.buffer], dim=1)
        target = torch.cat([item[1] for item in self.buffer], dim=1)
        n_samples = normalized.shape[1]
        x = normalized[..., : self.fit_features]
        weight = self.norm.base.weight[: self.fit_features].to(x.dtype)
        bias = self.norm.base.bias[: self.fit_features].to(x.dtype)

        # Shrinkage in units of the sample count, so `ridge` means the same thing
        # whatever the buffer holds: ridge=1 weighs the pretrained affine as much as
        # all observed samples together.
        lam = self.ridge * float(n_samples)
        s_x = x.sum(dim=1)
        s_y = target.sum(dim=1)
        if self.fit_weight:
            s_xx = (x * x).sum(dim=1)
            s_xy = (x * target).sum(dim=1)
            a11 = s_xx + lam
            a22 = float(n_samples) + lam
            a12 = s_x
            r1 = s_xy + lam * weight
            r2 = s_y + lam * bias
            # Strictly positive: s_xx * n >= s_x^2 by Cauchy-Schwarz, and lam > 0.
            det = a11 * a22 - a12 * a12
            gamma = (a22 * r1 - a12 * r2) / det
            beta = (a11 * r2 - a12 * r1) / det
        else:
            gamma = weight.expand_as(s_x)
            beta = (s_y - weight * s_x + lam * bias) / (float(n_samples) + lam)

        delta_weight = torch.zeros(
            x.shape[0], self.features, device=x.device, dtype=x.dtype
        )
        delta_bias = torch.zeros_like(delta_weight)
        delta_weight[:, : self.fit_features] = gamma - weight
        delta_bias[:, : self.fit_features] = beta - bias
        self.norm.set_affine(delta_weight, delta_bias)
        self._fits += 1

        # Did the fit reduce the quantity it fits? A ratio near 1 means the episode's
        # evidence does not support recalibration at this ridge, which is the honest
        # signal to log -- the method is then deliberately close to frozen.
        base_residual = (weight * x + bias - target).pow(2).mean()
        fit_residual = (
            gamma.unsqueeze(1) * x + beta.unsqueeze(1) - target
        ).pow(2).mean()
        # Over every affine feature, of which the action block is held at zero: a fixed
        # 1.2% dilution of the RMS on these bases, and comparable across cells, which
        # matters more for a diagnostic than exactness.
        base_scale = torch.cat([weight, bias]).pow(2).mean().sqrt().clamp_min(1e-12)
        delta_scale = torch.cat([delta_weight, delta_bias], dim=-1).pow(2).mean().sqrt()
        # How much does the emitted correction differ *between* episodes? ~0 would mean
        # the fit is picking up something shared rather than episode-specific.
        dispersion = float("nan")
        if delta_weight.shape[0] > 1:
            stacked = torch.cat([delta_weight, delta_bias], dim=-1)
            dispersion = float(
                (stacked.std(dim=0).mean() / delta_scale.clamp_min(1e-12)).cpu()
            )
        return {
            f"{self.log_prefix}/applied": 1,
            f"{self.log_prefix}/samples": float(n_samples),
            f"{self.log_prefix}/buffer_transitions": float(len(self.buffer)),
            f"{self.log_prefix}/delta_rms_rel": float((delta_scale / base_scale).cpu()),
            f"{self.log_prefix}/delta_dispersion": dispersion,
            f"{self.log_prefix}/fit_residual_ratio": float(
                (fit_residual / base_residual.clamp_min(1e-12)).cpu()
            ),
            f"{self.log_prefix}/base_residual": float(base_residual.cpu()),
            f"{self.log_prefix}/fits": float(self._fits),
        }

    # -- TestTimeAdapter protocol ---------------------------------------------

    def on_episode_start(self, obs_0, goal):
        """Clear the installed affine delta and every buffered statistic."""
        self._guard.restore()
        self.norm.clear_affine()
        self.buffer.clear()
        self._capturing = False
        self._captured = None
        self._observed = 0
        self._fits = 0
        self._last_logs = {}
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Turn the executed chunk into one regression sample per executed action.

        The rollout is at simulator resolution, so each planner action's target is the
        frame ``frameskip`` steps later. Pairing every action with the *chunk's* final
        observation instead would fit a one-step affine against a multi-step outcome.
        """
        steps = int(actions.shape[1])
        frameskip = int(frameskip)
        if actions.dim() != 3 or steps < 1:
            raise ValueError(
                f"LN-Recal expected executed actions (B, T, D), got {tuple(actions.shape)}"
            )
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        required = 1 + steps * frameskip
        if any(value.shape[1] < required for value in rollout_obs.values()):
            raise ValueError(
                f"LN-Recal needs {required} rollout frames for {steps} actions at "
                f"frameskip {frameskip}"
            )
        if any(value.shape[1] != 1 for value in obs_0.values()):
            raise ValueError("LN-Recal expects a one-frame current observation")

        source = {
            key: _concat_time(
                obs_0[key], value[:, frameskip : steps * frameskip : frameskip]
            )
            for key, value in rollout_obs.items()
        }
        target = {
            key: value[:, frameskip : (steps + 1) * frameskip : frameskip]
            for key, value in rollout_obs.items()
        }
        batch = int(actions.shape[0])
        flat_source = {
            key: value.reshape(batch * steps, 1, *value.shape[2:])
            for key, value in source.items()
        }
        flat_target = {
            key: value.reshape(batch * steps, 1, *value.shape[2:])
            for key, value in target.items()
        }
        flat_actions = actions.reshape(batch * steps, 1, actions.shape[-1]).to(self.device)

        normalized, observed = self._transition_samples(
            self._prepare_obs(flat_source), flat_actions, self._prepare_obs(flat_target)
        )
        tokens = normalized.shape[1]
        normalized = normalized.reshape(batch, steps, tokens, self.features)
        observed = observed.reshape(batch, steps, tokens, self.fit_features)
        for step in range(steps):
            # One buffer entry per executed action, each (B, tokens, features), so the
            # window is measured in transitions and the fit stays per-episode.
            self.buffer.append((normalized[:, step], observed[:, step]))
        self._observed += steps
        return {f"{self.log_prefix}/observed_transitions": float(self._observed)}

    def before_plan(self, obs):
        """Refit from scratch and install the correction. No accumulation."""
        if len(self.buffer) < self.min_transitions:
            # Nothing executed yet (or not enough): stay exactly frozen.
            self.norm.clear_affine()
            logs = {
                f"{self.log_prefix}/applied": 0,
                f"{self.log_prefix}/buffer_transitions": float(len(self.buffer)),
            }
        else:
            logs = self._fit()
        logs[f"{self.log_prefix}/ridge"] = self.ridge
        logs[f"{self.log_prefix}/fit_weight"] = float(self.fit_weight)
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
