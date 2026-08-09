"""HOVER -- horizon-matched online adaptation with a fresh-evidence brake.

Same correction as AdaJEPA -- a rank-2 LoRA (plus the final LayerNorm's affine
delta) on the predictor's last block, fitted online by AdamW on the world model's own
prediction loss -- and three deliberate differences from it, each aimed at something
the leaderboard already measures.

**1. The loss is the rollout the planner actually solves through.**  AdaJEPA fits
single-step prediction: encode the state the chunk started from, predict once, match
the state that came out.  The planner does not use the predictor that way.  It rolls
it forward ``goal_H / frameskip`` model steps open loop, feeding each prediction back
in as the next step's context, and scores the *end* of that rollout against the goal.
A predictor that is accurate for one step and drifts over five is exactly a predictor
whose plans are wrong, and single-step adaptation cannot see that drift.  HOVER rolls
the correction forward ``horizon`` steps under the actions that were really executed,
against the states the environment really reached, accumulating loss at every step --
i.e. it fits open-loop rollout consistency, mirroring ``VWorldModel.rollout``
including its growing ``num_hist`` context window.  ``horizon: 1`` recovers a
single-step objective, so the selection grid contains that arm rather than assuming
the multi-step one is better.

**2. The update is braked by evidence it has not seen.**  AdaJEPA's correction is an
optimizer trajectory that only ever accumulates: on PushObj it ends more than 2x
further from the goal than frozen on 17% of episodes, and its paired distance gap
grows with replan index.  Nothing in the method can notice.  HOVER scores its current
correction against the frozen predictor on the transition that *just arrived* -- a
sample the previous update was fitted before, so this is out-of-sample by
construction, needs no held-out split, and needs no episode outcome.  When the
correction is worse than frozen by more than ``brake_ratio`` there, the delta is reset
to zero and the optimizer is rebuilt: adaptation restarts from the frozen model
instead of compounding a correction that fresh evidence says is hurting.

**3. Target latents are cached, so a gradient step is predictor-only.**  The encoder
is frozen here (only predictor LoRA parameters are trained), so every encoding in the
buffer is a constant.  AdaJEPA re-encodes its whole buffer inside every gradient step
-- ``steps * buffer_size`` encoder forwards per replan.  HOVER encodes each transition
once, when it arrives (two forwards per replan, independent of ``steps``), and a
gradient step touches the predictor only.

What HOVER does *not* change: the parameterization (rank-2 LoRA on
``predlast_all`` plus that block's LayerNorm), the optimizer, the gradient clipping,
and the two tunable axes' cost.  The comparison against AdaJEPA is meant to be about
the objective and the brake, not about capacity.

Determinism: the only stochastic element is the LoRA ``A`` initialization, drawn from
a CPU generator seeded by the declared ``lora_init_seed``.  ``B`` starts at zero, so
the correction starts at exactly zero and the first replan of an episode plans with
the frozen model.
"""

from __future__ import annotations

import contextlib
import logging
from collections import deque

import torch

from models.hyper_lora import install_predictor_hyper_targets
from paarbench.adapter import BaseWeightGuard
from utils import move_to_device

log = logging.getLogger(__name__)

_EPS = 1e-12


class HoverAdapter:
    """Online predictor LoRA fitted to multi-step rollouts, with a reset brake."""

    def __init__(
        self,
        wm,
        preprocessor,
        steps=5,
        pred_lr=2.0e-3,
        horizon=3,
        buffer_size=8,
        target_scope="predlast_all",
        lora_rank=2,
        lora_scale=1.0,
        lora_init_seed=0,
        grad_clip_norm=1.0,
        brake_ratio=1.25,
        refresh_interval=1,
        require_single_episode=True,
        log_prefix="hover",
        **kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.steps = int(steps)
        self.pred_lr = float(pred_lr)
        self.horizon = int(horizon)
        self.buffer_size = int(buffer_size)
        self.target_scope = str(target_scope)
        self.lora_rank = int(lora_rank)
        self.lora_scale = float(lora_scale)
        self.lora_init_seed = int(lora_init_seed)
        self.grad_clip_norm = grad_clip_norm
        self.brake_ratio = float(brake_ratio)
        self.refresh_interval = int(refresh_interval)
        self.require_single_episode = bool(require_single_episode)
        self.log_prefix = str(log_prefix)

        if self.steps < 0:
            raise ValueError("HOVER steps must be non-negative.")
        if self.horizon < 1:
            raise ValueError("HOVER horizon must be at least 1 model step.")
        if self.buffer_size < 1:
            raise ValueError("HOVER buffer_size must be positive.")
        if self.refresh_interval < 1:
            raise ValueError("HOVER refresh_interval must be positive.")
        if self.brake_ratio <= 1.0:
            raise ValueError(
                "HOVER brake_ratio must exceed 1: it is the factor by which the "
                "correction may be worse than frozen on fresh evidence before the "
                "delta is reset, so brake_ratio <= 1 would reset on noise alone."
            )
        if self.lora_rank <= 0:
            raise ValueError("HOVER lora_rank must be positive.")

        self.num_hist = max(int(getattr(wm, "num_hist", 1) or 1), 1)
        self.concat_dim = int(getattr(wm, "concat_dim", 1))
        self.action_dim = int(getattr(wm, "action_dim", 0))

        self.lora_targets = []
        self.norm_targets = []
        self.lora_parameters = torch.nn.ParameterList()
        self._trainable = []
        self._zeroable = []
        self.optimizer = None
        if self.steps > 0:
            self._install_correction()
            self.optimizer = self._new_optimizer()

        # Snapshot AFTER installing the correction, so the guard covers the LoRA
        # parameters as well as the base weights: restoring it returns A to its seeded
        # initialization and B to zero, i.e. a correction of exactly zero.
        self._guard = BaseWeightGuard(wm)

        # Per-episode state; all of it is reset in on_episode_start.
        self._buffer = deque(maxlen=self.buffer_size)
        self._observed = 0
        self._adapted = False
        self._brakes = 0
        self._last_logs = {}

    # -- correction ------------------------------------------------------------

    def _install_correction(self):
        for param in self.wm.parameters():
            param.requires_grad = False

        self.lora_targets, self.norm_targets = install_predictor_hyper_targets(
            self.wm.predictor,
            scope=self.target_scope,
            rank=self.lora_rank,
            scale=self.lora_scale,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.lora_init_seed)
        for target in self.lora_targets:
            spec = target.spec
            A = torch.empty(spec.rank, spec.in_features)
            torch.nn.init.normal_(
                A, mean=0.0, std=1.0 / max(spec.in_features, 1) ** 0.5,
                generator=generator,
            )
            A = torch.nn.Parameter(A.to(self.device))
            B = torch.nn.Parameter(
                torch.zeros(spec.out_features, spec.rank, device=self.device)
            )
            target.module.set_lora(A, B)
            self.lora_parameters.extend([A, B])
            self._trainable.extend([A, B])
            # B alone carries the correction's magnitude: B = 0 means a delta of
            # exactly zero whatever A holds, so a brake need not touch A.
            self._zeroable.append(B)
        for target in self.norm_targets:
            delta_weight = torch.nn.Parameter(
                torch.zeros(target.spec.features, device=self.device)
            )
            delta_bias = torch.nn.Parameter(
                torch.zeros(target.spec.features, device=self.device)
            )
            target.module.set_affine(delta_weight, delta_bias)
            self.lora_parameters.extend([delta_weight, delta_bias])
            self._trainable.extend([delta_weight, delta_bias])
            self._zeroable.extend([delta_weight, delta_bias])

        if not self._trainable:
            raise ValueError(
                f"No correction parameters for HOVER target_scope='{self.target_scope}'."
            )
        log.info(
            "HOVER installed %s LoRA modules and %s LayerNorm modules "
            "(rank=%s, trainable tensors=%s).",
            len(self.lora_targets), len(self.norm_targets), self.lora_rank,
            len(self._trainable),
        )

    def _new_optimizer(self):
        return torch.optim.AdamW([{"params": self._trainable, "lr": self.pred_lr}])

    def _reset_correction(self):
        """Return to the frozen predictor and drop the optimizer trajectory."""
        with torch.no_grad():
            for tensor in self._zeroable:
                tensor.zero_()
        self.optimizer = self._new_optimizer()
        self._adapted = False

    @contextlib.contextmanager
    def _frozen_predictor(self):
        """Evaluate as if no correction were installed, then put it back.

        The parameters are detached from the wrappers rather than zeroed: zeroing and
        restoring would have to round-trip through a copy, and the optimizer holds
        references to these exact objects.
        """
        saved_lora = [
            (t.module, t.module.lora_A, t.module.lora_B) for t in self.lora_targets
        ]
        saved_norm = [
            (t.module, t.module.affine_delta_weight, t.module.affine_delta_bias)
            for t in self.norm_targets
        ]
        try:
            for target in self.lora_targets:
                target.module.clear_lora()
            for target in self.norm_targets:
                target.module.clear_affine()
            yield
        finally:
            for module, A, B in saved_lora:
                module.set_lora(A, B)
            for module, weight, bias in saved_norm:
                module.set_affine(weight, bias)

    # -- observation -----------------------------------------------------------

    def _transform(self, obs):
        return move_to_device(self.preprocessor.transform_obs(obs), self.device)

    def _boundary_frames(self, rollout_obs, chunk_len, frameskip):
        """The ``chunk_len + 1`` observations bounding the executed model actions.

        Indexed backwards from the end of the rollout.  ``rollout_obs`` is documented
        as ``1 + T * frameskip`` frames, but the evaluator replays the episode's whole
        action sequence from its initial condition on every replan, so in practice it
        is the entire trajectory so far and only its *tail* belongs to the chunk that
        was just executed.  Counting from the end is correct under either reading;
        counting from the front silently pairs the current state with the endpoint of
        the episode's first action.
        """
        frames = min(int(value.shape[1]) for value in rollout_obs.values())
        last = frames - 1
        first = last - chunk_len * int(frameskip)
        if first < 0:
            raise ValueError(
                f"HOVER needs {1 + chunk_len * frameskip} rollout frames to bound "
                f"{chunk_len} executed action(s) at frameskip {frameskip}, got {frames}."
            )
        return [
            self._transform({k: v[:, idx: idx + 1] for k, v in rollout_obs.items()})
            for idx in range(first, last + 1, int(frameskip))
        ]

    def _append(self, obs_before, action, obs_after):
        """Cache one executed model step as (latent input, latent target).

        Both are constants for the rest of the episode: only predictor LoRA
        parameters are trained, so the encoder, the proprio encoder and the action
        encoder that produced them never move.
        """
        with torch.no_grad():
            z_in = self.wm.encode(obs_before, action).detach()
            z_tgt = {k: v.detach() for k, v in self.wm.encode_obs(obs_after).items()}
        self._buffer.append({"z_in": z_in, "z_tgt": z_tgt})

    # -- losses ----------------------------------------------------------------

    def _graft_action(self, z_pred, z_ref):
        """Replace a prediction's action slots with the next executed action's.

        The same substitution ``VWorldModel.replace_actions_from_z`` makes during a
        rollout, but reading the already-encoded action out of a cached latent instead
        of re-running the action encoder, and out of place so the autograd graph
        through the predicted tokens stays intact.
        """
        if self.concat_dim == 1:
            if self.action_dim <= 0:
                return z_pred
            return torch.cat(
                [z_pred[..., : -self.action_dim], z_ref[..., -self.action_dim:]], dim=-1
            )
        return torch.cat([z_pred[:, :, :-1, :], z_ref[:, :, -1:, :]], dim=2)

    def _step_loss(self, z_pred_frame, targets):
        z_obs, _ = self.wm.separate_emb(z_pred_frame)
        visual = torch.cat([t["visual"] for t in targets], dim=0)
        proprio = torch.cat([t["proprio"] for t in targets], dim=0)
        return (self.wm.emb_criterion(z_obs["visual"], visual)
                + self.wm.emb_criterion(z_obs["proprio"], proprio))

    def _rollout_loss(self, horizon):
        """Open-loop rollout loss over every window the buffer supports.

        Window ``i`` starts from buffered state ``i`` and is rolled forward until it
        runs out of observed states, so at depth ``j`` the surviving windows are
        exactly ``i <= n - 1 - j`` -- a prefix of the batch, which is why the windows
        can share one batched predictor call per depth and be trimmed by slicing.
        Every window contributes a single-step term at ``j = 0``, so the single-step
        objective is nested inside this one rather than replaced by it.
        """
        items = list(self._buffer)
        n = len(items)
        batch = items[0]["z_in"].shape[0]
        depth = max(min(int(horizon), n), 1)

        z = torch.cat([item["z_in"] for item in items], dim=0)
        total = None
        depths = 0
        for j in range(depth):
            width = n - j
            if width <= 0:
                break
            z = z[: width * batch]
            z_pred = self.wm.predict(z[:, -self.num_hist:])
            z_new = z_pred[:, -1:]
            step_loss = self._step_loss(
                z_new, [items[i + j]["z_tgt"] for i in range(width)]
            )
            total = step_loss if total is None else total + step_loss
            depths += 1
            if j + 1 >= depth or width - 1 <= 0:
                break
            kept = (width - 1) * batch
            refs = torch.cat([items[i + j + 1]["z_in"] for i in range(width - 1)], dim=0)
            z = torch.cat(
                [z[:kept], self._graft_action(z_new[:kept], refs)], dim=1
            )
        return total / max(depths, 1), depths

    @torch.no_grad()
    def _fresh_step_loss(self):
        """Single-step loss on the newest transition, whatever is installed.

        Single-step even though the fit is multi-step: this transition arrived after
        the last update, so both its input and its target are unseen, and it is the
        only quantity here for which that is true. A deeper rollout ending at the
        newest state would have to start inside the buffer and pass through states the
        last update was fitted on, biasing the ratio in the adapted arm's favour --
        the direction in which the brake fails to fire.
        """
        item = self._buffer[-1]
        z_pred = self.wm.predict(item["z_in"][:, -self.num_hist:])
        return float(self._step_loss(z_pred[:, -1:], [item["z_tgt"]]).detach().cpu())

    def _fit(self):
        logs = {}
        loss_value = 0.0
        depths = 0
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, depths = self._rollout_loss(self.horizon)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable, float(self.grad_clip_norm)
                )
            self.optimizer.step()
            loss_value = float(loss.detach().cpu())
        if self.steps > 0:
            self._adapted = True
        logs[f"{self.log_prefix}/loss"] = loss_value
        logs[f"{self.log_prefix}/rollout_depth"] = float(depths)
        return logs

    # -- TestTimeAdapter protocol ---------------------------------------------

    def on_episode_start(self, obs_0, goal):
        """Restore the frozen model and drop every trace of the previous episode.

        ``BaseWeightGuard`` was constructed after the correction was installed, so a
        restore also returns A to its seeded initialization and B to zero -- there is
        no separate correction to clear.  The optimizer is rebuilt rather than zeroed:
        Adam's moment estimates crossing an episode boundary is the same leak in a
        subtler form.
        """
        self._guard.restore()
        self._buffer.clear()
        self._observed = 0
        self._adapted = False
        self._brakes = 0
        self._last_logs = {}
        if self.steps > 0 and self._trainable:
            self.optimizer = self._new_optimizer()
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Cache every executed model step, aligned to its own frame boundaries."""
        if actions.dim() != 3:
            raise ValueError(f"Expected actions (B, T, D), got {tuple(actions.shape)}")
        if self.require_single_episode and actions.shape[0] != 1:
            raise ValueError(
                "HOVER owns a mutable optimizer trajectory, so it requires batch size "
                "1: one gradient step over a batched cohort would average unrelated "
                "episodes into a single correction. Its method.yaml declares "
                "requires_episode_isolation: true; run it through the harness."
            )
        chunk_len = int(actions.shape[1])
        frames = self._boundary_frames(rollout_obs, chunk_len, frameskip)
        actions = actions.detach().to(self.device)
        for step in range(chunk_len):
            self._append(frames[step], actions[:, step: step + 1], frames[step + 1])
        self._observed += 1
        return {f"{self.log_prefix}/buffer_size": float(len(self._buffer))}

    def before_plan(self, obs):
        """Brake on fresh evidence, then refit -- if this replan is due an update.

        Ordering matters and is the method: the newest transition arrived *after* the
        last update, so scoring the correction on it here, before fitting it, is the
        only out-of-sample check available inside an episode. Doing it after the fit
        would score the correction on data it had just been trained on.
        """
        if self.steps == 0 or self._observed == 0 or not self._buffer:
            return {}
        if (self._observed - 1) % self.refresh_interval != 0:
            logs = {
                f"{self.log_prefix}/refresh_skipped": 1.0,
                f"{self.log_prefix}/buffer_size": float(len(self._buffer)),
            }
            self._last_logs = logs
            return logs

        adapted_loss = self._fresh_step_loss()
        with self._frozen_predictor():
            frozen_loss = self._fresh_step_loss()
        ratio = adapted_loss / max(frozen_loss, _EPS)
        braked = 0.0
        if self._adapted and ratio > self.brake_ratio:
            self._reset_correction()
            self._brakes += 1
            braked = 1.0

        logs = {
            f"{self.log_prefix}/fresh_loss": adapted_loss,
            f"{self.log_prefix}/fresh_loss_frozen": frozen_loss,
            f"{self.log_prefix}/fresh_loss_ratio": ratio,
            f"{self.log_prefix}/braked": braked,
            f"{self.log_prefix}/brakes": float(self._brakes),
            f"{self.log_prefix}/refresh_skipped": 0.0,
            f"{self.log_prefix}/buffer_size": float(len(self._buffer)),
        }
        logs.update(self._fit())
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
