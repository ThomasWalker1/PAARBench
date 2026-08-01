"""Stochastic-restoration online gradient test-time adaptation.

This is AdaJEPA-style online gradient adaptation with CoTTA's stochastic
restoration mechanism: immediately after every optimizer update, independently
restore each optimizable scalar to its pretrained value with probability ``p``.
The optimizer trajectory still accumulates, but the leak back to the snapshot is
intended to limit drift across replans.

Only the restoration mechanism is taken from CoTTA (Wang et al., CVPR 2022). The
teacher-student EMA and augmentation-averaged pseudo-label mechanisms require a
prediction target this latent-world-model setting does not expose.
"""

import logging
from collections import deque

import torch

from models.hyper_lora import install_predictor_hyper_targets
from paarbench.adapter import BaseWeightGuard
from utils import move_to_device

log = logging.getLogger(__name__)


class RestoreTTAAdapter:
    """Online JEPA test-time adaptation for MPC chunk transitions."""

    def __init__(
        self,
        wm,
        preprocessor,
        steps=1,
        buffer_size=5,
        buffer_strategy="recent",
        pred_lr=5e-4,
        enc_lr=1e-5,
        action_lr=5e-4,
        update_scope="predlast_enclast",
        lora_rank=4,
        lora_scale=1.0,
        lora_init_seed=None,
        stop_grad=True,
        grad_clip_norm=None,
        refresh_interval=1,
        restore_probability=0.01,
        require_single_episode=False,
        log_prefix="restore_tta",
        **kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.steps = int(steps)
        self.buffer_size = int(buffer_size)
        self.buffer_strategy = str(buffer_strategy)
        self.pred_lr = float(pred_lr)
        self.enc_lr = float(enc_lr)
        self.action_lr = float(action_lr)
        self.update_scope = str(update_scope)
        self.lora_rank = int(lora_rank)
        self.lora_scale = float(lora_scale)
        self.lora_init_seed = None if lora_init_seed is None else int(lora_init_seed)
        self.stop_grad = bool(stop_grad)
        self.grad_clip_norm = grad_clip_norm
        self.refresh_interval = int(refresh_interval)
        self.restore_probability = float(restore_probability)
        self.require_single_episode = bool(require_single_episode)
        self.log_prefix = log_prefix
        self.buffer = deque()
        self.optimizer = None
        self.param_groups = []
        self.lora_targets = []
        self.norm_targets = []
        self.lora_parameters = torch.nn.ParameterList()
        self.norm_parameters = torch.nn.ParameterList()

        if self.steps < 0:
            raise ValueError("restore_tta adaptation steps must be non-negative.")
        if self.buffer_size <= 0:
            raise ValueError("restore_tta buffer_size must be positive.")
        if self.refresh_interval < 1:
            raise ValueError("restore_tta refresh_interval must be positive.")
        if not 0.0 <= self.restore_probability <= 1.0:
            raise ValueError("restore_probability must be between 0 and 1.")
        if self.buffer_strategy not in {"recent", "hard"}:
            raise ValueError("buffer_strategy must be 'recent' or 'hard'.")

        if self.steps > 0:
            self._init_trainable_params()

        # Snapshot AFTER installing any LoRA targets, so it covers whatever this
        # configuration will actually mutate. Everything below is per-episode state
        # and must be reset in on_episode_start.
        self._guard = BaseWeightGuard(wm)
        # The guard snapshots after LoRA targets are installed, so this mapping
        # covers every optimizer-owned parameter restoration can legitimately touch.
        self._optimizer_parameter_names = {
            id(param): name for name, param in self.wm.named_parameters()
        }
        self._observed = 0
        self._last_logs = {}

    def _module_params(self, module, predicate):
        params = []
        names = []
        for name, param in module.named_parameters():
            if predicate(name, param):
                param.requires_grad = True
                params.append(param)
                names.append(name)
        return params, names

    def _predictor_selector(self):
        layers = getattr(getattr(self.wm.predictor, "transformer", None), "layers", None)
        depth = len(layers) if layers is not None else 0
        last_idx = max(depth - 1, 0)
        if self.update_scope.startswith("predfirstlast"):
            target_indices = {0, last_idx}
            include_final_norm = True
        elif self.update_scope.startswith("predfirst"):
            target_indices = {0}
            include_final_norm = False
        else:
            target_indices = {last_idx}
            # The paper's predlast variants include Transformer.norm.
            include_final_norm = self.update_scope.startswith("predlast")

        def select(name, _param):
            if self.update_scope in {"all", "all_predictor"}:
                return True
            if "pred" not in self.update_scope:
                return False
            if depth == 0:
                return True
            if include_final_norm and name.startswith("transformer.norm."):
                return True
            return any(
                name.startswith(f"transformer.layers.{idx}.")
                for idx in target_indices
            )

        return select

    def _encoder_selector(self):
        last_prefixes = (
            "projection.",
            "post_norm.",
            "rb5.",
            "base_model.blocks.11.",
            "base_model.norm.",
            "projector.",
        )

        def select(name, _param):
            if self.update_scope == "all":
                return True
            if "enc" not in self.update_scope or "encfrozen" in self.update_scope:
                return False
            if "encfirst" in self.update_scope:
                return name.startswith(("rb1.", "base_model.blocks.0."))
            return name.startswith(last_prefixes)

        return select

    def _action_selector(self):
        return lambda _name, _param: self.update_scope in {"all", "action", "predlast_action"}

    def _init_trainable_params(self):
        for param in self.wm.parameters():
            param.requires_grad = False

        if self.update_scope.startswith("lora_"):
            self._init_lora_params()
            return

        groups = []
        selected_names = {}

        pred_params, pred_names = self._module_params(
            self.wm.predictor, self._predictor_selector()
        )
        if pred_params:
            groups.append({"params": pred_params, "lr": self.pred_lr})
            selected_names["predictor"] = pred_names

        enc_params, enc_names = self._module_params(
            self.wm.encoder, self._encoder_selector()
        )
        if enc_params:
            groups.append({"params": enc_params, "lr": self.enc_lr})
            selected_names["encoder"] = enc_names

        action_params, action_names = self._module_params(
            self.wm.action_encoder, self._action_selector()
        )
        if action_params:
            groups.append({"params": action_params, "lr": self.action_lr})
            selected_names["action_encoder"] = action_names

        if not groups:
            raise ValueError(
                f"No parameters selected for restore_tta update_scope='{self.update_scope}'."
            )

        self.param_groups = groups
        self.optimizer = torch.optim.AdamW(groups)
        log.info(
            "restore_tta selected %s trainable parameter tensors: %s",
            sum(len(g["params"]) for g in groups),
            {k: len(v) for k, v in selected_names.items()},
        )

    def _init_lora_params(self):
        if self.lora_rank <= 0:
            raise ValueError("LoRA-restore_tta lora_rank must be positive.")
        target_scope = self.update_scope.removeprefix("lora_")
        self.lora_targets, self.norm_targets = install_predictor_hyper_targets(
            self.wm.predictor,
            scope=target_scope,
            rank=self.lora_rank,
            scale=self.lora_scale,
        )
        params = []
        lora_generator = None
        if self.lora_init_seed is not None:
            lora_generator = torch.Generator(device="cpu")
            lora_generator.manual_seed(self.lora_init_seed)
        for target in self.lora_targets:
            if lora_generator is None:
                A_tensor = torch.empty(
                    target.spec.rank,
                    target.spec.in_features,
                    device=self.device,
                )
                torch.nn.init.normal_(
                    A_tensor,
                    mean=0.0,
                    std=1.0 / max(target.spec.in_features, 1) ** 0.5,
                )
            else:
                A_tensor = torch.empty(
                    target.spec.rank,
                    target.spec.in_features,
                    device="cpu",
                )
                torch.nn.init.normal_(
                    A_tensor,
                    mean=0.0,
                    std=1.0 / max(target.spec.in_features, 1) ** 0.5,
                    generator=lora_generator,
                )
                A_tensor = A_tensor.to(self.device)
            A = torch.nn.Parameter(A_tensor)
            B = torch.nn.Parameter(
                torch.zeros(
                    target.spec.out_features,
                    target.spec.rank,
                    device=self.device,
                )
            )
            self.lora_parameters.extend([A, B])
            target.module.set_lora(A, B)
            params.extend([A, B])
        for target in self.norm_targets:
            delta_weight = torch.nn.Parameter(torch.zeros(target.spec.features, device=self.device))
            delta_bias = torch.nn.Parameter(torch.zeros(target.spec.features, device=self.device))
            self.norm_parameters.extend([delta_weight, delta_bias])
            target.module.set_affine(delta_weight, delta_bias)
            params.extend([delta_weight, delta_bias])
        if not params:
            raise ValueError(f"No LoRA parameters selected for update_scope='{self.update_scope}'.")
        self.param_groups = [{"params": params, "lr": self.pred_lr}]
        self.optimizer = torch.optim.AdamW(self.param_groups)
        log.info(
            "LoRA-restore_tta selected %s LoRA modules and %s LayerNorm modules "
            "(rank=%s, trainable tensors=%s).",
            len(self.lora_targets),
            len(self.norm_targets),
            self.lora_rank,
            len(params),
        )

    def _prepare_transition(self, obs_0, action, obs_1):
        if self.require_single_episode and action.shape[0] != 1:
            raise ValueError(
                "restore_tta episode isolation requires batch size 1. "
                "Run independent planning processes and aggregate their metrics."
            )
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_1 = move_to_device(
            self.preprocessor.transform_obs(obs_1), self.device
        )
        if action.dim() != 3:
            raise ValueError(f"Expected action shape (B,T,D), got {tuple(action.shape)}")
        if action.shape[1] != 1:
            action = action[:, :1]
        return {
            "obs_0": {k: v.detach().clone() for k, v in trans_obs_0.items()},
            "action": action.detach().clone().to(self.device),
            "obs_1": {k: v.detach().clone() for k, v in trans_obs_1.items()},
        }

    def _transition_loss(self, item):
        z_src = self.wm.encode(item["obs_0"], item["action"])
        z_pred = self.wm.predict(z_src)
        z_pred_obs, _ = self.wm.separate_emb(z_pred)
        z_tgt_obs = self.wm.encode_obs(item["obs_1"])
        if self.stop_grad:
            z_tgt_obs = {k: v.detach() for k, v in z_tgt_obs.items()}

        visual_loss = self.wm.emb_criterion(
            z_pred_obs["visual"][:, -1:], z_tgt_obs["visual"][:, -1:]
        )
        proprio_loss = self.wm.emb_criterion(
            z_pred_obs["proprio"][:, -1:], z_tgt_obs["proprio"][:, -1:]
        )
        return visual_loss + proprio_loss, visual_loss.detach(), proprio_loss.detach()

    @torch.no_grad()
    def _score_transition(self, item):
        was_training = self.wm.training
        self.wm.eval()
        loss, _, _ = self._transition_loss(item)
        if was_training:
            self.wm.train()
        return float(loss.detach().cpu())

    def _trim_buffer(self):
        while len(self.buffer) > self.buffer_size:
            if self.buffer_strategy == "recent":
                self.buffer.popleft()
            else:
                min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["score"])
                del self.buffer[min_idx]

    def append(self, obs_0, action, obs_1):
        item = self._prepare_transition(obs_0, action, obs_1)
        item["score"] = self._score_transition(item)
        self.buffer.append(item)
        self._trim_buffer()
        return item["score"]

    def step(self):
        if self.steps == 0 or not self.buffer:
            return {
                f"{self.log_prefix}/loss": 0.0,
                f"{self.log_prefix}/visual_loss": 0.0,
                f"{self.log_prefix}/proprio_loss": 0.0,
                f"{self.log_prefix}/buffer_size": len(self.buffer),
            }

        self.wm.train()
        logs = {}
        restored_elements = 0
        restoration_candidates = 0
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            losses = []
            visual_losses = []
            proprio_losses = []
            for item in self.buffer:
                loss, visual_loss, proprio_loss = self._transition_loss(item)
                losses.append(loss)
                visual_losses.append(visual_loss)
                proprio_losses.append(proprio_loss)
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            if self.grad_clip_norm is not None:
                params = [p for group in self.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, float(self.grad_clip_norm))
            self.optimizer.step()
            # CoTTA-style restoration is applied after every optimizer update.
            restored, candidates = self._stochastic_restore()
            restored_elements += restored
            restoration_candidates += candidates
            logs = {
                f"{self.log_prefix}/loss": float(total_loss.detach().cpu()),
                f"{self.log_prefix}/visual_loss": float(torch.stack(visual_losses).mean().cpu()),
                f"{self.log_prefix}/proprio_loss": float(torch.stack(proprio_losses).mean().cpu()),
                f"{self.log_prefix}/buffer_size": len(self.buffer),
            }

        logs[f"{self.log_prefix}/restore_probability"] = self.restore_probability
        logs[f"{self.log_prefix}/restored_elements"] = restored_elements
        logs[f"{self.log_prefix}/restored_fraction"] = (
            restored_elements / restoration_candidates if restoration_candidates else 0.0
        )
        self.wm.eval()
        if self.buffer_strategy == "hard":
            for item in self.buffer:
                item["score"] = self._score_transition(item)
        return logs

    @torch.no_grad()
    def _stochastic_restore(self):
        """Leak optimizer-owned weights back to the pretrained snapshot.

        The guard stores CPU copies of the post-installation base state. Sampling
        only optimizer-owned parameters is equivalent to sampling the full model:
        every other parameter is frozen and therefore still equals its snapshot.
        """
        restored = 0
        candidates = 0
        seen = set()
        for group in self.param_groups:
            for parameter in group["params"]:
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                name = self._optimizer_parameter_names.get(id(parameter))
                if name is None:
                    raise RuntimeError(
                        "restore_tta optimizer owns a parameter outside the world "
                        "model, so it has no pretrained snapshot to restore."
                    )
                reference = self._guard._snapshot[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if tuple(reference.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"restore_tta snapshot shape mismatch for {name}: "
                        f"{tuple(reference.shape)} != {tuple(parameter.shape)}"
                    )
                candidates += parameter.numel()
                if self.restore_probability == 0.0:
                    continue
                mask = torch.rand_like(parameter) < self.restore_probability
                parameter.copy_(torch.where(mask, reference, parameter))
                restored += int(mask.sum().item())
        return restored, candidates

    # -- TestTimeAdapter protocol ---------------------------------------------

    def on_episode_start(self, obs_0, goal):
        """Restore the base model and drop every trace of the previous episode.

        New relative to the predecessor, which had no reset at all: it built one
        adapter per planning process and each process evaluated exactly one batch,
        so the question never arose.  Under the protocol an adapter may be reused,
        and an optimizer trajectory surviving an episode boundary would silently
        contaminate the next episode in a way that looks like a real effect.

        The optimizer is rebuilt rather than zeroed, so Adam's moment estimates go
        with it -- carrying momentum across episodes is the same leak in a subtler
        form.
        """
        self._guard.restore()
        self.buffer.clear()
        self._observed = 0
        self._last_logs = {}
        if self.steps > 0 and self.param_groups:
            self.optimizer = torch.optim.AdamW(self.param_groups)
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Buffer the executed chunk.

        Only the chunk's final observation is used, paired with the chunk's *first*
        action (``_prepare_transition`` truncates to ``action[:, :1]``).  That
        pairing is the predecessor's semantics; it is preserved deliberately rather
        than corrected, because changing it would change the numbers this port is
        validated against.  Revisit it as a separate, measured change.
        """
        final_obs = {k: v[:, -1:] for k, v in rollout_obs.items()}
        pred_err = self.append(obs_0, actions, final_obs)
        self._observed += 1
        return {f"{self.log_prefix}/pre_update_loss": pred_err}

    def before_plan(self, obs):
        """Take the scheduled optimizer step, if this replan is due one.

        The refresh schedule lives here, in the method, rather than in the planner.
        Indexing matches the predecessor exactly: it stepped after observing
        transition ``i`` when ``i % refresh_interval == 0`` (0-based), which --
        observed here one replan later -- is
        ``(self._observed - 1) % refresh_interval == 0``.
        """
        if self._observed == 0:
            # First solve of the episode: nothing has been executed yet.
            return {}
        if (self._observed - 1) % self.refresh_interval == 0:
            logs = self.step()
            logs[f"{self.log_prefix}/refresh_skipped"] = 0
        else:
            logs = {
                f"{self.log_prefix}/loss": 0.0,
                f"{self.log_prefix}/visual_loss": 0.0,
                f"{self.log_prefix}/proprio_loss": 0.0,
                f"{self.log_prefix}/buffer_size": len(self.buffer),
                f"{self.log_prefix}/refresh_skipped": 1,
            }
        logs[f"{self.log_prefix}/refresh_interval"] = self.refresh_interval
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
