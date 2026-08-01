from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch

from models.hyper_lora import (
    HyperAdapterGate,
    build_generator,
    apply_hyper_tensors,
    clear_hyper_tensors,
    fold_hyper_tensors,
    full_rank_for_scope,
    install_predictor_hyper_targets,
    lora_parameter_budget,
    module_parameter_count,
    restore_base_weights,
    snapshot_base_weights,
)
from models.hyper_context import OrderedTransitionContext, transition_feature
from utils import move_to_device

log = logging.getLogger(__name__)


class HyperJEPAAdapter:
    """Amortized predictor adaptation via context-conditioned LoRA."""

    def __init__(
        self,
        wm,
        preprocessor,
        context_mode="transition_buffer",
        context_feature_kind="residual_action",
        context_aggregator="transformer_query",
        context_model_dim=128,
        context_depth=2,
        context_heads=4,
        context_dropout=0.0,
        adapter_gate=False,
        gate_hidden_dim=64,
        gate_init_bias=0.0,
        target_scope="predlast_all",
        rank=4,
        lora_scale=1.0,
        freeze_lora_A=False,
        fixed_a_scale=1.0,
        hidden_dim=512,
        depth=2,
        init_scale=0.0,
        generator="mlp",
        chunk_size=512,
        chunk_embed_dim=64,
        head_hidden_dim=128,
        checkpoint_path=None,
        refresh="episode_start",
        refresh_interval=1,
        transition_buffer_size=5,
        freeze_base=True,
        log_prefix="hyperjepa",
        context_perturb="none",
        query_perturb="none",
        fold_weights=False,
        **kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.context_mode = str(context_mode)
        self.context_feature_kind = str(context_feature_kind)
        if self.context_feature_kind not in {"residual_action", "latent_residual_action"}:
            raise ValueError("Unknown HyperJEPA context_feature_kind.")
        self.context_aggregator = str(context_aggregator)
        if self.context_aggregator not in {"mean", "transformer", "transformer_query"}:
            raise ValueError("Unknown HyperJEPA context_aggregator.")
        if self.context_aggregator in {"transformer", "transformer_query"} and self.context_mode != "transition_buffer":
            raise ValueError("Transformer aggregation is currently supported for transition_buffer only.")
        self.refresh = str(refresh)
        self.refresh_interval = int(refresh_interval)
        self.log_prefix = str(log_prefix)
        # Mechanism ablations for the claim that the correction is *conditioned* on
        # the episode's own transition evidence:
        #   swap -- give each episode a different episode's transitions (roll by 1)
        #   mean -- replace transitions by the batch mean, so the emitted correction
        #           is constant across episodes (a hypernet-parameterized static LoRA)
        #   zero -- ablate the evidence entirely
        # The state query is deliberately left untouched: it carries current state,
        # not dynamics evidence, so only the evidence is ablated.
        self.context_perturb = str(context_perturb)
        if self.context_perturb not in {"none", "swap", "mean", "zero"}:
            raise ValueError("HyperJEPA context_perturb must be none|swap|mean|zero.")
        # The transformer_query aggregator conditions on TWO things: the transition
        # evidence (above) and the episode's current state (the query). Ablating only
        # the evidence still leaves state conditioning, so the emitted correction stays
        # episode-dependent. Ablate both to obtain a genuinely constant correction --
        # i.e. the hypernetwork reduced to a learned static LoRA.
        self.query_perturb = str(query_perturb)
        if self.query_perturb not in {"none", "swap", "mean", "zero"}:
            raise ValueError("HyperJEPA query_perturb must be none|swap|mean|zero.")
        if self.refresh not in {"episode_start", "every_mpc"}:
            raise ValueError("HyperJEPA refresh must be 'episode_start' or 'every_mpc'.")
        if self.refresh_interval < 1:
            raise ValueError("HyperJEPA refresh_interval must be positive.")

        if freeze_base:
            for param in self.wm.parameters():
                param.requires_grad = False

        self.target_scope = str(target_scope)
        if str(rank).lower() == "full":
            rank = full_rank_for_scope(self.wm.predictor, self.target_scope)
        self.rank = int(rank)
        self.lora_targets, self.norm_targets = install_predictor_hyper_targets(
            self.wm.predictor,
            scope=self.target_scope,
            rank=self.rank,
            scale=float(lora_scale),
        )
        self.specs = [target.spec for target in self.lora_targets] + [target.spec for target in self.norm_targets]
        self.world_model_parameters = module_parameter_count(self.wm)
        self.generator = None
        self.freeze_lora_A = bool(freeze_lora_A)
        self.fixed_a_scale = float(fixed_a_scale)
        self.context_encoder = None
        self.context_model_dim = int(context_model_dim)
        self.context_depth = int(context_depth)
        self.context_heads = int(context_heads)
        self.context_dropout = float(context_dropout)
        self.adapter_gate_enabled = bool(adapter_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_init_bias = float(gate_init_bias)
        self.adapter_gate = None
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.init_scale = float(init_scale)
        self.generator_kind = str(generator)
        self.chunk_size = int(chunk_size)
        self.chunk_embed_dim = int(chunk_embed_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.checkpoint_path = checkpoint_path
        self.applied = False
        # Deployment mode: fold the emitted correction into the predictor weights once
        # per replan instead of routing every planner rollout through the LoRA wrapper
        # (RESULTS §12 -- the wrapper costs ~0.80 s of the 100-step GD-MPC replan).
        # Single-episode only; see fold_hyper_tensors.
        self.fold_weights = bool(fold_weights)
        self._base_snapshot = None
        self._mpc_refresh_calls = 0
        self.transition_buffer_size = int(transition_buffer_size)
        self.transition_buffer = deque(maxlen=self.transition_buffer_size)
        if self.transition_buffer_size < 1:
            raise ValueError("transition_buffer_size must be positive")

        # Protocol state: how many executed chunks this episode has observed, and the
        # most recent hook's logs.  Both reset in on_episode_start.
        self._observed = 0
        self._last_logs = {}

    def _prepare_obs(self, obs):
        obs_t = self.preprocessor.transform_obs(obs)
        return move_to_device(obs_t, self.device)

    @staticmethod
    def _perturb_batch(tensor, mode, name):
        """Ablate episode-specific information along the batch dimension."""
        if mode == "none":
            return tensor
        if mode == "zero":
            return torch.zeros_like(tensor)
        if mode == "mean":
            return tensor.mean(dim=0, keepdim=True).expand_as(tensor).contiguous()
        if tensor.shape[0] < 2:
            raise ValueError(
                f"{name}='swap' needs more than one episode in the batch; run the "
                "batched evaluator (n_evals>1), not one episode per process."
            )
        # Roll by one: a derangement, so no episode keeps its own information.
        return tensor.roll(1, dims=0)

    def _perturb_context_features(self, features):
        return self._perturb_batch(features, self.context_perturb, "context_perturb")

    @torch.no_grad()
    def encode_context(self, obs):
        if self.context_mode == "transition_buffer":
            if not self.transition_buffer:
                raise ValueError("No observed transitions are available for transition_buffer context")
            features = torch.stack(list(self.transition_buffer), dim=1)
            features = self._perturb_context_features(features)
            if self.context_aggregator == "mean":
                return features.mean(dim=1)
            query = None
            if self.context_aggregator == "transformer_query":
                obs_t = self._prepare_obs(obs)
                query = self._perturb_batch(
                    self._state_query(obs_t), self.query_perturb, "query_perturb"
                )
            self._ensure_context_encoder(features.shape[-1], None if query is None else query.shape[-1])
            return self.context_encoder(features, query=query)
        obs_t = self._prepare_obs(obs)
        if self.context_mode == "state_only":
            # No transition buffer: the whole context is the current observation's
            # pooled latent, so this arm is never frozen for want of evidence and
            # its correction changes only when the state does.
            return self._perturb_batch(
                self._state_query(obs_t), self.query_perturb, "query_perturb"
            )
        z = self.wm.encode_obs(obs_t)
        visual = z["visual"]
        proprio = z["proprio"]

        if self.context_mode == "first_frame":
            visual = visual[:, :1]
            proprio = proprio[:, :1]
        elif self.context_mode == "hist_frames":
            pass
        else:
            raise ValueError(
                f"Unsupported HyperJEPA context_mode='{self.context_mode}'. Implemented "
                "modes: first_frame, hist_frames, state_only, transition_buffer."
            )

        visual_ctx = visual.mean(dim=(1, 2))
        proprio_ctx = proprio.mean(dim=1)
        return torch.cat([visual_ctx, proprio_ctx], dim=-1)

    def _checkpoint_payload(self):
        if not hasattr(self, "_loaded_payload"):
            self._loaded_payload = None
            if self.checkpoint_path:
                self._loaded_payload = torch.load(Path(self.checkpoint_path), map_location=self.device)
                metadata = self._loaded_payload.get("metadata", {})
                # An architecture ablation grid makes silent mis-loading the main
                # hazard: several of these mismatches keep every tensor shape
                # valid, so strict state-dict loading would not catch them.
                for field, requested, default in (
                    ("context_aggregator", self.context_aggregator, "mean"),
                    ("context_feature_kind", self.context_feature_kind, "residual_action"),
                    ("context_mode", self.context_mode, "transition_buffer"),
                    ("target_scope", self.target_scope, "predlast_all"),
                    ("generator", self.generator_kind, "mlp"),
                ):
                    saved = metadata.get(field, default)
                    if str(saved) != str(requested):
                        raise ValueError(
                            f"HyperJEPA checkpoint {field} is '{saved}', but planning "
                            f"requested '{requested}'."
                        )
                saved_rank = metadata.get("rank")
                if saved_rank is not None and int(saved_rank) != int(self.rank):
                    raise ValueError(
                        f"HyperJEPA checkpoint rank is {saved_rank}, but planning "
                        f"requested {self.rank}."
                    )
                saved_transitions = metadata.get("context_transitions")
                if (
                    saved_transitions is not None
                    and int(saved_transitions) != self.transition_buffer_size
                ):
                    raise ValueError(
                        f"HyperJEPA checkpoint used {saved_transitions} context "
                        f"transitions, but planning requested {self.transition_buffer_size}."
                    )
                saved_gate = bool(metadata.get("adapter_gate", False))
                if saved_gate != self.adapter_gate_enabled:
                    raise ValueError(
                        "HyperJEPA checkpoint adapter_gate is "
                        f"'{saved_gate}', but planning requested '{self.adapter_gate_enabled}'."
                    )
                saved_freeze_a = bool(metadata.get("freeze_lora_A", False))
                if saved_freeze_a != self.freeze_lora_A:
                    raise ValueError(
                        "HyperJEPA checkpoint freeze_lora_A is "
                        f"'{saved_freeze_a}', but planning requested '{self.freeze_lora_A}'."
                    )
        return self._loaded_payload

    @torch.no_grad()
    def _state_query(self, obs_t):
        z = self.wm.encode_obs(obs_t)
        return torch.cat([z["visual"].mean(dim=(1, 2)), z["proprio"].mean(dim=1)], dim=-1)

    def _ensure_context_encoder(self, context_dim: int, query_dim: int | None = None):
        if self.context_aggregator not in {"transformer", "transformer_query"} or self.context_encoder is not None:
            return
        self.context_encoder = OrderedTransitionContext(
            input_dim=context_dim,
            model_dim=self.context_model_dim,
            depth=self.context_depth,
            heads=self.context_heads,
            max_context_length=self.transition_buffer_size,
            dropout=self.context_dropout,
            query_dim=query_dim,
        ).to(self.device)
        payload = self._checkpoint_payload()
        if payload is not None:
            if "context_encoder" not in payload:
                raise ValueError("Transformer HyperJEPA checkpoint has no context_encoder state")
            self.context_encoder.load_state_dict(payload["context_encoder"])
            log.info("Loaded HyperJEPA ordered context encoder from %s", self.checkpoint_path)
        self.context_encoder.eval()

    def _ensure_adapter_gate(self, context_dim: int):
        if not self.adapter_gate_enabled or self.adapter_gate is not None:
            return
        self.adapter_gate = HyperAdapterGate(
            context_dim=context_dim,
            hidden_dim=self.gate_hidden_dim,
            init_bias=self.gate_init_bias,
        ).to(self.device)
        payload = self._checkpoint_payload()
        if payload is not None:
            if "adapter_gate" not in payload:
                raise ValueError("Gated HyperJEPA checkpoint has no adapter_gate state")
            self.adapter_gate.load_state_dict(payload["adapter_gate"])
            log.info("Loaded HyperJEPA adapter gate from %s", self.checkpoint_path)
        self.adapter_gate.eval()

    def _ensure_generator(self, context_dim: int):
        if self.generator is not None:
            return
        self.generator = build_generator(
            self.generator_kind,
            context_dim=context_dim,
            specs=self.specs,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            init_scale=self.init_scale,
            freeze_lora_A=self.freeze_lora_A,
            fixed_a_scale=self.fixed_a_scale,
            chunk_size=self.chunk_size,
            chunk_embed_dim=self.chunk_embed_dim,
            head_hidden_dim=self.head_hidden_dim,
        ).to(self.device)
        payload = self._checkpoint_payload()
        if payload is not None:
            state = payload.get("hyper_lora", payload)
            self.generator.load_state_dict(state)
            log.info("Loaded HyperJEPA generator checkpoint from %s", self.checkpoint_path)
        self.generator.eval()

    def apply(self, obs):
        context = self.encode_context(obs)
        self._ensure_generator(context.shape[-1])
        self._ensure_adapter_gate(context.shape[-1])
        with torch.no_grad():
            tensors = self.generator(context)
            gate = self.adapter_gate(context) if self.adapter_gate is not None else None
        if self.fold_weights:
            # Restore first: the correction is recomputed from scratch each replan, so
            # folding onto already-folded weights would accumulate them.
            if self._base_snapshot is None:
                self._base_snapshot = snapshot_base_weights(self.lora_targets, self.norm_targets)
            else:
                restore_base_weights(self.lora_targets, self.norm_targets, self._base_snapshot)
            fold_hyper_tensors(self.lora_targets, self.norm_targets, tensors, gate=gate)
        else:
            apply_hyper_tensors(self.lora_targets, self.norm_targets, tensors, gate=gate)
        self.applied = True
        generated_rms = torch.stack(
            [tensor.pow(2).mean() for pair in tensors.values() for tensor in pair]
        ).mean().sqrt()
        # How much does the emitted correction actually vary BETWEEN episodes?
        # This is the conditioning strength: ~0 means the hypernetwork is emitting a
        # near-constant correction, i.e. behaving like a learned static LoRA
        # regardless of what the context says.
        dispersion = float("nan")
        flat = [
            tensor.flatten(start_dim=1)
            for pair in tensors.values()
            for tensor in pair
            if tensor.dim() > 1 and tensor.shape[0] > 1
        ]
        if flat:
            stacked = torch.cat(flat, dim=1)
            scale = stacked.pow(2).mean().sqrt().clamp_min(1e-12)
            dispersion = float((stacked.std(dim=0).mean() / scale).cpu())
        return {
            f"{self.log_prefix}/applied": 1,
            f"{self.log_prefix}/context_dim": context.shape[-1],
            f"{self.log_prefix}/num_lora_modules": len(self.lora_targets),
            f"{self.log_prefix}/num_layer_norm_modules": len(self.norm_targets),
            f"{self.log_prefix}/total_lora_params": lora_parameter_budget(
                self.specs,
                freeze_lora_A=self.freeze_lora_A,
            ),
            f"{self.log_prefix}/generator_params": module_parameter_count(self.generator),
            f"{self.log_prefix}/world_model_params": self.world_model_parameters,
            f"{self.log_prefix}/generated_lora_rms": float(generated_rms.cpu()),
            f"{self.log_prefix}/generated_lora_dispersion": dispersion,
            f"{self.log_prefix}/adapter_gate": float(gate.mean().cpu()) if gate is not None else 1.0,
        }

    def maybe_apply_episode_start(self, obs):
        if self.context_mode == "transition_buffer":
            # Preserve the frozen model until an action has produced feedback.
            return {}
        if not self.applied:
            return self.apply(obs)
        return {}

    @torch.no_grad()
    def append_transition(self, obs_0, action, obs_1):
        if self.context_mode != "transition_buffer":
            return {}
        if action.ndim != 3 or action.shape[1] != 1:
            raise ValueError(
                "append_transition expects one action chunk. Use append_executed_transitions "
                "for a multi-action MPC rollout."
            )
        obs_0_t = self._prepare_obs(obs_0)
        obs_1_t = self._prepare_obs(obs_1)
        action = action.to(self.device)
        self.transition_buffer.append(
            transition_feature(
                self.wm,
                obs_0_t,
                action,
                obs_1_t,
                feature_kind=self.context_feature_kind,
            )
        )
        return {f"{self.log_prefix}/transition_buffer_size": len(self.transition_buffer)}

    @torch.no_grad()
    def append_executed_transitions(self, obs_0, actions, observed_obs, frameskip: int):
        """Append one properly aligned feature for every executed MPC action.

        ``observed_obs`` contains raw environment observations at every simulator
        step, including the initial state.  Each planner action corresponds to
        ``frameskip`` simulator steps, so its target is sampled at that boundary.
        """
        if self.context_mode != "transition_buffer":
            return {}
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError(f"Expected executed actions (B, T, D), got {tuple(actions.shape)}")
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        steps = actions.shape[1]
        required_frames = 1 + steps * int(frameskip)
        if any(value.shape[1] < required_frames for value in observed_obs.values()):
            raise ValueError(
                f"Observed rollout is too short for {steps} actions at frameskip {frameskip}"
            )
        if any(value.shape[1] != 1 for value in obs_0.values()):
            raise ValueError("Online transition adaptation expects a one-frame current observation")

        def concat_time(initial, later):
            if isinstance(initial, torch.Tensor):
                return torch.cat([initial, later], dim=1)
            if isinstance(initial, np.ndarray):
                return np.concatenate([initial, later], axis=1)
            raise TypeError(f"Unsupported observation type: {type(initial)}")

        source = {
            key: concat_time(obs_0[key], value[:, frameskip : steps * frameskip : frameskip])
            for key, value in observed_obs.items()
        }
        target = {
            key: value[:, frameskip : (steps + 1) * frameskip : frameskip]
            for key, value in observed_obs.items()
        }
        batch = actions.shape[0]
        source_flat = {
            key: value.reshape(batch * steps, 1, *value.shape[2:]) for key, value in source.items()
        }
        target_flat = {
            key: value.reshape(batch * steps, 1, *value.shape[2:]) for key, value in target.items()
        }
        action_flat = actions.reshape(batch * steps, 1, actions.shape[-1]).to(self.device)
        features = transition_feature(
            self.wm,
            self._prepare_obs(source_flat),
            action_flat,
            self._prepare_obs(target_flat),
            feature_kind=self.context_feature_kind,
        ).reshape(batch, steps, -1)
        for step in range(steps):
            self.transition_buffer.append(features[:, step])
        return {
            f"{self.log_prefix}/transition_buffer_size": len(self.transition_buffer),
            f"{self.log_prefix}/transition_features_added": steps,
        }

    def maybe_refresh_after_mpc(self, obs=None):
        if self.refresh == "every_mpc":
            self._mpc_refresh_calls += 1
            if (self._mpc_refresh_calls - 1) % self.refresh_interval != 0:
                return {
                    f"{self.log_prefix}/refresh_skipped": 1,
                    f"{self.log_prefix}/refresh_interval": self.refresh_interval,
                }
            if self.context_mode == "transition_buffer" and not self.transition_buffer:
                return {}
            logs = self.apply(obs)
            logs[f"{self.log_prefix}/refresh_skipped"] = 0
            logs[f"{self.log_prefix}/refresh_interval"] = self.refresh_interval
            return logs
        return {}

    def clear(self):
        clear_hyper_tensors(self.lora_targets, self.norm_targets)
        if self._base_snapshot is not None:
            # Leave the predictor exactly as it was found, so a later episode does not
            # inherit the previous one's folded correction.
            restore_base_weights(self.lora_targets, self.norm_targets, self._base_snapshot)
        self.applied = False
        self._mpc_refresh_calls = 0
        self.transition_buffer.clear()


    # -- TestTimeAdapter protocol ---------------------------------------------
    #
    # This method has TWO independent switches, and the protocol has to keep them
    # independent or the port silently becomes a different method:
    #
    #   context_mode  -- what the correction is conditioned on.  Under
    #                    'transition_buffer' the frozen model is deliberately left
    #                    alone until an action has produced feedback, so there is
    #                    nothing to apply at episode start.
    #   refresh       -- how often the correction is recomputed.  'episode_start'
    #                    emits once; 'every_mpc' regenerates before each replan,
    #                    always from the frozen weights, never by accumulation.
    #
    # The planner sees only their product, which is why both stay in here.

    def on_episode_start(self, obs_0, goal):
        """Restore the base model, drop the buffer, and apply if this mode applies.

        ``clear()`` is the restoration: it drops the emitted LoRA tensors and, in
        fold mode, copies the snapshotted base weights back over the predictor.  In
        non-fold mode the base weights are never mutated at all -- the correction
        rides in a wrapper -- so bit-identity holds trivially there.
        """
        self.clear()
        self._observed = 0
        self._last_logs = {}
        logs = dict(self.maybe_apply_episode_start(obs_0))
        self._last_logs = logs
        return logs

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Append one aligned context feature per executed action.

        This is why the protocol hands over the whole rollout rather than just its
        final frame: each planner action spans ``frameskip`` simulator steps, and the
        feature for that action is built from the observations at its own boundaries.
        """
        logs = dict(self.append_executed_transitions(
            obs_0, actions, rollout_obs, frameskip=frameskip
        ))
        self._observed += 1
        self._last_logs = logs
        return logs

    def before_plan(self, obs):
        """Regenerate the correction from frozen weights, if this replan is due one.

        Gated on having observed something, so the first solve of an episode keeps
        whatever ``on_episode_start`` decided.  Indexing matches the predecessor: it
        called ``maybe_refresh_after_mpc`` at the END of every replan, so call *n*
        there is call *n* here, one replan later, and the internal
        ``_mpc_refresh_calls`` counter is untouched.
        """
        if self._observed == 0:
            return {}
        logs = dict(self.maybe_refresh_after_mpc(obs))
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
