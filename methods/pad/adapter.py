"""PAD: inverse-dynamics adaptation of the world-model encoder at deployment.

The auxiliary head is intentionally owned by this adapter rather than by the base
world model.  It is pretrained offline, then updated together with the encoder from
executed transitions during a single episode.  That is the PAD shape which exercises
the benchmark's reset contract beyond LoRA-style edits to base parameters.
"""

from collections import deque
from pathlib import Path

import torch
from torch import nn

from paarbench.adapter import BaseWeightGuard
from utils import move_to_device


class InverseDynamicsHead(nn.Module):
    """Predict the executed action from pooled consecutive world-model latents."""

    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, z_0, z_1):
        return self.net(torch.cat((z_0, z_1), dim=-1))


class PADAdapter:
    """Policy Adaptation during Deployment for the encoder of a latent world model."""

    def __init__(
        self,
        wm,
        preprocessor,
        checkpoint_path,
        latent_dim=394,
        action_dim=2,
        head_hidden_dim=256,
        encoder_lr=1.0e-5,
        head_lr=1.0e-4,
        steps=1,
        buffer_size=5,
        grad_clip_norm=1.0,
        refresh_interval=1,
        require_single_episode=True,
        log_prefix="pad",
        **_kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr)
        self.steps = int(steps)
        self.buffer_size = int(buffer_size)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.refresh_interval = int(refresh_interval)
        self.require_single_episode = bool(require_single_episode)
        self.log_prefix = str(log_prefix)
        if self.steps < 0 or self.buffer_size <= 0 or self.refresh_interval <= 0:
            raise ValueError("PAD requires non-negative steps and positive buffer/refresh sizes")

        self.inverse_head = InverseDynamicsHead(
            2 * self.latent_dim, self.action_dim, head_hidden_dim
        ).to(self.device)
        self.owned_modules = (self.inverse_head,)
        self._load_checkpoint(checkpoint_path)

        for parameter in self.wm.parameters():
            parameter.requires_grad = False
        for parameter in self.wm.encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.inverse_head.parameters():
            parameter.requires_grad = True
        self._param_groups = [
            {"params": list(self.wm.encoder.parameters()), "lr": self.encoder_lr},
            {"params": list(self.inverse_head.parameters()), "lr": self.head_lr},
        ]
        self.optimizer = self._make_optimizer()

        # This must include the head.  Resetting only wm would leak its online
        # trajectory into the next episode even though the base model was pristine.
        self._guard = BaseWeightGuard(self.wm, *self.owned_modules)
        self.buffer = deque()
        self._observed = 0
        self._last_logs = {}

    def _load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"PAD inverse-dynamics checkpoint not found: {path}; "
                "see docs/CHECKPOINTS.md"
            )
        payload = torch.load(path, map_location=self.device)
        expected = {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
        }
        for field, value in expected.items():
            saved = payload.get(field)
            if saved is not None and int(saved) != value:
                raise ValueError(
                    f"PAD checkpoint {field}={saved}, but method.yaml requests {value}"
                )
        self.inverse_head.load_state_dict(payload["inverse_head"])

    def _make_optimizer(self):
        return torch.optim.Adam(self._param_groups)

    def _prepare_transition(self, obs_0, action, obs_1):
        if self.require_single_episode and action.shape[0] != 1:
            raise ValueError(
                "PAD owns a shared encoder/optimizer and requires batch size 1. "
                "Declare requires_episode_isolation: true."
            )
        if action.dim() != 3:
            raise ValueError(f"PAD expected action (B,T,D), got {tuple(action.shape)}")
        return {
            "obs_0": {
                key: value.detach().clone()
                for key, value in move_to_device(self.preprocessor.transform_obs(obs_0), self.device).items()
            },
            "action": action[:, :1].detach().clone().to(self.device),
            "obs_1": {
                key: value.detach().clone()
                for key, value in move_to_device(self.preprocessor.transform_obs(obs_1), self.device).items()
            },
        }

    def _pooled_latent(self, obs):
        latents = self.wm.encode_obs(obs)
        # Visual has a singleton spatial-token axis for this base; flattening every
        # non-batch/non-time axis keeps this adapter compatible with a future encoder
        # that exposes several tokens instead.
        pieces = [
            value[:, -1].flatten(start_dim=1)
            for _name, value in sorted(latents.items())
        ]
        pooled = torch.cat(pieces, dim=-1)
        if pooled.shape[-1] != self.latent_dim:
            raise ValueError(
                f"PAD encoder emitted {pooled.shape[-1]} latent features, expected "
                f"{self.latent_dim}; use a matching pretrained inverse head."
            )
        return pooled

    def _inverse_loss(self, item):
        z_0 = self._pooled_latent(item["obs_0"])
        z_1 = self._pooled_latent(item["obs_1"])
        prediction = self.inverse_head(z_0, z_1)
        target = item["action"][:, 0]
        return torch.nn.functional.mse_loss(prediction, target)

    @torch.no_grad()
    def _score_transition(self, item):
        was_training = self.wm.training
        self.wm.eval()
        self.inverse_head.eval()
        loss = self._inverse_loss(item)
        if was_training:
            self.wm.train()
        return float(loss.detach().cpu())

    def _step(self):
        if self.steps == 0 or not self.buffer:
            return {f"{self.log_prefix}/inverse_loss": 0.0,
                    f"{self.log_prefix}/buffer_size": len(self.buffer)}
        self.wm.train()
        self.inverse_head.train()
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            loss = torch.stack([self._inverse_loss(item) for item in self.buffer]).mean()
            loss.backward()
            if self.grad_clip_norm is not None:
                params = [p for group in self._param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
            self.optimizer.step()
        self.wm.eval()
        self.inverse_head.eval()
        return {f"{self.log_prefix}/inverse_loss": float(loss.detach().cpu()),
                f"{self.log_prefix}/buffer_size": len(self.buffer)}

    def on_episode_start(self, obs_0, goal):
        self._guard.restore()
        self.optimizer = self._make_optimizer()
        self.buffer.clear()
        self._observed = 0
        self._last_logs = {}
        self.wm.eval()
        self.inverse_head.eval()
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        final_obs = {key: value[:, -1:] for key, value in rollout_obs.items()}
        item = self._prepare_transition(obs_0, actions, final_obs)
        item["score"] = self._score_transition(item)
        self.buffer.append(item)
        while len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self._observed += 1
        return {f"{self.log_prefix}/pre_update_inverse_loss": item["score"]}

    def before_plan(self, obs):
        if self._observed == 0:
            return {}
        if (self._observed - 1) % self.refresh_interval == 0:
            logs = self._step()
            logs[f"{self.log_prefix}/refresh_skipped"] = 0
        else:
            logs = {f"{self.log_prefix}/inverse_loss": 0.0,
                    f"{self.log_prefix}/buffer_size": len(self.buffer),
                    f"{self.log_prefix}/refresh_skipped": 1}
        logs[f"{self.log_prefix}/refresh_interval"] = self.refresh_interval
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
