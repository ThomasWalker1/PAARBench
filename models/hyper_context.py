"""Transition features available to HyperLoRA during closed-loop planning."""

from __future__ import annotations

import torch
from torch import nn


def _flatten_latent(obs_latent):
    visual = obs_latent["visual"].mean(dim=(1, 2))
    proprio = obs_latent["proprio"].mean(dim=1)
    return torch.cat([visual, proprio], dim=-1)


def transition_feature(wm, obs_0, action, obs_1, feature_kind="residual_action"):
    """Encode a prediction residual and action for one observed transition.

    The feature uses only quantities available after an MPC action chunk has
    been executed: the previous observation, executed action, and resulting
    observation. It intentionally leaves the visual encoder frozen.
    """
    if action.ndim != 3 or action.shape[1] != 1:
        raise ValueError(
            "transition_feature expects exactly one observed action chunk with "
            f"shape (B, 1, D), got {tuple(action.shape)}"
        )
    z_src = wm.encode(obs_0, action)
    z_pred = wm.predict(z_src)
    z_src_obs, _ = wm.separate_emb(z_src)
    z_pred_obs, _ = wm.separate_emb(z_pred)
    z_tgt_obs = wm.encode_obs(obs_1)

    visual_residual = (z_pred_obs["visual"] - z_tgt_obs["visual"]).mean(dim=(1, 2))
    proprio_residual = (z_pred_obs["proprio"] - z_tgt_obs["proprio"]).mean(dim=1)
    residual = torch.cat([visual_residual, proprio_residual], dim=-1)
    action_feature = wm.encode_act(action).mean(dim=1)
    if feature_kind == "residual_action":
        return torch.cat([residual, action_feature], dim=-1)
    if feature_kind == "latent_residual_action":
        return torch.cat(
            [
                _flatten_latent(z_src_obs),
                _flatten_latent(z_pred_obs),
                _flatten_latent(z_tgt_obs),
                residual,
                action_feature,
            ],
            dim=-1,
        )
    raise ValueError(
        "feature_kind must be 'residual_action' or 'latent_residual_action', "
        f"got '{feature_kind}'"
    )


def transition_buffer_features(wm, obs, act, feature_kind="residual_action"):
    """Return ordered residual/action features for consecutive transitions."""
    transitions = obs["visual"].shape[1] - 1
    if transitions < 1:
        raise ValueError("A transition buffer requires at least two observations")
    features = []
    for idx in range(transitions):
        obs_0 = {key: value[:, idx : idx + 1] for key, value in obs.items()}
        obs_1 = {key: value[:, idx + 1 : idx + 2] for key, value in obs.items()}
        features.append(
            transition_feature(
                wm,
                obs_0,
                act[:, idx : idx + 1],
                obs_1,
                feature_kind=feature_kind,
            )
        )
    return torch.stack(features, dim=1)


def transition_buffer_feature(wm, obs, act, feature_kind="residual_action"):
    """Mean feature for all consecutive transitions in an offline window."""
    return transition_buffer_features(wm, obs, act, feature_kind=feature_kind).mean(dim=1)


class OrderedTransitionContext(nn.Module):
    """Encode a short, ordered buffer of observed transition features.

    The encoder deliberately operates on frozen-model residual/action features,
    rather than raw pixels.  It preserves the information shown useful by the
    mean-buffer baseline while making recency and ordering available to the
    generator.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        max_context_length: int = 8,
        dropout: float = 0.0,
        query_dim: int | None = None,
    ):
        super().__init__()
        if input_dim < 1 or model_dim < 1 or depth < 1 or heads < 1:
            raise ValueError("OrderedTransitionContext dimensions and depth must be positive")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if max_context_length < 1:
            raise ValueError("max_context_length must be positive")
        self.input_dim = int(input_dim)
        self.max_context_length = int(max_context_length)
        self.query_dim = int(query_dim) if query_dim is not None else None
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, model_dim)
        self.position = nn.Parameter(torch.empty(1, self.max_context_length, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.query_proj = (
            nn.Sequential(nn.LayerNorm(self.query_dim), nn.Linear(self.query_dim, model_dim))
            if self.query_dim is not None
            else None
        )
        self.query_attention = (
            nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
            if self.query_dim is not None
            else None
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, self.input_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a context vector using the newest valid transition as readout.

        ``features`` has shape ``(batch, transitions, feature_dim)``.  ``valid``
        optionally marks valid prefix tokens, allowing future callers to pad
        short episode prefixes without changing their interpretation.
        """
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected transition features (B, T, {self.input_dim}), got {tuple(features.shape)}"
            )
        batch, steps, _ = features.shape
        if steps < 1 or steps > self.max_context_length:
            raise ValueError(
                f"Transition count must be in [1, {self.max_context_length}], got {steps}"
            )
        if valid is None:
            valid = torch.ones(batch, steps, dtype=torch.bool, device=features.device)
        if valid.shape != (batch, steps) or not valid.any(dim=1).all():
            raise ValueError("valid must contain at least one token for every batch item")
        if self.query_dim is None and query is not None:
            raise ValueError("This context encoder was created without a state query")
        if self.query_dim is not None:
            if query is None or query.shape != (batch, self.query_dim):
                raise ValueError(
                    f"Expected state query (B, {self.query_dim}), got "
                    f"{None if query is None else tuple(query.shape)}"
                )

        tokens = self.input_proj(self.input_norm(features)) + self.position[:, :steps]
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        if self.query_proj is not None:
            query_token = self.query_proj(query).unsqueeze(1)
            readout, _ = self.query_attention(
                query_token,
                encoded,
                encoded,
                key_padding_mask=~valid,
                need_weights=False,
            )
            readout = readout.squeeze(1) + query_token.squeeze(1)
        else:
            newest = valid.long().sum(dim=1) - 1
            readout = encoded[torch.arange(batch, device=features.device), newest]
        return self.output_proj(self.output_norm(readout))
