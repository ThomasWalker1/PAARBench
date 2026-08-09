"""GPU-free checks that context features are built from the right frames.

The transition buffer is what the hypernetwork conditions on, so a feature assembled
from the wrong pair of frames does not fail -- it produces a well-formed correction
conditioned on evidence about a state the episode has left. That is invisible in every
log the harness writes, which is why it is asserted here.

The stub world model below carries each frame's index as its latent value, so the pair
the adapter actually encoded can be read back out of the feature it produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paarbench import methods  # noqa: E402

VISUAL_DIM = 4
PROPRIO_DIM = 2
ACTION_DIM = 2
DIM = VISUAL_DIM + PROPRIO_DIM + ACTION_DIM
PATCHES = 3
FRAMESKIP = 5


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))


class _MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Identity(), nn.Linear(dim, dim), nn.Identity(), nn.Identity(),
            nn.Linear(dim, dim),
        )


class _Transformer(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList([_Attn(dim), _MLP(dim)]) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)


class _Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = _Transformer(dim)


class StubWorldModel(nn.Module):
    """Latents whose value is the index of the frame they were encoded from."""

    def __init__(self):
        super().__init__()
        self.predictor = _Predictor(DIM)
        self.num_hist = 3
        self.concat_dim = 1
        self.action_dim = ACTION_DIM
        self.proprio_dim = PROPRIO_DIM

    def predict(self, z):
        # Identity: the residual in the feature is then (source - target), which
        # makes the encoded frame pair readable straight off the feature vector.
        return z

    def separate_emb(self, z):
        tail = PROPRIO_DIM + ACTION_DIM
        z_obs = {
            "visual": z[..., :-tail],
            "proprio": z[..., -tail:-ACTION_DIM][:, :, 0, :],
        }
        return z_obs, z[..., -ACTION_DIM:]

    def encode_obs(self, obs):
        tag = obs["visual"].reshape(obs["visual"].shape[0], -1)[:, :1]
        return {
            "visual": tag.reshape(-1, 1, 1, 1).expand(-1, 1, PATCHES, VISUAL_DIM),
            "proprio": tag.reshape(-1, 1, 1).expand(-1, 1, PROPRIO_DIM),
        }

    def encode_act(self, act):
        return act

    def encode(self, obs, act):
        z_obs = self.encode_obs(obs)
        proprio = z_obs["proprio"].unsqueeze(2).expand(-1, -1, PATCHES, -1)
        action = act[:, :1].reshape(act.shape[0], 1, 1, -1).expand(-1, -1, PATCHES, -1)
        return torch.cat([z_obs["visual"], proprio, action], dim=-1)


class TaggedPreprocessor:
    def transform_obs(self, obs):
        return {
            "visual": torch.as_tensor(obs["visual"], dtype=torch.float32),
            "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32),
        }


BLOCK = VISUAL_DIM + PROPRIO_DIM


def build(**params):
    """An adapter whose features carry absolute latents, not just their difference.

    ``latent_residual_action`` packs ``[source, prediction, target, residual, action]``.
    Only the absolute blocks identify *which* frames were encoded: a residual alone is
    the same for every correctly-strided pair, so a test reading it could not tell a
    tail-indexed rollout from a front-indexed one.
    """
    wm = StubWorldModel()
    params.setdefault("context_feature_kind", "latent_residual_action")
    return methods.load("hyperjepa").build(
        wm=wm, preprocessor=TaggedPreprocessor(), **params
    )


def trajectory(frames):
    """The replayed rollout: frame ``i`` carries the value ``i``."""
    index = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1, 1, 1)
    return {"visual": index, "proprio": index.reshape(1, frames, 1)}


def encoded_pair(feature):
    """Recover the ``(source, target)`` frame indices one feature was built from."""
    flat = feature.reshape(-1)
    return int(flat[0]), int(flat[2 * BLOCK])


def test_features_come_from_the_tail_of_the_replayed_rollout():
    """The evaluator replays from the episode start, so only the tail is new.

    Replan ``n`` executes the action spanning frames ``5n -> 5n+5``. Reading from the
    front of the rollout instead pairs the same ``0 -> 5`` forever, which is the
    failure this test exists for.
    """
    adapter = build()
    adapter.on_episode_start({}, {})
    pairs = []
    for replan in range(4):
        traj = trajectory(1 + (replan + 1) * FRAMESKIP)
        adapter.on_transition(
            {k: v[:, -1:] for k, v in traj.items()},
            torch.zeros(1, 1, ACTION_DIM), traj, FRAMESKIP,
        )
        pairs.append(encoded_pair(adapter.transition_buffer[-1]))
    assert pairs == [(0, 5), (5, 10), (10, 15), (15, 20)]


def test_a_multi_action_chunk_covers_its_own_frames():
    """Two actions per chunk: the second chunk is 10 -> 15 -> 20, not 0 -> 5 -> 10."""
    adapter = build(transition_buffer_size=8)
    adapter.on_episode_start({}, {})
    for replan in range(2):
        traj = trajectory(1 + (replan + 1) * 2 * FRAMESKIP)
        adapter.on_transition(
            {k: v[:, -1:] for k, v in traj.items()},
            torch.zeros(1, 2, ACTION_DIM), traj, FRAMESKIP,
        )
    assert [encoded_pair(f) for f in adapter.transition_buffer] == [
        (0, 5), (5, 10), (10, 15), (15, 20)
    ]


def test_one_feature_is_appended_per_executed_action():
    adapter = build(transition_buffer_size=8)
    adapter.on_episode_start({}, {})
    logs = adapter.on_transition(
        {k: v[:, -1:] for k, v in trajectory(1 + 3 * FRAMESKIP).items()},
        torch.zeros(1, 3, ACTION_DIM), trajectory(1 + 3 * FRAMESKIP), FRAMESKIP,
    )
    assert logs["hyperjepa/transition_features_added"] == 3
    assert len(adapter.transition_buffer) == 3


def test_a_rollout_too_short_for_the_chunk_is_an_error():
    adapter = build()
    adapter.on_episode_start({}, {})
    traj = trajectory(1 + FRAMESKIP)
    with pytest.raises(ValueError, match="too few to bound"):
        adapter.on_transition(
            {k: v[:, -1:] for k, v in traj.items()},
            torch.zeros(1, 2, ACTION_DIM), traj, FRAMESKIP,
        )


def test_the_buffer_is_dropped_at_the_episode_boundary():
    adapter = build()
    adapter.on_episode_start({}, {})
    traj = trajectory(1 + FRAMESKIP)
    adapter.on_transition(
        {k: v[:, -1:] for k, v in traj.items()},
        torch.zeros(1, 1, ACTION_DIM), traj, FRAMESKIP,
    )
    assert len(adapter.transition_buffer) == 1
    adapter.on_episode_start({}, {})
    assert len(adapter.transition_buffer) == 0
