"""GPU-free checks of HOVER's two pieces of index arithmetic.

Both are the kind of thing that fails quietly rather than loudly: a rollout window
that trains on the wrong target, or a chunk boundary read off the wrong end of the
observed trajectory, still produces a loss that goes down and a number that looks
like a result. Neither needs a checkpoint, so they run in the GPU-free suite.

The stub world model below is the smallest thing ``install_predictor_hyper_targets``
and the adapter will accept. Every observation carries its own frame index as its
value, so the pairing the adapter chose can be read back out of the tensors it
compared.
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
    """``transformer.layers.<i>.0.to_qkv`` and friends -- the names the installer walks."""

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
    """Latents whose value is the index of the frame they came from."""

    def __init__(self):
        super().__init__()
        self.predictor = _Predictor(DIM)
        self.num_hist = 3
        self.concat_dim = 1
        self.action_dim = ACTION_DIM
        self.proprio_dim = PROPRIO_DIM
        self.emb_criterion = nn.MSELoss()
        self.predict_calls = []

    # The adapter reaches the correction through predict(), so the stub routes
    # through the wrapped modules rather than around them: the LoRA and LayerNorm
    # deltas are in the graph, and a gradient step is a real gradient step.
    def predict(self, z):
        self.predict_calls.append(tuple(z.shape))
        attn = self.predictor.transformer.layers[-1][0].to_qkv
        return self.predictor.transformer.norm(attn(z)[..., :DIM])

    def separate_emb(self, z):
        tail = PROPRIO_DIM + ACTION_DIM
        z_obs = {
            "visual": z[..., :-tail],
            "proprio": z[..., -tail:-ACTION_DIM][:, :, 0, :],
        }
        return z_obs, z[..., -ACTION_DIM:]

    def encode_obs(self, obs):
        tag = obs["visual"].reshape(-1)[0]
        batch = obs["visual"].shape[0]
        return {
            "visual": torch.full((batch, 1, PATCHES, VISUAL_DIM), float(tag)),
            "proprio": torch.full((batch, 1, PROPRIO_DIM), float(tag)),
        }

    def encode(self, obs, act):
        z_obs = self.encode_obs(obs)
        proprio = z_obs["proprio"].unsqueeze(2).expand(-1, -1, PATCHES, -1)
        action = act[:, :1].reshape(act.shape[0], 1, 1, -1).expand(-1, -1, PATCHES, -1)
        return torch.cat([z_obs["visual"], proprio, action], dim=-1)


class TaggedPreprocessor:
    """``transform_obs`` for observations that are just their own frame index."""

    def transform_obs(self, obs):
        return {
            "visual": torch.as_tensor(obs["visual"], dtype=torch.float32),
            "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32),
        }


def build(**params):
    wm = StubWorldModel()
    adapter = methods.load("adajepa_v2").build(
        wm=wm, preprocessor=TaggedPreprocessor(), **params
    )
    return wm, adapter


def trajectory(frames):
    """``rollout_obs`` for a replayed episode: frame ``i`` has value ``i``."""
    index = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1, 1, 1)
    return {"visual": index, "proprio": index.reshape(1, frames, 1)}


def tags(item_target):
    return float(item_target["proprio"].reshape(-1)[0])


def test_chunk_boundaries_are_read_from_the_end_of_the_rollout():
    """The evaluator replays the whole episode, so only the tail is the new chunk.

    After three replans at frameskip 5 the executed action spans frames 10 -> 15.
    Reading from the front would pair the current state with frame 5, the endpoint of
    the episode's *first* action, and keep doing so for the rest of the episode.
    """
    _wm, adapter = build()
    adapter.on_episode_start({}, {})
    for replan in range(3):
        adapter.on_transition(
            {}, torch.zeros(1, 1, ACTION_DIM), trajectory(1 + (replan + 1) * 5), 5
        )
    # One transition per replan, targeting frames 5, 10, 15.
    assert [tags(item["z_tgt"]) for item in adapter._buffer] == [5.0, 10.0, 15.0]


def test_a_multi_action_chunk_becomes_one_transition_per_action():
    _wm, adapter = build()
    adapter.on_episode_start({}, {})
    adapter.on_transition({}, torch.zeros(1, 3, ACTION_DIM), trajectory(1 + 3 * 5), 5)
    assert [tags(item["z_tgt"]) for item in adapter._buffer] == [5.0, 10.0, 15.0]


def test_a_rollout_too_short_for_the_chunk_is_an_error_not_a_guess():
    _wm, adapter = build()
    adapter.on_episode_start({}, {})
    with pytest.raises(ValueError, match="rollout frames"):
        adapter.on_transition({}, torch.zeros(1, 2, ACTION_DIM), trajectory(6), 5)


def _observed_targets(adapter, horizon):
    """Which targets each rollout depth was scored against."""
    seen = []
    original = adapter._step_loss

    def spy(z_pred, targets):
        seen.append([tags(t) for t in targets])
        return original(z_pred, targets)

    adapter._step_loss = spy
    _loss, depths = adapter._rollout_loss(horizon)
    adapter._step_loss = original
    return seen, depths


def _fill_buffer(adapter, transitions):
    adapter.on_episode_start({}, {})
    for replan in range(transitions):
        adapter.on_transition(
            {}, torch.zeros(1, 1, ACTION_DIM), trajectory(1 + (replan + 1) * 5), 5
        )


def test_every_window_is_rolled_to_the_target_it_can_reach():
    """Depth ``j`` keeps windows ``i <= n-1-j``, and scores window ``i`` on step ``i+j``.

    With four transitions (targets 5, 10, 15, 20) and horizon 3: depth 0 is the
    single-step term on all four, depth 1 drops the newest start, depth 2 drops the
    two newest. Anything else is training on a target the window cannot reach.
    """
    _wm, adapter = build(horizon=3)
    _fill_buffer(adapter, 4)
    seen, depths = _observed_targets(adapter, horizon=3)
    assert depths == 3
    assert seen == [
        [5.0, 10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [15.0, 20.0],
    ]


def test_horizon_one_is_the_single_step_objective():
    """The arm the selection grid contains, so it has to be exactly that arm."""
    _wm, adapter = build(horizon=1)
    _fill_buffer(adapter, 4)
    seen, depths = _observed_targets(adapter, horizon=1)
    assert depths == 1
    assert seen == [[5.0, 10.0, 15.0, 20.0]]


def test_horizon_deeper_than_the_buffer_stops_at_the_buffer():
    _wm, adapter = build(horizon=8)
    _fill_buffer(adapter, 3)
    seen, depths = _observed_targets(adapter, horizon=8)
    assert depths == 3
    assert seen == [[5.0, 10.0, 15.0], [10.0, 15.0], [15.0]]


def test_context_grows_to_num_hist_like_the_planner_rollout():
    """A deep window must be predicted from history, not from one frame each time."""
    wm, adapter = build(horizon=5, buffer_size=8)
    _fill_buffer(adapter, 6)
    wm.predict_calls.clear()
    adapter._rollout_loss(horizon=5)
    # (windows, context frames): one call per depth, context capped at num_hist.
    assert [(shape[0], shape[1]) for shape in wm.predict_calls] == [
        (6, 1), (5, 2), (4, 3), (3, 3), (2, 3),
    ]


def test_the_brake_only_fires_once_a_correction_exists():
    """At a zero delta the ratio is exactly 1, so nothing to brake."""
    _wm, adapter = build(brake_ratio=1.001)
    _fill_buffer(adapter, 1)
    logs = adapter.before_plan({})
    assert logs["adajepa_v2/fresh_loss_ratio"] == pytest.approx(1.0)
    assert logs["adajepa_v2/braked"] == 0.0


def test_the_brake_resets_the_correction_and_the_optimizer():
    _wm, adapter = build(brake_ratio=1.001, pred_lr=1.0)
    _fill_buffer(adapter, 3)
    adapter.before_plan({})              # fits; the delta becomes nonzero
    assert any(float(t.abs().sum()) > 0 for t in adapter._zeroable)
    optimizer = adapter.optimizer
    adapter._reset_correction()
    assert all(float(t.abs().sum()) == 0 for t in adapter._zeroable)
    assert adapter.optimizer is not optimizer, "Adam moments survived the reset"


def test_a_brake_ratio_at_or_below_one_is_refused():
    with pytest.raises(ValueError, match="brake_ratio"):
        build(brake_ratio=1.0)
