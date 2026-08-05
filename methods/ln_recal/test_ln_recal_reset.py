"""LN-Recal's own reset and memorylessness checks, against a real base model.

Why these live here rather than being left to ``tests/test_adapter_reset.py``: that
suite's probe finds an installed correction either by scanning ``state_dict()`` for
tensors whose names end in ``affine_delta_weight``/``affine_delta_bias``, or by reading
``lora_A``/``lora_B``/``affine_delta_weight``/``affine_delta_bias`` attributes off a
declared target module. ``HyperLayerNorm.set_affine`` assigns plain tensors, so they never
enter ``state_dict()`` -- and the probe supplies no executed transition, while LN-Recal
correctly installs nothing before it has seen one. So both halves of the generic check
skip rather than fail, and the benchmark's reset guarantee is vacuous for this method.

The guarantee is therefore discharged here, on the mechanism the method actually uses.
Marked ``integration`` because a base checkpoint is needed, so it is excluded from the
default suite and skips in CI (which collects ``methods/`` but has no checkpoint). Run it
with

    .venv/bin/python -m pytest methods/ln_recal -m integration -q
"""

import pytest
import torch

from paarbench import methods
from paarbench.adapter import BaseWeightGuard
from paarbench.settings import SETTINGS

_BASE = SETTINGS["pushobj"].base_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _BASE.is_dir(),
                       reason="no base checkpoint staged; see docs/CHECKPOINTS.md"),
]


class _PassthroughPreprocessor:
    """The adapter only ever calls ``transform_obs``; fabricated obs are already tensors."""

    def transform_obs(self, obs):
        return {key: torch.as_tensor(value).float() for key, value in obs.items()}


@pytest.fixture(scope="module")
def world_model():
    from paarbench.world_model import load_world_model

    return load_world_model(_BASE, "latest")


def _build(world_model, **overrides):
    return methods.load("ln_recal").build(
        wm=world_model, preprocessor=_PassthroughPreprocessor(), **overrides
    )


def _fabricate(world_model, batch=3, steps=2, frameskip=5, seed=0):
    """One executed chunk's worth of observations, shaped as the planner delivers them."""
    generator = torch.Generator().manual_seed(seed)
    device = next(world_model.parameters()).device
    proprio_dim = world_model.proprio_encoder.patch_embed.in_channels
    action_dim = world_model.action_encoder.patch_embed.in_channels
    frames = 1 + steps * frameskip

    def rand(*shape):
        return torch.rand(*shape, generator=generator).to(device)

    obs_0 = {"visual": rand(batch, 1, 3, 224, 224),
             "proprio": rand(batch, 1, proprio_dim)}
    rollout = {"visual": rand(batch, frames, 3, 224, 224),
               "proprio": rand(batch, frames, proprio_dim)}
    rollout["visual"][:, 0] = obs_0["visual"][:, 0]
    rollout["proprio"][:, 0] = obs_0["proprio"][:, 0]
    actions = rand(batch, steps, action_dim)
    return obs_0, actions, rollout, frameskip


def _installed(adapter):
    return (adapter.norm.affine_delta_weight, adapter.norm.affine_delta_bias)


def test_a_transition_installs_a_per_episode_correction(world_model):
    adapter = _build(world_model)
    adapter.on_episode_start({}, {})
    assert _installed(adapter) == (None, None), "frozen until something is observed"

    obs_0, actions, rollout, frameskip = _fabricate(world_model)
    adapter.on_transition(obs_0, actions, rollout, frameskip)
    logs = adapter.before_plan({key: value for key, value in obs_0.items()})

    delta_weight, delta_bias = _installed(adapter)
    assert delta_weight is not None and delta_bias is not None
    # One row per episode: the correction is conditioned on that episode's transitions.
    assert delta_weight.shape == (obs_0["visual"].shape[0], adapter.features)
    assert logs[f"{adapter.log_prefix}/applied"] == 1
    # The action feature block has no observed target and must be left alone.
    assert torch.count_nonzero(delta_weight[:, adapter.fit_features:]) == 0
    assert torch.count_nonzero(delta_bias[:, adapter.fit_features:]) == 0
    # Episodes see different transitions, so they must not receive the same correction.
    assert delta_bias.std(dim=0).abs().max() > 0


def test_the_fit_reduces_what_it_fits(world_model):
    """A least-squares solve that does not lower its own objective is a bug."""
    adapter = _build(world_model, ridge=0.01)
    adapter.on_episode_start({}, {})
    adapter.on_transition(*_fabricate(world_model))
    logs = adapter.before_plan({})
    assert logs[f"{adapter.log_prefix}/fit_residual_ratio"] <= 1.0


def test_ridge_shrinks_the_correction_toward_frozen(world_model):
    """`ridge` is the method's only real knob; large values must approach no-op."""
    deltas = {}
    for ridge in (0.01, 1.0, 1000.0):
        adapter = _build(world_model, ridge=ridge)
        adapter.on_episode_start({}, {})
        adapter.on_transition(*_fabricate(world_model))
        adapter.before_plan({})
        deltas[ridge] = float(torch.cat(_installed(adapter), dim=-1).pow(2).mean().sqrt())
    assert deltas[0.01] > deltas[1.0] > deltas[1000.0]
    assert deltas[1000.0] < 1e-2 * deltas[0.01]


def test_on_episode_start_clears_the_correction_and_the_base_model(world_model):
    """The reset contract, on the surface this method can actually reach."""
    adapter = _build(world_model)
    adapter.on_episode_start({}, {})
    adapter.on_transition(*_fabricate(world_model))
    adapter.before_plan({})
    assert _installed(adapter) != (None, None)

    guard = BaseWeightGuard(world_model)
    # Perturb everything the adapter can reach: the installed correction, and (as a
    # deliberate over-reach) the guarded base affine it is a delta on.
    with torch.no_grad():
        for tensor in _installed(adapter):
            tensor.add_(1.0)
        adapter.norm.base.weight.add_(1e-3)
    assert not guard.is_pristine()

    adapter.on_episode_start({}, {})

    assert _installed(adapter) == (None, None), (
        "the affine delta survived the episode boundary; episode N+1 would start with "
        "episode N's correction installed"
    )
    assert not guard.drifted(), guard.drifted()[:3]
    assert len(adapter.buffer) == 0


def test_reset_is_idempotent(world_model):
    adapter = _build(world_model)
    adapter.on_episode_start({}, {})
    guard = BaseWeightGuard(world_model)
    adapter.on_episode_start({}, {})
    assert guard.is_pristine(), guard.drifted()[:3]


def test_the_fit_is_memoryless(world_model):
    """The claim behind a ~0 compounding slope, tested rather than asserted.

    An installed correction must not influence the next fit. The statistics are taken
    before the affine, so a fit performed with a correction installed has to be
    bit-identical to the same fit performed from the frozen model.
    """
    transition = _fabricate(world_model, seed=1)

    fresh = _build(world_model, ridge=0.1)
    fresh.on_episode_start({}, {})
    fresh.on_transition(*transition)
    fresh.before_plan({})
    reference = tuple(tensor.clone() for tensor in _installed(fresh))

    # Same buffer contents, but reached with a correction already installed and a
    # different correction history behind it.
    warm = _build(world_model, ridge=0.1)
    warm.on_episode_start({}, {})
    warm.on_transition(*_fabricate(world_model, seed=2))
    warm.before_plan({})
    warm.buffer.clear()
    warm.on_transition(*transition)
    warm.before_plan({})

    for installed, expected in zip(_installed(warm), reference):
        assert torch.equal(installed, expected), (
            "the fit depends on the correction that was installed when it ran, so it "
            "accumulates across replans"
        )
