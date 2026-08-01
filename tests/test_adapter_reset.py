"""Base-model restoration across episode boundaries.

The harness depends on an adapter leaving no trace of the previous episode. A method
that gets this subtly wrong produces results that look real, so it is checked rather
than trusted.

``BaseWeightGuard`` is unit-tested here. Whether each *method* actually calls it is an
integration test that needs a real world model and a checkpoint; see
``test_registered_methods_reset`` at the bottom, which skips unless one is staged.
"""

import pytest
import torch
from torch import nn

from paarbench.adapter import BaseWeightGuard, NullAdapter


def _toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))


def test_guard_reports_pristine_before_anything_happens():
    model = _toy()
    assert BaseWeightGuard(model).is_pristine()


def test_guard_detects_drift():
    model = _toy()
    guard = BaseWeightGuard(model)
    with torch.no_grad():
        model[0].weight += 1.0
    assert not guard.is_pristine()
    assert any("0.weight" in name for name in guard.drifted())


def test_guard_restores_bit_identically():
    model = _toy()
    guard = BaseWeightGuard(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    # An optimizer trajectory, i.e. exactly what an online method does.
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
    assert not guard.is_pristine()

    guard.restore()
    assert guard.is_pristine()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), f"{name} not bit-identical"


def test_guard_restores_buffers_not_just_parameters():
    """LayerNorm has no buffers, but BatchNorm's running stats are the classic leak."""
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    guard = BaseWeightGuard(model)
    model.train()
    for _ in range(3):
        model(torch.randn(8, 4))
    assert not guard.is_pristine(), "running stats should have moved"
    guard.restore()
    assert guard.is_pristine()


def test_guard_survives_repeated_restore():
    model = _toy()
    guard = BaseWeightGuard(model)
    for _ in range(3):
        with torch.no_grad():
            model[0].weight += 1.0
        guard.restore()
    assert guard.is_pristine()


def test_null_adapter_never_touches_anything():
    model = _toy()
    guard = BaseWeightGuard(model)
    a = NullAdapter()
    a.on_episode_start({}, {})
    a.before_plan({})
    a.on_transition({}, torch.zeros(1, 1, 2), {}, 5)
    assert guard.is_pristine()
    assert a.metrics() == {}


# -- integration: every registered method must actually reset -----------------
#
# Loads a real base world model, so these are slow and need a staged checkpoint.
# Run them with:  pytest tests/test_adapter_reset.py -m integration
# They are excluded from the default run (see pyproject.toml) so the GPU-free suite
# stays a couple of seconds.

from paarbench import methods  # noqa: E402
from paarbench.settings import SETTINGS  # noqa: E402

_BASE = SETTINGS["pushobj"].base_path


def _shipped_methods():
    try:
        return [m.name for m in methods.discover()]
    except Exception:  # noqa: BLE001 - collection must not fail on a broken method
        return []


@pytest.fixture(scope="module")
def world_model():
    from paarbench.world_model import load_world_model

    return load_world_model(_BASE, "latest")


_CORRECTION_SUFFIXES = ("lora_A", "lora_B", "affine_delta_weight", "affine_delta_bias")


def _reachable(wm, adapter):
    """The tensors an adapter can actually modify.

    Two surfaces: parameters it marked trainable (an optimizer will move those), and
    the correction slots it installed on the model (a generated or folded correction
    lands in those). Anything else in the model is not the adapter's to touch, and a
    method is not responsible for restoring damage it did not cause -- requiring that
    would force every method to carry a full-model snapshot for nothing.
    """
    names = {n for n, p in wm.named_parameters() if p.requires_grad}
    names |= {n for n in wm.state_dict() if n.endswith(_CORRECTION_SUFFIXES)}
    return sorted(names)


@pytest.mark.integration
@pytest.mark.skipif(not _BASE.is_dir(),
                    reason="no base checkpoint staged; see docs/CHECKPOINTS.md")
@pytest.mark.parametrize("name", _shipped_methods())
def test_registered_methods_undo_their_own_mutations(world_model, name):
    """``on_episode_start`` must leave no trace of the previous episode.

    Perturbs the adapter's own reachable surface directly rather than running a
    rollout: what is under test is that the reset works, not that the method adapts.
    Direct perturbation is the stronger check -- a method that resets only the subset
    of its surface it happened to touch this episode passes a realistic rollout and
    fails here.
    """
    method = methods.load(name)
    adapter = method.build(wm=world_model, preprocessor=None)

    guard = BaseWeightGuard(world_model)
    assert guard.is_pristine()

    reachable = _reachable(world_model, adapter)
    assert reachable, (
        f"{name} declares no trainable parameters and installs no correction slots, "
        f"so it has no way to affect the model at all -- which cannot be right for an "
        f"adaptation method."
    )

    state = world_model.state_dict()
    with torch.no_grad():
        for tensor_name in reachable:
            tensor = state[tensor_name]
            if tensor.dtype.is_floating_point:
                tensor.add_(torch.randn_like(tensor) * 1e-3)
    assert not guard.is_pristine(), "perturbation did not take"

    adapter.on_episode_start({}, {})

    drifted = guard.drifted()
    assert not drifted, (
        f"{name}.on_episode_start left {len(drifted)} tensor(s) altered, e.g. "
        f"{drifted[:3]}. A method must undo everything it can change: restore base "
        f"weights with paarbench.adapter.BaseWeightGuard, and clear any correction "
        f"it installed."
    )


@pytest.mark.integration
@pytest.mark.skipif(not _BASE.is_dir(), reason="no base checkpoint staged")
@pytest.mark.parametrize("name", _shipped_methods())
def test_reset_is_idempotent(world_model, name):
    """Two resets in a row must be indistinguishable from one."""
    adapter = methods.load(name).build(wm=world_model, preprocessor=None)
    adapter.on_episode_start({}, {})
    guard = BaseWeightGuard(world_model)
    adapter.on_episode_start({}, {})
    assert guard.is_pristine(), guard.drifted()[:3]


@pytest.mark.integration
@pytest.mark.skipif(not _BASE.is_dir(), reason="no base checkpoint staged")
def test_world_model_loads_frozen_and_in_eval_mode(world_model):
    """The state the planner hands an adapter: no grads, not training."""
    assert not world_model.training
    assert not any(p.requires_grad for p in world_model.parameters())
