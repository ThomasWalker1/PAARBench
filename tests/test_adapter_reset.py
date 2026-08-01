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


def _world_model_available():
    from paarbench.settings import SETTINGS

    return any(s.base_path.is_dir() for s in SETTINGS.values() if s.enabled)


@pytest.mark.skipif(not _world_model_available(),
                    reason="no base checkpoint staged; see docs/CHECKPOINTS.md")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_registered_methods_reset():
    """Placeholder for the cross-episode bit-identity check on real adapters.

    Not yet implemented: constructing a world model outside plan.py's Hydra
    machinery needs a loader the harness does not expose yet. Until it does, the
    guarantee rests on each method calling BaseWeightGuard.restore() in
    on_episode_start, which is unit-tested above but not enforced per method.
    """
    pytest.skip("world-model loader not yet factored out of plan.py")
