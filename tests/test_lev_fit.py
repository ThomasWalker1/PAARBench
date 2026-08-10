"""LEV's closed-form fit, its leave-one-out veto, and its horizon decay.

The whole method is algebra on sufficient statistics: the fit never sees a design
matrix and the held-out score never re-evaluates a correction on data. That is what
makes it cheap, and it is also exactly the kind of derivation that produces a plausible
number when it is wrong. So each identity is checked against a direct computation that
does hold the data and does refit.

No GPU, no checkpoint: the world model here is a stub whose only job is to have
parameters, a predictor and a rollout for the adapter to wrap.
"""

import numpy as np
import pytest
import torch
from torch import nn

from methods.lev.adapter import LEVAdapter


class _StubWorldModel(nn.Module):
    """Enough of a world model to construct an adapter against.

    ``predict`` is deliberately not the identity: the correction is applied to whatever
    comes out of it, so an identity predictor would hide an error that scaled the
    output.
    """

    def __init__(self, channels=4, action_dim=2):
        super().__init__()
        self.dummy = nn.Linear(2, 2)
        self.concat_dim = 1
        self.num_hist = 3
        self.action_dim = action_dim
        self.proprio_dim = 2
        self.num_action_repeat = 1
        self.num_proprio_repeat = 1
        self.channels = channels

    def predict(self, z):
        return z * 2.0

    def rollout(self, obs_0, act):
        return {}, None


def _adapter(ridge=1.0, decay=0.9, buffer_size=15):
    return LEVAdapter(
        _StubWorldModel(), preprocessor=None,
        ridge=ridge, decay=decay, buffer_size=buffer_size,
    )


def _folds(n_folds=4, batch=2, patches=7, channels=3, seed=0):
    """Random (prediction, residual) folds, plus the adapter's statistics for them."""
    generator = torch.Generator().manual_seed(seed)
    xs = [torch.randn(batch, patches, channels, generator=generator) for _ in range(n_folds)]
    rs = [torch.randn(batch, patches, channels, generator=generator) for _ in range(n_folds)]
    stats = [LEVAdapter._stats(x, r) for x, r in zip(xs, rs)]
    return xs, rs, stats


def _reference_fit(xs, rs, ridge, b, c):
    """Ridge fit of ``r ~ a * u + b`` by explicitly solving the 2x2 normal equations.

    Independent of the adapter's claim that standardizing makes the system diagonal:
    this builds the design matrix and solves it, so if the columns are not actually
    orthogonal the two disagree.
    """
    x = np.concatenate([np.asarray(f[b, :, c]) for f in xs])
    r = np.concatenate([np.asarray(f[b, :, c]) for f in rs])
    n = x.size
    mean = x.mean()
    sigma = np.sqrt(max(x.var(), 0.0) + 1e-8)
    u = (x - mean) / sigma
    design = np.stack([u, np.ones_like(u)], axis=1)
    penalty = np.diag([ridge * n, 0.0])          # slope penalized, intercept is not
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ r)
    return coefficients[0], mean, sigma, coefficients[1]


def test_solve_matches_an_explicitly_solved_ridge_regression():
    adapter = _adapter(ridge=0.3)
    xs, rs, stats = _folds()
    slope, mean, sigma, intercept = adapter._solve(adapter._totals(stats))

    for b in range(slope.shape[0]):
        for c in range(slope.shape[1]):
            want = _reference_fit(xs, rs, 0.3, b, c)
            got = (slope[b, c], mean[b, c], sigma[b, c], intercept[b, c])
            for name, w, g in zip(("slope", "mean", "sigma", "intercept"), want, got):
                assert float(g) == pytest.approx(float(w), rel=1e-4, abs=1e-6), name


def test_large_ridge_degenerates_to_a_pure_offset():
    """The claim the axis rests on: ridge -> inf is the classical disturbance observer."""
    _, rs, stats = _folds()
    total = _adapter(ridge=1e9)._totals(stats)
    slope, _mean, _sigma, intercept = _adapter(ridge=1e9)._solve(total)

    assert float(slope.abs().max()) < 1e-6
    # ...and what survives is the mean residual, channel by channel.
    pooled = torch.cat(rs, dim=1).mean(dim=1)
    assert torch.allclose(intercept, pooled.double(), atol=1e-5)


def test_a_channel_with_no_spread_keeps_its_offset_and_loses_its_slope():
    """Dividing by a spread that is not there does not merely add noise.

    The correction is installed in raw coordinates as ``slope/sigma * x + (intercept -
    slope*mean/sigma)``. When ``sigma`` is near zero those two terms both blow up and
    cancel on application, so an unidentifiable slope destroys the offset that *is*
    identifiable. The floor drops the slope and leaves the offset alone.
    """
    adapter = _adapter(ridge=0.0)
    x = torch.full((1, 7, 2), 5.0)
    x[..., 1] += torch.linspace(0.0, 1.0, 7).reshape(1, 7)   # only channel 1 varies
    for _ in range(4):
        adapter._buffer.append(LEVAdapter._stats(x, torch.full((1, 7, 2), 0.4)))

    slope, _mean, _sigma, intercept = adapter._solve(adapter._totals(list(adapter._buffer)))
    assert float(slope[0, 0]) == 0.0
    assert float(intercept[0, 0]) == pytest.approx(0.4, abs=1e-6)

    adapter._fit()
    assert torch.isfinite(adapter._gain).all()
    assert torch.isfinite(adapter._bias).all()
    # The flat channel still gets its offset, undamaged.
    z = torch.zeros(1, 1, 7, 4)
    assert float(adapter._correct(z, depth=0)[0, 0, 0, 0]) == pytest.approx(0.4, abs=1e-5)


def test_held_out_sse_matches_evaluating_the_correction_on_the_held_out_data():
    adapter = _adapter(ridge=0.7)
    xs, rs, stats = _folds()
    slope, mean, sigma, intercept = adapter._solve(adapter._totals(stats[1:]))

    got = adapter._held_out_sse(stats[0], slope, mean, sigma, intercept)

    delta = (slope.unsqueeze(1) * (xs[0] - mean.unsqueeze(1)) / sigma.unsqueeze(1)
             + intercept.unsqueeze(1))
    want = ((rs[0] - delta) ** 2).sum(dim=1)
    assert torch.allclose(got, want, atol=1e-4)


def test_held_out_sse_of_a_zero_correction_is_the_frozen_error():
    """The comparison kappa makes is against the frozen model, so this has to hold."""
    adapter = _adapter()
    _, _, stats = _folds()
    zero = torch.zeros_like(stats[0].m1)
    got = adapter._held_out_sse(stats[0], zero, zero, torch.ones_like(zero), zero)
    assert torch.allclose(got, stats[0].s, atol=1e-4)


def test_kappa_is_one_when_the_residual_is_a_constant_offset():
    """A correction that generalizes perfectly should not be shrunk at all."""
    adapter = _adapter(ridge=1e6)          # offset-only, which is what the data is
    generator = torch.Generator().manual_seed(1)
    offset = torch.tensor([0.5, -2.0, 3.0])
    for _ in range(4):
        x = torch.randn(2, 7, 3, generator=generator)
        adapter._buffer.append(LEVAdapter._stats(x, offset.expand_as(x).clone()))

    logs = adapter._fit()
    assert logs["lev/kappa"] == pytest.approx(1.0, abs=1e-4)


def test_kappa_is_zero_when_the_correction_does_not_generalize():
    """Residuals that flip sign fold to fold: any fit is worse than doing nothing."""
    adapter = _adapter(ridge=1e6)
    x = torch.ones(2, 7, 3)
    for fold in range(4):
        residual = torch.full_like(x, 1.0 if fold % 2 == 0 else -1.0)
        adapter._buffer.append(LEVAdapter._stats(x, residual))

    logs = adapter._fit()
    assert logs["lev/kappa"] == 0.0
    assert logs["lev/kappa_zero_frac"] == 1.0


def test_a_vetoed_correction_leaves_the_prediction_untouched():
    adapter = _adapter(ridge=1e6)
    x = torch.ones(2, 7, 3)
    for fold in range(4):
        adapter._buffer.append(
            LEVAdapter._stats(x, torch.full_like(x, 1.0 if fold % 2 == 0 else -1.0))
        )
    adapter._fit()

    z = torch.randn(2, 1, 7, 5)
    assert torch.equal(adapter._correct(z, depth=0), z)


def test_the_correction_decays_with_rollout_depth():
    adapter = _adapter(ridge=1e6, decay=0.5)
    generator = torch.Generator().manual_seed(2)
    offset = torch.tensor([0.5, -2.0, 3.0])
    for _ in range(4):
        x = torch.randn(2, 7, 3, generator=generator)
        adapter._buffer.append(LEVAdapter._stats(x, offset.expand_as(x).clone()))
    adapter._fit()

    z = torch.zeros(2, 1, 7, 5)
    depth0 = adapter._correct(z, depth=0)
    depth3 = adapter._correct(z, depth=3)

    # kappa is 1 on this data, so depth 0 applies the offset itself.
    assert torch.allclose(depth0[..., :3], offset.expand(2, 1, 7, 3), atol=1e-4)
    assert torch.allclose(depth3[..., :3], depth0[..., :3] * 0.5 ** 3, atol=1e-4)
    # The action channels are never touched.
    assert torch.equal(depth3[..., 3:], z[..., 3:])


def test_decay_of_one_applies_the_correction_undamped():
    """The null cell of the decay axis: the method without its horizon mechanism."""
    adapter = _adapter(ridge=1e6, decay=1.0)
    x = torch.ones(2, 7, 3)
    for _ in range(4):
        adapter._buffer.append(LEVAdapter._stats(x, torch.full_like(x, 0.25)))
    adapter._fit()

    z = torch.zeros(2, 1, 7, 5)
    assert torch.allclose(adapter._correct(z, 0), adapter._correct(z, 20), atol=1e-6)


def test_a_batch_the_statistics_do_not_cover_is_planned_frozen():
    adapter = _adapter(ridge=1e6)
    x = torch.ones(2, 7, 3)
    for _ in range(4):
        adapter._buffer.append(LEVAdapter._stats(x, torch.full_like(x, 0.25)))
    adapter._fit()

    z = torch.randn(5, 1, 7, 5)              # five episodes, statistics for two
    assert torch.equal(adapter._correct(z, depth=0), z)
    assert adapter._batch_mismatch == 1


def test_predict_is_wrapped_and_respects_the_frozen_context():
    adapter = _adapter(ridge=1e6, decay=1.0)
    x = torch.ones(2, 7, 3)
    for _ in range(4):
        adapter._buffer.append(LEVAdapter._stats(x, torch.full_like(x, 0.25)))
    adapter._fit()

    z = torch.zeros(2, 1, 7, 5)
    corrected = adapter.wm.predict(z)
    assert torch.allclose(corrected[..., :3], torch.full((2, 1, 7, 3), 0.25), atol=1e-6)

    with adapter._frozen():
        assert torch.equal(adapter.wm.predict(z), z * 2.0)


def test_rollout_resets_the_depth_counter():
    adapter = _adapter()
    adapter._depth = 17
    adapter.wm.rollout(obs_0={}, act=None)
    assert adapter._depth == 0


def test_episode_start_clears_the_correction_and_the_buffer():
    adapter = _adapter(ridge=1e6)
    x = torch.ones(2, 7, 3)
    for _ in range(4):
        adapter._buffer.append(LEVAdapter._stats(x, torch.full_like(x, 0.25)))
        adapter._hist.append(torch.zeros(2, 1, 7, 5))
    adapter._fit()
    assert adapter._enabled

    adapter.on_episode_start({}, {})

    assert not adapter._enabled
    assert adapter._corr is None
    assert adapter._kappa_max == 0.0
    assert len(adapter._buffer) == 0
    assert len(adapter._hist) == 0
    z = torch.randn(2, 1, 7, 5)
    assert torch.equal(adapter.wm.predict(z), z * 2.0)


def test_before_plan_does_nothing_until_the_veto_has_something_to_hold_out():
    adapter = _adapter()
    assert adapter.before_plan({})["lev/kappa"] == 0.0

    adapter._buffer.append(LEVAdapter._stats(torch.ones(2, 7, 3), torch.ones(2, 7, 3)))
    assert adapter.before_plan({})["lev/kappa"] == 0.0
    assert adapter._corr is None


# -- end to end over one executed chunk ---------------------------------------
#
# The stub above never exercises `on_transition`, which is where all of LEV's compute
# is and where the latent layout is reconstructed by hand. The stub below makes the
# observation *be* the latent, so the whole path -- frame alignment, obs/action channel
# split, context assembly, statistics -- can be checked against an error the test put
# there on purpose.

_PATCHES, _EMB, _PROPRIO, _ACTION = 4, 6, 2, 2
_CHANNELS = _EMB + _PROPRIO
_STEP = 0.25
_OFFSET = torch.tensor([0.3, -0.7, 1.1, 0.0, 0.5, -0.2, 0.9, -1.3])


class _IdentityPreprocessor:
    """Pass-through, but counting.

    The count is the point. `transform_obs` is not idempotent on real data -- it
    rearranges (b t h w c) to (b t c h w) and resizes -- so an adapter that transforms
    the same frame twice corrupts it, and an identity stub that did not count would
    sail straight past that. It happened.
    """

    def __init__(self):
        self.calls = 0

    def transform_obs(self, obs):
        self.calls += 1
        return {key: value.clone() for key, value in obs.items()}


class _LinearWorldModel(_StubWorldModel):
    """A world model whose one-step error is a known constant.

    The latent of frame ``k`` is ``k * _STEP``; ``predict`` returns the next frame's
    latent minus ``_OFFSET``. So the residual LEV fits is exactly ``_OFFSET`` at every
    step and every patch, and a correct implementation must recover it.
    """

    def __init__(self, frameskip):
        super().__init__(action_dim=_ACTION)
        self.frameskip = frameskip

    def encode_obs(self, obs):
        return {"visual": obs["visual"], "proprio": obs["proprio"]}

    def encode_act(self, act):
        return act

    def predict(self, z):
        step = torch.zeros(z.shape[-1])
        step[:_CHANNELS] = _STEP * self.frameskip - _OFFSET
        return z + step


def _linear_rollout(batch, frames):
    """``frames`` observations of a latent moving at a constant rate."""
    ramp = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1, 1) * _STEP
    visual = ramp.expand(batch, frames, _PATCHES, _EMB).clone()
    proprio = (torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1) * _STEP
               ).expand(batch, frames, _PROPRIO).clone()
    return {"visual": visual, "proprio": proprio}


def test_one_executed_chunk_recovers_the_predictor_error_it_was_given():
    batch, chunk_len, frameskip = 2, 5, 3
    wm = _LinearWorldModel(frameskip)
    adapter = LEVAdapter(wm, _IdentityPreprocessor(), ridge=1e6, decay=1.0)
    adapter.on_episode_start({}, {})

    rollout_obs = _linear_rollout(batch, 1 + chunk_len * frameskip)
    actions = torch.zeros(batch, chunk_len, _ACTION)
    adapter.on_transition({}, actions, rollout_obs, frameskip)

    assert len(adapter._buffer) == chunk_len
    # Exactly one preprocessor pass per bounding frame, and no more.
    assert adapter.preprocessor.calls == chunk_len + 1
    logs = adapter.before_plan({})
    assert logs["lev/kappa"] == pytest.approx(1.0, abs=1e-4)

    # The fitted offset is the error the stub predictor was built with.
    intercept = adapter._corr[4]
    assert torch.allclose(
        intercept, _OFFSET.double().expand(batch, _CHANNELS), atol=1e-4
    )

    # And applying it cancels that error: the corrected prediction is the truth.
    z = torch.zeros(batch, 1, _PATCHES, _CHANNELS + _ACTION)
    corrected = adapter.wm.predict(z)
    assert torch.allclose(
        corrected[..., :_CHANNELS],
        torch.full((batch, 1, _PATCHES, _CHANNELS), _STEP * frameskip),
        atol=1e-4,
    )


def test_a_second_chunk_extends_the_buffer_rather_than_restarting_it():
    batch, chunk_len, frameskip = 2, 5, 3
    adapter = LEVAdapter(_LinearWorldModel(frameskip), _IdentityPreprocessor())
    adapter.on_episode_start({}, {})
    actions = torch.zeros(batch, chunk_len, _ACTION)

    for chunks in (1, 2, 3):
        frames = 1 + chunks * chunk_len * frameskip
        adapter.on_transition({}, actions, _linear_rollout(batch, frames), frameskip)
        assert len(adapter._buffer) == chunks * chunk_len
    # num_hist - 1 real frames are retained as context for the next chunk's first step.
    assert len(adapter._hist) == adapter.num_hist - 1


def test_a_short_rollout_is_an_error_rather_than_a_misaligned_fit():
    adapter = LEVAdapter(_LinearWorldModel(3), _IdentityPreprocessor())
    with pytest.raises(ValueError, match="rollout frames"):
        adapter.on_transition({}, torch.zeros(2, 5, _ACTION), _linear_rollout(2, 4), 3)


def test_rejects_a_token_concatenated_base_model():
    wm = _StubWorldModel()
    wm.concat_dim = 0
    with pytest.raises(NotImplementedError, match="concat_dim"):
        LEVAdapter(wm, preprocessor=None)


@pytest.mark.parametrize("decay", [-0.1, 1.5])
def test_rejects_a_decay_outside_the_unit_interval(decay):
    with pytest.raises(ValueError, match="decay"):
        _adapter(decay=decay)


def test_rejects_a_buffer_too_small_to_hold_anything_out():
    with pytest.raises(ValueError, match="buffer_size"):
        _adapter(buffer_size=1)
