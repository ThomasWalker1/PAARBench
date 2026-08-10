"""LEV -- a closed-form latent offset, scaled by a held-out veto, decayed over horizon.

Every other method on this board corrects the *weights*: by gradient (``adajepa``,
``adajepa_v2``, ``pad``) or by a network trained offline to emit them (``hyperlora``,
``static_lora``).  LEV corrects neither.  The predictor is left exactly as it was
loaded, and the correction is an **offset added to what it predicts**, solved in closed
form from the transitions the controller has already executed this episode.  There is
no optimizer, no gradient, no trained checkpoint, and nothing carried between replans.

This is the disturbance observer of offset-free MPC, moved into a learned latent space
and given a way to distrust itself.  Three parts.

**1. The correction is solved, not learned.**  For every executed model step we have
what the frozen predictor said the next latent would be (``x``) and what the encoder
says it actually was (``y``).  LEV fits, per latent channel and pooled over patches,

    y - x  ~=  a * x + b

by ridge regression, and adds that residual back to the predictor's output during
planning.  Standardizing ``x`` per channel makes the slope and intercept columns
orthogonal, so the normal equations are diagonal and the "fit" is six sums and a
division -- no matrix solve, no iteration.  ``ridge`` is the only thing it needs: at
``ridge -> inf`` the slope vanishes and the method degenerates to a pure per-channel
offset, which is the classical disturbance observer; at ``ridge -> 0`` it is ordinary
least squares.  The grid spans both.

**2. It is scaled by evidence it was not fitted on.**  Sufficient statistics are kept
per executed step, so refitting with one step held out is a re-sum of a handful of
``(B, C)`` tensors -- no re-encoding, no forward pass, no iteration.  LEV does that for
every step in the buffer, scores each resulting correction *on the step it was held out
of*, and compares it there against the frozen predictor.  The fraction of held-out
squared error the correction removes,

    kappa = clamp(1 - SSE_corrected / SSE_frozen, 0, 1),

multiplies the correction.  A correction that cannot beat doing nothing out of sample
is multiplied by zero and the planner solves through the frozen model.  This is the
statistically standard form of what ``adajepa_v2`` approximates with a single fresh
transition and a threshold, and unlike that brake it needs no extra forward pass and no
tuned ratio: it is read off statistics already in memory.

**3. It decays over the planning horizon.**  The fit and the veto are both one-step
quantities, and the planner rolls the predictor forward 25 (Push) or 50 (maze) model
steps open loop, feeding each corrected prediction back in as the next step's context.
Trusting a one-step-validated offset equally at depth 25 is exactly what the
compounding column measures.  LEV applies the correction at depth ``t`` with weight
``kappa * decay ** t``, so the planner falls back to the model it was trained with as
it extrapolates past its evidence.  ``decay: 1.0`` -- no decay at all -- is in the
selection grid, so the mechanism has to earn its place rather than be assumed.

Cost.  All the encoding and all the predictor forwards happen in ``on_transition``,
once per executed step, and are reduced immediately to six scalars per channel; the
buffer holds statistics, not tensors.  ``before_plan`` therefore runs no forward pass at
all -- it is arithmetic on ``(B, C)`` tensors.  Nothing is per-episode-mutable in the
model, so a whole cohort adapts in one batch and the method does **not** need
``requires_episode_isolation``.

Scope.  ``concat_dim == 1`` only, which is what all four benchmark base models use.
The correction is affine per latent channel, pooled over the patch axis -- which on
every base model here is one pooled visual token rather than a grid, so the pooling is
nominal and the real restriction is the affine form.
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict, deque

import torch

from paarbench.adapter import BaseWeightGuard
from utils import move_to_device

log = logging.getLogger(__name__)

_EPS = 1e-8

# Smallest across-patch spread, relative to a channel's own magnitude, at which the
# slope of the fit is still identifiable. Below it only the offset is applied.
_SIGMA_FLOOR = 1e-3


class _Stats:
    """Per-executed-step sufficient statistics of the one-step residual.

    Six sums per (batch element, latent channel), pooled over patches.  Everything the
    fit, the leave-one-out refit and the held-out score need is a linear function of
    these, which is why the buffer can drop the latents that produced them.
    """

    __slots__ = ("n", "m1", "m2", "q0", "q1", "s")

    def __init__(self, n, m1, m2, q0, q1, s):
        self.n = n     # float, patches pooled
        self.m1 = m1   # sum x
        self.m2 = m2   # sum x^2
        self.q0 = q0   # sum r
        self.q1 = q1   # sum x*r
        self.s = s     # sum r^2


class LEVAdapter:
    """Closed-form per-channel latent offset with a leave-one-out veto."""

    def __init__(
        self,
        wm,
        preprocessor,
        ridge=1.0,
        decay=0.9,
        buffer_size=15,
        log_prefix="lev",
        **kwargs,
    ):
        self.wm = wm
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device
        self.dtype = next(wm.parameters()).dtype
        self.ridge = float(ridge)
        self.decay = float(decay)
        self.buffer_size = int(buffer_size)
        self.log_prefix = str(log_prefix)

        if self.ridge < 0.0:
            raise ValueError("LEV ridge must be non-negative.")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(
                "LEV decay is a per-model-step trust factor and must lie in [0, 1]; "
                "1.0 applies the correction undamped over the whole horizon."
            )
        if self.buffer_size < 2:
            raise ValueError(
                "LEV buffer_size must be at least 2: the veto refits with one step "
                "held out, which needs at least one step left to fit on."
            )

        self.concat_dim = int(getattr(wm, "concat_dim", 1))
        if self.concat_dim != 1:
            raise NotImplementedError(
                f"LEV supports concat_dim == 1 (channel-concatenated latents), got "
                f"{self.concat_dim}. Every benchmark base model uses 1; a token-"
                f"concatenated model needs a different obs/action split than the one "
                f"below, and shipping an untested branch for it would be worse than "
                f"this error."
            )
        self.num_hist = max(int(getattr(wm, "num_hist", 1) or 1), 1)
        self.action_dim = int(getattr(wm, "action_dim", 0))
        self.proprio_dim = int(getattr(wm, "proprio_dim", 0))
        self.num_action_repeat = int(getattr(wm, "num_action_repeat", 1))
        self.num_proprio_repeat = int(getattr(wm, "num_proprio_repeat", 1))

        # The model is never mutated, so this guard only ever confirms that.  It is
        # constructed and called anyway: "restore the base model" is an invariant the
        # harness tests, and a method that skips it because it believes it is clean is
        # exactly the method that stops being clean without anyone noticing.
        self._guard = BaseWeightGuard(wm)

        # The correction rides on the predictor's *output*, so it is installed by
        # wrapping the two model entry points rather than by touching any weight.
        # `rollout` is where the planner solves, and it is also the only place a
        # meaningful rollout depth exists -- which is what the decay is indexed by.
        self._orig_predict = wm.predict
        self._orig_rollout = wm.rollout
        wm.predict = self._predict_hook
        wm.rollout = self._rollout_hook

        # Per-episode state. All of it is cleared in on_episode_start.
        self._buffer = deque(maxlen=self.buffer_size)
        self._hist = deque(maxlen=max(self.num_hist - 1, 0))
        self._corr = None
        self._gain = None
        self._bias = None
        self._kappa_max = 0.0
        self._enabled = False
        self._depth = 0
        self._batch_mismatch = 0
        self._last_logs = {}

    # -- latent layout ---------------------------------------------------------
    #
    # With concat_dim == 1 a latent frame is (B, T, P, D) with D = C + action_dim,
    # where the leading C channels are the observation part (visual embedding followed
    # by the tiled proprio embedding) and the trailing action_dim channels carry the
    # action. The correction is applied to the first C channels only: the action
    # channels are overwritten by `replace_actions_from_z` at every rollout step and
    # dropped by `separate_emb` at the end, so correcting them is at best wasted and at
    # worst a correction fitted on a quantity the planner then discards.

    def _obs_part(self, z_dct):
        """The observation channels of a latent, built exactly as ``encode`` builds them."""
        visual = z_dct["visual"]                       # (B, T, P, E)
        patches = visual.shape[2]
        proprio = z_dct["proprio"].unsqueeze(2)        # (B, T, 1, p)
        proprio = proprio.expand(-1, -1, patches, -1)
        proprio = proprio.repeat(1, 1, 1, self.num_proprio_repeat)
        return torch.cat([visual, proprio], dim=-1)    # (B, T, P, C)

    def _act_part(self, act, patches):
        act_emb = self.wm.encode_act(act).unsqueeze(2)  # (B, T, 1, a)
        act_emb = act_emb.expand(-1, -1, patches, -1)
        return act_emb.repeat(1, 1, 1, self.num_action_repeat)

    # -- the installed correction ----------------------------------------------

    @contextlib.contextmanager
    def _frozen(self):
        """Evaluate the predictor as loaded, then re-arm the correction.

        The rollout depth is saved too: fitting calls ``predict`` directly rather than
        through ``rollout``, and letting those calls advance the counter would index
        the decay by how much fitting happened rather than by rollout depth.
        """
        enabled, depth = self._enabled, self._depth
        self._enabled = False
        try:
            yield
        finally:
            self._enabled, self._depth = enabled, depth

    def _rollout_hook(self, obs_0, act):
        self._depth = 0
        return self._orig_rollout(obs_0=obs_0, act=act)

    def _predict_hook(self, z):
        z_pred = self._orig_predict(z)
        if self._enabled and self._corr is not None:
            z_pred = self._correct(z_pred, self._depth)
        self._depth += 1
        return z_pred

    def _correct(self, z_pred, depth):
        """``z + kappa * decay**depth * (a * z + b)``, in one pass over the latent.

        Written as a single ``addcmul`` rather than as the fit's own
        ``slope * (x - mean) / sigma + intercept`` because this runs inside the
        planner's inner loop -- once per model step, per gradient step, per replan --
        and each intermediate would be another tensor the size of the latent held live
        by autograd. The gain and bias are folded into raw coordinates at fit time and
        zero-padded across the action channels, so those pass through untouched without
        a slice and a concatenation.
        """
        if z_pred.shape[0] != self._gain.shape[0]:
            # A rollout batched differently from the episodes the statistics came from
            # cannot be matched up element-wise. Planning frozen is the safe reading;
            # broadcasting someone else's correction is not.
            self._batch_mismatch += 1
            return z_pred
        # Bounded on the host from a scalar cached at fit time. Reading the weight off
        # the GPU here instead would synchronize once per predictor call -- tens of
        # thousands of times per replan -- which is the whole latency claim.
        weight = self.decay ** depth
        if self._kappa_max * weight < _EPS:
            return z_pred
        return torch.addcmul(self._bias * weight, z_pred, self._gain * weight + 1.0)

    # -- observation -----------------------------------------------------------

    def _transform(self, obs):
        return move_to_device(self.preprocessor.transform_obs(obs), self.device)

    def _boundary_frames(self, rollout_obs, chunk_len, frameskip):
        """The ``chunk_len + 1`` observations bounding the executed model actions.

        Indexed backwards from the end. ``rollout_obs`` is documented as
        ``1 + T * frameskip`` frames, but the evaluator replays the episode's whole
        action sequence from its initial condition on every replan, so in practice it
        is the entire trajectory so far and only its tail belongs to the chunk that was
        just executed. Counting from the end is correct under either reading; counting
        from the front pairs the current state with the endpoint of the episode's first
        action. (Same reasoning, and the same bug avoided, as in ``adajepa_v2``.)

        Returned as a list, one transformed frame at a time, because that is the unit
        the encoder is run on -- see ``_encode_frames``.
        """
        frames = min(int(value.shape[1]) for value in rollout_obs.values())
        last = frames - 1
        first = last - chunk_len * int(frameskip)
        if first < 0:
            raise ValueError(
                f"LEV needs {1 + chunk_len * frameskip} rollout frames to bound "
                f"{chunk_len} executed action(s) at frameskip {frameskip}, got {frames}."
            )
        return [
            self._transform({k: v[:, idx: idx + 1] for k, v in rollout_obs.items()})
            for idx in range(first, last + 1, int(frameskip))
        ]

    def _encode_frames(self, frames):
        """Encode the chunk's bounding frames, one frame per encoder call.

        Stacking the whole chunk into a single ``encode_obs`` is the obvious
        optimization and it is the wrong one here. The visual encoder is a ViT over
        ``B`` images per frame, so its activations scale with frames-per-call, and the
        benchmark reports peak memory as a frontier column: measured on ``pushobj`` at
        ``B = 50``, six frames in one call peaks at 3849 MB against 642 MB one frame at
        a time, while every predictor call in this hook together accounts for 14 MB.
        What the batched version buys is kernel launches, on a hook that already runs
        at a third of the gradient methods' latency. Six calls of the same total work
        is the right trade.

        Note the returned latent is ``(B, T, P, ...)`` with ``P = 1`` on every base
        model the benchmark ships: these encoders emit one pooled visual token rather
        than a patch grid.
        """
        encoded = [self.wm.encode_obs(frame) for frame in frames]
        return {
            key: torch.cat([frame[key] for frame in encoded], dim=1)
            for key in encoded[0]
        }

    def _frozen_predictions(self, z_full, chunk_len):
        """One-step frozen predictions for every executed step, in as few calls as possible.

        Step ``j`` is predicted from the ``num_hist`` real frames ending at ``j`` --
        the same context width the planner's rollout uses at that depth, taken from the
        trajectory rather than from a single frame. Early episode steps have a shorter
        history, exactly as the planner's own first rollout steps do. Steps sharing a
        context length share one batched predictor call, so a chunk costs at most
        ``num_hist`` forwards however long it is.
        """
        available = list(self._hist) + [z_full[:, j: j + 1] for j in range(chunk_len)]
        offset = len(self._hist)
        by_length = defaultdict(list)
        for j in range(chunk_len):
            end = offset + j
            context = available[max(0, end - self.num_hist + 1): end + 1]
            by_length[len(context)].append((j, torch.cat(context, dim=1)))

        predictions = [None] * chunk_len
        for _length, items in by_length.items():
            stacked = torch.cat([context for _, context in items], dim=0)
            predicted = self._orig_predict(stacked)[:, -1]        # (k*B, P, D)
            chunks = predicted.chunk(len(items), dim=0)
            for (j, _context), prediction in zip(items, chunks):
                predictions[j] = prediction
        return predictions

    @staticmethod
    def _stats(x, r):
        """Reduce one step's (B, P, C) prediction and residual to its six sums.

        Accumulated in double precision. The fit recovers the variance as
        ``m2/n - mean^2`` and the slope numerator as ``q1 - mean*q0``, both differences
        of like-sized quantities; in float32 a channel whose across-patch spread is
        small next to its magnitude loses most of its significant digits there. The
        tensors this produces are ``(B, C)``, so carrying them at double costs nothing
        downstream.
        """
        x, r = x.double(), r.double()
        return _Stats(
            n=float(x.shape[1]),
            m1=x.sum(dim=1),
            m2=(x * x).sum(dim=1),
            q0=r.sum(dim=1),
            q1=(x * r).sum(dim=1),
            s=(r * r).sum(dim=1),
        )

    # -- the fit ---------------------------------------------------------------

    @staticmethod
    def _totals(items):
        total = _Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for item in items:
            total.n += item.n
            total.m1 = total.m1 + item.m1
            total.m2 = total.m2 + item.m2
            total.q0 = total.q0 + item.q0
            total.q1 = total.q1 + item.q1
            total.s = total.s + item.s
        return total

    def _solve(self, total):
        """Ridge fit of ``r ~ a * x + b``, per batch element and per channel.

        Standardizing ``x`` by its own mean and standard deviation over the buffer
        makes the two design columns orthogonal, so the normal equations are diagonal
        and this is division rather than a solve. The intercept is deliberately left
        unpenalized: it is the pure offset, the part of the correction with the most
        evidence behind it, and ``ridge`` exists to shrink the *slope* -- to interpolate
        between an affine correction and the classical disturbance observer.
        """
        mean = total.m1 / total.n
        variance = (total.m2 / total.n - mean * mean).clamp_min(0.0)
        sigma = (variance + _EPS).sqrt()
        intercept = total.q0 / total.n
        slope = ((total.q1 - mean * total.q0) / sigma) / (total.n * (1.0 + self.ridge))

        # A channel whose predicted value barely varies across the buffer has nothing
        # to fit a slope against, and the fit would divide by that near-zero spread. In
        # raw coordinates the correction is `slope/sigma * x + (intercept -
        # slope*mean/sigma)`, two terms that grow as `mean/sigma` and cancel when
        # applied -- so an unidentifiable slope does not merely add noise, it destroys
        # the offset that *is* identifiable. Dropping the slope leaves exactly that
        # offset. The floor is relative to the channel's own scale, and is applied here
        # rather than at the end so the held-out score grades the correction that will
        # actually be installed.
        identifiable = sigma > _SIGMA_FLOOR * (mean.abs() + sigma)
        return slope * identifiable, mean, sigma, intercept

    @staticmethod
    def _held_out_sse(item, slope, mean, sigma, intercept):
        """Squared error of a correction on a step it was not fitted on.

        Written out from the step's own sums rather than from its latents, which is why
        the buffer does not have to keep them: with the correction rewritten in raw
        coordinates as ``delta = a*x + c``,
        ``sum (r - a*x - c)^2 = s - 2a*q1 - 2c*q0 + a^2*m2 + 2ac*m1 + c^2*n``.
        """
        a = slope / sigma
        c = intercept - a * mean
        return (item.s
                - 2.0 * a * item.q1
                - 2.0 * c * item.q0
                + a * a * item.m2
                + 2.0 * a * c * item.m1
                + c * c * item.n)

    def _fit(self):
        """Fit the correction and scale it by what it is worth out of sample."""
        items = list(self._buffer)
        total = self._totals(items)

        corrected = 0.0
        frozen = 0.0
        for index, held_out in enumerate(items):
            rest = self._totals(items[:index] + items[index + 1:])
            slope, mean, sigma, intercept = self._solve(rest)
            corrected = corrected + self._held_out_sse(
                held_out, slope, mean, sigma, intercept
            ).sum(dim=-1)
            frozen = frozen + held_out.s.sum(dim=-1)

        kappa = (1.0 - corrected / frozen.clamp_min(_EPS)).clamp(0.0, 1.0)  # (B,)

        slope, mean, sigma, intercept = self._solve(total)
        self._corr = (kappa, slope, mean, sigma, intercept)

        # Fold the standardization, the trust scale and the action channels into a
        # single gain and bias, once, so the inner loop is one fused op. Zero in the
        # action channels leaves them exactly as the predictor emitted them, which is
        # what `replace_actions_from_z` and `separate_emb` then expect.
        scale = kappa.unsqueeze(-1)
        gain = scale * slope / sigma
        bias = scale * (intercept - slope * mean / sigma)
        pad = (0, self.action_dim)
        self._gain = torch.nn.functional.pad(gain, pad).reshape(
            gain.shape[0], 1, 1, -1).to(self.dtype)
        self._bias = torch.nn.functional.pad(bias, pad).reshape(
            bias.shape[0], 1, 1, -1).to(self.dtype)
        self._kappa_max = float(kappa.max())
        self._enabled = True
        return {
            f"{self.log_prefix}/kappa": float(kappa.mean()),
            f"{self.log_prefix}/kappa_zero_frac": float((kappa <= 0.0).float().mean()),
            f"{self.log_prefix}/offset_norm": float(intercept.norm(dim=-1).mean()),
            f"{self.log_prefix}/slope_norm": float(slope.norm(dim=-1).mean()),
            f"{self.log_prefix}/folds": float(len(items)),
        }

    # -- TestTimeAdapter protocol ----------------------------------------------

    def on_episode_start(self, obs_0, goal):
        """Drop every trace of the previous episode.

        The correction lives entirely in ``self._corr``; no model tensor is ever
        written, so restoring the guard is a confirmation rather than a repair. Both
        happen regardless: the buffer, the history and the correction are what would
        leak, and the guard is what the harness checks.
        """
        self._guard.restore()
        self._buffer.clear()
        self._hist.clear()
        self._corr = None
        self._gain = None
        self._bias = None
        self._kappa_max = 0.0
        self._enabled = False
        self._depth = 0
        self._batch_mismatch = 0
        self._last_logs = {}
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Encode the chunk, score the frozen predictor on it, keep only the statistics.

        All of the method's compute is here, and it is proportional to the chunk rather
        than to anything tunable: one encoder pass per bounding frame, at most
        ``num_hist`` predictor passes, and then the latents are discarded.
        """
        if actions.dim() != 3:
            raise ValueError(f"Expected actions (B, T, D), got {tuple(actions.shape)}")
        chunk_len = int(actions.shape[1])
        if chunk_len < 1:
            return {}

        frames = self._boundary_frames(rollout_obs, chunk_len, frameskip)
        actions = actions.detach().to(self.device)

        with torch.no_grad(), self._frozen():
            encoded = self._encode_frames(frames)                 # chunk_len + 1 frames
            obs_part = self._obs_part(encoded)                    # (B, L, P, C)
            patches = obs_part.shape[2]
            act_part = self._act_part(actions[:, :chunk_len], patches)
            z_full = torch.cat([obs_part[:, :chunk_len], act_part], dim=-1)

            channels = obs_part.shape[-1]
            predictions = self._frozen_predictions(z_full, chunk_len)
            for j, prediction in enumerate(predictions):
                x = prediction[..., :channels]                    # (B, P, C)
                residual = obs_part[:, j + 1] - x
                self._buffer.append(self._stats(x, residual))

            for j in range(chunk_len):
                self._hist.append(z_full[:, j: j + 1])

        return {f"{self.log_prefix}/buffer_size": float(len(self._buffer))}

    def before_plan(self, obs):
        """Refit from the buffered statistics. No forward pass, no gradient, no state.

        The correction is solved from scratch every time rather than updated, so it
        cannot accumulate across replans and there is nothing for a later replan to
        inherit from an earlier one.
        """
        if len(self._buffer) < 2:
            # The veto scores a refit on a step it was held out of, so it needs at
            # least two. Until then LEV plans with the frozen model rather than with an
            # unvalidated correction.
            self._corr = None
            self._gain = None
            self._bias = None
            self._kappa_max = 0.0
            self._enabled = False
            logs = {
                f"{self.log_prefix}/kappa": 0.0,
                f"{self.log_prefix}/buffer_size": float(len(self._buffer)),
            }
            self._last_logs = logs
            return logs

        with torch.no_grad():
            logs = self._fit()
        logs[f"{self.log_prefix}/buffer_size"] = float(len(self._buffer))
        logs[f"{self.log_prefix}/batch_mismatch"] = float(self._batch_mismatch)
        self._last_logs = logs
        return logs

    def metrics(self):
        return dict(self._last_logs)
