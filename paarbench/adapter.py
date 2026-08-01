"""The standardized test-time adaptation interface.

This is the benchmark's core extension point. A contributed method implements
``TestTimeAdapter`` and nothing else: the planner calls these hooks uniformly and has
no per-method branch, so adding a method never requires editing benchmark internals.

See ``methods/README.md`` for how to contribute one, and ``docs/ADAPTER_PROTOCOL.md``
for why each hook has the signature it has.


What a method may and may not see
---------------------------------

A method may use anything computable *during* an episode from observations and
executed transitions. It may **not** see the episode's success or its final distance
to goal. That is what makes the setting what it is — an episode's outcome is known
only once it is over — and it is enforced structurally: no hook below is passed an
outcome, and the planner never offers one.

Choosing hyperparameters from outcomes is legitimate, but it is a *selection rule*
(``paarbench/selection.py``), it may only read the selection cohort, and its cost is
declared and reported.


Batching
--------

By default the planner evaluates a whole cohort as one batch: every observation below
carries a leading batch dimension of ``n_evals`` and all episodes step in lockstep.
"Episode" hooks are therefore per-batch-of-episodes.

That is fine for a method whose correction is naturally per-episode. It is **wrong**
for a method holding mutable shared state -- model weights and an optimizer -- because
one gradient step then averages unrelated episodes into a single correction. Such a
method declares ``requires_episode_isolation: true`` in its ``method.yaml`` and the
harness gives each episode its own process and its own adapter instance.

The failure mode this prevents is silent: batching such a method produces a plausible
number rather than an error. AdaJEPA scored 0.553 batched against 0.677 isolated on the
same cohort, with nothing in the logs to indicate a problem.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import torch

ObsDict = Dict[str, torch.Tensor]
"""Observation batch: modality -> ``(B, T, ...)``, e.g. ``visual``, ``proprio``."""


@runtime_checkable
class TestTimeAdapter(Protocol):
    """Everything a TTA method may do inside the control loop.

    Implementations must be stateless across episodes after ``on_episode_start``.
    Every hook returns a dict of scalars to log; return ``{}`` if there is nothing
    to say.
    """

    def on_episode_start(self, obs_0: ObsDict, goal: ObsDict) -> Mapping[str, Any]:
        """Reset all per-episode state and undo everything this adapter changed.

        After this returns, nothing the adapter can reach may differ from its state
        before the episode: **base weights it marked trainable** must be restored
        (``BaseWeightGuard`` below does that), and **any correction it installed**
        must be cleared. It is not responsible for changes it did not make.

        Without this, cross-episode leakage silently changes results — an episode
        inheriting the previous one's correction produces a number that looks real.
        ``tests/test_adapter_reset.py`` enforces it per method against a real world
        model: it perturbs the adapter's whole reachable surface, calls this hook, and
        requires bit-identity. Resetting only the part touched this episode passes a
        realistic rollout and fails that test.

        May also apply an initial correction, for methods conditioned only on the
        initial observation.
        """

    def on_transition(
        self,
        obs_0: ObsDict,
        actions: torch.Tensor,
        rollout_obs: ObsDict,
        frameskip: int,
    ) -> Mapping[str, Any]:
        """One executed action chunk and everything the environment produced.

        ``actions`` is ``(B, T, action_dim)``, the actions actually executed this
        replan. ``rollout_obs`` is the *full* observed rollout at simulator
        resolution — ``1 + T * frameskip`` frames including the initial state — so a
        method can align one feature per executed action. Methods that only want the
        endpoint take ``rollout_obs[..., -1:]``.

        This hook is for *observing*. Buffer here; do the work in ``before_plan``.
        """

    def before_plan(self, obs: ObsDict) -> Mapping[str, Any]:
        """Last hook before the planner solves. Apply or refresh any correction.

        Kept separate from ``on_transition`` because that is exactly where methods
        differ — taking an optimizer step on buffered transitions versus regenerating
        a correction from scratch at plan time — and collapsing them loses the
        distinction the compounding-slope metric measures.

        **Any per-replan refresh schedule belongs here, inside the method.**

        Called before every solve, including the first of an episode, at which point
        no transition has been observed yet.
        """

    def metrics(self) -> Mapping[str, float]:
        """Per-replan scalars for the record (loss, correction norm, buffer size...).

        Must not contain anything derived from episode outcome.
        """


class NullAdapter:
    """The do-nothing reference: the frozen base model.

    Reported on every setting, always, as the baseline every metric is relative to.
    Also the trivial conformance case for the protocol.
    """

    def on_episode_start(self, obs_0: ObsDict, goal: ObsDict) -> Mapping[str, Any]:
        return {}

    def on_transition(
        self,
        obs_0: ObsDict,
        actions: torch.Tensor,
        rollout_obs: ObsDict,
        frameskip: int,
    ) -> Mapping[str, Any]:
        return {}

    def before_plan(self, obs: ObsDict) -> Mapping[str, Any]:
        return {}

    def metrics(self) -> Mapping[str, float]:
        return {}


class BaseWeightGuard:
    """Snapshot base-model weights and restore them exactly.

    Provided by the benchmark rather than left to each method, because "restore the
    base model between episodes" is an invariant the harness depends on, and a method
    that gets it subtly wrong contaminates its own results in a way that looks like a
    real effect. ``tests/test_adapter_reset.py`` asserts bit-identity across an
    episode boundary for every registered method.

    Snapshots are kept on the CPU: a full base is ~400 MB and the GPUs are also
    running the planner.

    Usage::

        self._guard = BaseWeightGuard(wm)          # in __init__, before any mutation
        ...
        self._guard.restore()                      # in on_episode_start
    """

    def __init__(self, module: torch.nn.Module):
        self._module = module
        self._snapshot = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in module.state_dict().items()
        }

    def restore(self) -> None:
        with torch.no_grad():
            for name, tensor in self._module.state_dict().items():
                tensor.copy_(self._snapshot[name])

    def is_pristine(self) -> bool:
        """Are the live weights bit-identical to the snapshot?"""
        with torch.no_grad():
            for name, tensor in self._module.state_dict().items():
                if not torch.equal(tensor.detach().cpu(), self._snapshot[name]):
                    return False
        return True

    def drifted(self) -> list:
        """Names of parameters that differ from the snapshot. For test failure messages."""
        with torch.no_grad():
            return [
                name
                for name, tensor in self._module.state_dict().items()
                if not torch.equal(tensor.detach().cpu(), self._snapshot[name])
            ]
