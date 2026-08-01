"""Template adapter. Copy the directory, then replace everything below.

This one does nothing at all, on purpose: it is the smallest thing that satisfies the
protocol, so you can run it end to end before writing any real logic and confirm the
plumbing works. It should score exactly what the frozen baseline scores.
"""

from paarbench.adapter import BaseWeightGuard


class TemplateAdapter:
    def __init__(self, wm, preprocessor, scale=1.0):
        self.wm = wm
        self.preprocessor = preprocessor
        self.scale = float(scale)

        # Snapshot the base model BEFORE anything mutates it, so on_episode_start can
        # put it back exactly. Costs ~400 MB of host RAM; kept off the GPU deliberately.
        self._guard = BaseWeightGuard(wm)

        # Per-episode state. Everything here must be reset in on_episode_start.
        self._buffer = []
        self._replans = 0

    def on_episode_start(self, obs_0, goal):
        """Reset per-episode state and restore the base model."""
        self._guard.restore()
        self._buffer.clear()
        self._replans = 0
        return {}

    def on_transition(self, obs_0, actions, rollout_obs, frameskip):
        """Observe one executed chunk. Buffer here; do the work in before_plan.

        obs_0        modality -> (B, 1, ...)   observation the chunk started from
        actions      (B, T, action_dim)        what was actually executed
        rollout_obs  modality -> (B, 1 + T * frameskip, ...)   the full rollout
        frameskip    simulator steps per planner action

        For just the endpoint: {k: v[:, -1:] for k, v in rollout_obs.items()}
        """
        self._buffer.append(actions.shape[1])
        return {}

    def before_plan(self, obs):
        """Apply or refresh the correction. Called before every solve, including the
        first of an episode, when nothing has been observed yet."""
        self._replans += 1
        return {}

    def metrics(self):
        """Per-replan scalars written to the record. No outcome-derived quantities."""
        return {
            "template/replans": float(self._replans),
            "template/buffered_chunks": float(len(self._buffer)),
        }
