"""Gymnasium-Robotics PointMaze wrapper used by the generated Minari dataset.

This intentionally has the small legacy interface consumed by PlanEvaluator.
It must not be used for the Temporal Straightening/AdaJEPA paper benchmark.
"""
import numpy as np
import gym


class MinariPointMazeWrapper(gym.Env):
    def __init__(self, density_scale=1.0, damping_scale=1.0, **_kwargs):
        import gymnasium as gymnasium
        import gymnasium_robotics  # noqa: F401

        self._gymnasium = gymnasium
        self._env = gymnasium.make(
            "PointMaze_Medium-v3",
            continuing_task=True,
            reset_target=True,
            max_episode_steps=1_000_000,
            render_mode="rgb_array",
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_dim = 2
        self._goal = None

        # AdaJEPA-style eval-time dynamics shift. On the D4RL maze AdaJEPA rebuilds
        # the MJCF (geom density -> mass+inertia; joint damping). Our medium is the
        # gymnasium/Minari env (fixed at model-build time to match the training
        # renders), so we scale the compiled MjModel in place to the same effect:
        #   density_scale multiplies the point mass body mass and inertia,
        #   damping_scale multiplies the slide-joint damping.
        self.density_scale = float(density_scale)
        self.damping_scale = float(damping_scale)
        if self.density_scale != 1.0 or self.damping_scale != 1.0:
            model = self._base.point_env.model
            if self.density_scale != 1.0:
                model.body_mass[:] *= self.density_scale
                model.body_inertia[:] *= self.density_scale
            if self.damping_scale != 1.0:
                model.dof_damping[:] *= self.damping_scale

    @property
    def _base(self):
        return self._env.unwrapped

    def _set_goal(self, goal):
        self._goal = np.asarray(goal, dtype=np.float64).copy()
        self._base.goal = self._goal.copy()
        self._base.update_target_site_pos()

    def _obs(self):
        state = np.asarray(self._base.point_env.data.qpos.tolist() + self._base.point_env.data.qvel.tolist(), dtype=np.float32)
        frame = np.asarray(self._env.render(), dtype=np.uint8)
        # The Minari generator records this exact 480 -> 224 AREA resize.
        if frame.shape[:2] != (224, 224):
            import cv2
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return {"visual": frame, "proprio": state}

    def sample_distant_cell_init_goal_states(self, seed, min_cell_distance=3.0, cell_jitter=0.25):
        rng = np.random.RandomState(seed)
        # Gymnasium's maze coordinates are cell centres; these are exactly the
        # locations used by the generator's WaypointController.
        cells = np.argwhere(np.asarray(self._base.maze.maze_map) != 1)
        while True:
            start_i, goal_i = rng.choice(len(cells), size=2, replace=False)
            start_cell, goal_cell = cells[start_i], cells[goal_i]
            start = self._base.maze.cell_rowcol_to_xy(start_cell)
            goal = self._base.maze.cell_rowcol_to_xy(goal_cell)
            if np.linalg.norm(start - goal) > min_cell_distance:
                break
        def state(xy):
            return np.concatenate([xy + rng.uniform(-cell_jitter, cell_jitter, size=2), np.zeros(2)]).astype(np.float32)
        return state(start), state(goal)

    def sample_random_init_goal_states(self, seed):
        return self.sample_distant_cell_init_goal_states(seed, min_cell_distance=0.0)

    def prepare(self, seed, init_state):
        init_state = np.asarray(init_state, dtype=np.float64)
        self._env.reset(seed=int(seed))
        self._base.point_env.set_state(init_state[:2], init_state[2:])
        self._set_goal(init_state[:2] if self._goal is None else self._goal)
        return self._obs(), init_state.astype(np.float32)

    def rollout(self, seed, init_state, actions):
        self.prepare(seed, init_state)
        obses = [self._obs()]
        states = [np.asarray(init_state, dtype=np.float32)]
        for action in actions:
            self._env.step(np.asarray(action, dtype=np.float32))
            obses.append(self._obs())
            states.append(np.asarray(self._base.point_env.data.qpos.tolist() + self._base.point_env.data.qvel.tolist(), dtype=np.float32))
        return {key: np.stack([obs[key] for obs in obses]) for key in obses[0]}, np.stack(states)

    def eval_state(self, goal_state, cur_state):
        d = np.linalg.norm(np.asarray(goal_state)[:2] - np.asarray(cur_state)[:2])
        return {"success": d < 0.5, "state_dist": d}

    def update_env(self, _env_info):
        return None

    def close(self):
        self._env.close()
