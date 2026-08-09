"""Point-mass MuJoCo maze environments used by the PAARBench maze tracks.

This is intentionally self-contained: it retains the AdaJEPA coordinate convention
and dynamics controls without importing D4RL or executing code from another checkout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


MEDIUM_MAZE = (
    "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
)
OFF_TARGET = np.asarray([10.0, 10.0], dtype=np.float32)


def parse_maze(spec: str) -> np.ndarray:
    lines = spec.strip().split("\\")
    if not lines or len({len(line) for line in lines}) != 1:
        raise ValueError("maze specification must be a non-empty rectangular grid")
    values = {"#": 10, "O": 11, " ": 11, "0": 11, "G": 12}
    try:
        return np.asarray([[values[cell] for cell in line] for line in lines], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unknown maze tile {exc.args[0]!r}") from None


def _model_xml(spec: str, *, density_scale: float, damping_scale: float) -> str:
    maze = parse_maze(spec)
    walls = []
    for row, col in zip(*np.where(maze == 10)):
        walls.append(
            f'<geom name="wall_{row}_{col}" type="box" pos="{row + 1} {col + 1} 0" '
            'size=".5 .5 .2" material="wall" conaffinity="1"/>'
        )
    return f"""<mujoco model="paarbench_point_maze">
  <compiler inertiafromgeom="true" angle="radian" coordinate="local"/>
  <option timestep=".01" gravity="0 0 0" iterations="20" integrator="Euler"/>
  <default><joint damping="{float(damping_scale)}" limited="false"/>
    <geom friction=".5 .1 .1" density="{1000.0 * float(density_scale)}"
          margin=".002" condim="1" contype="2" conaffinity="1"/></default>
  <asset><material name="wall" rgba=".7 .5 .3 1"/><material name="target" rgba=".6 .3 .3 1"/></asset>
  <worldbody>
    <geom name="ground" size="40 40 .25" pos="0 0 -.1" type="plane" contype="1" conaffinity="0"/>
    <body name="particle" pos="1.2 1.2 0"><geom name="particle_geom" type="sphere" size=".1"
        rgba="0 0 1 0" contype="1"/><site name="particle_site" pos="0 0 0" size=".2" rgba=".3 .6 .3 1"/>
      <joint name="ball_x" type="slide" pos="0 0 0" axis="1 0 0"/>
      <joint name="ball_y" type="slide" pos="0 0 0" axis="0 1 0"/></body>
    <site name="target_site" pos="0 0 0" size=".2" material="target"/>
    {" ".join(walls)}
  </worldbody>
  <actuator><motor joint="ball_x" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
             <motor joint="ball_y" ctrlrange="-1 1" ctrllimited="true" gear="100"/></actuator>
</mujoco>"""


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """PointMaze with controlled initialization and dict observations."""

    def __init__(
        self,
        maze_spec: str = MEDIUM_MAZE,
        density_scale: float = 1.0,
        damping_scale: float = 1.0,
        **_: object,
    ):
        self.maze_spec = maze_spec
        self.maze = parse_maze(maze_spec)
        # AdaJEPA's visual maze bases hide the target marker so that the model
        # observes only agent state and layout; goals are supplied as goal images.
        self._target = OFF_TARGET.copy()
        self._init_state: np.ndarray | None = None
        self._render_mode = False
        self._xml = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        self._xml.write(
            _model_xml(
                maze_spec, density_scale=density_scale, damping_scale=damping_scale,
            )
        )
        self._xml.flush()
        self._xml.close()
        mujoco_env.MujocoEnv.__init__(self, self._xml.name, frame_skip=5)
        utils.EzPickle.__init__(self, maze_spec, density_scale, damping_scale)
        self.empty_locations = list(zip(*np.where(self.maze != 10)))
        self.observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
                "proprio": gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            }
        )

    def __del__(self):
        path = getattr(self, "_xml", None)
        if path is not None:
            Path(path.name).unlink(missing_ok=True)

    def _state(self) -> np.ndarray:
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel)).astype(np.float32)

    def _observation(self) -> dict:
        state = self._state()
        visual = self.sim.render(224, 224) if self._render_mode else state
        return {"visual": visual, "proprio": state}

    def _set_target(self, target: np.ndarray) -> None:
        self._target = np.asarray(target[:2], dtype=np.float32)
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.asarray(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def prepare_for_render(self) -> None:
        if self._render_mode:
            return
        self._render_mode = True
        # Match AdaJEPA's first-person setup exactly.  Rendering once creates the
        # MuJoCo context; subsequent start/goal renders retain this top-down view.
        warmup_state = np.asarray([1.0856, 1.9746, 0.0098, 0.0217], dtype=np.float32)
        self.set_state(warmup_state[:2], warmup_state[2:])
        self.sim.render(224, 224)
        if self.sim.render_contexts:
            camera = self.sim.render_contexts[0].cam
            camera.azimuth = 90
            camera.elevation = -90
        self.sim.render(224, 224)

    def prepare(self, seed: int, init_state: np.ndarray):
        # Maze bases are trained on RGB observations.  Gym's default reset path
        # returns state-only placeholders, so enable the offscreen renderer before
        # materializing any planner start/goal context.
        self.prepare_for_render()
        self.seed(seed)
        self._init_state = np.asarray(init_state, dtype=np.float32)
        return self.reset()

    def reset_model(self):
        state = self._init_state
        if state is None:
            cell = self.empty_locations[self.np_random.randint(len(self.empty_locations))]
            state = np.asarray([cell[0], cell[1], 0.0, 0.0], dtype=np.float32)
        self.set_state(state[:2], state[2:4])
        self._set_target(self._target)
        return self._observation(), state

    def reset(self):
        self.sim.reset()
        return self.reset_model()

    def step(self, action):
        self.set_state(self.sim.data.qpos, np.clip(self.sim.data.qvel, -5.0, 5.0))
        self.do_simulation(np.clip(action, -1.0, 1.0), self.frame_skip)
        state = self._state()
        self._set_target(self._target)
        distance = float(np.linalg.norm(state[:2] - self._target))
        return self._observation(), float(np.exp(-distance)), False, {"state": state}

    def step_multiple(self, actions: np.ndarray):
        observations, states = [], []
        for action in actions:
            observation, _, _, info = self.step(action)
            observations.append(observation)
            states.append(info["state"])
        return observations, np.asarray(states)

    def rollout(self, seed: int, init_state: np.ndarray, actions: np.ndarray):
        """Roll one environment; ``SubprocVectorEnv`` distributes batches to workers."""
        obs, state = self.prepare(int(seed), init_state)
        trajectory = {key: [value] for key, value in obs.items()}
        states = [state]
        for action in actions:
            obs, _, _, info = self.step(action)
            for key, value in obs.items():
                trajectory[key].append(value)
            states.append(info["state"])
        return {key: np.asarray(value) for key, value in trajectory.items()}, np.asarray(states)

    def eval_state(self, goal_state: np.ndarray, current_state: np.ndarray) -> dict:
        state_distance = np.linalg.norm(np.asarray(goal_state) - np.asarray(current_state), axis=-1)
        position_distance = np.linalg.norm(
            np.asarray(goal_state)[..., :2] - np.asarray(current_state)[..., :2], axis=-1
        )
        return {"success": position_distance < 0.5, "state_dist": state_distance}
