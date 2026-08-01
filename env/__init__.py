from gym.envs.registration import register
import os
from pathlib import Path

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# Alias used by AdaJEPA's released bases (env.name="pushobj"); same wrapper/env,
# shape set per-episode via update_env. Mirrors adajepa/env/__init__.py.
register(
    id="pushobj",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

glew_candidates = [
    Path("/usr/include/GL/glew.h"),
    Path("/usr/local/include/GL/glew.h"),
]

local_deps_dir = os.environ.get(
    "LOCAL_DEPS_DIR", str(Path(__file__).resolve().parent.parent / ".local-deps")
)
glew_candidates.append(Path(local_deps_dir) / "usr/include/GL/glew.h")

for include_root in os.environ.get("CPATH", "").split(":"):
    if include_root:
        glew_candidates.append(Path(include_root) / "GL/glew.h")

try:
    if not any(path.exists() for path in glew_candidates):
        raise ImportError("GL/glew.h was not found")
    from .pointmaze import U_MAZE, MEDIUM_MAZE

    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )

    # AdaJEPA-style dynamics-shift variants of the umaze env. Identical maze and
    # task to `point_maze`; only MuJoCo physics differ, for eval-time OOD tests.
    _umaze_shift_base = {
        "maze_spec": U_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 23.85,
        "ref_max_score": 161.86,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
    }
    register(
        id="point_maze_lowmass",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=300,
        kwargs={**_umaze_shift_base, "mass_scale": 0.2},
    )
    register(
        id="point_maze_highdamp",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=300,
        kwargs={**_umaze_shift_base, "damping_scale": 20.0},
    )

    register(
        id="point_maze_medium",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=600,
        kwargs={
            "maze_spec": MEDIUM_MAZE,
            "reward_type": "sparse",
            "reset_target": False,
            "ref_min_score": 13.13,
            "ref_max_score": 277.39,
            "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5",
        },
    )
except Exception as exc:
    print(f"Skipping PointMaze registration because MuJoCo is unavailable: {exc}")

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="point_maze_medium_minari",
    entry_point="env.minari_pointmaze:MinariPointMazeWrapper",
    max_episode_steps=1_000_000,
)

# AdaJEPA-style eval-time dynamics-shift variant of the Minari medium maze.
# Same maze/task/renders as point_maze_medium_minari; only MuJoCo physics differ.
# NOTE: only DAMPING is a meaningful axis here. This gymnasium PointMaze is
# damping-dominated (the point reaches terminal velocity ~= force/damping almost
# immediately), so its motion is independent of mass -- density_scale is inert on
# this env (verified: 25x mass change -> identical trajectory, even after
# mj_setConst). AdaJEPA's D4RL maze differs (lower damping, so mass matters), but
# we only ship the axis that actually shifts our env.
register(
    id="point_maze_medium_minari_highdamp",
    entry_point="env.minari_pointmaze:MinariPointMazeWrapper",
    max_episode_steps=1_000_000,
    kwargs={"damping_scale": 20.0},
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
