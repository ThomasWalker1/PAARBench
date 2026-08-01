import os
import numpy as np
import gym
from env.pointmaze.maze_model import MazeEnv, U_MAZE, U_MAZE_EVAL
from utils import aggregate_dct

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

class PointMazeWrapper(MazeEnv):
    def __init__(self, **kwargs):
        # mass_scale / damping_scale (AdaJEPA-style dynamics shifts) are consumed
        # by MazeEnv, which applies them at model-build time (see maze_model.py).
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        if getattr(self, "str_maze_spec", None) in (U_MAZE, U_MAZE_EVAL):
            def generate_state_umaze():
                valid = False
                while not valid:
                    x = rs.uniform(0.5, 3.1)
                    y = rs.uniform(0.5, 3.1)
                    valid = (
                        ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))
                        or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
                    )
                state = np.array([
                    x,
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                ])
                return state

            return generate_state_umaze(), generate_state_umaze()

        def sample_location(exclude=None):
            loc = self.empty_and_goal_locations[rs.randint(len(self.empty_and_goal_locations))]
            if exclude is not None and tuple(loc) == tuple(exclude):
                loc = self.empty_and_goal_locations[rs.randint(len(self.empty_and_goal_locations))]
            return np.array(loc)

        init_loc = sample_location()
        goal_loc = sample_location(exclude=init_loc)

        def loc_to_state(loc_xy):
            qpos = loc_xy + rs.uniform(low=-0.25, high=0.25, size=2)
            qvel = np.array([
                rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
            ])
            return np.concatenate([qpos, qvel], axis=0)

        return loc_to_state(init_loc), loc_to_state(goal_loc)

    def sample_distant_cell_init_goal_states(
        self,
        seed,
        min_cell_distance=3.0,
        cell_jitter=0.25,
    ):
        """Sample start/goal states from distinct open cells far apart in the maze."""
        rs = np.random.RandomState(seed)
        locations = np.array(self.empty_and_goal_locations, dtype=np.float32)
        if len(locations) < 2:
            raise ValueError("PointMaze needs at least two open cells for distant goals.")

        for _ in range(10000):
            init_idx, goal_idx = rs.choice(len(locations), size=2, replace=False)
            init_loc = locations[init_idx]
            goal_loc = locations[goal_idx]
            if np.linalg.norm(init_loc - goal_loc) > float(min_cell_distance):
                break
        else:
            raise ValueError(
                f"Failed to sample PointMaze cells farther than {min_cell_distance}."
            )

        def loc_to_state(loc_xy):
            qpos = loc_xy + rs.uniform(
                low=-float(cell_jitter),
                high=float(cell_jitter),
                size=2,
            )
            qvel = np.zeros(2, dtype=np.float32)
            return np.concatenate([qpos, qvel], axis=0).astype(np.float32)

        return loc_to_state(init_loc), loc_to_state(goal_loc)
    
    def sample_bfs_distance_init_goal_states(
        self,
        seed,
        min_cell_distance=3,
        max_cell_distance=5,
        cell_jitter=0.2,
    ):
        """Sample start/goal from open cells at a controlled BFS (shortest-path)
        cell distance. Unlike sample_distant_cell_init_goal_states (euclidean, can
        pick cross-wall / unreachable goals -> the medium 0% dead end), this uses
        4-connected BFS over the maze grid so goals are genuinely reachable at a
        bounded difficulty. Used for held-out-layout eval (Track B / B5)."""
        from collections import deque

        rs = np.random.RandomState(seed)
        arr = self.maze_arr  # WALL=10, EMPTY=11, GOAL=12
        H, W = arr.shape

        def navigable(r, c):
            return 0 <= r < H and 0 <= c < W and arr[r, c] != 10

        opens = [tuple(map(int, loc)) for loc in self.empty_and_goal_locations]

        def bfs(src):
            d = {src: 0}
            q = deque([src])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if navigable(nr, nc) and (nr, nc) not in d:
                        d[(nr, nc)] = d[(r, c)] + 1
                        q.append((nr, nc))
            return d

        for _ in range(10000):
            src = opens[rs.randint(len(opens))]
            dist = bfs(src)
            cands = [c for c, dd in dist.items()
                     if min_cell_distance <= dd <= max_cell_distance]
            if cands:
                goal = cands[rs.randint(len(cands))]
                break
        else:
            raise ValueError(
                f"No open cell pair at BFS distance [{min_cell_distance},"
                f"{max_cell_distance}] in this maze."
            )

        def loc_to_state(loc_xy):
            qpos = np.asarray(loc_xy, np.float32) + rs.uniform(
                -float(cell_jitter), float(cell_jitter), size=2
            )
            return np.concatenate([qpos, np.zeros(2, np.float32)]).astype(np.float32)

        return loc_to_state(src), loc_to_state(goal)

    def update_env(self, env_info):
        pass
    
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
