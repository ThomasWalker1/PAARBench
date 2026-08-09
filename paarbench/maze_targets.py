"""Deterministic, serializable episode definitions for the maze settings.

AdaJEPA samples maze starts and goals in each evaluation process.  That makes an
episode depend on the sampler and its dependency versions.  PAARBench writes the
samples once and evaluates every method against exactly those definitions instead.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


MEDIUM_HARD_CELLS = np.asarray(
    [
        (1, 1), (1, 2), (1, 5), (1, 6), (2, 1), (2, 2), (2, 4),
        (2, 5), (2, 6), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2),
        (4, 4), (4, 5), (4, 6), (5, 1), (5, 3), (5, 4), (5, 6),
        (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
    ],
    dtype=np.float32,
)
QVEL_LOW, QVEL_HIGH = -5.2262554, 5.2262554


def _state_from_cell(rng: np.random.RandomState, cell: np.ndarray) -> np.ndarray:
    pos = cell + rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    velocity = rng.uniform(QVEL_LOW, QVEL_HIGH, size=2).astype(np.float32)
    return np.concatenate((pos, velocity)).astype(np.float32)


def medium_hard_targets(seed: int, n_episodes: int) -> dict:
    """Match AdaJEPA's PointMaze-Medium hard-goal sampler."""
    starts, goals = [], []
    for episode in range(n_episodes):
        rng = np.random.RandomState(seed * n_episodes + episode + 1)
        # Retain the upstream parity draw so its published protocol is reproduced.
        rng.random()
        while True:
            start_idx, goal_idx = rng.choice(len(MEDIUM_HARD_CELLS), size=2, replace=False)
            if np.linalg.norm(MEDIUM_HARD_CELLS[start_idx] - MEDIUM_HARD_CELLS[goal_idx]) >= 3.0:
                break
        starts.append(_state_from_cell(rng, MEDIUM_HARD_CELLS[start_idx]))
        goals.append(_state_from_cell(rng, MEDIUM_HARD_CELLS[goal_idx]))
    return {
        "schema_version": 1,
        "environment": "point_maze_medium",
        "source": "medium_hard",
        "seed": seed,
        "start_states": np.stack(starts),
        "goal_states": np.stack(goals),
    }


def _open_cells(spec: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    rows = spec.strip().split("\\")
    grid = np.asarray([[cell in {"O", " ", "0", "G"} for cell in row] for row in rows])
    return list(zip(*np.where(grid))), grid


def _bfs(grid: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_cell = row + d_row, col + d_col
            if (
                0 <= next_cell[0] < grid.shape[0]
                and 0 <= next_cell[1] < grid.shape[1]
                and grid[next_cell]
                and next_cell not in distances
            ):
                distances[next_cell] = distances[(row, col)] + 1
                queue.append(next_cell)
    return distances


def _diverse_state(rng: np.random.Generator, cell: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        [cell[0] + rng.uniform(-0.1, 0.1), cell[1] + rng.uniform(-0.1, 0.1), 0.0, 0.0],
        dtype=np.float32,
    )


def diverse_targets(
    maze_specs_path: Path,
    *,
    seed: int,
    n_episodes: int,
    min_block_radius: int = 3,
    max_block_radius: int = 5,
) -> dict:
    """Generate balanced, BFS-distance-controlled targets from a layout corpus."""
    specs = torch.load(maze_specs_path, map_location="cpu")
    map_indices = sorted(int(index) for index in specs)
    if not map_indices:
        raise ValueError(f"{maze_specs_path} contains no maze specifications")
    rng = np.random.default_rng(seed)
    starts, goals, chosen_specs, chosen_maps, block_dists = [], [], [], [], []
    for episode in range(n_episodes):
        map_index = map_indices[episode % len(map_indices)]
        spec = specs[map_index]
        cells, grid = _open_cells(spec)
        if not cells:
            raise ValueError(f"maze {map_index} contains no open cells")
        candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for start in cells:
            distances = _bfs(grid, start)
            for goal, distance in distances.items():
                if min_block_radius <= distance <= max_block_radius:
                    candidates.append((start, goal, distance))
        if not candidates:
            raise ValueError(
                f"maze {map_index} has no goal within BFS band "
                f"[{min_block_radius}, {max_block_radius}]"
            )
        start, goal, distance = candidates[int(rng.integers(len(candidates)))]
        starts.append(_diverse_state(rng, start))
        goals.append(_diverse_state(rng, goal))
        chosen_specs.append(spec)
        chosen_maps.append(map_index)
        block_dists.append(distance)
    return {
        "schema_version": 1,
        "environment": "diverse_maze",
        "source": "diverse_bfs",
        "seed": seed,
        "start_states": np.stack(starts),
        "goal_states": np.stack(goals),
        "maze_specs": chosen_specs,
        "map_indices": np.asarray(chosen_maps, dtype=np.int64),
        "block_dists": np.asarray(block_dists, dtype=np.int64),
    }


def validate_targets(targets: dict, *, expected_n: int | None = None) -> None:
    required = {"schema_version", "environment", "start_states", "goal_states"}
    missing = required.difference(targets)
    if missing:
        raise ValueError(f"maze targets are missing keys: {sorted(missing)}")
    starts, goals = np.asarray(targets["start_states"]), np.asarray(targets["goal_states"])
    if starts.ndim != 2 or starts.shape != goals.shape or starts.shape[1] != 4:
        raise ValueError("maze target states must have matching shape (episodes, 4)")
    if expected_n is not None and len(starts) != expected_n:
        raise ValueError(f"maze targets define {len(starts)} episodes, expected {expected_n}")
    if targets["environment"] == "diverse_maze":
        specs = targets.get("maze_specs", [])
        if len(specs) != len(starts):
            raise ValueError("diverse maze targets need one maze_specs entry per episode")


def write_targets(path: Path, targets: dict) -> None:
    """Validate then atomically stage a target corpus."""
    validate_targets(targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(targets, temporary)
    temporary.replace(path)
