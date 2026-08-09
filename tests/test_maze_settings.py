from pathlib import Path

import numpy as np
import torch

from paarbench.maze_targets import (
    diverse_targets,
    medium_hard_targets,
    validate_targets,
    write_targets,
)
from paarbench.runner import build_command
from paarbench.settings import get


def test_maze_ood_settings_inherit_medium_selection():
    medium = get("maze_medium")
    assert medium.has_selection_cohort
    assert get("maze_medium_low_density").inherits_selection_from == "maze_medium"
    assert get("maze_medium_high_damping").inherits_selection_from == "maze_medium"
    assert get("maze_diverse").inherits_selection_from == "maze_medium"


def test_maze_target_paths_are_cohort_specific():
    medium = get("maze_medium")
    assert medium.targets_path("medium", 0) != medium.targets_path("medium", 100)
    assert medium.targets_path("medium", 100) == Path("data/maze_eval/maze_medium/seed_100.pkl")


def test_medium_targets_are_deterministic_and_distant():
    first = medium_hard_targets(seed=100, n_episodes=4)
    second = medium_hard_targets(seed=100, n_episodes=4)
    np.testing.assert_array_equal(first["start_states"], second["start_states"])
    np.testing.assert_array_equal(first["goal_states"], second["goal_states"])
    distances = np.linalg.norm(first["start_states"][:, :2] - first["goal_states"][:, :2], axis=1)
    # Cells are at least 3 units apart; independently sampled ±0.25 jitter is
    # deliberately retained for AdaJEPA parity.
    assert np.all(distances >= 3.0 - np.sqrt(0.5))
    validate_targets(first, expected_n=4)


def test_diverse_targets_use_bfs_band_and_round_robin_layouts(tmp_path):
    specs = {
        7: "#####\\#OOO#\\#OOO#\\#OOO#\\#####",
        9: "#####\\#OOO#\\#OOO#\\#OOO#\\#####",
    }
    path = tmp_path / "maze_specs.pth"
    torch.save(specs, path)
    targets = diverse_targets(path, seed=0, n_episodes=4, min_block_radius=3, max_block_radius=5)
    validate_targets(targets, expected_n=4)
    assert targets["map_indices"].tolist() == [7, 9, 7, 9]
    assert np.all((targets["block_dists"] >= 3) & (targets["block_dists"] <= 5))
    out = tmp_path / "targets.pkl"
    write_targets(out, targets)
    validate_targets(torch.load(out, map_location="cpu"), expected_n=4)


def test_maze_runner_command_uses_immutable_corpus_and_dynamics_override(tmp_path):
    setting = get("maze_medium_low_density")
    command = build_command(
        setting, seed=100, shape="medium", out_dir=tmp_path, n_evals=2,
    )
    assert "goal_source=maze_file" in command
    assert "+maze_target_path=data/maze_eval/maze_medium/seed_100.pkl" in command
    assert "dataset_path=data/point_maze_medium" in command
    assert "++env_kwargs_override.density_scale=0.2" in command
    assert "goal_H=50" in command


def test_diverse_ood_setting_uses_medium_dataset_metadata(tmp_path):
    command = build_command(
        get("maze_diverse"), seed=100, shape="heldout_layouts", out_dir=tmp_path, n_evals=2,
    )
    assert "+maze_target_path=data/maze_eval/maze_diverse/seed_100.pkl" in command
    assert "dataset_path=data/point_maze_medium" in command
    assert "goal_H=50" in command
