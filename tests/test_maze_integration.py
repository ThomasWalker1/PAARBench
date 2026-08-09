"""Opt-in real-artifact smoke checks for the maze extension."""

from pathlib import Path

import pytest

from paarbench.settings import get
from paarbench.world_model import load_world_model


@pytest.mark.integration
@pytest.mark.parametrize("setting_id", ("maze_medium", "maze_diverse"))
def test_maze_checkpoint_loads_with_its_saved_hydra_config(setting_id):
    setting = get(setting_id)
    checkpoint = Path(setting.base_path) / "checkpoints" / "model_latest.pth"
    if not checkpoint.is_file():
        pytest.skip(f"maze checkpoint not staged: {checkpoint}")
    model = load_world_model(setting.base_path, device="cpu")
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
