import torch

from datasets.pusht_dset import (
    ACTION_MEAN,
    PROPRIO_MEAN,
    STATE_MEAN,
    load_pusht_evaluation_metadata,
)


def test_evaluation_metadata_uses_fixed_training_constants_without_data():
    transform = object()
    _, trajectories = load_pusht_evaluation_metadata(
        transform=transform,
        data_path="a/path/that/does/not/exist",
        with_velocity=True,
    )
    metadata = trajectories["valid"]

    assert metadata.transform is transform
    assert (metadata.action_dim, metadata.state_dim, metadata.proprio_dim) == (2, 7, 4)
    assert torch.equal(metadata.action_mean, ACTION_MEAN)
    assert torch.equal(metadata.state_mean, STATE_MEAN)
    assert torch.equal(metadata.proprio_mean, PROPRIO_MEAN)


def test_evaluation_metadata_respects_velocity_and_normalization_options():
    _, trajectories = load_pusht_evaluation_metadata(
        with_velocity=False,
        normalize_action=False,
    )
    metadata = trajectories["valid"]

    assert (metadata.action_dim, metadata.state_dim, metadata.proprio_dim) == (2, 5, 2)
    assert torch.equal(metadata.action_mean, torch.zeros(2))
    assert torch.equal(metadata.state_std, torch.ones(5))
    assert torch.equal(metadata.proprio_std, torch.ones(2))
