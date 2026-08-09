"""Multi-layout PointMaze loader used by the diverse-maze checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from .point_maze_dset import PointMazeDataset
from .traj_dset import TrajSlicerDataset, get_train_val_sliced


class DiverseMazeDataset(PointMazeDataset):
    """PointMaze tensors plus the layout that generated each trajectory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(self.data_path)
        indices, specs = root / "map_indices.pth", root / "maze_specs.pth"
        self.map_indices = torch.load(indices, map_location="cpu") if indices.is_file() else None
        self.maze_specs = torch.load(specs, map_location="cpu") if specs.is_file() else None
        if self.map_indices is not None:
            self.map_indices = self.map_indices[: len(self)]

    def get_map_idx(self, episode_index: int):
        return None if self.map_indices is None else int(self.map_indices[episode_index])

    def get_maze_spec(self, episode_index: int):
        if self.maze_specs is None or self.map_indices is None:
            return None
        return self.maze_specs[self.get_map_idx(episode_index)]


def load_diverse_maze_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/diverse_maze",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_preprocessed=False,
    use_frame_files=False,
):
    root = Path(data_path)
    common = dict(
        n_rollout=n_rollout, transform=transform, normalize_action=normalize_action,
        use_preprocessed=use_preprocessed, use_frame_files=use_frame_files,
    )
    if (root / "train").is_dir() and (root / "val").is_dir():
        train = DiverseMazeDataset(data_path=str(root / "train"), **common)
        valid = DiverseMazeDataset(data_path=str(root / "val"), **common)
        for attribute in (
            "action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std",
        ):
            setattr(valid, attribute, getattr(train, attribute))
        frame_count = num_hist + num_pred
        return (
            {"train": TrajSlicerDataset(train, frame_count, frameskip),
             "valid": TrajSlicerDataset(valid, frame_count, frameskip)},
            {"train": train, "valid": valid},
        )
    dataset = DiverseMazeDataset(data_path=str(root), **common)
    train, valid, train_slices, valid_slices = get_train_val_sliced(
        traj_dataset=dataset, train_fraction=split_ratio,
        num_frames=num_hist + num_pred, frameskip=frameskip,
    )
    return {"train": train_slices, "valid": valid_slices}, {"train": train, "valid": valid}
