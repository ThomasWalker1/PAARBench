"""PointMaze trajectory loader compatible with released AdaJEPA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


class PointMazeDataset(TrajDataset):
    """Lazy image loading with checkpoint-compatible normalization metadata."""

    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        use_preprocessed: bool = False,
        use_frame_files: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        # Released AdaJEPA hydra configs request preprocessed per-frame files; the
        # staged PAARBench MediumMaze zip ships per-episode uint8 HWC tensors.
        # Prefer the requested layout when present, otherwise decode episode files.
        if use_frame_files:
            frame_probe = self.data_path / "obses" / "episode_000_frame_000.pth"
            episode_probe = self.data_path / "obses" / "episode_000.pth"
            if not frame_probe.is_file() and episode_probe.is_file():
                use_frame_files = False
                use_preprocessed = False
        self.use_frame_files = use_frame_files
        self.use_preprocessed = use_preprocessed
        self.states = torch.load(self.data_path / "states.pth", map_location="cpu").float()
        self.actions = torch.load(self.data_path / "actions.pth", map_location="cpu").float()
        self.actions = self.actions / action_scale
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth", map_location="cpu")
        if n_rollout is not None:
            self.states = self.states[:n_rollout]
            self.actions = self.actions[:n_rollout]
            self.seq_lengths = self.seq_lengths[:n_rollout]
        self.proprios = self.states.clone()
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios)
        else:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.cat(
            [values[index, :length] for index, length in enumerate(self.seq_lengths)]
        )
        return valid.mean(dim=0), valid.std(dim=0)

    def get_seq_length(self, index: int) -> int:
        return int(self.seq_lengths[index])

    def get_all_actions(self) -> torch.Tensor:
        return torch.cat(
            [self.actions[index, :length] for index, length in enumerate(self.seq_lengths)]
        )

    def _episode_visual(self, index: int) -> torch.Tensor:
        directory = self.data_path / "obses"
        candidates = (
            directory / f"episode_{index:06d}.pth",
            directory / f"episode_{index:03d}.pth",
        )
        for candidate in candidates:
            if candidate.is_file():
                return torch.load(candidate, map_location="cpu")
        raise FileNotFoundError(f"no image tensor for PointMaze episode {index}: {candidates}")

    def get_frames(self, index: int, frames):
        frames = list(frames)
        if self.use_frame_files:
            images = []
            for frame in frames:
                path = self.data_path / "obses" / f"episode_{index:03d}_frame_{frame:03d}.pth"
                images.append(torch.load(path, map_location="cpu"))
            visual = torch.stack(images)
        else:
            visual = self._episode_visual(index)[frames]
        if not self.use_preprocessed:
            visual = rearrange(visual.float() / 255.0, "t h w c -> t c h w")
            if self.transform:
                visual = self.transform(visual)
        return (
            {"visual": visual, "proprio": self.proprios[index, frames]},
            self.actions[index, frames],
            self.states[index, frames],
            {},
        )

    def __getitem__(self, index: int):
        return self.get_frames(index, range(self.get_seq_length(index)))

    def __len__(self) -> int:
        return len(self.seq_lengths)


def load_point_maze_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/point_maze",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_preprocessed=False,
    use_frame_files=False,
):
    dataset = PointMazeDataset(
        data_path=data_path, n_rollout=n_rollout, transform=transform,
        normalize_action=normalize_action, use_preprocessed=use_preprocessed,
        use_frame_files=use_frame_files,
    )
    train, valid, train_slices, valid_slices = get_train_val_sliced(
        traj_dataset=dataset, train_fraction=split_ratio,
        num_frames=num_hist + num_pred, frameskip=frameskip,
    )
    return {"train": train_slices, "valid": valid_slices}, {"train": train, "valid": valid}
