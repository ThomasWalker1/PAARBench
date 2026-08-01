import torch
import decord
import pickle
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])


def episode_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True, # agent's velocity
        use_hdf5: bool = False,
        hdf5_path: Optional[str] = None,
        hdf5_cache_mb: int = 256,
        hdf5_cache_nslots: int = 200003,
        hdf5_cache_w0: float = 0.75,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.use_hdf5 = use_hdf5
        self.hdf5_path = Path(hdf5_path) if hdf5_path else self.data_path / "obses.h5"
        self.hdf5_cache_mb = hdf5_cache_mb
        self.hdf5_cache_nslots = hdf5_cache_nslots
        self.hdf5_cache_w0 = hdf5_cache_w0
        self._h5_file = None
        self._h5_frames = None
        self._h5_episode_starts = None
        self._h5_episode_id_to_pos = None
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            # Prefer per-dataset stats (norm_stats.pt) when present; else hardcoded
            # pusht_noise constants. Stats are in the loader's native units:
            # action stats post-/action_scale, proprio order [ax,ay,vx,vy].
            stats_file = self.data_path / "norm_stats.pt"
            if stats_file.exists():
                ns = torch.load(stats_file)
                self.action_mean = ns["action_mean"]
                self.action_std = ns["action_std"]
                self.state_mean = ns["state_mean"][:self.state_dim]
                self.state_std = ns["state_std"][:self.state_dim]
                self.proprio_mean = ns["proprio_mean"][:self.proprio_dim]
                self.proprio_std = ns["proprio_std"][:self.proprio_dim]
            else:
                self.action_mean = ACTION_MEAN
                self.action_std = ACTION_STD
                self.state_mean = STATE_MEAN[:self.state_dim]
                self.state_std = STATE_STD[:self.state_dim]
                self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
                self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = self.load_visual_frames(idx, frames)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def load_visual_frames(self, idx, frames):
        if self.use_hdf5:
            image = self._load_frames_from_hdf5(idx, frames)
        else:
            vid_dir = self.data_path / "obses"
            reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
            image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0

    def _ensure_hdf5_open(self):
        if self._h5_file is not None:
            return
        if not self.hdf5_path.exists():
            raise ValueError(f"Failed to load HDF5 observations from {self.hdf5_path}")
        self._h5_file = h5py.File(
            self.hdf5_path,
            "r",
            rdcc_nbytes=int(self.hdf5_cache_mb) * 1024 * 1024,
            rdcc_nslots=int(self.hdf5_cache_nslots),
            rdcc_w0=float(self.hdf5_cache_w0),
        )
        self._h5_frames = self._h5_file["frames"]
        self._h5_episode_starts = self._h5_file["episode_starts"]
        self._h5_episode_id_to_pos = self._build_hdf5_episode_index()

    def _build_hdf5_episode_index(self):
        if "episode_ids" in self._h5_file:
            episode_ids = self._h5_file["episode_ids"][:]
            return {int(ep_id): pos for pos, ep_id in enumerate(episode_ids)}

        h5_lengths = np.diff(self._h5_episode_starts[:])
        expected_lengths = np.asarray(self.seq_lengths, dtype=np.int64)
        if (
            len(h5_lengths) == len(expected_lengths)
            and np.array_equal(h5_lengths, expected_lengths)
        ):
            return {idx: idx for idx in range(len(expected_lengths))}

        obs_dir = self.data_path / "obses"
        episode_files = sorted(obs_dir.glob("episode_*.mp4"))
        lex_episode_ids = [episode_id_from_path(path) for path in episode_files]
        if len(lex_episode_ids) != len(h5_lengths):
            raise ValueError(
                f"HDF5 episode count mismatch for {self.hdf5_path}: "
                f"{len(h5_lengths)} HDF5 episodes vs {len(lex_episode_ids)} files"
            )

        lex_lengths = np.asarray(
            [self.seq_lengths[episode_id] for episode_id in lex_episode_ids],
            dtype=np.int64,
        )
        if np.array_equal(h5_lengths, lex_lengths):
            return {episode_id: pos for pos, episode_id in enumerate(lex_episode_ids)}

        raise ValueError(
            f"Could not infer HDF5 episode ordering for {self.hdf5_path}"
        )

    def _load_frames_from_hdf5(self, episode_idx: int, frame_indices):
        self._ensure_hdf5_open()
        if isinstance(frame_indices, range):
            frame_indices = list(frame_indices)
        elif isinstance(frame_indices, torch.Tensor):
            frame_indices = frame_indices.tolist()
        else:
            frame_indices = list(frame_indices)
        h5_episode_idx = self._h5_episode_id_to_pos[episode_idx]
        episode_start = int(self._h5_episode_starts[h5_episode_idx])
        absolute_indices = [episode_start + frame_idx for frame_idx in frame_indices]
        raw_visual = self._h5_frames[absolute_indices]
        return torch.from_numpy(np.asarray(raw_visual))

    def __del__(self):
        h5_file = getattr(self, "_h5_file", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    use_hdf5=False,
    hdf5_path_train=None,
    hdf5_path_val=None,
    hdf5_cache_mb=256,
    hdf5_cache_nslots=200003,
    hdf5_cache_w0=0.75,
    shuffle_slices=True,
    shuffle_block_size=0,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        use_hdf5=use_hdf5,
        hdf5_path=hdf5_path_train,
        hdf5_cache_mb=hdf5_cache_mb,
        hdf5_cache_nslots=hdf5_cache_nslots,
        hdf5_cache_w0=hdf5_cache_w0,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        use_hdf5=use_hdf5,
        hdf5_path=hdf5_path_val,
        hdf5_cache_mb=hdf5_cache_mb,
        hdf5_cache_nslots=hdf5_cache_nslots,
        hdf5_cache_w0=hdf5_cache_w0,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(
        train_dset,
        num_frames,
        frameskip,
        shuffle_slices=shuffle_slices,
        shuffle_block_size=shuffle_block_size,
    )
    val_slices = TrajSlicerDataset(
        val_dset,
        num_frames,
        frameskip,
        shuffle_slices=False,
    )

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
