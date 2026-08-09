#!/usr/bin/env python3
"""Brief offline pretraining for PAD's inverse-dynamics head.

This deliberately trains only the auxiliary head on frozen world-model latents.  It
does not alter a benchmark base checkpoint and never reads an evaluation cohort.
"""

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets.img_transforms import default_transform
from datasets.diverse_maze_dset import DiverseMazeDataset
from datasets.point_maze_dset import PointMazeDataset
from datasets.pusht_dset import PushTDataset
from paarbench.world_model import load_world_model

from adapter import InverseDynamicsHead


def pooled_latent(wm, obs):
    latents = wm.encode_obs(obs)
    return torch.cat(
        [value[:, -1].flatten(start_dim=1) for _name, value in sorted(latents.items())],
        dim=-1,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("data/pushobj_multishape"))
    parser.add_argument("--base", type=Path,
                        default=Path("checkpoints/pushobj_shape_shift"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/pad/pushobj_inverse_dynamics.pth"))
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frameskip", type=int, default=5,
                        help="environment actions represented by one model transition")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-kind",
        choices=("auto", "push", "point_maze", "diverse_maze"),
        default="auto",
        help="auto infers a maze loader from --base; explicit is safer for new bases",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"dataset does not exist: {args.data}")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = args.dataset_kind
    if kind == "auto":
        kind = (
            "diverse_maze" if "diverse" in str(args.base)
            else "point_maze" if "maze" in str(args.base)
            else "push"
        )
    data_path = args.data / "train" if (args.data / "train").is_dir() else args.data
    transform = default_transform(img_size=224)
    if kind == "push":
        dataset = PushTDataset(
            data_path=str(data_path), transform=transform,
            normalize_action=True, with_velocity=True,
        )
    elif kind == "point_maze":
        dataset = PointMazeDataset(
            data_path=str(data_path), transform=transform, normalize_action=True,
        )
    else:
        dataset = DiverseMazeDataset(
            data_path=str(data_path), transform=transform, normalize_action=True,
        )
    wm = load_world_model(args.base, device=device)
    wm.eval()
    generator = torch.Generator().manual_seed(args.seed)

    features, actions = [], []
    # One sampled consecutive pair per decoder open keeps this brief while spanning
    # the full offline distribution. The head sees frozen latents, exactly as it will
    # at the beginning of a deployment episode.
    for _ in range(args.pairs):
        episode = int(torch.randint(len(dataset), (1,), generator=generator))
        length = dataset.get_seq_length(episode)
        if length <= args.frameskip:
            continue
        frame = int(torch.randint(length - args.frameskip, (1,), generator=generator))
        obs, _action, _state, _info = dataset.get_frames(
            episode, [frame, frame + args.frameskip]
        )
        with torch.no_grad():
            first = {
                key: value[:1].unsqueeze(0).to(device)
                for key, value in obs.items()
            }
            second = {
                key: value[1:].unsqueeze(0).to(device)
                for key, value in obs.items()
            }
            z = torch.cat((pooled_latent(wm, first), pooled_latent(wm, second)), dim=-1)
        features.append(z.cpu())
        # The MPC planner concatenates the five simulator actions it executes into
        # one model action. Train on precisely that target, not a single raw action.
        actions.append(
            dataset.actions[episode, frame:frame + args.frameskip].reshape(1, -1).float()
        )
    features = torch.cat(features)
    actions = torch.cat(actions)
    latent_dim = features.shape[-1] // 2
    head = InverseDynamicsHead(features.shape[-1], actions.shape[-1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        order = torch.randperm(len(features), generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x, y = features[idx].to(device), actions[idx].to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(head.net(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch + 1}/{args.epochs}: inverse_mse={sum(losses) / len(losses):.6f}",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "inverse_head": head.state_dict(),
        "latent_dim": int(latent_dim),
        "action_dim": int(actions.shape[-1]),
        "head_hidden_dim": 256,
        "pairs": args.pairs,
        "epochs": args.epochs,
        "source_data": str(args.data),
        "base": str(args.base),
        "seed": args.seed,
        "frameskip": args.frameskip,
    }, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
