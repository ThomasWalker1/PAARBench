#!/usr/bin/env python3
"""Publish evaluation goal segments to Hugging Face Hub.

Uploads ``data/pushobj_eval/`` and ``data/pushobj_eval.zip`` to
``ThomasWalker1/paarbench-data``. Requires a write token::

    export HF_TOKEN=hf_...
    python scripts/upload_targets.py

After upload, pin the printed commit hash in ``scripts/download_targets.py``
as ``DEFAULT_REVISION``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_REPO_ID = "ThomasWalker1/paarbench-data"
REMOTE_ROOT = "pushobj_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="local data root containing pushobj_eval/ (default: <repo>/data)",
    )
    parser.add_argument(
        "--commit-message", default="Add PushObj/PushT evaluation goal segments",
        help="Hub commit message",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    eval_root = data_dir / "pushobj_eval"
    archive = data_dir / "pushobj_eval.zip"
    if not eval_root.is_dir():
        raise SystemExit(f"missing staged targets: {eval_root}")
    if not archive.is_file():
        raise SystemExit(f"missing archive to publish: {archive}")

    api = HfApi()
    who = api.whoami()
    print(f"Uploading as {who.get('name')} to {HF_REPO_ID} ...", flush=True)

    api.upload_folder(
        folder_path=str(eval_root),
        path_in_repo=REMOTE_ROOT,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"  uploaded {REMOTE_ROOT}/", flush=True)

    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=f"{REMOTE_ROOT}.zip",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"  uploaded {REMOTE_ROOT}.zip", flush=True)

    info = api.repo_info(HF_REPO_ID, repo_type="dataset")
    print(
        "\nUpload complete. Pin this revision in scripts/download_targets.py:\n"
        f"  DEFAULT_REVISION = \"{info.sha}\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
