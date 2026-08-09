#!/usr/bin/env python3
"""Download PushObj/PushT evaluation goal segments into data/pushobj_eval/.

Benchmark evaluation requires one ``plan_targets.pkl`` per shape under::

    data/pushobj_eval/val_<shape>/plan_targets.pkl

The release is hosted on Hugging Face Hub at ``ThomasWalker1/paarbench-data`` and
verified against ``TARGETS.sha256``. A local ``data/pushobj_eval.zip`` is accepted
when present; Google Drive is a last-resort fallback.

Examples::

    python scripts/download_targets.py
    python scripts/download_targets.py --dest data
    python scripts/download_targets.py --source drive
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "TARGETS.sha256"
HF_REPO_ID = "ThomasWalker1/paarbench-data"
DEFAULT_REVISION = "b0568eff6a55932c6ad03e6c8c86b4a5331a6b98"
DEFAULT_DRIVE_ARCHIVE_ID = "1LNoPl-3XTlBFGSPg6DFqNXF3rv01LbUf"
REQUIRED_SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")
REMOTE_PREFIX = "pushobj_eval/"


def manifest_entries() -> dict[str, str]:
    """Return ``remote path -> sha256`` for the PushObj/PushT Hub release.

    ``TARGETS.sha256`` also lists the tiny tracked maze corpora under
    ``data/maze_eval/``; those are regenerated with
    ``scripts/generate_maze_targets.py`` (or already present in git) and are
    not fetched from the Hub.
    """
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        digest, local_path = line.split(maxsplit=1)
        local_path = local_path.strip()
        data_prefix = "data/"
        if not local_path.startswith(data_prefix):
            raise ValueError(f"unexpected target manifest path: {local_path}")
        remote_path = local_path[len(data_prefix):]
        if remote_path == "pushobj_eval.zip":
            continue
        if remote_path.startswith("maze_eval/"):
            continue
        if not remote_path.startswith(REMOTE_PREFIX):
            raise ValueError(f"unexpected target manifest path: {local_path}")
        entries[remote_path] = digest
    return entries


def verify(entries: dict[str, str], data_dir: Path) -> None:
    """Fail if a staged goal file is missing or differs from the manifest."""
    problems = []
    for relative_path, expected in entries.items():
        path = data_dir / relative_path
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            problems.append(
                f"checksum mismatch: {path}\n  expected {expected}\n  actual   {actual}"
            )
    if problems:
        raise RuntimeError("evaluation-target verification failed:\n" + "\n".join(problems))


def extract_archive(archive: Path, dest_root: Path) -> Path:
    eval_root = dest_root / "pushobj_eval"
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        if not members:
            raise RuntimeError(f"{archive} is empty")
        root_prefix = ""
        if members[0].startswith("pushobj_eval/"):
            root_prefix = "pushobj_eval/"
        eval_root.mkdir(parents=True, exist_ok=True)
        for member in members:
            rel = member[len(root_prefix):] if member.startswith(root_prefix) else member
            if not rel:
                continue
            target = eval_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return eval_root


def download_from_hub(revision: str, dest_root: Path) -> None:
    entries = manifest_entries()
    patterns = sorted(f"{remote}" for remote in entries)
    print(
        f"Downloading evaluation targets from {HF_REPO_ID}@{revision} ...",
        flush=True,
    )
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=dest_root,
    )


def download_from_drive(archive_id: str, zip_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for --source drive; install with "
            "`uv pip install gdown`."
        ) from exc

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=archive_id, output=str(zip_path), quiet=False)
    if not zip_path.is_file() or zip_path.stat().st_size < 10_000:
        raise RuntimeError(
            f"download of {zip_path} looks invalid. Google Drive may be "
            "rate-limiting; retry later or use --source hub."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data"),
        help="destination root (default: <repo>/data)",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="Hub tag or full commit hash (default: benchmark-pinned revision)",
    )
    parser.add_argument(
        "--source", choices=("auto", "hub", "local", "drive"), default="auto",
        help="auto=Hub, then local zip, then Drive (default: auto)",
    )
    parser.add_argument(
        "--archive-id", default=DEFAULT_DRIVE_ARCHIVE_ID,
        help="Google Drive file id for pushobj_eval.zip (--source drive)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="keep pushobj_eval.zip after extracting a local or Drive archive",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest)
    if not dest_root.is_absolute():
        dest_root = REPO_ROOT / dest_root
    dest_root.mkdir(parents=True, exist_ok=True)
    entries = manifest_entries()
    eval_root = dest_root / "pushobj_eval"

    sources = {
        "hub": ["hub"],
        "local": ["local"],
        "drive": ["drive"],
        "auto": ["hub", "local", "drive"],
    }[args.source]
    errors: list[str] = []

    for source in sources:
        try:
            if source == "hub":
                download_from_hub(args.revision, dest_root)
            elif source == "local":
                zip_path = dest_root / "pushobj_eval.zip"
                if not zip_path.is_file():
                    raise FileNotFoundError(f"local archive not found: {zip_path}")
                print(f"Extracting {zip_path} ...", flush=True)
                extract_archive(zip_path, dest_root)
                if not args.keep_zip:
                    zip_path.unlink(missing_ok=True)
            else:
                zip_path = dest_root / "pushobj_eval.zip"
                print("Downloading evaluation targets from Google Drive ...", flush=True)
                download_from_drive(args.archive_id, zip_path)
                extract_archive(zip_path, dest_root)
                if not args.keep_zip:
                    zip_path.unlink(missing_ok=True)
            verify(entries, dest_root)
            print(f"Verified {len(entries)} goal files under {eval_root}")
            return 0
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    raise RuntimeError(
        "could not stage evaluation targets:\n" + "\n".join(errors)
    )


if __name__ == "__main__":
    raise SystemExit(main())
