import sys
from pathlib import Path

import pytest

from scripts import download_targets


def test_manifest_lists_all_required_shapes():
    entries = download_targets.manifest_entries()
    assert len(entries) == len(download_targets.REQUIRED_SHAPES)
    for shape in download_targets.REQUIRED_SHAPES:
        assert f"pushobj_eval/val_{shape}/plan_targets.pkl" in entries


def test_verify_rejects_missing_or_modified_files(tmp_path):
    entries = {"pushobj_eval/val_T/plan_targets.pkl": "a" * 64}
    with pytest.raises(RuntimeError, match="missing"):
        download_targets.verify(entries, tmp_path)

    path = tmp_path / "pushobj_eval/val_T/plan_targets.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the expected segments file")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_targets.verify(entries, tmp_path)


def test_verify_accepts_matching_file(tmp_path):
    import hashlib

    content = b"segments"
    path = tmp_path / "pushobj_eval/val_T/plan_targets.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    download_targets.verify(
        {"pushobj_eval/val_T/plan_targets.pkl": hashlib.sha256(content).hexdigest()},
        tmp_path,
    )


def test_local_source_extracts_staged_zip(tmp_path, monkeypatch):
    archive = Path(download_targets.REPO_ROOT) / "data" / "pushobj_eval.zip"
    if not archive.is_file():
        pytest.skip("pushobj_eval.zip not staged in the workspace")

    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "pushobj_eval.zip").write_bytes(archive.read_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_targets.py", "--dest", str(dest), "--source", "local", "--keep-zip"],
    )
    assert download_targets.main() == 0
    download_targets.verify(download_targets.manifest_entries(), dest)
