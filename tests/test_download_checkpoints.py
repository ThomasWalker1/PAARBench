from pathlib import Path

import pytest

from scripts import download_checkpoints


def test_checkpoint_groups_partition_the_manifest():
    settings = download_checkpoints.manifest_entries("settings")
    methods = download_checkpoints.manifest_entries("methods")
    all_files = download_checkpoints.manifest_entries("all")

    assert settings
    assert methods
    assert settings.keys().isdisjoint(methods)
    assert set(settings) | set(methods) == set(all_files)
    assert all(path.startswith(("pushobj_shape_shift/", "pusht_visual_shift/"))
               for path in settings)


def test_verify_rejects_missing_or_modified_files(tmp_path):
    entries = {"one/file.pth": "a" * 64}
    with pytest.raises(RuntimeError, match="missing"):
        download_checkpoints.verify(entries, tmp_path)

    path = tmp_path / "one/file.pth"
    path.parent.mkdir()
    path.write_bytes(b"not the expected checkpoint")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_checkpoints.verify(entries, tmp_path)


def test_verify_accepts_matching_file(tmp_path):
    import hashlib

    content = b"checkpoint"
    path = tmp_path / "file.pth"
    path.write_bytes(content)
    download_checkpoints.verify(
        {"file.pth": hashlib.sha256(content).hexdigest()}, Path(tmp_path)
    )
