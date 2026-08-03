import hashlib

import pytest

from scripts import download_data


def test_split_inventory_matches_sha256sum_tree_format(tmp_path):
    split = tmp_path / "pushobj_multishape" / "train"
    split.mkdir(parents=True)
    (split / "a.bin").write_bytes(b"alpha")
    (split / "z.bin").write_bytes(b"omega")

    lines = "".join(
        f"{hashlib.sha256(content).hexdigest()}  train/{name}\n"
        for name, content in (("a.bin", b"alpha"), ("z.bin", b"omega"))
    )
    inventory = download_data.split_inventory(split, tmp_path / "pushobj_multishape")

    assert inventory == {
        "file_count": 2,
        "total_bytes": 10,
        "tree_sha256": hashlib.sha256(lines.encode()).hexdigest(),
    }


def test_verify_accepts_matching_tree_and_rejects_changes(tmp_path):
    split = tmp_path / "dataset" / "train"
    split.mkdir(parents=True)
    artifact = split / "sample.bin"
    artifact.write_bytes(b"trajectory")
    expected = download_data.split_inventory(split, tmp_path / "dataset")
    manifest = {"root": "dataset", "splits": {"train": expected}}

    download_data.verify(tmp_path, manifest)
    artifact.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="mismatch"):
        download_data.verify(tmp_path, manifest)
