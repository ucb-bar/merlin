"""Portable source exports bind only the small canonical source roster."""

import hashlib
import json

import pytest
from merlin_timing.export import ASSETS, ASSETS_V1, SCHEMA_V1, export_cpp_core, verify_cpp_core


def test_export_is_complete_and_has_no_compiler_dependency(tmp_path):
    output = tmp_path / "portable"
    manifest = export_cpp_core(output)
    assert verify_cpp_core(output) == manifest
    assert [row["path"] for row in manifest["members"]] == list(ASSETS)
    assert sum(row["bytes"] for row in manifest["members"]) < 1024 * 1024
    for row in manifest["members"]:
        data = (output / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert "merlin/" not in " ".join(row["path"] for row in manifest["members"])


@pytest.mark.parametrize("change", ["modified", "missing", "additional", "directory", "linked"])
def test_modified_export_refuses(tmp_path, change):
    output = tmp_path / "portable"
    export_cpp_core(output)
    header = output / ASSETS[0]
    if change == "modified":
        header.write_bytes(header.read_bytes() + b"\n")
    elif change == "missing":
        header.unlink()
    elif change == "additional":
        (output / "unlisted.py").write_text("raise RuntimeError('not a source member')\n")
    elif change == "directory":
        (output / "unlisted").mkdir()
    else:
        header.unlink()
        header.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        verify_cpp_core(output)


def test_export_cannot_overwrite_or_follow_a_destination(tmp_path):
    output = tmp_path / "portable"
    export_cpp_core(output)
    with pytest.raises(ValueError):
        export_cpp_core(output)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError):
        export_cpp_core(alias)


def test_original_v1_source_roster_remains_readable_without_upgrade(tmp_path):
    output = tmp_path / "original"
    document = export_cpp_core(output)
    for name in set(ASSETS) - set(ASSETS_V1):
        (output / name).unlink()
    document["schema"] = SCHEMA_V1
    document["members"] = [row for row in document["members"] if row["path"] in ASSETS_V1]
    path = output / "source-manifest.json"
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    assert verify_cpp_core(output) == document
    assert path.read_bytes() == original
