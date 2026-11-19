"""Nested image plumbing retains containment and complete directory accounting.

These synthetic byte identities exercise publication, not device execution.
"""

import hashlib
import json
from pathlib import Path

import pytest

from merlin.semantic_compiler.target_binding import verify_native_publication


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publication(root):
    artifacts, rows = {}, []
    for index, name in enumerate(("square", "publish")):
        relative = "segments/region/" + name
        folder = root / relative
        folder.mkdir(parents=True)
        binary, execution = bytes((19, index, 0, 0)), b"{}"
        detail = dict(artifact_kind="program", engine="merlin_native", request_digest="source-" + name,
                      target_identity="test-target", binary_sha256=sha(binary), execution_plan_sha256=sha(execution))
        files = {"program.bin": binary, "execution_plan.json": execution,
                 "manifest.json": json.dumps(detail).encode()}
        for member, data in files.items():
            (folder / member).write_bytes(data)
            artifacts[relative + "/" + member] = sha(data)
        rows.append(dict(index=index, path=relative, request_digest=detail["request_digest"],
                         manifest_sha256=sha(files["manifest.json"]), binary_sha256=sha(binary)))
    data = b'{"source":"bound-region"}\n'
    (root / "segments/region/source-plan.json").write_bytes(data)
    artifacts["segments/region/source-plan.json"] = sha(data)
    return dict(artifact_kind="program_set", engine="merlin_native", request_digest="whole-source",
                target_identity="test-target", segments=rows, stage_artifacts=artifacts)


def verify(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest))
    verify_native_publication(root, manifest, engine="merlin_native", request_digest="whole-source",
                              target_identity="test-target")


def test_two_nested_images_and_authenticated_region_metadata_pass(tmp_path):
    verify(tmp_path, publication(tmp_path))


@pytest.mark.parametrize("path", ["segments", "segments//region/square", "segments/./region/square",
                                  "segments/region/square/", "segments/../region/square",
                                  "/segments/region/square", "segments/region\\square", "other/region/square"])
def test_noncanonical_or_escaping_segment_path_refuses(tmp_path, path):
    manifest = publication(tmp_path)
    manifest["segments"][0]["path"] = path
    with pytest.raises(ValueError, match="unsafe or repeated segment path"):
        verify(tmp_path, manifest)


@pytest.mark.parametrize("overlap", [False, True])
def test_duplicate_and_overlapping_program_roots_refuse(tmp_path, overlap):
    manifest = publication(tmp_path)
    manifest["segments"][1]["path"] = manifest["segments"][0]["path"] + ("/child" if overlap else "")
    with pytest.raises(ValueError, match="unsafe or repeated segment path"):
        verify(tmp_path, manifest)


@pytest.mark.parametrize("kind", ["directory", "file", "root-file", "reserved-manifest"])
def test_unaccounted_entries_and_intermediate_native_manifest_refuse(tmp_path, kind):
    manifest = publication(tmp_path)
    relative = "segments/extra.json" if kind == "root-file" else "segments/region/extra"
    if kind == "reserved-manifest":
        relative = "segments/region/manifest.json"
    path = tmp_path / relative
    if kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"{}")
        if kind in {"root-file", "reserved-manifest"}:
            manifest["stage_artifacts"][relative] = sha(b"{}")
    with pytest.raises(ValueError, match="unaccounted segment"):
        verify(tmp_path, manifest)


def test_region_metadata_and_individual_image_identity_remain_authenticated(tmp_path):
    manifest = publication(tmp_path)
    metadata = tmp_path / "segments/region/source-plan.json"
    before = metadata.read_bytes()
    metadata.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="stage artifact changed"):
        verify(tmp_path, manifest)
    metadata.write_bytes(before)
    manifest["segments"][1]["request_digest"] = "substituted-source"
    with pytest.raises(ValueError, match="identity differs"):
        verify(tmp_path, manifest)


def test_linked_intermediate_is_rejected_before_hashing_any_member(tmp_path, monkeypatch):
    from merlin.semantic_compiler import target_binding
    manifest = publication(tmp_path)
    (tmp_path / "segments/region").rename(tmp_path / "kept")
    (tmp_path / "segments/region").symlink_to(tmp_path / "kept", target_is_directory=True)
    def forbidden(_path):
        raise AssertionError("linked member must not be hashed")
    monkeypatch.setattr(target_binding, "digest_file", forbidden)
    with pytest.raises(ValueError, match="stage artifact changed"):
        verify(tmp_path, manifest)


@pytest.mark.parametrize("member", ["manifest.json", "program.bin", "execution_plan.json"])
def test_leaf_files_cannot_alias_another_location_without_stage_map(tmp_path, member):
    manifest = publication(tmp_path)
    # Legacy leaves may omit the optional artifact map. The immutable image
    # identity still cannot be supplied by a linked file.
    del manifest["stage_artifacts"]
    (tmp_path / "segments/region/source-plan.json").unlink()
    path = tmp_path / "segments/region/square" / member
    kept = tmp_path / ("kept-" + member)
    path.rename(kept)
    path.symlink_to(kept)
    with pytest.raises(ValueError):
        verify(tmp_path, manifest)
