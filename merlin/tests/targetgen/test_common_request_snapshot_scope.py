"""Common-request custody controls; formal results are deliberately unsupported."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common import bundle_snapshot as BW
from merlin.common.paths import repo_root
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate

_PAYLOAD_BYTES = 1 << 20


@pytest.fixture
def original(tmp_path, monkeypatch):
    # Mutation controls must never write through another snapshot's CAS inode.
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    from merlin.verify import lean_checked, lean_isa_semantics

    repo = repo_root()
    source = repo / "src"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inputs = tmp_path / "selected-inputs"
    inputs.mkdir()
    padding = inputs / "payload.bin"
    padding.write_bytes(b"x" * _PAYLOAD_BYTES)
    selected = [inputs / "semantics.json", inputs / "facts.json"]
    for path in selected:
        path.write_bytes(b"{}\n")
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=source, host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(source / row["path"].removeprefix("merlin/python/"))
    bundle["allowed"].append({"path": str(inputs), "kind": "directory", "note": "synthetic custody inputs"})
    archive = tmp_path / "original.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    service = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo)

    def toolchain(_work):
        # Exercise the same separate manifest and closure lookup pattern.
        for path in selected:
            BW.snapshot_input_paths(workspace, bundle, [path], repo=repo)
        return SimpleNamespace(timeout_s=60)

    def semantics(_selection, *, resolve_path):
        for path in selected:
            resolve_path(str(path)).read_bytes()
        return SimpleNamespace(registry=object(), effect_contract=lambda **kwargs: object())

    receipt = SimpleNamespace(
        status="unsupported",
        verified=False,
        scope=("synthetic custody control",),
        to_dict=lambda: {"status": "unsupported"},
    )
    monkeypatch.setattr(service, "_toolchain", toolchain)
    monkeypatch.setattr(lean_isa_semantics, "reconstruct_isa_semantics", semantics)
    monkeypatch.setattr(lean_checked, "verify_authored_transform", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(lean_checked, "accepted_authored_receipt", lambda *args, **kwargs: False)
    frozen = BW.snapshot_path(BW.bundle_snapshot_root(workspace), padding, repo)
    return SimpleNamespace(
        service=service,
        padding=frozen,
        root=tmp_path,
        checked=lean_checked,
        request={
            "schema": "merlin.common_check_request.v1",
            "before_ir": "before",
            "after_ir": "after",
            "entry": "entry",
        },
    )


def _reads(monkeypatch, padding):
    digest = BW.hashlib.file_digest
    paths = []

    def observe(stream, *args, **kwargs):
        paths.append(Path(stream.name))
        return digest(stream, *args, **kwargs)

    monkeypatch.setattr(BW.hashlib, "file_digest", observe)
    return lambda: paths.count(padding)


@pytest.mark.parametrize("value", [[], ["final_verification"]])
def test_new_host_ordering_option_is_not_implicitly_granted_to_a_frozen_profile(original, monkeypatch, value):
    def forbidden(_work):
        pytest.fail("unsupported new options must be refused before selecting or running a toolchain")

    monkeypatch.setattr(original.service, "_toolchain", forbidden)
    request = {
        "schema": "merlin.common_transform_request.v1",
        "source_ir": "builtin.module {}",
        "options": {"pass_order": value},
    }
    with pytest.raises(ValueError, match="outside the original frozen substrate"):
        original.service.apply(request, work_root=original.root)


def test_each_actual_request_reopens_both_boundaries_without_lookup_rescans(original, monkeypatch):
    reads = _reads(monkeypatch, original.padding)
    for _ in range(2):
        result = original.service.check(original.request, work_root=original.root, timeout_s=60)
        assert result["status"] == "unsupported"
    assert reads() == 4  # Two independent requests; no reuse across calls.
    assert BW._ACTIVE_VERIFICATION.get() is None


def test_request_retains_fresh_boundaries_inside_an_outer_lookup_scope(original, monkeypatch):
    reads = _reads(monkeypatch, original.padding)
    owner = original.service
    with BW.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
        assert owner.check(original.request, work_root=original.root)["status"] == "unsupported"
    assert reads() == 4


def test_same_size_restored_mtime_mutation_refuses_before_returning_result(original, monkeypatch):
    before = original.padding.stat()
    returned = []

    def change(*args, **kwargs):
        original.padding.chmod(0o600)
        with original.padding.open("r+b") as stream:
            stream.write(b"y")
        os.utime(original.padding, ns=(before.st_atime_ns, before.st_mtime_ns))
        return SimpleNamespace(status="unsupported", verified=False, scope=(), to_dict=lambda: {})

    monkeypatch.setattr(original.checked, "verify_authored_transform", change)
    with pytest.raises(RuntimeError, match="content differs"):
        returned.append(original.service.check(original.request, work_root=original.root))
    assert not returned and BW._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize("mutation", ["marker", "member", "archive"])
def test_changed_owner_or_member_cannot_publish_a_request_result(original, monkeypatch, mutation):
    owner = original.service

    def change(*args, **kwargs):
        if mutation == "marker":
            path = BW.bundle_snapshot_root(owner.workspace) / BW.MARKER
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"\n")
        elif mutation == "member":
            original.padding.parent.chmod(0o700)
            (original.padding.parent / "undeclared.bin").write_bytes(b"new")
        else:
            owner.bundle_manifest.write_bytes(owner.bundle_manifest.read_bytes() + b"\n")
        return SimpleNamespace(status="unsupported", verified=False, scope=(), to_dict=lambda: {})

    monkeypatch.setattr(original.checked, "verify_authored_transform", change)
    with pytest.raises((RuntimeError, ValueError)):
        owner.check(original.request, work_root=original.root)
    assert BW._ACTIVE_VERIFICATION.get() is None
