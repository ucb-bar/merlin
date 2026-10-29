"""Original frozen grants admit exact API imports without granting the harness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import compiler_profile_admission as API
from merlin.targetgen.package_runtime import CertFailure, integrity_scan
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate


@pytest.fixture
def original(tmp_path):
    repo = repo_root()
    source = repo / "src"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=source, host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(source / row["path"].removeprefix("merlin/python/"))
    manifest = tmp_path / "original.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    candidate = workspace / "submission"
    candidate.mkdir()
    package = SimpleNamespace(directory=candidate, integrity_exempt=False, manifest={})
    return SimpleNamespace(workspace=workspace, bundle_manifest=manifest, repo=repo, package=package)


def _scope(original):
    return API.original_compiler_imports(
        workspace=original.workspace, bundle_manifest=original.bundle_manifest, repo=original.repo
    )


def test_common_helpers_work_only_under_the_original_profile_scope(original):
    source = original.package.directory / "tool.py"
    source.write_text("from merlin.transforms.normalization import simplify_integers\n")
    with pytest.raises(CertFailure, match="imports"):
        integrity_scan(original.package)
    with _scope(original) as admission:
        integrity_scan(original.package)
        assert admission.permits("merlin.transforms.normalization")
        assert not admission.permits("merlin.transforms")
        assert not admission.permits("merlin.verify.lean_checked")
    assert not admission.permits("merlin.transforms.normalization")
    with pytest.raises(CertFailure, match="imports"):
        integrity_scan(original.package)


@pytest.mark.parametrize(
    "module",
    [
        "merlin.verify.lean_checked",
        "merlin.verify.lean_checker",
        "merlin.runtime.reference",
        "merlin.targetgen.isa_spec",
        "merlin.transforms.unreviewed",
        "merlin.transforms.pipeline_state",
        "merlin_experiments.phase2.authoring",
    ],
)
def test_scoped_common_grants_do_not_admit_private_siblings_or_optional_tools(original, module):
    (original.package.directory / "tool.py").write_text("import " + module + "\n")
    with _scope(original), pytest.raises(CertFailure):
        integrity_scan(original.package)


def test_candidate_metadata_does_not_select_grants_or_an_exemption(original):
    (original.package.directory / "tool.py").write_text("import merlin.verify.lean_checker\n")
    original.package.manifest["compiler_tool_profile"] = {"common_public_modules": ["merlin.verify.lean_checker"]}
    with pytest.raises(CertFailure):
        integrity_scan(original.package)
    original.package.integrity_exempt = True
    with _scope(original), pytest.raises(CertFailure, match="exemption"):
        integrity_scan(original.package)


def test_mutated_admission_or_manifest_cannot_change_import_membership(original):
    with _scope(original) as admission:
        admission.modules = frozenset({"merlin.runtime.reference"})
        with pytest.raises(ValueError, match="membership"):
            API.active_compiler_imports()
        admission.modules = frozenset(admission.profile["common_public_modules"])
    original.bundle_manifest.write_text(original.bundle_manifest.read_text() + "\n")
    # A completed old instance is never reusable, including after archive edits.
    with pytest.raises(ValueError, match="host-issued"):
        admission.verify()


def test_writable_workspace_manifest_cannot_be_import_authority(original):
    manifest = original.workspace / "candidate.yaml"
    manifest.write_bytes(original.bundle_manifest.read_bytes())
    with pytest.raises(ValueError, match="original archived"):
        with API.original_compiler_imports(workspace=original.workspace, bundle_manifest=manifest, repo=original.repo):
            pytest.fail("author archive admitted")


def test_official_formal_entry_keeps_original_admission_and_caller_convention(original, monkeypatch):
    from merlin_experiments.phase1.feedback import formal
    from merlin_experiments.phase1.run_inputs import bundle_manifest_identity

    from merlin.targetgen import target_experiment

    manifest = original.bundle_manifest.parent / "input_bundle_manifest.yaml"
    manifest.write_bytes(original.bundle_manifest.read_bytes())
    bundle = yaml.safe_load(manifest.read_bytes())
    environment = manifest.parent / "environment.yaml"
    environment.write_text(
        yaml.safe_dump(
            {
                "compiler_tool_profile": bundle["compiler_tool_profile"],
                "bundle_manifest_sha256": bundle_manifest_identity(manifest, bundle),
            }
        )
    )
    (original.package.directory / "tool.py").write_text("import merlin.transforms.normalization\n")
    context = SimpleNamespace(repo=original.repo, descriptor=manifest.parent / "descriptor.yaml")
    original.package.directory.rename(manifest.parent / "submission")
    original.package.directory = manifest.parent / "submission"
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: SimpleNamespace(path=path))
    seen = []

    def grade(argv, *, context=None):
        seen.append(context)
        assert API.active_compiler_imports() is not None
        integrity_scan(original.package)
        return 17  # This control executes import custody only, never a numerical/RTL grade.

    monkeypatch.setattr(formal, "_main", grade)
    argv = ["--run-dir", str(manifest.parent), "--workspace", str(original.workspace)]
    assert formal.main(argv, context=context) == 17
    assert seen == [context] and API.active_compiler_imports() is None
    monkeypatch.setattr(formal, "resolve_context", lambda *_args: context)
    assert formal.main(argv) == 17
    assert seen[-1] is None  # Installed explicit-input requirements remain with the original body.
    with pytest.raises(ValueError, match="original frozen workspace"):
        formal.main(["--run-dir", str(manifest.parent)], context=context)
    environment.write_text(yaml.safe_dump({"compiler_tool_profile": bundle["compiler_tool_profile"]}))
    with pytest.raises(ValueError, match="original run record"):
        formal.main(argv, context=context)


def test_formal_help_and_unprofiled_legacy_keep_original_behavior(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import formal

    seen = []
    monkeypatch.setattr(formal, "_main", lambda argv, **kwargs: seen.append((argv, kwargs)) or 0)
    assert formal.main(["--help"]) == 0
    assert formal.main(["--run-dir", str(tmp_path)]) == 0
    assert seen[0][0] == ["--help"] and API.active_compiler_imports() is None


def test_snapshot_marker_cannot_be_rebound_with_the_same_payload(original):
    import json

    from merlin.common.bundle_snapshot import bundle_snapshot_root
    from merlin.targetgen.common_transforms import CommonTransformService

    owner = CommonTransformService(
        workspace=original.workspace, bundle_manifest=original.bundle_manifest, repo=original.repo
    )
    marker = bundle_snapshot_root(original.workspace) / "snapshot.json"
    with _scope(original) as admission:
        original_bytes = marker.read_bytes()
        original_mode = marker.stat().st_mode & 0o777
        marker.chmod(0o600)
        document = json.loads(original_bytes)
        document["untrusted_extra_metadata"] = "changed marker, same payload"
        marker.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="snapshot"):
            admission.verify()
        with pytest.raises(ValueError, match="source inputs"):
            owner._verify()
        marker.write_bytes(original_bytes)
        marker.chmod(original_mode)


def _overlay_setup(original, monkeypatch):
    from merlin.common import bundle_snapshot, read_lease_digest
    from merlin_experiments.phase1 import compiler_library

    manifest = original.bundle_manifest.parent / "input_bundle_manifest.yaml"
    manifest.write_bytes(original.bundle_manifest.read_bytes())
    bundle = yaml.safe_load(manifest.read_bytes())
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    # Simulate the disabled cross-call lease cache: actual payload reads remain
    # mandatory at both boundaries, while directory/file watches still protect
    # the bounded operation. This does not mock the snapshot verifier.
    monkeypatch.setattr(read_lease_digest, "digest", lambda _path: None)
    monkeypatch.setattr(read_lease_digest, "reuse", lambda *args, **kwargs: None)
    monkeypatch.setattr(compiler_library, "active_compiler_library", lambda: object())
    reads = []
    actual = bundle_snapshot._payload_content

    def measured(members):
        reads.append(1)
        return actual(members)

    monkeypatch.setattr(bundle_snapshot, "_payload_content", measured)
    return compiler_library, bundle_snapshot, manifest, bundle, reads


def test_library_overlay_batches_nested_checks_without_cross_call_cache(original, monkeypatch):
    library, snapshots, manifest, bundle, reads = _overlay_setup(original, monkeypatch)

    def assemble(policy):
        for _ in range(4):
            owner = API.active_compiler_imports()
            owner.verify()
            snapshots.snapshot_input_paths(original.workspace, owner.bundle, [], repo=original.repo)
        return policy

    monkeypatch.setattr(library.overlays, "apply", assemble)
    argv, env = library.sandbox_overlay(
        ["unchanged"], workspace=original.workspace, bundle=bundle,
        context=SimpleNamespace(repo=original.repo), private_run_dir=manifest.parent,
    )
    assert argv == ["unchanged"] and env == ""
    assert len(reads) == 2  # Fresh complete entry and exit; no stale cache authority.
    assert API.active_compiler_imports() is None


@pytest.mark.parametrize("changed", ["payload", "marker", "manifest", "policy"])
def test_library_overlay_rejects_mutations_before_publishing(original, monkeypatch, changed):
    library, snapshots, manifest, bundle, reads = _overlay_setup(original, monkeypatch)

    def assemble(policy):
        owner = API.active_compiler_imports()
        if changed == "payload":
            source = original.repo / owner.bundle["allowed"][0]["path"]
            path = snapshots.snapshot_input_paths(original.workspace, owner.bundle, [source], repo=original.repo)[0]
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"\n# changed\n")
        elif changed == "marker":
            path = snapshots.bundle_snapshot_root(original.workspace) / snapshots.MARKER
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"\n")
        elif changed == "manifest":
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        else:
            bundle["allowed"] = []
        return policy

    monkeypatch.setattr(library.overlays, "apply", assemble)
    with pytest.raises((ValueError, RuntimeError)):
        library.sandbox_overlay(
            [], workspace=original.workspace, bundle=bundle,
            context=SimpleNamespace(repo=original.repo), private_run_dir=manifest.parent,
        )
    assert API.active_compiler_imports() is None
