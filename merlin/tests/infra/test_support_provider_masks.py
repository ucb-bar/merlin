"""Relocated host support stays private beyond the backend entry module."""

import importlib
import json
import shutil
from types import SimpleNamespace

import pytest

from merlin.targetgen import target_registry
from merlin.targetgen.providers import ProviderError
from merlin.targetgen.sandbox import bwrap, cleanroom

surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


@pytest.fixture
def support(tmp_path, monkeypatch):
    root = tmp_path / "outside-checkout/support"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/target_contract.yaml").write_text("name: fixture_device\nplugin:\n  backend: backend\n")
    for directory in ("backend", "build_support", "conformance", "tools"):
        (root / directory).mkdir()
        (root / directory / "__init__.py").write_text("raise AssertionError('must not import support')\n")
    (root / "private_data.yaml").write_text("answer: withheld\n")
    repo = tmp_path / "checkout"
    repo.mkdir()
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    # Replace all reference shelves, including physical-checkout examples.
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(repo / "merlin/targets"))
    monkeypatch.setattr(target_registry, "targets_dir", lambda: repo / "merlin/targets")
    monkeypatch.setattr(target_registry, "generated_target_home", lambda: repo / "out/build/generated")
    monkeypatch.setattr(surfaces, "repo_root", lambda: repo)
    monkeypatch.setattr(surfaces, "artifacts_dir", lambda: repo / "out/artifacts")
    monkeypatch.setattr(surfaces, "MODULE_ACCESS", ())
    monkeypatch.setattr(surfaces, "contract_resource_roots", lambda *_: ())
    monkeypatch.setattr(surfaces, "experimenter_memory_dir", lambda: repo / "absent-memory")
    te = SimpleNamespace(
        target="fixture_device",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    return root, repo, te


def test_selected_support_root_masks_siblings_without_importing_them(support):
    root, _, te = support
    derived = surfaces.answer_surfaces(te)
    package = next(item for item in derived if item.path == root)
    assert package.origin == "backend"
    assert package.grantable == ("contracts",)
    exposed = ["--ro-bind", str(root), str(root)]
    assert package in bwrap.coverage_gap(exposed, derived)
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(exposed, derived), derived) == []
    denied = cleanroom.derive_deny_set(te)
    for rel in (
        "backend/__init__.py",
        "conformance/__init__.py",
        "build_support/__init__.py",
        "tools/__init__.py",
        "private_data.yaml",
    ):
        assert cleanroom.is_withheld(root / rel, denied)
        assert any(token in str(root / rel) for token in surfaces.audit_tokens(te)["answer"])
    contract = root / "contracts/target_contract.yaml"
    assert not cleanroom.is_withheld(contract, denied)
    assert not any(token in str(contract) for token in surfaces.audit_tokens(te)["answer"])


def test_descriptor_location_remains_masked_after_provider_selection(support):
    root, repo, te = support
    native = repo / "legacy-support"
    native.mkdir()
    te.backend_package = "legacy-support"
    derived = {item.path for item in surfaces.answer_surfaces(te) if item.origin == "backend"}
    assert derived == {root, native}


def test_candidate_compiler_on_same_shelf_is_not_host_support(support, monkeypatch):
    root, _, te = support
    candidate = root.parent / "candidate"
    candidate.mkdir()
    (candidate / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: compiled-result\ntarget: fixture_device\nrole: candidate_compiler\n"
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root.parent))
    assert surfaces._support_package_dirs() == [root]
    assert all(item.path != candidate for item in surfaces.answer_surfaces(te))


def test_invalid_support_discovery_refuses_instead_of_omitting_masks(support):
    root, _, te = support
    (root / "provider.yaml").write_text("schema: invalid\n")
    with pytest.raises(ProviderError):
        surfaces.answer_surfaces(te)
    with pytest.raises(cleanroom.CleanRoomRefused, match="answer surfaces could not be derived"):
        cleanroom.derive_deny_set(te)


@pytest.mark.parametrize("operation", ["--bind", "--ro-bind"])
@pytest.mark.parametrize("placement", ["same", "translated", "root-alias"])
def test_final_masks_private_subgrants_and_aliases(support, tmp_path, operation, placement):
    root, repo, te = support
    source = root / "build_support"
    destination = source if placement == "same" else tmp_path / "tool-view"
    if placement == "root-alias":
        source = tmp_path / "support-link"
        source.symlink_to(root, target_is_directory=True)
    argv = ["--ro-bind", str(root), str(root), operation, str(source), str(destination)]
    private = destination / ("build_support/__init__.py" if placement == "root-alias" else "__init__.py")
    derived = surfaces.answer_surfaces(te)
    assert bwrap.is_exposed(argv, private)
    assert bwrap.coverage_gap(argv, derived)
    result = bwrap.apply_final_answer_masks(argv, te, tmp_path / "workspace", {}, repo=repo)
    assert not bwrap.is_exposed(result, private)
    assert bwrap.coverage_gap(result, derived) == []


def test_only_explicit_contract_subgrant_survives(support, tmp_path):
    root, repo, te = support
    alias = tmp_path / "provider-view"
    broad = ["--ro-bind", str(root), str(alias)]
    contract = alias / "contracts/target_contract.yaml"
    assert not bwrap.is_exposed(bwrap.apply_final_answer_masks(broad, te, tmp_path / "ws", {}, repo=repo), contract)
    explicit = [*broad, "--ro-bind", str(root / "contracts"), str(alias / "contracts")]
    result = bwrap.apply_final_answer_masks(explicit, te, tmp_path / "ws", {}, repo=repo)
    assert bwrap.is_exposed(result, contract)
    assert not bwrap.is_exposed(result, alias / "build_support/__init__.py")


@pytest.mark.parametrize("root_alias", [False, True])
def test_frozen_private_grant_cannot_be_reintroduced_under_runtime_alias(support, tmp_path, root_alias):
    root, repo, te = support
    if root_alias:
        alias = tmp_path / "selected-root"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(root)}, {"path": str(root / "contracts")}], "denied": []}
    bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = bwrap.snapshot_input_paths(ws, bundle, [root], repo=repo)[0]
    alias = tmp_path / "frozen-tools"
    argv = [
        "--ro-bind",
        str(frozen),
        str(root),
        "--ro-bind",
        str(frozen / "build_support"),
        str(alias),
        "--ro-bind",
        str(frozen / "contracts"),
        str(root / "contracts"),
    ]
    assert bwrap.is_exposed(argv, alias / "__init__.py")
    result = bwrap.apply_final_answer_masks(argv, te, ws, bundle, repo=repo)
    assert not bwrap.is_exposed(result, alias / "__init__.py")
    assert bwrap.is_exposed(result, root / "contracts/target_contract.yaml")


def test_nested_support_beats_outer_contract_grant(support, tmp_path, monkeypatch):
    root, repo, te = support
    inner = root / "contracts/nested-support"
    (inner / "contracts").mkdir(parents=True)
    (inner / "contracts/target_contract.yaml").write_text("name: other_fixture\n")
    secret = inner / "build.py"
    secret.write_text("# private nested support\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root) + ":" + str(inner))
    alias = tmp_path / "contract-view"
    argv = [
        "--ro-bind",
        str(root / "contracts"),
        str(alias),
        "--ro-bind",
        str(secret),
        str(alias / "nested-support/build.py"),
    ]
    result = bwrap.apply_final_answer_masks(argv, te, tmp_path / "ws", {}, repo=repo)
    assert bwrap.is_exposed(result, alias / "target_contract.yaml")
    assert not bwrap.is_exposed(result, alias / "nested-support/build.py")


def test_public_contract_spelling_cannot_alias_private_file(support, tmp_path):
    root, repo, te = support
    pointer = root / "contracts/build_alias.py"
    pointer.symlink_to(root / "build_support/__init__.py")
    destination = tmp_path / "public.py"
    argv = ["--ro-bind", str(pointer), str(destination)]
    result = bwrap.apply_final_answer_masks(argv, te, tmp_path / "ws", {}, repo=repo)
    assert not bwrap.is_exposed(result, destination)


@pytest.mark.parametrize("change", ["retarget", "unlink"])
def test_frozen_ownership_survives_selection_and_interior_alias_changes(support, tmp_path, monkeypatch, change):
    root, repo, te = support
    alias = root / "contracts/leaked.py"
    alias.symlink_to(root / "build_support/__init__.py")
    public = root / "contracts/target_contract.yaml"
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(root)}, {"path": str(root / "contracts")}]}
    manifest = bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    expected = bwrap.snapshot_record(ws)
    frozen = bwrap.snapshot_input_paths(ws, bundle, [root], repo=repo)[0]
    if change == "retarget":
        alias.unlink()
        alias.symlink_to(public)
    else:
        shutil.rmtree(root)
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    assert bwrap.verify_snapshot_binding(ws, bundle, expected, repo=repo) == manifest
    bwrap.verify_bundle_snapshot(ws, bundle, repo=repo)
    destination = tmp_path / "translated"
    argv = [
        "--ro-bind",
        str(frozen),
        str(destination),
        "--ro-bind",
        str(frozen / "contracts"),
        str(destination / "contracts"),
        "--bind",
        str(frozen / "contracts/leaked.py"),
        str(destination / "deeper.py"),
    ]
    assert bwrap.is_exposed(argv, destination / "deeper.py")
    result = bwrap.apply_final_answer_masks(argv, te, ws, bundle, repo=repo)
    assert not bwrap.is_exposed(result, destination / "deeper.py")
    assert not bwrap.is_exposed(result, destination / "contracts/leaked.py")
    assert bwrap.is_exposed(result, destination / "contracts/target_contract.yaml")
    support_views = bwrap.snapshot_support_surfaces(ws, manifest)
    assert not bwrap.snapshot_public_member(frozen / "contracts/leaked.py", frozen / "contracts", support_views)
    assert bwrap.snapshot_public_member(frozen / "contracts/target_contract.yaml", frozen / "contracts", support_views)
    assert not bwrap.snapshot_public_member(frozen / "contracts/target_contract.yaml", frozen, support_views)


def test_owner_metadata_is_bound_to_host_marker_not_payload_digest(support, tmp_path):
    root, repo, _ = support
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(root)}]}
    manifest = bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    expected = bwrap.snapshot_record(ws)
    manifest["support_ownership"]["views"] = []
    marker = bwrap.bundle_snapshot_root(ws) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    assert bwrap.verify_bundle_snapshot(ws, bundle, repo=repo)["content_sha256"] == expected["content_sha256"]
    with pytest.raises(RuntimeError, match="marker"):
        bwrap.verify_snapshot_binding(ws, bundle, expected, repo=repo)


@pytest.mark.parametrize("tree_grant", [False, True])
def test_intermediate_private_alias_cannot_borrow_contracts_exemption(support, tmp_path, tree_grant):
    root, repo, te = support
    public = tmp_path / "public-bytes.txt"
    public.write_text("same public bytes")
    private_alias = root / "private-link"
    private_alias.symlink_to(public)
    alias = root / "contracts/public-link"
    alias.symlink_to(private_alias)
    root_alias = tmp_path / "provider-link"
    root_alias.symlink_to(root, target_is_directory=True)
    source = root_alias / "contracts" if tree_grant else root_alias / "contracts/public-link"
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(source)}]}
    manifest = bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = bwrap.snapshot_input_paths(ws, bundle, [source], repo=repo)[0]
    secret = frozen / "public-link" if tree_grant else frozen
    destination = tmp_path / "visible"
    exposed = destination / "public-link" if tree_grant else destination
    argv = ["--ro-bind", str(frozen), str(destination)]
    assert bwrap.is_exposed(argv, exposed)
    result = bwrap.apply_final_answer_masks(argv, te, ws, bundle, repo=repo)
    assert not bwrap.is_exposed(result, exposed)
    assert not bwrap.snapshot_public_member(secret, frozen, bwrap.snapshot_support_surfaces(ws, manifest))


def test_cache_hit_does_not_transfer_public_source_ownership(support, tmp_path, monkeypatch):
    root, repo, te = support
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    public = repo / "public.py"
    private = root / "build_support/__init__.py"
    public.write_bytes(private.read_bytes())
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(public)}, {"path": str(private)}]}
    bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen_public, frozen_private = bwrap.snapshot_input_paths(ws, bundle, [public, private], repo=repo)
    assert frozen_public.stat().st_ino == frozen_private.stat().st_ino
    argv = [
        "--ro-bind",
        str(frozen_public),
        str(tmp_path / "public-view"),
        "--ro-bind",
        str(frozen_private),
        str(tmp_path / "private-view"),
    ]
    result = bwrap.apply_final_answer_masks(argv, te, ws, bundle, repo=repo)
    assert bwrap.is_exposed(result, tmp_path / "public-view")
    assert not bwrap.is_exposed(result, tmp_path / "private-view")


def test_captured_overlapping_owner_survives_original_removal(support, tmp_path, monkeypatch):
    root, repo, te = support
    inner = root / "contracts/inner"
    (inner / "contracts").mkdir(parents=True)
    (inner / "contracts/target_contract.yaml").write_text("name: nested_fixture_device\n")
    (inner / "private.py").write_text("# private nested implementation\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root) + ":" + str(inner))
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(root / "contracts")}]}
    manifest = bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = bwrap.snapshot_input_paths(ws, bundle, [root / "contracts"], repo=repo)[0]
    shutil.rmtree(root)
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    destination = tmp_path / "contract-view"
    argv = ["--ro-bind", str(frozen), str(destination)]
    result = bwrap.apply_final_answer_masks(argv, te, ws, bundle, repo=repo)
    assert bwrap.is_exposed(result, destination / "target_contract.yaml")
    assert not bwrap.is_exposed(result, destination / "inner/private.py")
    assert not bwrap.is_exposed(result, destination / "inner/contracts/target_contract.yaml")
    assert not bwrap.snapshot_public_member(
        frozen / "inner/private.py", frozen, bwrap.snapshot_support_surfaces(ws, manifest)
    )


@pytest.mark.parametrize("version", [2, 3])
def test_historical_snapshot_is_inspectable_not_executable(support, tmp_path, version):
    root, repo, _ = support
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": str(root)}]}
    manifest = bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    manifest["version"] = version
    manifest.pop("support_ownership")
    marker = bwrap.bundle_snapshot_root(ws) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    before = marker.read_bytes()
    assert bwrap.verify_bundle_snapshot(ws, bundle, repo=repo) == manifest
    with pytest.raises(RuntimeError, match="newly frozen"):
        bwrap.base_argv(ws, bundle, repo=repo)
    assert marker.read_bytes() == before


def test_final_support_verification_catches_disabled_mask(support, tmp_path, monkeypatch):
    root, repo, te = support
    alias = tmp_path / "tools"
    argv = ["--ro-bind", str(root / "build_support"), str(alias)]
    monkeypatch.setattr(bwrap, "apply_answer_masks", lambda argv, _surfaces: argv)
    with pytest.raises(RuntimeError, match="support"):
        bwrap.apply_final_answer_masks(argv, te, tmp_path / "workspace", {}, repo=repo)
