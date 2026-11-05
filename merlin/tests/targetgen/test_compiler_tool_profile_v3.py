"""V3 extends exact host inputs while preserving preceding substrate memberships."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V1,
    COMMON_SUBSTRATE_V2,
    COMMON_SUBSTRATE_V3,
    attach_common_substrate,
    inherited_tool_profile,
    substrate_data,
    substrate_inventory,
    verify_tool_profile,
)


@pytest.fixture
def source_root(tmp_path):
    from merlin.verify import lean_contract

    root = tmp_path / "source"
    for group in substrate_inventory(COMMON_SUBSTRATE_V3):
        for module in group:
            path = root / (module.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('"""Independent exact-membership fixture."""\n')
    (root / "merlin/verify/lean_contract.py").write_bytes(Path(lean_contract.__file__).read_bytes())
    for name in substrate_data(COMMON_SUBSTRATE_V3):
        path = root / "merlin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("-- Independent mathematical source membership.\n")
    return root


def _descriptor():
    selected = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )
    return replace(selected, numeric_profile=None)


def _direct(root, version=COMMON_SUBSTRATE_V3):
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=version, python_source_root=root, host_index_bits=64)
    return bundle


def test_v3_adds_only_new_automatic_domains_and_keeps_original_versions(source_root):
    first, second, third = (
        _direct(source_root, version) for version in (COMMON_SUBSTRATE_V1, COMMON_SUBSTRATE_V2, COMMON_SUBSTRATE_V3)
    )
    profiles = [verify_tool_profile(bundle) for bundle in (first, second, third)]
    assert [profile["proof_policy"]["registry_version"] for profile in profiles] == ["1", "2", "3"]
    assert profiles[1]["common_public_modules"] == profiles[2]["common_public_modules"]
    assert set(profiles[2]["proof_policy"]["domains"]) - set(profiles[1]["proof_policy"]["domains"]) == {
        "resource_geometry_v1",
        "state_effect_schedule_v1",
    }
    assert profiles[2]["semantic_inputs"]["selection"] == "builtins_only"
    assert profiles[2]["semantic_inputs"]["software_spec"] is None
    assert profiles[2]["semantic_inputs"]["rtl_facts"] is None
    assert profiles[2]["semantic_inputs"]["target"] is None
    assert len(substrate_data(COMMON_SUBSTRATE_V1)) == len(substrate_data(COMMON_SUBSTRATE_V2)) == 1
    assert len(substrate_data(COMMON_SUBSTRATE_V3)) == 3
    with pytest.raises(ValueError, match="Phase 2 changed"):
        inherited_tool_profile(second, third)


def test_selected_facts_are_host_only_unless_original_level_already_grants_them(source_root, tmp_path):
    descriptor = _descriptor()
    facts_root = tmp_path / "selected-facts"
    facts_root.mkdir()
    facts = facts_root / "facts.json"
    facts.write_text(json.dumps({"facts": {"target": descriptor.target}}))
    bundles = generate_bundles(
        descriptor,
        arms=("raw_baseline", "merlin_rtlchecks", "merlin_isa", "merlin_lean"),
        common_substrate=COMMON_SUBSTRATE_V3,
        python_source_root=source_root,
        rtl_facts_root=facts_root,
        host_index_bits=64,
    )
    for bundle in bundles.values():
        profile = verify_tool_profile(bundle)
        assert profile["semantic_inputs"]["rtl_facts"] == {
            "path": str(facts),
            "sha256": hashlib.sha256(facts.read_bytes()).hexdigest(),
        }
        allowed, hosts = ({row["path"] for row in bundle[role]} for role in ("allowed", "host_inputs"))
        assert descriptor.experiment_resource("task") + "/" not in allowed
        assert descriptor.experiment_resource("task") + "/" in hosts
        if bundle["arm"] == "raw_baseline":
            assert str(facts) in hosts and str(facts_root) + "/" not in allowed
        else:
            assert str(facts) not in hosts and str(facts_root) + "/" in allowed
        for module in ("isa_spec", "isa_dialect_gen"):
            source = str(source_root / "merlin/targetgen" / (module + ".py"))
            author = "isa_dialect_generation" in bundle["tools"]
            assert (source in allowed) == author
            assert (source in hosts) != author
        assert any(
            row.get("module") == "merlin.verify.lean_state_projection" for row in profile["automatic_checker_sources"]
        )
        copied = copy.deepcopy(bundle)
        assert inherited_tool_profile(bundle, copied) == profile
        copied["compiler_tool_profile"]["semantic_inputs"]["registry_sha256"] = "a" * 64
        with pytest.raises(ValueError, match="identity changed"):
            inherited_tool_profile(bundle, copied)


def test_v3_semantic_inventory_is_required_and_cannot_be_forged_as_legacy(source_root):
    bundle = _direct(source_root)
    profile = bundle["compiler_tool_profile"]
    del profile["semantic_inputs"]
    document = {key: value for key, value in profile.items() if key != "sha256"}
    profile["sha256"] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="semantic input schema"):
        verify_tool_profile(bundle)


def _toolchain(tmp_path, source_root, *, complete):
    root = tmp_path / "mathematics"
    (root / "Merlin").mkdir(parents=True)
    executable = tmp_path / "selected-lean"
    executable.write_bytes(b"explicit unprobed executable fixture")
    names = ("Foundation", "ResourceGeometry", "StateEffects") if complete else ("Foundation",)
    files = []
    for name in names:
        source = source_root / "merlin/verify/lean/Merlin" / (name + ".lean")
        product = root / "Merlin" / (name + ".olean")
        product.write_bytes(b"explicit unprobed product fixture")
        files.extend(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (source, product)
        )
    manifest = tmp_path / "lean.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "merlin.lean_toolchain.v1",
                "executable": {"path": str(executable), "sha256": hashlib.sha256(executable.read_bytes()).hexdigest()},
                "library_roots": [str(root)],
                "files": files,
            }
        )
    )
    return manifest


def test_v3_refuses_foundation_only_manifest_without_reinterpreting_v2(source_root, tmp_path):
    manifest = _toolchain(tmp_path, source_root, complete=False)
    for version in (COMMON_SUBSTRATE_V2, COMMON_SUBSTRATE_V3):
        bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
        if version == COMMON_SUBSTRATE_V3:
            with pytest.raises(ValueError, match="packaged mathematical source"):
                attach_common_substrate(
                    bundle, version=version, python_source_root=source_root, lean_toolchain_manifest=manifest
                )
        else:
            attach_common_substrate(
                bundle, version=version, python_source_root=source_root, lean_toolchain_manifest=manifest
            )
            assert verify_tool_profile(bundle)["lean_toolchain"]["readiness"] == "declared_unprobed"


def test_v3_pins_current_sources_and_rejects_shadowed_mathematical_namespace(source_root, tmp_path):
    manifest = _toolchain(tmp_path, source_root, complete=True)
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(
        bundle, version=COMMON_SUBSTRATE_V3, python_source_root=source_root, lean_toolchain_manifest=manifest
    )
    assert verify_tool_profile(bundle)["lean_toolchain"]["readiness"] == "declared_unprobed"
    document = json.loads(manifest.read_text())
    old_root = tmp_path / "old-mathematics"
    (old_root / "Merlin").mkdir(parents=True)
    product = old_root / "Merlin/Foundation.olean"
    product.write_bytes(b"unprobed shadowing product")
    document["files"].append({"path": str(product), "sha256": hashlib.sha256(product.read_bytes()).hexdigest()})
    document["library_roots"].insert(0, str(old_root))
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="first selected namespace"):
        attach_common_substrate(
            {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []},
            version=COMMON_SUBSTRATE_V3,
            python_source_root=source_root,
            lean_toolchain_manifest=manifest,
        )
