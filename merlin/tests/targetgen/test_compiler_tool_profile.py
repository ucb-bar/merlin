"""Common infrastructure is invariant; optional tools cannot bypass level grants."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _ALL_ARMS, _ARMS, generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.tool_profile import (
    AUTOMATIC_CHECKER_DATA,
    AUTOMATIC_CHECKER_MODULES,
    COMMON_NAMESPACE_MODULES,
    COMMON_PUBLIC_MODULES,
    COMMON_SUBSTRATE_V1,
    inherited_tool_profile,
    tool_profile_prompt,
    verify_tool_profile,
)


@pytest.fixture
def selected_sources(tmp_path):
    """Real coverage registry bytes, small independent source-membership fixture."""
    from merlin.verify import lean_contract

    root = tmp_path / "selected-src"
    for module in (*COMMON_NAMESPACE_MODULES, *COMMON_PUBLIC_MODULES, *AUTOMATIC_CHECKER_MODULES):
        path = root / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Independent source membership fixture."""\n', encoding="utf-8")
    (root / "merlin/verify/lean_contract.py").write_bytes(Path(lean_contract.__file__).read_bytes())
    for relative in AUTOMATIC_CHECKER_DATA:
        path = root / "merlin" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("-- Independent verifier data fixture.\n")
    (root / "merlin/verify/cli.py").write_text('"""Historical independent verification client."""\n')
    return root


def _te():
    return load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )


def _bundles(root, *, arms=(), lean=None):
    return generate_bundles(
        _te(),
        arms=arms,
        common_substrate=COMMON_SUBSTRATE_V1,
        python_source_root=root,
        lean_toolchain_manifest=lean,
    )


def _paths(bundle, role):
    return {row["path"] for row in bundle[role]}


def test_historical_generation_is_unversioned_and_optional_levels_are_not_default():
    manifests = generate_bundles(_te())
    assert {row["arm"] for row in manifests.values()} == set(_ARMS)
    assert all("compiler_tool_profile" not in row for row in manifests.values())
    assert {"merlin_isa", "merlin_lean"}.isdisjoint(_ARMS)
    for arm in ("merlin_isa", "merlin_lean"):
        with pytest.raises(ValueError, match="versioned common"):
            generate_bundles(_te(), arms=(arm,))


def test_common_sources_and_preservation_policy_are_identical_on_every_level(selected_sources):
    profiles = [verify_tool_profile(row) for row in _bundles(selected_sources, arms=tuple(_ALL_ARMS)).values()]
    assert len(profiles) == len(_ALL_ARMS)
    for profile in profiles:
        assert profile["common_public_modules"] == list(COMMON_PUBLIC_MODULES)
        assert profile["common_sources"] == profiles[0]["common_sources"]
        assert profile["automatic_checker_sources"] == profiles[0]["automatic_checker_sources"]
        assert profile["proof_policy"] == profiles[0]["proof_policy"]
        assert profile["proof_policy"]["mode"] == "strict_within_coverage"
        assert profile["lean_toolchain"]["readiness"] == "unavailable"


def test_automatic_verifier_closure_is_pinned_host_only_on_lower_levels(selected_sources):
    for row in _bundles(selected_sources, arms=tuple(_ALL_ARMS)).values():
        profile = verify_tool_profile(row)
        pinned = profile["automatic_checker_sources"]
        assert [entry["module"] for entry in pinned] == [*AUTOMATIC_CHECKER_MODULES, None]
        for entry in pinned:
            selected = str(selected_sources / "merlin" / entry["path"].removeprefix("merlin/python/merlin/"))
            assert entry["sha256"] == hashlib.sha256(Path(selected).read_bytes()).hexdigest()
            author = "lean_proof_authoring" in row["tools"]
            assert (selected in _paths(row, "host_inputs")) != author
            assert (selected in _paths(row, "allowed")) == author
        assert not any(path.endswith("verify/lean/") for path in _paths(row, "host_inputs"))
    before = next(iter(_bundles(selected_sources, arms=("merlin_rtlchecks",)).values()))
    foundation = selected_sources / "merlin" / AUTOMATIC_CHECKER_DATA[0]
    foundation.write_text("-- Changed verifier definition.\n")
    after = next(iter(_bundles(selected_sources, arms=("merlin_rtlchecks",)).values()))
    assert before["compiler_tool_profile"]["sha256"] != after["compiler_tool_profile"]["sha256"]
    with pytest.raises(ValueError, match="Phase 1 compiler tool profile"):
        inherited_tool_profile(before, after)


def test_optional_leaf_grants_are_denied_on_other_levels_including_historical_verify(selected_sources):
    manifests = _bundles(selected_sources, arms=tuple(_ALL_ARMS))
    for row in manifests.values():
        allowed, denied = _paths(row, "allowed"), _paths(row, "denied")
        assert "merlin/" not in denied
        assert not any(path.endswith("/verify/") for path in allowed)
        for name in ("isa_dialect_generation", "lean_proof_authoring"):
            for logical in TR.spec(name).bundle_paths:
                selected = str(selected_sources / "merlin" / logical.removeprefix("merlin/python/merlin/"))
                if logical.endswith("/"):
                    selected += "/"
                if logical.endswith("lean_contract.py"):
                    assert selected in allowed
                elif name in row["tools"]:
                    assert selected in allowed and selected not in denied
                else:
                    assert selected not in allowed and selected in denied


def test_optional_tools_cannot_be_added_as_an_undeclared_ablation():
    for tool in ("isa_dialect_generation", "lean_proof_authoring"):
        with pytest.raises(ValueError, match="explicit EL5/EL6"):
            TR.arm_tools("merlin_rtlchecks", add=(tool,))


def test_tool_flags_cannot_change_new_profile_after_preparation(selected_sources, tmp_path):
    from merlin_experiments.phase1.task_staging import resolved_tools

    row = next(iter(_bundles(selected_sources, arms=("merlin_rtlchecks",)).values()))
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "input_bundle_manifest.yaml").write_text(yaml.safe_dump(row))
    (directory / "tools.txt").write_text("\n".join(row["tools"]) + "\n")
    assert resolved_tools(row["bundle_id"], bundle_dir=directory) == tuple(row["tools"])
    with pytest.raises(ValueError, match="immutable compiler tool profile"):
        resolved_tools(row["bundle_id"], bundle_dir=directory, add_tools=("lean_proof_authoring",))
    with pytest.raises(ValueError, match="immutable compiler tool profile"):
        resolved_tools(row["bundle_id"], bundle_dir=directory, drop_tools=("isa_tools",))


def test_optional_inventory_requires_the_frozen_profile_and_uses_longest_stem(selected_sources, tmp_path):
    from types import SimpleNamespace

    from merlin_experiments.phase1.task_staging import arm_from_bundle_id, task_tool_inventory_block

    for row in _bundles(selected_sources, arms=("merlin_isa", "merlin_lean")).values():
        assert arm_from_bundle_id(row["bundle_id"]) == row["arm"]
        directory = tmp_path / row["bundle_id"]
        directory.mkdir()
        (directory / "input_bundle_manifest.yaml").write_text(yaml.safe_dump(row))
        (directory / "tools.txt").write_text("\n".join(row["tools"]) + "\n")
        prompt = task_tool_inventory_block(
            SimpleNamespace(
                bundle_id=row["bundle_id"],
                bundle_dir=directory,
                add_tools=(),
                drop_tools=(),
            )
        )
        assert "automatic verification" in prompt
        assert "`isa_dialect_generation`" in prompt
        assert ("`lean_proof_authoring`" in prompt) == (row["arm"] == "merlin_lean")


def test_changed_profile_and_phase2_assistance_are_rejected(selected_sources):
    row = next(iter(_bundles(selected_sources, arms=("merlin_isa",)).values()))
    assert inherited_tool_profile(row, copy.deepcopy(row)) == row["compiler_tool_profile"]
    changed = copy.deepcopy(row)
    changed["compiler_tool_profile"]["proof_policy"]["mode"] = "advisory"
    with pytest.raises(ValueError, match="identity changed"):
        verify_tool_profile(changed)
    changed = copy.deepcopy(row)
    changed["allowed"].append({"path": "/additional-tool", "mode": "ro"})
    with pytest.raises(ValueError, match="Phase 1 target/spec/runtime grants"):
        inherited_tool_profile(row, changed)
    changed = next(iter(_bundles(selected_sources, arms=("merlin_lean",)).values()))
    with pytest.raises(ValueError, match="Phase 1 compiler tool profile"):
        inherited_tool_profile(row, changed)


def test_profile_prompt_only_names_optional_capabilities_actually_granted(selected_sources):
    for row in _bundles(selected_sources, arms=tuple(_ALL_ARMS)).values():
        prompt = tool_profile_prompt(row)
        assert "every experiment level" in prompt
        assert "timeouts cannot discharge" in prompt
        assert ("`isa_dialect_generation`" in prompt) == ("isa_dialect_generation" in row["tools"])
        assert ("`lean_proof_authoring`" in prompt) == ("lean_proof_authoring" in row["tools"])


def test_common_generation_requires_actual_reviewed_source_members(selected_sources):
    (selected_sources / "merlin/transforms/planning.py").unlink()
    with pytest.raises(ValueError, match="source is absent"):
        _bundles(selected_sources)


def test_explicit_toolchain_is_host_owned_except_on_lean_author_level(selected_sources, tmp_path):
    executable = tmp_path / "lean"
    library = tmp_path / "Primitive.olean"
    executable.write_bytes(b"independent executable fixture")
    library.write_bytes(b"independent import fixture")

    def member(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = tmp_path / "toolchain.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "merlin.lean_toolchain.v1",
                "executable": member(executable),
                "files": [member(library)],
                "library_roots": [str(tmp_path)],
                "dependency_revision": "independent-fixture",
                "timeout_s": 1,
                "memory_mb": 64,
            }
        )
    )
    for row in _bundles(
        selected_sources, arms=("merlin_rtlchecks", "merlin_isa", "merlin_lean"), lean=manifest
    ).values():
        profile = verify_tool_profile(row)
        assert profile["lean_toolchain"]["readiness"] == "declared_unprobed"
        author = "lean_proof_authoring" in row["tools"]
        for path in (manifest, executable, library):
            assert (str(path) in _paths(row, "allowed")) == author
            assert (str(path) in _paths(row, "host_inputs")) != author
        assert str(tmp_path) + "/" not in _paths(row, "allowed")
    library.write_bytes(b"changed imports")
    with pytest.raises(ValueError, match="bytes differ"):
        _bundles(selected_sources, lean=manifest)
