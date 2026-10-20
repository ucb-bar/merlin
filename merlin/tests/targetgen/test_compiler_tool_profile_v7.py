"""Versioned shared inventories isolate reviewed additions from frozen V6 access."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest
from test_compiler_tool_profile_v6 import bundle, original_owner, prepare, reseal

from merlin.common.paths import repo_root
from merlin.compile.scheduling import export, library_sources
from merlin.targetgen.package_runtime import _py_imports_merlin
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V6,
    COMMON_SUBSTRATE_V7,
    inherited_tool_profile,
    substrate_inventory,
    tool_profile_prompt,
    verify_tool_profile,
)
from merlin.targetgen.tool_profile_shared import reviewed_shared_manifest, shared_inventory


def test_v6_retains_its_reviewed_public_inventory_and_namespace_bytes():
    from merlin.targetgen import tool_profile_shared

    # This fingerprints public API names/glue, never compiler answers or data.
    inventory = dict(
        modules=library_sources.library_modules(6),
        public=export.public_modules(6),
        replacements=export.namespace_replacements(6),
        namespaces=tool_profile_shared._namespace_modules(6),
    )
    assert hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest() == (
        "e58384835f4ba7beaad0438c17769b320524e41d283e1165b75b6ae2001d024c"
    )


def test_explicit_v7_does_not_reinterpret_v6_manifest_or_inherited_grants(tmp_path):
    six = bundle(prepare(tmp_path / "six"))
    seven = bundle(prepare(tmp_path / "seven", version=7), version=COMMON_SUBSTRATE_V7)
    old, new = map(verify_tool_profile, (six, seven))
    assert old["proof_policy"] == new["proof_policy"]
    assert set(new["common_public_modules"]) - set(old["common_public_modules"]) == {
        "merlin.compile.scheduling.staged_dense"
    }
    original_rows = {row["module"]: row for row in old["common_sources"]}
    selected_rows = {row["module"]: row for row in new["common_sources"]}
    assert all(selected_rows[name] == row for name, row in original_rows.items())
    assert set(selected_rows) - set(original_rows) == {
        "merlin.compile.scheduling.staged_dense",
        "merlin.compile.scheduling.staged_dense_verification",
    }
    assert old["shared_scheduling_library"]["schema"] == "merlin.shared_scheduling_library.v1"
    assert "inventory_version" not in old["shared_scheduling_library"]
    assert new["shared_scheduling_library"]["schema"] == "merlin.shared_scheduling_library.v2"
    assert new["shared_scheduling_library"]["inventory_version"] == 7
    assert "Every V7 treatment" in tool_profile_prompt(seven)
    assert inherited_tool_profile(seven, copy.deepcopy(seven)) == new
    with pytest.raises(ValueError, match="Phase 2 changed"):
        inherited_tool_profile(six, seven)
    for root, version in ((tmp_path / "six", 7), (tmp_path / "seven", 6)):
        with pytest.raises(ValueError, match="reviewed SDK export"):
            reviewed_shared_manifest(root, version=version)


def test_future_reviewed_v7_roster_does_not_expand_legacy_receipts_or_v6_access(tmp_path, monkeypatch):
    # Simulate a separately reviewed V7 addition using an existing generic
    # source leaf. No candidate request accepts a supplied module roster.
    before = substrate_inventory(COMMON_SUBSTRATE_V6)
    six = bundle(prepare(tmp_path / "six"))
    extra = "merlin.transforms.loops"
    promoted = "merlin.compile.scheduling.dense_contracts"
    monkeypatch.setattr(library_sources, "LIBRARY_MODULES_V7", (*library_sources.LIBRARY_MODULES_V7, extra))
    monkeypatch.setattr(export, "PUBLIC_MODULES_V7", (*export.PUBLIC_MODULES_V7, promoted))
    assert extra not in library_sources.shared_library_sources()
    assert extra in library_sources.shared_library_sources(version=7)
    assert substrate_inventory(COMMON_SUBSTRATE_V6) == before
    assert verify_tool_profile(six)["common_public_modules"] == list(before[1])
    assert promoted not in before[1] and promoted in substrate_inventory(COMMON_SUBSTRATE_V7)[1]
    assert extra not in {row["module"] for row in reviewed_shared_manifest(tmp_path / "six")["members"]}
    seven = bundle(prepare(tmp_path / "seven", version=7), version=COMMON_SUBSTRATE_V7)
    assert promoted in verify_tool_profile(seven)["common_public_modules"]
    assert extra in {row["module"] for row in reviewed_shared_manifest(tmp_path / "seven", version=7)["members"]}
    six_owner, _ = original_owner(tmp_path / "old-owner")
    seven_owner, _ = original_owner(tmp_path / "new-owner", version=7)
    with six_owner.compiler_imports() as admission:
        assert _py_imports_merlin("import " + promoted, compiler_api=admission) is not None
    with seven_owner.compiler_imports() as admission:
        assert _py_imports_merlin("import " + promoted, compiler_api=admission) is None


@pytest.mark.parametrize("change", ["profile", "inventory_version", "namespace", "public", "member"])
def test_resealed_v7_cannot_relabel_the_selected_inventory(tmp_path, change):
    selected = bundle(prepare(tmp_path / "release", version=7), version=COMMON_SUBSTRATE_V7)
    profile = selected["compiler_tool_profile"]
    manifest = profile["shared_scheduling_library"]
    if change == "profile":
        profile["common_substrate"] = COMMON_SUBSTRATE_V6
    elif change == "inventory_version":
        manifest["inventory_version"] = 6
    elif change == "namespace":
        member = next(row for row in manifest["members"] if row["module"] == "merlin.sched.isa")
        member["sha256"] = "a" * 64
        next(row for row in profile["common_sources"] if row["module"] == "merlin.sched.isa.__init__")["sha256"] = (
            "a" * 64
        )
    elif change == "public":
        manifest["public_modules"].append("merlin.runtime.backends")
    else:
        manifest["members"].pop(1)
    manifest["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    reseal(selected)
    with pytest.raises(ValueError, match="API|namespace|membership|inventory"):
        verify_tool_profile(selected)


def test_v6_namespace_inventory_is_independent_of_v7_reviewed_replacements(monkeypatch):
    from merlin.targetgen import tool_profile_shared

    namespace_inventory = tool_profile_shared._namespace_modules(6)
    monkeypatch.setattr(
        tool_profile_shared, "_NAMESPACE_MODULES_V7", (*namespace_inventory, "merlin.reviewed_extension")
    )
    assert tool_profile_shared._namespace_modules(6) == namespace_inventory
    assert "merlin.reviewed_extension.__init__" not in shared_inventory((), (), version=6)[0]
    assert "merlin.reviewed_extension.__init__" in shared_inventory((), (), version=7)[0]
    six = export.namespace_replacements()
    monkeypatch.setattr(export, "_NAMESPACES_V7", {**six, "merlin.transforms": six["merlin.transforms"] + "# V7\n"})
    assert export.namespace_replacements(6) == six
    assert export.namespace_replacements(7) != six
    six["merlin.transforms"] = "caller mutation"
    assert export.namespace_replacements(6)["merlin.transforms"] != "caller mutation"


@pytest.mark.parametrize("version", [True, 0, 8, "7", None])
def test_unknown_inventory_version_refuses_before_writing(tmp_path, version):
    destination = tmp_path / "release"
    with pytest.raises(ValueError, match="version 6 or 7"):
        export.export_shared_scheduling(destination, license_directory=repo_root(), version=version)
    assert not destination.exists()


def test_v7_isolated_release_runs_existing_entries_without_host_backend_access(tmp_path):
    root = prepare(tmp_path / "release", version=7)
    public = substrate_inventory(COMMON_SUBSTRATE_V7)[1]
    dependencies = sorted(
        {str(importlib.util.find_spec(name).origin).rsplit("/", 2)[0] for name in ("xdsl", "numpy", "yaml", "z3")}
    )
    script = """import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root), *json.loads(sys.argv[3])]
for name in json.loads(sys.argv[2]):
    assert pathlib.Path(importlib.import_module(name).__file__).is_relative_to(root), name
from merlin.compile.scheduling.library_sources import library_modules, shared_library_sources
assert set(shared_library_sources()) == set(library_modules(6))
assert set(shared_library_sources(version=7)) == set(library_modules(7))
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
assert ScheduleBudgets().max_repairs >= 0
from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout, AccumulatorZero, CompletionFence, ComputeConsume,
    DescriptorIssue, OperandLoad, StagedDenseContract, StagedDenseProblem, rewrite_staged_dense,
)
contract = StagedDenseContract('neutral-facts', 'neutral-effects', 'ordered-signed',
    'synthetic-contract-only', 1, 2, 1, 8, 16, 32, 'read_at_compute_consume',
    'data_commands_preserve_pending_descriptor', 'raw_accumulator_identity', True, True, True)
operations = (AccumulatorZero(0,1,1), OperandLoad('lhs',0,0,0,1,1),
    OperandLoad('rhs',0,0,1,1,1), DescriptorIssue(1,0,1,1,1,True),
    CompletionFence(), ComputeConsume(0,1,1), CompletionFence(),
    AccumulatorReadout(0,0,0,1,1), CompletionFence())
actual, report = rewrite_staged_dense(operations, problem=StagedDenseProblem(1,1,1), contract=contract)
assert actual == operations and report['checks'][0]['outputs'] == 1
for name, module in tuple(sys.modules.items()):
    if name == 'merlin' or name.startswith('merlin.'):
        assert pathlib.Path(module.__file__).is_relative_to(root), name
try:
    import merlin.runtime.backends
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('withheld backend imported')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root), json.dumps(public), json.dumps(dependencies)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_each_v7_treatment_has_the_same_versioned_shared_library(tmp_path):
    from dataclasses import replace

    from merlin.targetgen.generate_bundles import generate_bundles
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release
    from merlin.targetgen.tool_registry import ARM_TOOLS

    arms = ("raw_baseline", "merlin_rtlchecks", "merlin_isa", "merlin_lean")
    tools = tuple(dict.fromkeys(tool for arm in arms for tool in ARM_TOOLS[arm]))
    root = tmp_path / "release"
    prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), tools=tools, version=7
    )
    descriptor = replace(
        load_target_experiment(repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"),
        numeric_profile=None,
    )
    bundles = generate_bundles(
        descriptor, arms=arms, common_substrate=COMMON_SUBSTRATE_V7, python_source_root=root, host_index_bits=64
    )
    profiles = [verify_tool_profile(value) for value in bundles.values()]
    assert all(profile["common_sources"] == profiles[0]["common_sources"] for profile in profiles)
    assert all(profile["shared_scheduling_library"] == profiles[0]["shared_scheduling_library"] for profile in profiles)
    assert all(profile["proof_policy"] == profiles[0]["proof_policy"] for profile in profiles)
    six, seven = shared_inventory((), (), version=6), shared_inventory((), (), version=7)
    assert six[0] == seven[0]
    assert set(seven[1]) - set(six[1]) == {"merlin.compile.scheduling.staged_dense"}
    assert set(seven[2]) - set(six[2]) == {"merlin.compile.scheduling.staged_dense_verification"}


def test_staged_api_is_not_a_live_v6_grant(tmp_path):
    old, _ = original_owner(tmp_path / "old")
    new, _ = original_owner(tmp_path / "new", version=7)
    with old.compiler_imports() as admission:
        assert _py_imports_merlin("import merlin.compile.scheduling.staged_dense", compiler_api=admission) is not None
    with new.compiler_imports() as admission:
        assert _py_imports_merlin("import merlin.compile.scheduling.staged_dense", compiler_api=admission) is None


def test_first_v7_roster_replays_without_new_staged_grants(tmp_path):
    root = prepare(tmp_path / "release", version=7)
    path = root / "shared-library.json"
    manifest = json.loads(path.read_bytes())
    old_modules = set(library_sources.library_modules(6))
    manifest["members"] = [row for row in manifest["members"] if row["module"] is None or row["module"] in old_modules]
    manifest["public_modules"] = list(export.public_modules(6))
    manifest["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    selected = bundle(root, version=COMMON_SUBSTRATE_V7)
    profile = verify_tool_profile(selected)
    assert "merlin.compile.scheduling.staged_dense" not in profile["common_public_modules"]
    assert {row["module"].removesuffix(".__init__") for row in profile["common_sources"]}.isdisjoint(
        {"merlin.compile.scheduling.staged_dense", "merlin.compile.scheduling.staged_dense_verification"}
    )
    assert inherited_tool_profile(selected, copy.deepcopy(selected)) == profile
