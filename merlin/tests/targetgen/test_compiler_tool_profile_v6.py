"""A new, immutable ordinary source release exposes the reviewed shared passes."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import _NAMESPACES, PUBLIC_MODULES
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.package_runtime import _py_imports_merlin
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V5,
    COMMON_SUBSTRATE_V6,
    attach_common_substrate,
    inherited_tool_profile,
    tool_profile_prompt,
    verify_tool_profile,
)
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release


def prepare(root):
    prepare_shared_source_release(root, python_source_root=repo_root() / "src", license_directory=repo_root())
    return root


def bundle(root, *, arm="raw_baseline", tools=()):
    selected = dict(arm=arm, tools=list(tools), allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(selected, version=COMMON_SUBSTRATE_V6, python_source_root=root, host_index_bits=64)
    for row in selected["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(root / row["path"].removeprefix("merlin/python/"))
    return selected


def original_owner(tmp_path):
    root = prepare(tmp_path / "release")
    selected = bundle(root)
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(selected))
    workspace = tmp_path / "original"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, selected, repo=repo_root())
    owner = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo_root())
    owner._verify()
    return owner, root


def reseal(selected):
    profile = selected["compiler_tool_profile"]
    profile["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in profile.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_release_requires_preparation_and_preserves_old_profiles(tmp_path):
    old = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(old, version=COMMON_SUBSTRATE_V5, python_source_root=repo_root() / "src")
    new = bundle(prepare(tmp_path / "release"))
    before, after = map(verify_tool_profile, (old, new))
    assert before["proof_policy"] == after["proof_policy"]
    assert "shared_scheduling_library" not in before
    assert set(PUBLIC_MODULES) <= set(after["common_public_modules"])
    assert "Every V6 treatment" in tool_profile_prompt(new)
    assert inherited_tool_profile(new, copy.deepcopy(new)) == after
    with pytest.raises(ValueError, match="Phase 2 changed"):
        inherited_tool_profile(old, new)
    with pytest.raises((OSError, ValueError)):
        bundle(repo_root() / "src")


def archived_roster(tmp_path):
    """Structural archived-profile fixture; this supplies no execution qualification."""
    root = prepare(tmp_path / "release")
    selected = bundle(root)
    profile = selected["compiler_tool_profile"]
    removed = {"merlin.common", "merlin.common.dense_float_ir"}
    manifest = profile["shared_scheduling_library"]
    old_paths = {str(root / row["path"]) for row in manifest["members"] if row["module"] in removed}
    manifest["members"] = [row for row in manifest["members"] if row["module"] not in removed]
    manifest["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    profile["common_sources"] = [row for row in profile["common_sources"]
                                 if row["module"].removesuffix(".__init__") not in removed]
    profile["common_implementation_modules"] = [name for name in profile["common_implementation_modules"]
                                               if name not in removed]
    selected["allowed"] = [row for row in selected["allowed"] if row["path"] not in old_paths]
    (root / "shared-library.json").write_text(json.dumps(manifest, indent=2) + "\n")
    profile["shared_scheduling_resources"][0]["sha256"] = hashlib.sha256(
        (root / "shared-library.json").read_bytes()).hexdigest()
    reseal(selected)
    return selected


def test_archived_v6_roster_keeps_exact_original_grants_after_new_release(tmp_path):
    selected = archived_roster(tmp_path)
    before = copy.deepcopy(selected)
    profile = verify_tool_profile(selected)
    assert selected == before
    assert inherited_tool_profile(selected, copy.deepcopy(selected)) == profile
    modules = {row["module"] for row in profile["common_sources"]}
    assert "merlin.common.__init__" not in modules and "merlin.common.dense_float_ir" not in modules


@pytest.mark.parametrize("mutation", ["new_live_grant", "foreign_release"])
def test_archived_v6_profile_cannot_acquire_current_or_foreign_grants(tmp_path, mutation):
    selected = archived_roster(tmp_path)
    profile = selected["compiler_tool_profile"]
    if mutation == "new_live_grant":
        profile["common_sources"].append({"module": "merlin.common.dense_float_ir",
            "path": "merlin/python/merlin/common/dense_float_ir.py", "sha256": "a" * 64})
    else:
        manifest = profile["shared_scheduling_library"]
        manifest["members"][1]["module"] = "merlin.runtime.reference"
        manifest["library_sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
        ).hexdigest()
    reseal(selected)
    with pytest.raises(ValueError, match="membership|inventory"):
        verify_tool_profile(selected)


def test_original_admission_allows_only_exact_public_leaves(tmp_path):
    owner, _ = original_owner(tmp_path)
    assert _py_imports_merlin("import merlin.compile.scheduling.schedule_pipeline") is not None
    with owner.compiler_imports() as admission:
        for module in PUBLIC_MODULES:
            assert _py_imports_merlin("import " + module, compiler_api=admission) is None
        for module in (
            "merlin",
            "merlin.runtime.backends",
            "merlin.runtime.reference",
            "merlin.compile.scheduling.storage_contracts",
            "merlin_experiments.phase2",
        ):
            assert _py_imports_merlin("import " + module, compiler_api=admission) is not None


@pytest.mark.parametrize("fault", ["namespace", "algorithm", "manifest"])
def test_changed_release_refuses_before_admission(tmp_path, fault):
    root = prepare(tmp_path / "release")
    path = (
        root
        / {
            "namespace": "merlin/sched/isa/__init__.py",
            "algorithm": "merlin/compile/scheduling/configuration.py",
            "manifest": "shared-library.json",
        }[fault]
    )
    path.write_bytes(path.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValueError):
        bundle(root)


def test_changed_namespace_hash_cannot_redefine_the_reviewed_initializer(tmp_path):
    selected = bundle(prepare(tmp_path / "release"))
    profile = selected["compiler_tool_profile"]
    manifest = profile["shared_scheduling_library"]
    row = next(row for row in profile["common_sources"] if row["module"] == "merlin.sched.isa.__init__")
    row["sha256"] = "a" * 64
    next(row for row in manifest["members"] if row["module"] == "merlin.sched.isa")["sha256"] = "a" * 64
    manifest["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    reseal(selected)
    with pytest.raises(ValueError, match="namespace replacement"):
        verify_tool_profile(selected)


def test_original_host_namespace_identity_is_independently_reopened(tmp_path):
    root = prepare(tmp_path / "release")
    selected = bundle(root)
    manifest = selected["compiler_tool_profile"]["shared_scheduling_library"]
    manifest["replaced_namespace_source_sha256"]["merlin.sched.isa"] = "a" * 64
    manifest["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    reseal(selected)
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(selected))
    workspace = tmp_path / "original"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, selected, repo=repo_root())
    owner = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo_root())
    with pytest.raises(ValueError, match="host source differs"):
        owner._verify()


def test_all_sdk_families_import_from_one_isolated_release(tmp_path):
    root = prepare(tmp_path / "release")
    # Third-party packages are explicitly available; the checkout and installed
    # Merlin must never fill a missing source. Namespaces are the SDK's exact bytes.
    script = """import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root), *__import__("json").loads(sys.argv[3])]
modules = __import__('json').loads(sys.argv[2])
for name in modules:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(root), name
from merlin.compile.scheduling.library_sources import shared_library_sources
assert shared_library_sources()
from merlin.transforms.pipeline import PipelineOptions
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
assert ScheduleBudgets().max_fusions >= 0
try:
    import merlin.runtime.backends
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('withheld backend present')
for name, module in tuple(sys.modules.items()):
    if name == 'merlin' or name.startswith('merlin.'):
        assert pathlib.Path(module.__file__).is_relative_to(root), (name, module.__file__)
        assert all(pathlib.Path(p).is_relative_to(root) for p in getattr(module, '__path__', ()))
"""
    dependencies = sorted(
        {str(Path(importlib.util.find_spec(name).origin).parent.parent) for name in ("xdsl", "numpy", "yaml", "z3")}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(root),
            json.dumps(PUBLIC_MODULES),
            json.dumps(dependencies),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for module, text in _NAMESPACES.items():
        assert (root / (module.replace(".", "/") + "/__init__.py")).read_text() == text


@pytest.mark.parametrize(
    "member", ["shared-library.json", "merlin/sched/isa/__init__.py", "merlin/compile/scheduling/configuration.py"]
)
def test_frozen_namespace_and_resource_drift_refuses_imports(tmp_path, member):
    owner, root = original_owner(tmp_path)
    frozen = bwrap.snapshot_input_paths(owner.workspace, owner.bundle, [root / member], repo=repo_root())[0]
    frozen.chmod(0o644)
    frozen.write_bytes(frozen.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="snapshot input content"):
        with owner.compiler_imports():
            pytest.fail("changed release admitted")


def test_all_treatments_receive_identical_shared_algorithms(tmp_path):
    from dataclasses import replace

    from merlin.targetgen.generate_bundles import generate_bundles
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.tool_registry import ARM_TOOLS

    root = tmp_path / "release"
    tools = tuple(
        dict.fromkeys(
            tool for arm in ("raw_baseline", "merlin_rtlchecks", "merlin_isa", "merlin_lean") for tool in ARM_TOOLS[arm]
        )
    )
    prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), tools=tools
    )
    descriptor = replace(
        load_target_experiment(repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"),
        numeric_profile=None,
    )
    selected = generate_bundles(
        descriptor,
        arms=("raw_baseline", "merlin_rtlchecks", "merlin_isa", "merlin_lean"),
        common_substrate=COMMON_SUBSTRATE_V6,
        python_source_root=root,
        host_index_bits=64,
    )
    profiles = [verify_tool_profile(value) for value in selected.values()]
    assert all(profile["common_sources"] == profiles[0]["common_sources"] for profile in profiles)
    assert all(profile["shared_scheduling_library"] == profiles[0]["shared_scheduling_library"] for profile in profiles)
    for value in selected.values():
        paths = [row["path"] for row in value["allowed"]]
        shared = {row["path"] for row in value["allowed"] if row.get("note") == "common compiler infrastructure"}
        assert all(Path(path).is_file() for path in shared)
        assert len(paths) == len(set(paths)), "duplicate source grant obscures the frozen import destination"
