"""Moves must preserve withheld identities, physical masks and persistent private resources.

Synthetic old/new layouts make these checks independent of whatever is currently installed. The
unmasked mount-table control proves that a missing relocated mask cannot yield a vacuous pass.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import access as A
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW

AS = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


def _write(root: Path, rel: str, text: str = "private answer\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_evaluation_package_move_does_not_mask_public_descriptor_helper(tmp_path, isolated_policy):
    grader = _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/eval/__init__.py").parent
    public = _write(tmp_path, "packages/merlin-experiments/src/merlin/benchharness/capsule_descriptor.py")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == grader and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {grader}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_moved_golden_exporter_is_masked_without_hiding_public_emitter(tmp_path, isolated_policy):
    exporter = _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/model_slice_export.py")
    emitter = _write(tmp_path, "src/merlin/targetgen/contract/matmul_interface.py")
    assert "merlin.targetgen.model_slice_export" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == exporter and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != emitter and surface.path not in emitter.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {exporter}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize(
    "module",
    [
        "merlin.targetgen.eval.gemmini_conformance",
        "merlin.targetgen.eval.gemmini_suite",
        "merlin.targetgen.eval.gemmini_dispatcher",
        "merlin.targetgen.agent.gemmini_kernel_slot",
        "merlin.targetgen.model_slice_export",
    ],
)
def test_evicted_gemmini_owners_retain_historical_private_identity(tmp_path, isolated_policy, module):
    assert module in A.declared_modules("grader")
    assert "gemmini_conformance" in A.declared_modules("grader")
    relative = "packages/merlin-experiments/src/" + module.replace(".", "/") + ".py"
    private = _write(tmp_path, relative)
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(s.path == private or s.path in private.parents for s in surfaces)
    mounted = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert BW.coverage_gap(mounted, surfaces)
    assert BW.coverage_gap(BW.apply_answer_masks(mounted, surfaces), surfaces) == []


def test_relocated_evaluation_cohort_keeps_grader_mask(tmp_path, isolated_policy):
    cohort = _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/evaluation_cohort.py")
    assert "merlin.targetgen.evaluation_cohort" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == cohort and surface.origin == "grader" for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {cohort}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize("module", ["merlin.targetgen.oracle_helpers.npu_emit", "atlas_program_emit"])
def test_program_emitters_retain_private_identity(tmp_path, isolated_policy, monkeypatch, module):
    assert module in A.declared_modules("oracle")
    if module == "atlas_program_emit":
        private = _write(tmp_path, "support/atlas_program_emit.py")
        monkeypatch.setattr(AS, "_support_package_dirs", lambda: [private.parent])
    else:
        private = _write(tmp_path, "src/" + module.replace(".", "/") + ".py")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private or surface.path in private.parents for surface in surfaces)
    mounted = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert BW.coverage_gap(mounted, surfaces)
    assert BW.coverage_gap(BW.apply_answer_masks(mounted, surfaces), surfaces) == []


def test_phase1_private_run_inputs_mask_does_not_hide_inert_context(tmp_path, isolated_policy):
    private = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/run_inputs.py")
    context = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/context.py")
    assert "merlin_experiments.phase1.run_inputs" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != context and surface.path not in context.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {private}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_provider_host_directory_is_masked_but_public_client_is_not(tmp_path, isolated_policy):
    provider = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/providers/codex_agent.py")
    public = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py")
    assert "merlin_experiments.phase1.providers" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == provider.parent and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {provider.parent}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize(
    "owner,member",
    [
        ("merlin_experiments.phase1.brokers", "isa_tools.py"),
        ("merlin_experiments.phase2", "campaign.py"),
        ("merlin_experiments.phase2", "broker.py"),
        ("merlin_experiments.phase2", "transcript_audit.py"),
        ("merlin_experiments.phase2", "functional_inputs.py"),
        ("merlin_experiments.phase1.telemetry", "evidence.py"),
    ],
)
def test_packaged_phase_tools_keep_host_only_directory_masks(tmp_path, isolated_policy, owner, member):
    relative = "packages/merlin-experiments/src/" + owner.replace(".", "/")
    private = _write(tmp_path, relative + "/" + member)
    public = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py")
    assert owner in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private.parent and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {private.parent}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize(
    "relative",
    [
        "src/merlin/targetgen/golden_provenance.py",
        "src/merlin/targetgen/capsule_inputs.py",
        "packages/merlin-experiments/src/merlin_experiments/source_snapshot.py",
        "packages/merlin-experiments/src/merlin_experiments/measured_launch.py",
        "packages/merlin-experiments/src/merlin_experiments/corpus/admission.py",
        "packages/merlin-experiments/src/merlin_experiments/corpus/preparation.py",
        "packages/merlin-experiments/src/merlin_experiments/corpus/release.py",
        "packages/merlin-experiments/src/merlin_experiments/phase1/treatments.py",
        "packages/merlin-experiments/src/merlin_experiments/phase1/corpus_inputs.py",
        "packages/merlin-experiments/src/merlin/targetgen/numeric_falsifiability.py",
        "packages/merlin-experiments/src/merlin/targetgen/rtl/gen_rocc_replay.py",
        "packages/merlin-analysis/src/merlin/verify/replay.py",
        "packages/merlin-analysis/src/merlin/verify/replay_layers.py",
    ],
)
def test_golden_metadata_and_relocated_audits_keep_grader_masks(tmp_path, isolated_policy, relative):
    private = _write(tmp_path, relative)
    assert A.module_name_for(relative) in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private and surface.origin == "grader" for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {private}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_managed_execution_service_and_clients_are_host_only(tmp_path, isolated_policy):
    private = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/execution/chia_native.py")
    assert "merlin_experiments.execution" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private.parent and surface.origin == "grader" for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {private.parent}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize("name", ["formal", "freeze"])
def test_formal_lifecycle_is_withheld_by_registered_feedback_namespace(tmp_path, isolated_policy, name):
    private = _write(tmp_path, f"packages/merlin-experiments/src/merlin_experiments/phase1/feedback/{name}.py")
    assert "merlin_experiments.phase1.feedback" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == private.parent and surface.origin == "grader" for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {private.parent}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_preflight_probes_actual_evaluator_and_rejects_missing_old_paths(tmp_path, isolated_policy):
    # Execute the real pure probe builder without importing the hardware/agent-launch harness.
    source = repo_root() / "merlin/experiments/capsule_bench/harness/preflight_sandbox.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_masks"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    build = namespace["_source_masks"]
    oracle = _write(tmp_path, "src/merlin/runtime/reference.py")
    simulator = _write(tmp_path, "src/merlin/runtime/simulator.py")
    with pytest.raises(SystemExit, match="absent required module merlin.targetgen.capsule_grade"):
        build(tmp_path)
    grader = _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/capsule_grade.py")
    assert set(build(tmp_path).values()) == {oracle, simulator, grader}


def test_package_owned_phase0_is_masked_as_a_whole_before_any_child_import(tmp_path, isolated_policy):
    package = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase0/__init__.py").parent
    _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase0/numerics.py")
    _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase0/golden_cache.py")
    public = _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py")
    assert "merlin_experiments.phase0" in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(isolated_policy)
    assert any(surface.path == package and surface.origin == "grader" for surface in surfaces)
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {package}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.fixture
def isolated_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(AS, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(AS, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(AS, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(AS, "_support_package_dirs", lambda: [])
    monkeypatch.setattr(AS, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    return SimpleNamespace(
        target="test_device",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("merlin/python/merlin/runtime/reference.py", "merlin.runtime.reference"),
        ("src/merlin/runtime/reference.py", "merlin.runtime.reference"),
        ("src/merlin/runtime/__init__.py", "merlin.runtime"),
        ("src/merlin/runtime/", "merlin.runtime"),
        (
            "packages/merlin-experiments/src/merlin_experiments/evaluation/program_oracle.py",
            "merlin_experiments.evaluation.program_oracle",
        ),
        ("packages/merlin-dse/src/merlin_dse/cost.py", "merlin_dse.cost"),
        ("packages/merlin-dse/src/merlin/dse/cost.py", "merlin.dse.cost"),
        ("packages/merlin-mining/src/merlin_mining/search.py", "merlin_mining.search"),
        ("packages/merlin-analysis/src/merlin_analysis/report.py", "merlin_analysis.report"),
    ],
)
def test_module_identity_does_not_depend_on_file_existence(path, module):
    assert A.module_name_for(path) == module
    assert AS.module_name_for(path) == module  # existing audit callers retain their import


@pytest.mark.parametrize(
    "path",
    [
        "src/merlin/../../private.py",
        "/src/merlin/runtime/reference.py",
        "packages/unrelated/src/merlin/runtime/reference.py",
        "merlin/contract/capsules/example.yaml",
        "src/merlin/not_python.yaml",
    ],
)
def test_unrelated_or_escaping_paths_are_not_module_identities(path):
    assert A.module_name_for(path) is None


@pytest.mark.parametrize("source", ["merlin/python/merlin", "src/merlin"])
def test_legacy_and_src_masks_have_nonvacuous_coverage(tmp_path, isolated_policy, source):
    oracle = _write(tmp_path, f"{source}/runtime/reference.py")
    grader = _write(tmp_path, f"{source}/targetgen/capsule_grade.py")
    public = _write(tmp_path, f"{source}/xdsl_dialects/interface.py", "# public grammar\n")
    surfaces = AS.answer_surfaces(isolated_policy)
    by_path = {surface.path: surface.origin for surface in surfaces}
    assert by_path[oracle] == "oracle"
    assert by_path[grader] == "grader"
    assert public not in by_path
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {oracle, grader}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_both_compatibility_copies_and_new_namespace_are_masked(tmp_path, isolated_policy):
    paths = {
        _write(tmp_path, "merlin/python/merlin/targetgen/program_oracle.py"),
        _write(tmp_path, "src/merlin/targetgen/program_oracle.py"),
        _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/evaluation/program_oracle.py"),
    }
    grader = _write(
        tmp_path, "packages/merlin-experiments/src/merlin_experiments/evaluation/capsule_grade/__init__.py"
    ).parent
    surfaces = AS.answer_surfaces(isolated_policy)
    by_path = {surface.path: surface.origin for surface in surfaces}
    assert paths <= by_path.keys()
    assert all(by_path[path] == "oracle" for path in paths)
    assert by_path[grader] == "grader"
    argv = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert len(BW.coverage_gap(argv, surfaces)) == 4
    assert BW.coverage_gap(BW.apply_answer_masks(argv, surfaces), surfaces) == []
    assert "merlin_experiments/evaluation/program_oracle" in AS.audit_tokens(isolated_policy)["answer"]


def test_split_program_values_keeps_oracle_masks(tmp_path, isolated_policy):
    paths = {
        _write(tmp_path, "src/merlin/targetgen/program_oracle.py"),
        _write(tmp_path, "src/merlin/targetgen/program_values.py"),
        _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/evaluation/program_values.py"),
    }
    surfaces = AS.answer_surfaces(isolated_policy)
    by_path = {surface.path: surface.origin for surface in surfaces}
    assert paths <= by_path.keys()
    assert all(by_path[path] == "oracle" for path in paths)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == paths
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []
    assert "merlin.targetgen.program_values" in A.declared_modules("oracle")


def test_split_package_runtime_and_certification_keep_grader_masks(tmp_path, isolated_policy):
    paths = {
        _write(tmp_path, "src/merlin/targetgen/oot_runner.py"),
        _write(tmp_path, "src/merlin/targetgen/package_runtime.py"),
        _write(tmp_path, "src/merlin/targetgen/oracle_policy.py"),
        _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/_capsule_bundle_worker.py"),
        _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/package_certification.py"),
    }
    surfaces = AS.answer_surfaces(isolated_policy)
    by_path = {surface.path: surface.origin for surface in surfaces}
    assert paths <= by_path.keys()
    assert all(by_path[path] == "grader" for path in paths)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == paths
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []
    assert {"merlin.targetgen.package_runtime", "merlin.targetgen.package_certification"} <= set(
        A.declared_modules("grader")
    )


def test_legacy_source_symlink_does_not_erase_either_mask(tmp_path, isolated_policy):
    relocated = _write(tmp_path, "src/merlin/runtime/reference.py")
    legacy = tmp_path / "merlin/python/merlin"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to("../../src/merlin", target_is_directory=True)
    old_path = legacy / "runtime/reference.py"
    assert old_path.resolve() == relocated
    surfaces = AS.answer_surfaces(isolated_policy)
    assert {surface.path for surface in surfaces} == {old_path, relocated}
    argv = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert len(BW.coverage_gap(argv, surfaces)) == 2
    assert BW.coverage_gap(BW.apply_answer_masks(argv, surfaces), surfaces) == []


def test_absent_sources_remain_declared_and_are_reported_unresolved(tmp_path, isolated_policy):
    missing = A.unresolved_modules(tmp_path)
    assert missing and len(missing) == len(A.MODULE_ACCESS)
    assert "merlin.targetgen.program_oracle" in AS.declared_oracle_modules()
    assert "merlin_experiments.evaluation.program_oracle" in AS.declared_oracle_modules()
    _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/evaluation/program_oracle.py")
    assert "merlin.targetgen.program_oracle" not in {item.identity for item in A.unresolved_modules(tmp_path)}


def test_installed_and_shadowed_evaluators_and_bytecode_are_masked(tmp_path, isolated_policy, monkeypatch):
    site = tmp_path / ".venv/lib/python3.12/site-packages"
    other_site = tmp_path / "alternate-python/site-packages"
    editable = tmp_path / "editable-contribution/merlin"
    source = _write(tmp_path, "src/merlin/targetgen/capsule_grade.py", "raise AssertionError('never import')\n")
    installed = _write(site, "merlin/targetgen/capsule_grade.py", "raise AssertionError('never import')\n")
    cached = _write(site, "merlin/targetgen/__pycache__/capsule_grade.cpython-312.pyc")
    sourceless = _write(site, "merlin/targetgen/program_values.pyc")
    alternate = _write(other_site, "merlin/targetgen/capsule_grade.py")
    contributed = _write(editable, "targetgen/package_certification.py")
    public = _write(site, "merlin/xdsl_dialects/interface.py")
    monkeypatch.setattr(
        A,
        "sys",
        SimpleNamespace(
            path=[str(other_site)],
            prefix=str(tmp_path / "python"),
            modules={"merlin": SimpleNamespace(__path__=[str(editable)])},
        ),
    )
    surfaces = AS.answer_surfaces(isolated_policy)
    expected = {source, installed, cached, sourceless, alternate, contributed}
    assert expected <= {surface.path for surface in surfaces}
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    # Broad toolchain bind exposes installed bytes even when checkout imports shadow them.
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == expected
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_installed_resources_outside_checkout_keep_masks_and_audit_tokens(tmp_path, isolated_policy, monkeypatch):
    checkout = tmp_path / "checkout"
    site = tmp_path / "installed/site-packages"
    monkeypatch.setattr(AS, "repo_root", lambda: checkout)
    monkeypatch.setattr(A, "sys", SimpleNamespace(path=[str(site)], prefix=str(tmp_path / "python"), modules={}))
    expected = _write(site, "merlin/_data/contract/examples/expected_result.json")
    golden = _write(site, "merlin_experiments/_data/contract/capsules/public/sample/golden.yaml")
    hidden = _write(site, "merlin_experiments/_data/contract/capsules/hidden/sample/capsule.yaml").parents[1]
    surfaces = AS.answer_surfaces(isolated_policy)
    assert {surface.path for surface in surfaces} == {expected, golden, hidden}
    unmasked = ["--ro-bind", str(site), str(site)]
    assert len(BW.coverage_gap(unmasked, surfaces)) == 3
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []
    assert "capsules/hidden" in AS.audit_tokens(isolated_policy)["answer"]


def test_borrowed_venv_masks_both_bind_destinations(tmp_path, isolated_policy):
    real = tmp_path / "borrowed-python"
    grader = _write(real, "lib/python3.12/site-packages/merlin/targetgen/capsule_grade.py")
    (tmp_path / ".venv").symlink_to(real, target_is_directory=True)
    alias = tmp_path / ".venv" / grader.relative_to(real)
    surfaces = AS.answer_surfaces(isolated_policy)
    assert {grader, alias} <= {surface.path for surface in surfaces}
    unmasked = ["--ro-bind", str(real), str(real), "--ro-bind", str(tmp_path / ".venv"), str(tmp_path / ".venv")]
    assert {surface.path for surface in BW.coverage_gap(unmasked, surfaces)} == {grader, alias}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_installed_grader_is_unreadable_in_real_bwrap(tmp_path, isolated_policy):
    executable = shutil.which("bwrap")
    if executable is None:
        pytest.skip("bubblewrap is not installed")
    base = [executable, "--die-with-parent", "--ro-bind", "/", "/"]
    probe = subprocess.run([*base, "--", sys.executable, "-c", "pass"], capture_output=True, text=True)
    if probe.returncode:
        pytest.skip(f"bubblewrap unavailable on this host: {probe.stderr.strip()}")
    grader = _write(tmp_path, ".venv/lib/python3.12/site-packages/merlin/targetgen/capsule_grade.py", "PRIVATE\n")
    read = ["--", sys.executable, "-c", "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())", str(grader)]
    exposed = subprocess.run([*base, *read], capture_output=True, text=True, check=True)
    assert exposed.stdout.strip() == "PRIVATE"  # negative control: the actual bind exposes wheel bytes
    protected = subprocess.run(
        [*BW.apply_answer_masks(base, AS.answer_surfaces(isolated_policy)), *read],
        capture_output=True,
        text=True,
        check=True,
    )
    assert protected.stdout.strip() == ""


def test_non_filesystem_active_namespace_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        A,
        "sys",
        SimpleNamespace(
            path=[],
            prefix=str(tmp_path / "python"),
            modules={"merlin": SimpleNamespace(__path__=[str(tmp_path / "bundle.zip/merlin")])},
        ),
    )
    with pytest.raises(RuntimeError, match="cannot mask non-filesystem harness namespace"):
        A.runtime_package_roots(tmp_path)


def test_editable_namespace_placeholders_resolve_loaded_metadata_only(tmp_path, isolated_policy, monkeypatch):
    package = tmp_path / "outside-editable/targetgen"
    grader = _write(package, "capsule_grade.py", "raise AssertionError('must not import')\n")
    placeholder = "__editable__.test.finder.__path_hook__"
    monkeypatch.setattr(
        A,
        "sys",
        SimpleNamespace(
            path=[placeholder],
            prefix=str(tmp_path / "python"),
            modules={
                "merlin": SimpleNamespace(__path__=[placeholder]),
                "test_finder": SimpleNamespace(
                    PATH_PLACEHOLDER=placeholder,
                    MAPPING={"merlin.targetgen": str(package)},
                    NAMESPACES={"merlin": []},
                ),
            },
        ),
    )
    surfaces = AS.answer_surfaces(isolated_policy)
    assert {surface.path for surface in surfaces} == {grader}
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert BW.coverage_gap(unmasked, surfaces)
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_real_release_wheel_files_are_covered_when_installed(tmp_path, isolated_policy):
    root = repo_root()
    candidates = (
        sorted((root / "out/build/dist").glob("merlin-*.whl")),
        sorted((root / "out/build/python/merlin-experiments/dist").glob("merlin_experiments-*.whl")),
    )
    if not all(candidates):
        pytest.skip("build core and experiments release wheels to exercise their installed bytes")
    site = tmp_path / ".venv/lib/python3.12/site-packages"
    for wheels in candidates:
        with zipfile.ZipFile(max(wheels, key=lambda path: path.stat().st_mtime_ns)) as archive:
            archive.extractall(site)
    grader = site / "merlin/targetgen/package_certification.py"
    oracle = site / "merlin/runtime/reference.py"
    expected_output = site / "merlin/_data/contract/examples/expected_command_buffer_g0.json"
    public = site / "merlin/xdsl_dialects/interface.py"
    assert all(path.is_file() for path in (grader, oracle, expected_output, public))
    surfaces = AS.answer_surfaces(isolated_policy)
    assert {grader, oracle, expected_output} <= {surface.path for surface in surfaces}
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path / ".venv"), str(tmp_path / ".venv")]
    assert {grader, oracle, expected_output} <= {surface.path for surface in BW.coverage_gap(unmasked, surfaces)}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize("resource_root", A.CONTRACT_RESOURCE_ROOTS)
def test_private_resource_masks_follow_each_declared_layout(tmp_path, isolated_policy, resource_root):
    golden = _write(tmp_path, f"{resource_root}/capsules/public/sample/golden.yaml")
    weight = _write(tmp_path, f"{resource_root}/capsules/public/sample/capsule.weights.safetensors")
    hidden = _write(tmp_path, f"{resource_root}/capsules/hidden/sample/capsule.yaml").parents[1]
    holdout = _write(tmp_path, f"{resource_root}/capsules/profiles/test.hidden.yaml")
    expected = _write(tmp_path, f"{resource_root}/examples/expected_command_buffer.json")
    surfaces = AS.answer_surfaces(isolated_policy)
    by_path = {surface.path: surface.origin for surface in surfaces}
    assert by_path[golden] == "golden"
    assert by_path[weight] == "weight"
    assert by_path[hidden] == "hidden"
    assert by_path[holdout] == "hidden"
    assert by_path[expected] == "example"
    argv = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert len(BW.coverage_gap(argv, surfaces)) == 5
    assert BW.coverage_gap(BW.apply_answer_masks(argv, surfaces), surfaces) == []
    tokens = AS.audit_tokens(isolated_policy)["answer"]
    assert "capsules/hidden" in tokens and ".hidden.yaml" in tokens
    assert "public/sample/capsule.weights.safetensors" in tokens


def test_recovery_keys_and_sources_survive_missing_optional_scorer(tmp_path, isolated_policy, monkeypatch):
    real_import = builtins.__import__

    def without_research(name, *args, **kwargs):
        if name.startswith(("merlin.perf", "merlin_experiments")):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    source = _write(tmp_path, "private-source/patch.diff")
    body = "provenance:\n  answer_surfaces:\n    - private-source\n"
    old = _write(tmp_path, f"out/artifacts/{A.KEY_TOPIC}/sample/{A.KEY_FILENAME}", body)
    folded = _write(tmp_path, f"out/artifacts/{'/'.join(A.KEY_TOPIC_FOLDED)}/sample/{A.KEY_FILENAME}", body)
    override = _write(tmp_path, f"operator-selected/{A.KEY_FILENAME}", body)
    monkeypatch.setenv(A.KEY_ENV, str(override))
    monkeypatch.setattr(builtins, "__import__", without_research)
    assert set(AS.recovery_key_files()) == {old, folded, override}
    surfaces = {surface.path: surface.origin for surface in AS.answer_surfaces(isolated_policy)}
    assert all(surfaces[path] == "recovery_key" for path in (old, folded, override, source.parent))
    assert A.KEY_FILENAME in AS.audit_tokens(isolated_policy)["answer"]
