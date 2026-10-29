"""Explicit import-closed releases retain algorithms, checks and access boundaries."""

import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import PUBLIC_MODULES, export_shared_scheduling
from merlin.compile.scheduling.library_sources import LIBRARY_MODULES
from merlin.targetgen.compiler_library import CompilerLibraryError, freeze_compiler_library
from merlin.targetgen.package_runtime import _py_imports_merlin


def review(root, manifest):
    return freeze_compiler_library(
        root,
        review_id="synthetic-test-only-import-review",
        public_modules=tuple(manifest["public_modules"]),
        sources=tuple((member["path"], member["module"]) for member in manifest["members"])
        + tuple((path, None) for path in manifest["review_resources"]),
    )


def test_legacy_imports_preserve_the_actual_core_objects():
    from merlin.sched import isa
    from merlin.sched.isa import types
    from merlin.transforms import pipeline, pipeline_state

    for name in (
        "AnalysisStamp",
        "PassRecord",
        "PassSpec",
        "PipelineContext",
        "PipelineSession",
        "validate_pass_order",
    ):
        assert getattr(pipeline, name) is getattr(pipeline_state, name)
    for name in ("OPERAND_KINDS", "InstrDef", "InstructionSet", "IsaError", "Operand"):
        assert getattr(isa, name) is getattr(types, name)


def test_export_is_deterministic_exact_and_import_closed(tmp_path):
    first = export_shared_scheduling(tmp_path / "first")
    second = export_shared_scheduling(tmp_path / "second")
    assert first == second
    contract = review(tmp_path / "first", first)
    assert {member.module for member in contract.members if member.module is not None} == set(LIBRARY_MODULES)
    for member in first["members"]:
        if member["module"] is None:
            assert (tmp_path / "first" / member["path"]).read_bytes() == (repo_root() / member["path"]).read_bytes()
        elif member["module"] not in first["replaced_namespace_source_sha256"]:
            assert (tmp_path / "first" / member["path"]).read_bytes() == (
                repo_root() / "src" / member["path"]
            ).read_bytes()
    assert first["admission"].startswith("requires_separate_host_review")
    assert set(first["external_package_versions"]) == {"xdsl", "numpy", "PyYAML", "z3-solver"}
    assert all(contract.permits(module) for module in PUBLIC_MODULES)
    assert _py_imports_merlin("import merlin.compile.scheduling.schedule_pipeline") is not None
    assert _py_imports_merlin("import merlin.compile.scheduling.schedule_pipeline", compiler_library=contract) is None
    for module in ("merlin", "merlin.transforms.pipeline", "merlin.runtime.backends", "merlin.runtime.reference"):
        assert not contract.permits(module)
        assert _py_imports_merlin("import " + module, compiler_library=contract) is not None


@pytest.mark.parametrize("name,selected_version", [
    ("merlin.compile.scheduling.compact_loaded_sum", 8),
    ("merlin.compile.scheduling.spatial_producer", 10),
    ("merlin.compile.scheduling.spatial_readout", 10),
    ("merlin.llvmlower.spatial_scratch_abi", 11),
    ("merlin.llvmlower.borrowed_effects", 11),
    ("merlin.compile.scheduling.configuration_contexts", 15),
    ("merlin.compile.scheduling.configuration_loop_cleanup", 15),
    ("merlin.compile.scheduling.dense_panel_readout", 15),
])
def test_compact_loaded_sum_is_available_only_in_prospective_reviewed_inventory(tmp_path, name, selected_version):
    from merlin.compile.scheduling.export import public_modules
    from merlin.compile.scheduling.library_sources import library_modules

    for version in range(6, selected_version):
        assert name not in library_modules(version)
        if version != 9:
            assert name not in public_modules(version)
    manifest = export_shared_scheduling(tmp_path / "prospective", version=selected_version)
    contract = review(tmp_path / "prospective", manifest)
    assert name in library_modules(selected_version)
    assert name in manifest["public_modules"]
    assert contract.permits(name)
    member = next(member for member in manifest["members"] if member["module"] == name)
    assert (tmp_path / "prospective" / member["path"]).read_bytes() == (
        repo_root() / "src" / member["path"]
    ).read_bytes()


def test_historical_inventory_membership_is_unchanged_by_spatial_v10():
    from merlin.compile.scheduling.export import public_modules
    from merlin.compile.scheduling.library_sources import library_modules

    # Public membership pinned before the spatial addition (ef421aaeb).
    owners = {
        6: "2e14b5081ae360398f9db8e215b47c15e7f185797db83970f86894d252cb6810",
        7: "53179f4919e8207d9338143625c84744a6cb5e7efd93464e3ed112ed1315a0da",
        8: "1e94a8a010e7e6adc0d1e72f8b7e1fe1cbbfb97d82b30ee1cf66bff8e18e1fad",
        9: "40a72750a7347a003c989c03d4d624eae89ff2d9697a5c309a44ef524357a0ed",
    }
    public = {
        6: "b0fd6d0fe93623410b63cb3872762e6bafdb8331edab2cbebb239a173cc321ca",
        7: "a05264440574e0da7501375f2af0cc83bb859c835dffd0e0a2e4b5f7b2efca49",
        8: "088d39800cd17abfa3f948440d4af16bc26a537d4e4776179bf0797972be2742",
    }
    for getter, pins in ((library_modules, owners), (public_modules, public)):
        for version, expected in pins.items():
            assert hashlib.sha256(json.dumps(getter(version), separators=(",", ":")).encode()).hexdigest() == expected


def test_changed_or_withheld_source_cannot_enter_a_reviewed_release(tmp_path):
    manifest = export_shared_scheduling(tmp_path / "library")
    contract = review(tmp_path / "library", manifest)
    path = tmp_path / "library/merlin/transforms/pipeline_state.py"
    path.write_text(path.read_text() + "\nimport merlin.runtime.reference\n")
    with pytest.raises(CompilerLibraryError, match="bytes changed"):
        contract.verify(tmp_path / "library")
    with pytest.raises(CompilerLibraryError, match="unreviewed library dependency"):
        review(tmp_path / "library", manifest)


def test_review_binds_the_version_and_dependency_manifest(tmp_path):
    manifest = export_shared_scheduling(tmp_path / "library")
    contract = review(tmp_path / "library", manifest)
    path = tmp_path / "library/shared-library.json"
    changed = {**manifest, "external_package_versions": {"unsupported": "unknown"}}
    path.write_text(json.dumps(changed))
    with pytest.raises(CompilerLibraryError, match="bytes changed"):
        contract.verify(tmp_path / "library")


def test_existing_or_linked_export_destinations_refuse(tmp_path):
    destination = tmp_path / "library"
    export_shared_scheduling(destination)
    with pytest.raises(ValueError, match="empty"):
        export_shared_scheduling(destination)
    linked = tmp_path / "linked"
    linked.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="unlinked"):
        export_shared_scheduling(linked)


def test_installed_style_sources_require_explicit_license_inputs(tmp_path, monkeypatch):
    from merlin.compile.scheduling import export

    installed = tmp_path / "site-packages"
    manifest = export_shared_scheduling(installed)
    writer = installed / "merlin/compile/scheduling/export.py"
    writer.write_bytes(Path(export.__file__).read_bytes())
    owners = {m["module"]: m["sha256"] for m in manifest["members"] if m["module"] is not None}
    monkeypatch.setattr(export, "__file__", str(writer))

    def selected_sources(*, version=6):
        assert version == 6
        return owners

    monkeypatch.setattr(export, "shared_library_sources", selected_sources)
    # A nearby interpreter/environment license is not a selected project input.
    (tmp_path / "LICENSE").write_text("unrelated environment license")
    (tmp_path / "NOTICE").write_text("unrelated environment notice")
    with pytest.raises(ValueError, match="explicit license_directory"):
        export_shared_scheduling(tmp_path / "unselected")
    selected = export_shared_scheduling(tmp_path / "selected", license_directory=installed)
    review(tmp_path / "selected", selected)
    for name in ("LICENSE", "NOTICE"):
        assert (tmp_path / "selected" / name).read_bytes() == (installed / name).read_bytes()


def test_export_refuses_source_changes_before_manifest_publication(tmp_path, monkeypatch):
    owner = importlib.import_module("merlin.compile.scheduling.block_schedule")
    source = tmp_path / "isolated_owner.py"
    source.write_bytes(Path(owner.__file__).read_bytes())
    monkeypatch.setattr(owner, "__file__", str(source))
    write = Path.write_bytes

    def changed(path, payload):
        result = write(path, payload)
        if path.name == "pipeline_state.py":
            write(source, source.read_bytes() + b"\n# implementation changed\n")
        return result

    monkeypatch.setattr(Path, "write_bytes", changed)
    with pytest.raises(ValueError, match="changed during export"):
        export_shared_scheduling(tmp_path / "library")
    assert not (tmp_path / "library/shared-library.json").exists()


def test_all_families_execute_from_only_the_exported_library(tmp_path, record_property):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("independent native C compiler unavailable")
    bundle, work = tmp_path / "library", tmp_path / "execution"
    manifest = export_shared_scheduling(bundle, version=8)
    review(bundle, manifest)
    work.mkdir()
    dependencies = sorted(
        {
            str(Path(importlib.util.find_spec(name).origin).parent.parent)
            for name in ("xdsl", "numpy", "yaml", "z3", "pytest")
        }
    )
    from test_shared_input_schedule import source as shared_input_source

    shared_source = tmp_path / "shared-input.mlir"
    shared_source.write_text(str(shared_input_source()))
    result = tmp_path / "execution.json"
    config = tmp_path / "worker.json"
    config.write_text(
        json.dumps(
            {
                "bundle": str(bundle),
                "work": str(work),
                "dependencies": dependencies,
                "fixtures": str(repo_root() / "merlin/tests/infra"),
                "shared_input_source": str(shared_source),
                "compiler": compiler,
                "result": str(result),
            }
        )
    )
    worker = repo_root() / "merlin/tests/fixtures/shared_sdk_worker.py"
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(worker), str(config)], capture_output=True, text=True, timeout=180
    )
    assert run.returncode == 0, run.stdout + run.stderr
    observed = json.loads(result.read_text())
    assert len(observed["completed"]) == 25
    assert {
        "packed_panel_transfers",
        "concurrent_copy_pipeline",
        "loop_configuration",
        "compact_command_native",
    } <= set(observed["completed"])
    assert "dense_float_io" in observed["completed"]
    assert "source_rows_compile" in observed["completed"]
    assert "configured_source_pointwise" in observed["completed"]
    assert "candidate_source_pointwise" in observed["completed"]
    assert {"source_spills_narrow", "source_spills_wide", "source_fragments_i32"} <= set(observed["completed"])
    assert observed["library_sha256"] == manifest["library_sha256"]
    assert set(observed["source_owners"]) == {row["module"] for row in manifest["members"] if row["module"] is not None}
    assert observed["bytecode_disabled"] and not list(bundle.rglob("__pycache__"))
    record_property("isolated_sdk_execution", json.dumps(observed))
    record_property("compiler_sha256", hashlib.sha256(Path(compiler).read_bytes()).hexdigest())


def test_tiled_convolution_runs_with_only_the_exported_stdlib_leaf(tmp_path):
    bundle = tmp_path / "library"
    manifest = export_shared_scheduling(bundle)
    review(bundle, manifest)
    # -I -S excludes the repository, site packages and inherited PYTHONPATH.
    program = """
import json, sys
sys.path.insert(0, sys.argv[1])
from merlin.compile.scheduling.block_schedule import (
    ConvContraction, Geometry, Compute, Store, schedule_tiled_convolution,
)
result = schedule_tiled_convolution(
    ConvContraction(1, 8, 8, 9, 3, 3, 17, padding=(1, 1, 1, 1)),
    Geometry(4, 256, 128, 128, True),
)
assert result.k_blocks < 27
assert result.m_blocks > 1 and result.n_blocks > 1
print(json.dumps([result.schedule.count(Compute), result.schedule.count(Store)]))
"""
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program, str(bundle)], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout) == [2160, 80]
    assert not list(bundle.rglob("__pycache__"))
