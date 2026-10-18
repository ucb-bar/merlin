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
        root, review_id="synthetic-test-only-import-review", public_modules=tuple(manifest["public_modules"]),
        sources=tuple((member["path"], member["module"]) for member in manifest["members"])
        + tuple((path, None) for path in manifest["review_resources"]),
    )


def test_legacy_imports_preserve_the_actual_core_objects():
    from merlin.sched import isa
    from merlin.sched.isa import types
    from merlin.transforms import pipeline, pipeline_state

    for name in (
        "AnalysisStamp", "PassRecord", "PassSpec", "PipelineContext", "PipelineSession", "validate_pass_order"
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
    monkeypatch.setattr(export, "shared_library_sources", lambda: owners)
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
    manifest = export_shared_scheduling(bundle)
    review(bundle, manifest)
    work.mkdir()
    dependencies = sorted({
        str(Path(importlib.util.find_spec(name).origin).parent.parent)
        for name in ("xdsl", "numpy", "yaml", "z3", "pytest")
    })
    result = tmp_path / "execution.json"
    config = tmp_path / "worker.json"
    config.write_text(json.dumps({"bundle": str(bundle), "work": str(work), "dependencies": dependencies,
                                  "fixtures": str(repo_root() / "merlin/tests/infra"),
                                  "compiler": compiler, "result": str(result)}))
    worker = repo_root() / "merlin/tests/fixtures/shared_sdk_worker.py"
    run = subprocess.run([sys.executable, "-I", "-S", "-B", str(worker), str(config)],
                         capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    observed = json.loads(result.read_text())
    assert len(observed["completed"]) == 17
    assert "configured_source_pointwise" in observed["completed"]
    assert {"source_spills_narrow", "source_spills_wide", "source_fragments_i32"} <= set(observed["completed"])
    assert observed["library_sha256"] == manifest["library_sha256"]
    assert len(observed["source_owners"]) == len(LIBRARY_MODULES)
    assert observed["bytecode_disabled"] and not list(bundle.rglob("__pycache__"))
    record_property("isolated_sdk_execution", json.dumps(observed))
    record_property("compiler_sha256", hashlib.sha256(Path(compiler).read_bytes()).hexdigest())
