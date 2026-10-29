"""Actual LLVM/object/link custody publishes only owned function instructions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import candidate_build_views as views
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_analysis_tool import _request, _world
from test_reference_machine_artifacts import observer
from test_reference_views import inputs

from merlin.common import invocation_record as invocation
from merlin.compile.analysis.disassembly import objdump_bin, tokenize_text
from merlin.compile.analysis.schedule_structure import analyze_normalized_commands
from merlin.kernels.machine_evidence import machine_projection


@pytest.fixture
def built(tmp_path):
    translate, clang = shutil.which("mlir-translate"), os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    linker = os.environ.get("MERLIN_TEST_LINK_DRIVER") or shutil.which("riscv64-unknown-elf-gcc")
    if not all((translate, clang, linker)):
        pytest.skip("selected LLVM and cross link driver required; no target execution")
    workspace, host = tmp_path / "workspace", tmp_path / "host"
    workspace.mkdir()
    world = _world(workspace)
    generated = host / "generated"
    generated.mkdir(parents=True)
    source, cb, _ = inputs()
    (generated / "input.interface.mlir").write_bytes(source)
    (generated / "command_buffer.json").write_text(json.dumps(cb))
    (generated / "lowered.llvm.mlir").write_text(r'''module {
      llvm.mlir.global @owned(0 : i64) : i64
      llvm.func @kernel_entry(%a:!llvm.ptr,%b:!llvm.ptr,%c:!llvm.ptr,%d:!llvm.ptr,%e:!llvm.ptr) {
        "llvm.inline_asm"() <{asm_string="la t0, owned\0Alui t1, 4352\0A.insn r 0x2b, 3, 4, x0, x0, t1",
          constraints="~{x5},~{x6}",has_side_effects}> : () -> ()
        llvm.return
      }
    }''')
    def run(stage, argv, before, after):
        result = invocation.run(argv, directory=generated, stage=stage, inputs=(before,), outputs=(after,),
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    run("llvm_translation", [translate, "--mlir-to-llvmir", str(generated / "lowered.llvm.mlir"),
                             "-o", str(generated / "kernel.ll")],
        generated / "lowered.llvm.mlir", generated / "kernel.ll")
    run("object", [clang, "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
                   "-O2", "-c", str(generated / "kernel.ll"), "-o", str(generated / "kernel.o")],
        generated / "kernel.ll", generated / "kernel.o")
    run("elf", [linker, "-nostdlib", "-nostartfiles", "-Wl,-e,kernel_entry", "-Wl,--no-relax",
                str(generated / "kernel.o"), "-o", str(generated / "package_kernel.elf")],
        generated / "kernel.o", generated / "package_kernel.elf")
    manifest = host / "input_bundle_manifest.yaml"
    manifest.write_text("# original host publication root; not an author input\n")
    publication = views.publish(workspace=workspace, public_root=workspace / "selfcheck_out/requests/owned/case",
                                private_root=views.registry_root(manifest), generated=generated,
                                compiler=workspace / "submission", entry_symbol="kernel_entry")
    return workspace, world, generated, manifest, publication


def test_real_linked_function_removes_relocation_refusal_and_keeps_immutable_source_identity(built, monkeypatch):
    workspace, world, generated, manifest, publication = built
    decoder = dict(semantics=observer(), target="array_fixture",
                   structure=dict(instruction_roles={"FETCH_PANEL": ["operand_load"]}))
    monkeypatch.setattr(analysis, "_operand_decoder", lambda *args, **kwargs: decoder)
    raw_object = subprocess.check_output([objdump_bin(), "-d", "-r", str(generated / "kernel.o")], text=True)
    rows = tokenize_text(raw_object)
    _, before = machine_projection(rows, model=analyze_normalized_commands([]), evidence_id="object", decoder=decoder)
    assert before["register_analysis"]["reason"] == "unresolved_object_relocations_require_linked_bytes"
    assert before["decoded_sites"] == 0 and before["role_sites"] == {"operand_load": 1}
    assert before["field_profiles"] == {}
    with reference.invocation(world["path"], world["sha"]):
        response = analysis.handle(_request("candidate", publication["path"], format="candidate_build"),
                                   workspace=workspace, bundle_manifest=manifest, facts=world["facts"])
    assert response["status"] == "analyzed", response
    report = response["report"]
    projection, = report["regions"]
    projection = projection["mechanism_projection"]
    assert projection["register_analysis"]["status"] == "analyzed"
    assert projection["field_profiles"]["FETCH_PANEL.height"] == [dict(value=17, sites=1)]
    assert report["producer_binding"]["status"] == "observed_native_build_consumption"
    assert report["producer_binding"]["compiler_emission_binding"] == "unknown"
    assert response["artifact_to_current_compiler_binding"] == "unknown"
    assert projection["decoded_sites"] == 1
    assert len(projection["field_profiles"]) == 2
    assert report["producer_binding"]["semantic_correspondence"] == "unknown"
    # An asynchronous author edit cannot rewrite an original published build's identity.
    (workspace / "submission/backend.py").write_text("# edited after original build\n")
    record = views.load(workspace / publication["path"], workspace=workspace, bundle_manifest=manifest)
    assert record["compiler_sha256"] == publication["compiler_sha256"]
    assert not list((workspace / publication["path"]).parent.glob("*.elf"))
    assert response["qualification_issued"] is False and response["all_pass"] is None


def test_relocation_on_the_custom_site_does_not_supply_instruction_identity():
    from dataclasses import replace

    rows = tokenize_text("0: 0860302b .insn r 0x2b,3,4,x0,x0,t1\n4: 00008067 ret\n")
    rows[0] = replace(rows[0], relocations=("R_RISCV_32 unknown",))
    decoder = dict(semantics=observer(), target="array_fixture",
                   structure=dict(instruction_roles={"FETCH_PANEL": ["operand_load"]}))
    _, report = machine_projection(rows, model=analyze_normalized_commands([]),
                                   evidence_id="relocated", decoder=decoder)
    assert not report["role_sites"] and report["decoded_sites"] == 0


def test_author_manifest_without_private_publication_and_mutated_leaf_are_refused(built):
    workspace, _, _, manifest, publication = built
    path = workspace / publication["path"]
    (path.parent / "kernel.linked.dis").write_text("0: 00008067 ret\n")
    with pytest.raises(ValueError, match="published_leaf_changed"):
        views.load(path, workspace=workspace, bundle_manifest=manifest)
    path.write_text(json.dumps(dict(schema=views.SCHEMA, publication="0" * 64)))
    with pytest.raises(FileNotFoundError):
        views.load(path, workspace=workspace, bundle_manifest=manifest)


def test_publication_rejects_symlink_leaves_and_ancestors(built, tmp_path):
    workspace, _, _, manifest, publication = built
    path = workspace / publication["path"]
    leaf = path.parent / "kernel.linked.dis"
    outside = tmp_path / "outside.dis"
    leaf.rename(outside)
    leaf.symlink_to(outside)
    with pytest.raises(OSError):
        views.load(path, workspace=workspace, bundle_manifest=manifest)
    leaf.unlink()
    outside.rename(leaf)
    directory = path.parent
    moved = directory.with_name("moved")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(OSError):
        views.load(path, workspace=workspace, bundle_manifest=manifest)


def test_atomic_writer_ignores_predictable_temp_and_rejects_redirected_parent(tmp_path, monkeypatch):
    root = tmp_path / "public"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"unchanged")
    (root / "leaf.publishing").symlink_to(sentinel)
    views._write(root / "leaf", b"retained")
    assert sentinel.read_bytes() == b"unchanged"
    original_replace = os.replace
    def replace_after_swap(source, destination, **kwargs):
        root.rename(tmp_path / "detached")
        root.symlink_to(outside, target_is_directory=True)
        return original_replace(source, destination, **kwargs)
    monkeypatch.setattr(os, "replace", replace_after_swap)
    with pytest.raises(OSError):
        views._write(root / "other", b"must not escape")
    assert not (outside / "other").exists()
    assert sentinel.read_bytes() == b"unchanged"


def test_analysis_uses_retained_payloads_after_public_leaf_replacement(built, monkeypatch):
    workspace, world, _, manifest, publication = built
    original_load = views._load
    def replace_after_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        for role in ("source", "command_buffer", "emitted", "binary"):
            Path(result[0]["leaves"][role]["path"]).write_bytes(b"changed after verified read")
        return result
    monkeypatch.setattr(views, "_load", replace_after_load)
    decoder = dict(semantics=observer(), target="array_fixture",
                   structure=dict(instruction_roles={"FETCH_PANEL": ["operand_load"]}))
    monkeypatch.setattr(analysis, "_operand_decoder", lambda *args, **kwargs: decoder)
    with reference.invocation(world["path"], world["sha"]):
        response = analysis.handle(_request("candidate", publication["path"], format="candidate_build"),
                                   workspace=workspace, bundle_manifest=manifest, facts=world["facts"])
    assert response["status"] == "analyzed", response
    projection, = response["report"]["regions"]
    assert projection["mechanism_projection"]["field_profiles"]["FETCH_PANEL.height"] == [dict(value=17, sites=1)]


def test_missing_compiler_owner_never_claims_compiler_emission_binding(built):
    workspace, _, generated, manifest, publication = built
    record = views.load(workspace / publication["path"], workspace=workspace, bundle_manifest=manifest)
    assert views.producer_binding(record)["compiler_emission_binding"] == "unknown"
    with pytest.raises(ValueError, match="original_compiler_changed"):
        views.publish(workspace=workspace, public_root=workspace / "another",
                      private_root=views.registry_root(manifest), generated=generated,
                      compiler=workspace / "submission", entry_symbol="kernel_entry",
                      compiler_snapshot_sha256="0" * 64)


def test_optional_diagnostic_timeout_is_unknown_and_interrupt_is_preserved(monkeypatch):
    def timeout(**kwargs):
        raise subprocess.TimeoutExpired(["owned disassembler"], 30)
    monkeypatch.setattr(views, "publish", timeout)
    assert views.publish_feedback()["status"] == "unknown"
    assert views.publish_feedback()["error_type"] == "TimeoutExpired"
    def interrupted(**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(views, "publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        views.publish_feedback()


@pytest.mark.parametrize("name", ["input.interface.mlir", "command_buffer.json", "lowered.llvm.mlir"])
def test_initial_emission_pins_cannot_be_replaced_by_fresh_publication_pins(built, monkeypatch, name):
    from merlin_experiments.phase1.feedback import candidate_observations

    workspace, _, generated, manifest, _ = built
    original = candidate_observations.reopen_native_products
    def replace_after_native_join(*args, **kwargs):
        result = original(*args, **kwargs)
        (generated / name).write_bytes(b"changed after original native join")
        return result
    monkeypatch.setattr(candidate_observations, "reopen_native_products", replace_after_native_join)
    with pytest.raises(ValueError, match="original_sources_changed"):
        views.publish(workspace=workspace, public_root=workspace / "another",
                      private_root=views.registry_root(manifest), generated=generated,
                      compiler=workspace / "submission", entry_symbol="kernel_entry")
