"""Actual ordinary transport while provenance facets remain explicitly isolated.

The private structural fixture returns a pointer entry without output stores.
It cannot issue fresh origin, original source selection, ISA or static authority.
Its only purpose is to test the controller's real complete transport evidence,
private snapshot reopening and refusal despite successful large-shape linking.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.component_compile_sources import CompileOnlySourceMember
from merlin_experiments.phase1 import component_compile_roles as R
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import invocation_record
from merlin.common.paths import data_path
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def unused_renderer(*args, **kwargs):
    raise AssertionError("source-only role allocated a numerical harness")


def diagnostic_gate(*, elf, evidence_root):
    evidence_root.mkdir()
    path = evidence_root / "structural-diagnostic.json"
    result = {"status": "accepted", "elf_sha256": file_digest(elf), "scope": "unit transport, no ISA authority"}
    path.write_text(json.dumps(result))
    return {**result, "report_path": str(path), "report_sha256": file_digest(path)}


class ActualCommands:
    def build_package(self, pkg, **kwargs):
        assert not pkg.manifest.get("build")

    def run_entrypoint(self, pkg, name, source, output=None, *, timeout, invocation_directory, **kwargs):
        argv = P._resolve_argv(pkg, name, source, output)
        return invocation_record.run(
            [sys.executable, "-I", "-B", *argv[1:]], directory=invocation_directory, stage=name,
            inputs=(Path(source),), outputs=(Path(output),) if output is not None else (),
            dependencies=tuple(pkg.directory.rglob("*.*")), cwd=pkg.directory,
            capture_output=True, text=True, timeout=timeout,
        )


@pytest.fixture
def transport(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    compiler = os.environ.get("MERLIN_TEST_RISCV_GCC")
    if not compiler or not toolchain.mlir_translate().is_file() or not Path(toolchain.clang()).is_file():
        pytest.skip("requires explicitly selected stock translation/object/link tools")
    monkeypatch.setattr(base, "get_backend", lambda *args: pytest.fail("source-only role discovered a backend"))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    shape = (1 << 30, 7)
    tensors = {name: {"shape": list(shape), "dtype": "i32", "role": name} for name in ("input", "output")}
    buffer = {
        "abi_version": "0.1", "target": "structural_control", "commands": [], "tensors": tensors,
        "kernel_abi": {"kind": "whole_program", "outputs": ["output"], "args": [
            {"tensor": "input", "access": "read"}, {"tensor": "output", "access": "write"},
        ]},
    }
    (candidate / "buffer.json").write_text(json.dumps(buffer))
    (candidate / "emitted.mlir").write_text(
        "module { llvm.func @control_entry(%a: !llvm.ptr, %b: !llvm.ptr) { llvm.return } }\n"
    )
    names = ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    (candidate / "manifest.yaml").write_text(json.dumps({
        "artifact_type": "mlir_oot_target_backend", "target": "structural_control", "language": "python",
        "authoring": {"mode": "hand_curated"}, "integrity_exempt": False, "entrypoints": {"tool": "driver.py"},
        "commands": {name: {"argv": ["python3", "driver.py", name, "{input_mlir}",
                                     *(["{output_json}"] if name == "emit_command_buffer" else [])]} for name in names},
    }))
    (candidate / "driver.py").write_text(
        "from pathlib import Path\nimport sys\ncommand, source, *rest = sys.argv[1:]\n"
        "text = Path(source).read_text()\n"
        "if command == 'lower_interface_to_target': sys.stdout.write(text)\n"
        "elif command == 'emit_command_buffer': Path(rest[0]).write_bytes(Path('buffer.json').read_bytes())\n"
        "elif command == 'lower_target_to_llvm': sys.stdout.write(Path('emitted.mlir').read_text())\n"
        "elif command != 'parse': raise ValueError(command)\n"
    )
    abi = CompileOnlySourceAbi(
        (CompileOnlyTensor("input", shape, "i32"),), (CompileOnlyTensor("output", shape, "i32"),)
    )
    members = []
    for name, cohort in (("original-large", "functional_guard"), ("private-transfer", "withheld_transfer")):
        source = tmp_path / "originals" / name / "source.mlir"
        source.parent.mkdir(parents=True)
        source.write_text(
            "module { func.func @source(%a: tensor<1073741824x7xi32>) -> tensor<1073741824x7xi32> {\n"
            "func.return %a : tensor<1073741824x7xi32> } }\n"
        )
        members.append(CompileOnlySourceMember(
            name, cohort, "compile_only", "mlir", source, file_digest(source), abi,
            ("semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage"),
        ))
    roster = SimpleNamespace(members=tuple(members), sha256="unissued-structural-fixture")
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    link_script = tmp_path / "link.ld"
    link_script.write_text(
        "ENTRY(main)\nSECTIONS { . = 0x80000000; .text : { *(.text*) } .data : { *(.data*) } .bss : { *(.bss*) } }\n"
    )
    owner, cc = Path(__file__).resolve(), Path(compiler).resolve(strict=True)
    pins = tuple((str(path), file_digest(path)) for path in (owner, cc, link_script))
    recipe = HarnessBuildRecipe(
        cc, (), (), link_script, 0x80000000,
        ("-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffunction-sections", "-nostdlib"),
        ldflags=("-Wl,--gc-sections",), kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
    )
    build = BuildOnlyService("structural_control", recipe, unused_renderer, pins)
    gate = LinkedElfAdmissionService("structural_control", diagnostic_gate, ((str(owner), file_digest(owner)),))
    view = tmp_path / "view"
    library_root = view / "compiler"
    (library_root / "merlin").mkdir(parents=True)
    (library_root / "merlin/__init__.py").write_text("")
    (library_root / "merlin/portable.py").write_text("VALUE = 1\n")
    library = freeze_compiler_library(
        library_root,
        review_id="unit transport isolation; no semantic authority",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    origin = SimpleNamespace(
        inputs=SimpleNamespace(
            hardware=SimpleNamespace(target="structural_control"),
            view=SimpleNamespace(root=view),
            runtime=(),
            library=library,
            compiler_transport=None,
            corpus_root=tmp_path / "numeric-corpus",
        )
    )
    monkeypatch.setattr(R, "_selection", lambda **kwargs: {"scope": "unit provenance facets isolated, no authority"})
    monkeypatch.setattr(R, "qualified_package_execution", lambda **kwargs: P.scoped_package_executor(ActualCommands()))
    return dict(
        roster=roster, compiler_origin=origin, candidate=candidate, contract_root=contract, build_service=build,
        instruction_check=SimpleNamespace(admission_service=lambda: gate),
        readelf=Path(shutil.which("readelf")).resolve(strict=True), evidence_root=tmp_path / "evidence", timeout_s=60,
    )


def test_actual_every_original_source_links_but_static_unknown_prevents_complete_qualification(transport):
    evaluation = R.evaluate_component_compile_roles(**transport)
    document = evaluation.verify()
    assert document["compilation_denominator"] == {"required": 2, "linked_and_policy_accepted": 2}
    assert document["static_denominator"] == {"required": 8, "proved": 0, "unknown": 8}
    assert len(document["unresolved"]) == 8 and len(document["members"]) == 2
    assert document["numerical_execution"] == document["physical_effects"] == document["performance"] == "not_attempted"
    for row in document["members"]:
        observed = json.loads(Path(row["transport_report"]).read_bytes())
        assert {"parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact", "elf"} <= {
            record["stage"] for record in observed["invocations"]
        }
        assert observed["inputs"]["package_root"] == str(evaluation.compiler_snapshot)
    with pytest.raises(StageGateError, match="compilation/static roles remain incomplete"):
        evaluation.require_complete()
    original_sha = file_digest(transport["candidate"] / "driver.py")
    (evaluation.compiler_snapshot / "driver.py").write_text("raise RuntimeError('changed clone')\n")
    assert file_digest(transport["candidate"] / "driver.py") == original_sha
    with pytest.raises(StageGateError, match="private snapshot changed"):
        evaluation.verify()


def test_failed_original_compilation_keeps_full_private_denominator_and_never_proves_static_refusal(transport):
    roster = transport["roster"]
    roster.members = (replace(roster.members[0], expectation="static_refusal"), roster.members[1])
    transport["candidate"].joinpath("driver.py").write_text("raise RuntimeError('deliberate diagnostic failure')\n")
    evaluation = R.evaluate_component_compile_roles(**transport)
    document = evaluation.verify()
    assert document["compilation_denominator"] == {"required": 2, "linked_and_policy_accepted": 0}
    assert document["static_denominator"] == {"required": 8, "proved": 0, "unknown": 8}
    assert len(document["unresolved"]) == 11
    assert "original-large:original_static_refusal" in document["unresolved"]
    assert all(row["failure"] and row["compilation_status"] == "unavailable" for row in document["members"])
    assert all(Path(row["transport_report"]).is_file() for row in document["members"])
    with pytest.raises(StageGateError, match="compilation/static roles remain incomplete"):
        evaluation.require_complete()


def test_private_transport_evidence_cannot_enter_the_original_numerical_corpus_grant(transport):
    owner = transport["compiler_origin"].inputs.corpus_root / "evaluation"
    transport["evidence_root"] = owner
    with pytest.raises(StageGateError, match="outside all input grants"):
        R.evaluate_component_compile_roles(**transport)
    assert not owner.exists()
