"""Actual stock-tool structural transport, without a compiler correctness claim.

The fixture package deliberately emits a pointer entry with no output stores.
Its four normal commands exercise package dispatch, translation, object/link
generation and reopening. It is neither a Phase 1 compiler seed nor a semantic
checker. The diagnostic artifact gate is not an independently qualified ISA
authority; the experimental caller must supply that separate live owner.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.common.paths import data_path
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compile_only_execution import compile_source_only, verify_compile_only_report
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor, prepare_linkage
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def _unused_renderer(*_args, **_kwargs):
    raise AssertionError("compile-only attempted numerical harness rendering")


def _private_build_only_renderer(_cb, *, inputs, readback_policy):
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64

    # This diagnostic never executes; it tests exact full-value BUILD bindings.
    assert inputs == {"input": [0]} and readback_policy.transport == FULL_VALUES_B64
    return (
        "extern void fixture_entry(void *, void *);\n"
        "int main(void) { long input = 0; char output = 0; fixture_entry(&input, &output); return 0; }\n"
    )


def _diagnostic_artifact_gate(*, elf, evidence_root):
    evidence_root.mkdir()
    report = evidence_root / "diagnostic.json"
    result = {"status": "accepted", "elf_sha256": file_digest(elf), "scope": "transport test, no ISA authority"}
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": file_digest(report)}


def _diagnostic_refusal(*, elf, evidence_root):
    result = _diagnostic_artifact_gate(elf=elf, evidence_root=evidence_root)
    report = Path(result["report_path"])
    stored = json.loads(report.read_text())
    stored["status"] = "refused"
    report.write_text(json.dumps(stored))
    return {**result, "status": "refused", "report_sha256": file_digest(report)}


class _ActualCommandExecutor:
    """Private test process owner; source snapshots are bound by the real transport."""

    def build_package(self, package, *, timeout=1800):
        assert not package.manifest.get("build")

    def run_entrypoint(
        self,
        package,
        name,
        source,
        output=None,
        *,
        timeout,
        invocation_directory,
        write_bytecode=False,
        artifact_profile=None,
    ):
        assert not write_bytecode and artifact_profile is None
        argv = P._resolve_argv(package, name, source, output)
        assert argv[0] == "python3"
        return invocation_record.run(
            [sys.executable, "-I", "-B", *argv[1:]],
            directory=invocation_directory,
            stage=name,
            inputs=(Path(source),),
            outputs=(Path(output),) if output is not None else (),
            dependencies=tuple(package.directory.rglob("*.py")),
            cwd=package.directory,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def _buffer(shape=(1 << 38,)):
    return {
        "abi_version": "0.1",
        "target": "transport_fixture",
        "commands": [],
        "tensors": {
            name: {"shape": list(shape), "dtype": "f32", "role": role}
            for name, role in (("input", "input"), ("output", "output"))
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "input", "access": "read"}, {"tensor": "output", "access": "write"}],
            "outputs": ["output"],
        },
    }


def _original_abi(shape=(1 << 38,)):
    return CompileOnlySourceAbi(
        (CompileOnlyTensor("input", shape, "f32"),), (CompileOnlyTensor("output", shape, "f32"),)
    )


_LLVM = "module { llvm.func @fixture_entry(%a: !llvm.ptr, %b: !llvm.ptr) { llvm.return } }\n"


@pytest.fixture
def actual_transport(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    compiler = os.environ.get("MERLIN_TEST_RISCV_GCC")
    if not compiler:
        pytest.skip("requires explicitly selected stock RISC-V linker/compiler")
    if not toolchain.mlir_translate().is_file() or not Path(toolchain.clang()).is_file():
        pytest.skip("requires selected stock LLVM translator and object compiler")
    monkeypatch.setattr(base, "get_backend", lambda *_args: pytest.fail("compile-only discovered a backend"))
    package = tmp_path / "candidate"
    package.mkdir()
    commands = ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "transport_fixture",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "driver.py"},
        "commands": {
            name: {
                "argv": [
                    "python3",
                    "driver.py",
                    name,
                    "{input_mlir}",
                    *(["{output_json}"] if name == "emit_command_buffer" else []),
                ]
            }
            for name in commands
        },
    }
    (package / "manifest.yaml").write_text(json.dumps(manifest))
    (package / "buffer.json").write_text(json.dumps(_buffer()))
    (package / "emitted.mlir").write_text(_LLVM)
    (package / "driver.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "command, source, *rest = sys.argv[1:]\ntext = Path(source).read_text()\n"
        "if command == 'lower_interface_to_target': sys.stdout.write(text)\n"
        "elif command == 'emit_command_buffer': Path(rest[0]).write_bytes(Path('buffer.json').read_bytes())\n"
        "elif command == 'lower_target_to_llvm': sys.stdout.write(Path('emitted.mlir').read_text())\n"
        "elif command != 'parse': raise ValueError(command)\n"
    )
    source = tmp_path / "original" / "source.mlir"
    source.parent.mkdir()
    source.write_text(
        "module { func.func @source(%a: tensor<274877906944xf32>) -> tensor<274877906944xf32> {\n"
        "func.return %a : tensor<274877906944xf32> } }\n"
    )
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    script = sdk / "link.ld"
    script.write_text(
        "ENTRY(main)\nSECTIONS { . = 0x80000000; .text : { *(.text*) } .data : { *(.data*) } .bss : { *(.bss*) } }\n"
    )
    selected_compiler = Path(compiler).resolve(strict=True)
    owner = Path(__file__).resolve()
    pins = tuple((str(path), file_digest(path)) for path in (owner, selected_compiler, script))
    recipe = HarnessBuildRecipe(
        selected_compiler,
        (),
        (),
        script,
        0x80000000,
        (
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-nostdlib",
        ),
        kernel_stack_frame=KernelStackFramePolicy("fixture_entry", 1024),
        ldflags=("-Wl,--gc-sections",),
    )
    build = BuildOnlyService("transport_fixture", recipe, _unused_renderer, pins)
    gate = LinkedElfAdmissionService(
        "transport_fixture", _diagnostic_artifact_gate, ((str(owner), file_digest(owner)),)
    )
    arguments = dict(
        package_dir=package,
        source=source,
        original_abi=_original_abi(),
        contract_root=contract,
        target="transport_fixture",
        output_root=tmp_path / "observation",
        build_service=build,
        elf_admission=gate,
        readelf=Path(shutil.which("readelf")).resolve(strict=True),
        timeout_s=60,
    )
    return arguments, build, gate


def _reviewed_library_candidate(arguments, tmp_path, *, imported="portable"):
    root = tmp_path / "reviewed-library"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def identity(value):\n    return value\n")
    library = freeze_compiler_library(
        root,
        review_id="explicit diagnostic source review; no semantic authority",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    driver = arguments["package_dir"] / "driver.py"
    driver.write_text(
        "import sys\n"
        + f"sys.path.insert(0, {str(root)!r})\n"
        + f"from merlin.{imported} import identity\n"
        + driver.read_text().replace("text = Path(source).read_text()", "text = identity(Path(source).read_text())")
    )
    return library, root


def test_real_ordinary_compile_uses_only_explicit_reviewed_library(actual_transport, tmp_path):
    arguments, build, gate = actual_transport
    library, root = _reviewed_library_candidate(arguments, tmp_path)
    with P.scoped_package_executor(_ActualCommandExecutor()):
        report = compile_source_only(**arguments, compiler_library=library, compiler_library_root=root)
    reopened = verify_compile_only_report(
        arguments["output_root"] / "compile_only_result.json",
        build_service=build,
        elf_admission=gate,
        compiler_library=library,
        compiler_library_root=root,
    )
    assert reopened == json.loads(json.dumps(report)) and report["compilation_status"] == "linked"
    assert report["inputs"]["compiler_library"]["contract_sha256"] == library.sha256
    lowering = [
        invocation_record.verify(Path(row["path"]))
        for row in report["invocations"]
        if row["stage"] == "compile_only_source_lowering"
    ]
    assert len(lowering) == 1
    assert {str(root / member.path) for member in library.members} <= {
        pin["path"] for pin in lowering[0]["dependencies"]
    }
    with pytest.raises(ValueError, match="explicit compiler library"):
        verify_compile_only_report(
            arguments["output_root"] / "compile_only_result.json", build_service=build, elf_admission=gate
        )


def test_completed_library_report_refuses_later_member_mutation(actual_transport, tmp_path):
    arguments, build, gate = actual_transport
    library, root = _reviewed_library_candidate(arguments, tmp_path)
    with P.scoped_package_executor(_ActualCommandExecutor()):
        compile_source_only(**arguments, compiler_library=library, compiler_library_root=root)
    (root / "merlin/portable.py").write_text("def identity(value):\n    return 0\n")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_compile_only_report(
            arguments["output_root"] / "compile_only_result.json",
            build_service=build,
            elf_admission=gate,
            compiler_library=library,
            compiler_library_root=root,
        )


@pytest.mark.parametrize("defect", ["unselected", "sibling", "changed"])
def test_real_library_refusal_precedes_any_ordinary_compiler_command(actual_transport, tmp_path, defect):
    arguments, _, _ = actual_transport
    library, root = _reviewed_library_candidate(
        arguments, tmp_path, imported="sibling" if defect == "sibling" else "portable"
    )
    if defect == "changed":
        (root / "merlin/portable.py").write_text("def identity(value):\n    return 0\n")
    selected = {} if defect == "unselected" else {"compiler_library": library, "compiler_library_root": root}
    with P.scoped_package_executor(_ActualCommandExecutor()):
        with pytest.raises((P.CertFailure, ValueError), match="imports|bytes changed"):
            compile_source_only(**arguments, **selected)
    assert not list(arguments["output_root"].rglob("invocation.json"))


def test_library_change_during_actual_lowering_refuses_before_object(actual_transport, tmp_path):
    arguments, _, _ = actual_transport
    library, root = _reviewed_library_candidate(arguments, tmp_path)

    class MutatingCommand(_ActualCommandExecutor):
        def run_entrypoint(self, *args, **kwargs):
            result = super().run_entrypoint(*args, **kwargs)
            (root / "merlin/portable.py").write_text("def identity(value):\n    return 0\n")
            return result

    with P.scoped_package_executor(MutatingCommand()):
        with pytest.raises(ValueError, match="bytes changed"):
            compile_source_only(**arguments, compiler_library=library, compiler_library_root=root)
    assert not (arguments["output_root"] / "build").exists()
    records = [json.loads(path.read_text()) for path in arguments["output_root"].rglob("invocation.json")]
    assert any(row["stage"] == "parse" for row in records)
    assert not any(row["stage"] in {"object", "elf"} for row in records)


def _compile(arguments):
    with P.scoped_package_executor(_ActualCommandExecutor()):
        return compile_source_only(**arguments)


def test_stock_translation_object_link_keep_large_abi_without_values(actual_transport):
    arguments, build, gate = actual_transport
    result = _compile(arguments)
    assert result["compilation_status"] == "linked"
    assert result["execution"] == result["numeric"] == "not_attempted"
    assert set(result["static_obligations"].values()) == {"UNKNOWN"}
    assert result["retained_entry"]["size_bytes"] > 0
    harness = (arguments["output_root"] / "build" / "harness.c").read_text()
    assert "(void *)0" in harness and "274877906944" not in harness
    assert "OUT_" not in harness and "golden" not in harness
    stages = {row["stage"] for row in result["invocations"]}
    assert {
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
        "llvm_translation",
        "object",
        "harness_object",
        "elf",
        "compile_only_linked_entry",
    } <= stages
    assert "functional_engine" not in stages
    report = arguments["output_root"] / "compile_only_result.json"
    assert verify_compile_only_report(report, build_service=build, elf_admission=gate)["numeric"] == "not_attempted"


@pytest.mark.parametrize(
    "mutation",
    ["source", "package_member", "contract_member", "product", "invocation", "instruction_selection", "recipe"],
)
def test_actual_compile_records_reopen_sources_membership_products_and_selections(actual_transport, mutation):
    arguments, build, gate = actual_transport
    _compile(arguments)
    if mutation == "source":
        arguments["source"].write_text("module {}")
    elif mutation == "package_member":
        (arguments["package_dir"] / "added.py").write_text("# changed actual member roster\n")
    elif mutation == "contract_member":
        (arguments["contract_root"] / "added.json").write_text("{}")
    elif mutation == "product":
        (arguments["output_root"] / "build" / "kernel.o").write_bytes(b"changed actual object")
    elif mutation == "invocation":
        next(arguments["output_root"].rglob("invocation.json")).unlink()
    elif mutation == "instruction_selection":
        gate = replace(gate, source_pins=build.source_pins)
    else:
        build = replace(build, recipe=replace(build.recipe, load_address=0x80001000))
    with pytest.raises(ValueError, match="changed|differs"):
        verify_compile_only_report(
            arguments["output_root"] / "compile_only_result.json", build_service=build, elf_admission=gate
        )


def test_source_abi_mismatch_refuses_before_object_or_link(actual_transport):
    arguments, _, _ = actual_transport
    arguments["original_abi"] = _original_abi((7,))
    with pytest.raises(ValueError, match="shape or dtype"):
        _compile(arguments)
    result = json.loads((arguments["output_root"] / "compile_only_result.json").read_text())
    assert result["compilation_status"] == "unavailable" and result["execution"] == "not_attempted"
    assert not (arguments["output_root"] / "build").exists()


def test_linked_policy_refusal_retains_compilation_without_execution(actual_transport):
    arguments, build, gate = actual_transport
    gate = replace(gate, evaluator=_diagnostic_refusal)
    arguments["elf_admission"] = gate
    result = _compile(arguments)
    assert result["compilation_status"] == "refused_by_instruction_policy"
    assert result["execution"] == result["numeric"] == "not_attempted"
    assert result["elf"]["sha256"] == result["instruction_policy"]["elf_sha256"]
    with pytest.raises(ValueError, match="no complete linked observation"):
        verify_compile_only_report(
            arguments["output_root"] / "compile_only_result.json", build_service=build, elf_admission=gate
        )


def test_saved_link_report_cannot_gain_a_static_correctness_claim(actual_transport):
    arguments, build, gate = actual_transport
    result = _compile(arguments)
    result["static_obligations"]["semantic_coverage"] = "PASS"
    report = arguments["output_root"] / "compile_only_result.json"
    report.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="cannot qualify a static semantic"):
        verify_compile_only_report(report, build_service=build, elf_admission=gate)


def test_missing_scoped_executor_refuses_before_candidate_or_products(actual_transport):
    arguments, _, _ = actual_transport
    with pytest.raises(ValueError, match="ordinary scoped package executor"):
        compile_source_only(**arguments)
    assert not arguments["output_root"].exists()


def test_real_declared_package_build_obeys_total_transport_budget(actual_transport):
    arguments, _, _ = actual_transport
    package, output = arguments["package_dir"], arguments["output_root"]
    script = package / "slow_build.py"
    script.write_text(
        "from pathlib import Path\nimport sys, time\n"
        "owner = Path(sys.argv[1])\n(owner / 'started').write_text('started')\n"
        "time.sleep(3)\n(owner / 'finished').write_text('finished')\n"
    )
    manifest = package / "manifest.yaml"
    declared = json.loads(manifest.read_text())
    declared["build"] = {"command": [sys.executable, script.name], "tool_output": "driver.py"}
    manifest.write_text(json.dumps(declared))
    arguments["timeout_s"] = 1

    class TimedBuild(_ActualCommandExecutor):
        def build_package(self, package, *, timeout=1800):
            process = invocation_record.run(
                [*package.manifest["build"]["command"], str(output)],
                directory=output,
                stage="declared_package_build",
                cwd=package.directory,
                dependencies=(script,),
                outputs=(output / "started", output / "finished"),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            process.check_returncode()

    started = time.monotonic()
    with P.scoped_package_executor(TimedBuild()), pytest.raises(subprocess.TimeoutExpired):
        compile_source_only(**arguments)
    assert time.monotonic() - started < 2.8
    assert (output / "started").read_text() == "started"
    assert not (output / "finished").exists() and not (output / "generated").exists()
    report = json.loads((output / "compile_only_result.json").read_text())
    assert report["compilation_status"] == "unavailable"
    assert report["failure"]["type"] == "TimeoutExpired"
    assert report["execution"] == report["numeric"] == "not_attempted"
    assert len(report["retained_unqualified_records"]) == 1


def _oversized_llvm():
    return (
        "module {\n"
        "llvm.func @fixture_entry(%a: !llvm.ptr, %b: !llvm.ptr) {\n"
        "%count = llvm.mlir.constant(8192 : i64) : i64\n"
        "%buffer = llvm.alloca %count x i8 {alignment = 64 : i64} : (i64) -> !llvm.ptr\n"
        "%index = llvm.load %a : !llvm.ptr -> i64\n"
        "%address = llvm.getelementptr %buffer[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n"
        "%value = llvm.load %b : !llvm.ptr -> i8\n"
        "llvm.store %value, %address : i8, !llvm.ptr\n"
        "%last = llvm.getelementptr %buffer[8191] : (!llvm.ptr) -> !llvm.ptr, i8\n"
        "%result = llvm.load %last : !llvm.ptr -> i8\n"
        "llvm.store %result, %b : i8, !llvm.ptr\n"
        "llvm.return\n} }\n"
    )


def test_actual_stack_repair_links_revised_object_and_reopens_original_observations(actual_transport):
    arguments, build, gate = actual_transport
    (arguments["package_dir"] / "emitted.mlir").write_text(_oversized_llvm())
    _compile(arguments)
    work = arguments["output_root"] / "build"
    receipt = json.loads((work / "kernel.stack_frame.json").read_text())
    assert receipt["repair"]["transform"] == "stack_arena_bind"
    assert receipt["repair"]["frame_bytes_before"] > 1024 >= receipt["frame_bytes"]
    assert (work / "kernel.ll").is_file() and (work / "kernel.o").is_file()
    assert (work / "kernel.arena.ll").is_file() and (work / "kernel.arena.o").is_file()
    assert receipt["llvm_ir_sha256"] == file_digest(work / "kernel.arena.ll")
    assert receipt["object_sha256"] == file_digest(work / "kernel.arena.o")
    rows = [invocation_record.verify(path) for path in arguments["output_root"].rglob("invocation.json")]
    link = next(row for row in rows if row["stage"] == "elf")
    assert str(work / "kernel.arena.o") in link["argv"] and str(work / "kernel.o") not in link["argv"]
    verify_compile_only_report(
        arguments["output_root"] / "compile_only_result.json", build_service=build, elf_admission=gate
    )


def test_full_value_receipt_reopens_the_object_actually_selected_after_stack_repair(actual_transport):
    from merlin.targetgen.contract import compile as compiler
    from merlin.targetgen.contract import readback_policy as RB

    arguments, build, _ = actual_transport
    build = replace(build, renderer=_private_build_only_renderer)
    work = arguments["output_root"].parent / "full_value_build"
    obj = compiler.llvm_mlir_to_object(
        _oversized_llvm(), work, target=build.target, _build_service=build, build_timeout_s=30
    )
    assert obj.name == "kernel.arena.o"
    policy = RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    cb = _buffer((1,))
    elf = compiler.link_elf(
        cb,
        obj,
        work,
        target=build.target,
        inputs={"input": [0]},
        _build_service=build,
        readback_policy=policy,
        build_timeout_s=30,
    )
    recipe, pins = RB.selected_build_inputs(build.target, build.recipe.with_effective_abi(), build, policy=policy)
    exact = RB.require_build_receipt(
        work / RB.BUILD_RECEIPT,
        policy=policy,
        cb=cb,
        target=build.target,
        recipe_record=recipe,
        source_pins=pins,
        object_path=obj,
        harness_path=work / "harness.c",
        elf_path=elf,
    )
    current = RB.require_current_build_receipt(
        cb=cb, target=build.target, workdir=work, elf_path=elf, policy=policy, build_service=build
    )
    assert current == exact and current["kernel_object_sha256"] == file_digest(obj)
    for path in work.rglob("invocation.json"):
        invocation_record.verify(path)
    for name, reason in (("../outside.o", "unsafe"), ("kernel.o", "does not bind")):
        selected = {key: value for key, value in current.items() if key != "build_identity_sha256"}
        selected["kernel_object_name"] = name
        selected["build_identity_sha256"] = RB.canonical_sha256(selected)
        (work / RB.BUILD_RECEIPT).write_text(json.dumps(selected))
        with pytest.raises(ValueError, match=reason):
            RB.require_current_build_receipt(
                cb=cb, target=build.target, workdir=work, elf_path=elf, policy=policy, build_service=build
            )


def test_report_cannot_qualify_a_transport_without_normal_execution_owner(tmp_path):
    with pytest.raises(ValueError, match="exact bounded"):
        compile_source_only(
            package_dir=tmp_path,
            source=tmp_path,
            original_abi=_original_abi(),
            contract_root=tmp_path,
            target="fixture",
            output_root=tmp_path,
            build_service=None,
            elf_admission=None,
            readelf=tmp_path,
            timeout_s=1,
        )


def test_legacy_unnamed_readback_receipt_keeps_original_object_convention(actual_transport):
    from merlin.targetgen.contract import compile as compiler
    from merlin.targetgen.contract import readback_policy as RB

    arguments, build, _ = actual_transport
    build = replace(build, renderer=_private_build_only_renderer)
    work = arguments["output_root"].parent / "legacy_build"
    obj = compiler.llvm_mlir_to_object(_LLVM, work, target=build.target, _build_service=build, build_timeout_s=30)
    assert obj.name == "kernel.o"
    cb, policy = _buffer((1,)), RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    elf = compiler.link_elf(
        cb,
        obj,
        work,
        target=build.target,
        inputs={"input": [0]},
        _build_service=build,
        readback_policy=policy,
        build_timeout_s=30,
    )
    path = work / RB.BUILD_RECEIPT
    legacy = json.loads(path.read_text())
    legacy.pop("kernel_object_name")
    legacy.pop("build_identity_sha256")
    legacy["build_identity_sha256"] = RB.canonical_sha256(legacy)
    path.write_text(json.dumps(legacy))
    assert (
        RB.require_current_build_receipt(
            cb=cb, target=build.target, workdir=work, elf_path=elf, policy=policy, build_service=build
        )
        == legacy
    )


@pytest.mark.parametrize("defect", ["missing_output", "extra_output", "pointer_arity", "nonpointer", "entry"])
def test_static_roster_and_entry_defects_are_refused(defect):
    cb, llvm, entry = _buffer(), _LLVM, "fixture_entry"
    if defect == "missing_output":
        cb["kernel_abi"]["outputs"] = []
    elif defect == "extra_output":
        cb["kernel_abi"]["outputs"].append("extra")
    elif defect == "pointer_arity":
        llvm = llvm.replace(", %b: !llvm.ptr", "")
    elif defect == "nonpointer":
        llvm = llvm.replace("%b: !llvm.ptr", "%b: i64")
    else:
        entry = "missing"
    with pytest.raises(ValueError, match="original|entry|ABI"):
        prepare_linkage(cb=cb, lowered_mlir=llvm, entry_symbol=entry, original_abi=_original_abi())
