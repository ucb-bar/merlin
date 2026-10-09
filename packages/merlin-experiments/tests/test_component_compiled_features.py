"""Actual stock compile consumption; no ISA, isolation or numeric qualification."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import component_compiled_features as F
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, exact_tree_record, sha256_file
from merlin_experiments.phase2.feedback_protocol import FeedbackValueLimits

from merlin.common import invocation_record as I
from merlin.common.paths import data_path
from merlin.llvmlower import toolchain
from merlin.perf.compiled_static_features import StaticFeatureLimits
from merlin.perf.component_cost import ComponentCostScope
from merlin.perf.component_source_demand import SourceDemandLimits
from merlin.targetgen import component_program
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compile_only_execution import compile_source_only
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def renderer(*_args, **_kwargs):
    raise AssertionError("no numerical harness rendering in static feature controls")


def artifact_gate(*, elf, evidence_root):
    evidence_root.mkdir()
    product = evidence_root / "diagnostic.json"
    result = {"status": "accepted", "elf_sha256": sha256_file(elf), "scope": "structural test; ISA UNKNOWN"}
    product.write_text(json.dumps(result))
    return {**result, "report_path": str(product), "report_sha256": sha256_file(product)}


class OwnedCommands:
    """Execute only this owned fixture, without claiming compiler isolation."""

    def build_package(self, package, *, timeout):
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
        return I.run(
            [sys.executable, "-I", "-B", *argv[1:]],
            directory=invocation_directory,
            stage=name,
            inputs=(source,),
            outputs=(output,) if output is not None else (),
            dependencies=tuple(package.directory.glob("*.py")),
            cwd=package.directory,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=timeout,
        )


WIDE = """%v = llvm.load %a {alignment = 1 : i64} : !llvm.ptr -> i16
llvm.store %v, %b {alignment = 1 : i64} : i16, !llvm.ptr"""
SPLIT = """%a_int = llvm.ptrtoint %a : !llvm.ptr to i64
%b_int = llvm.ptrtoint %b : !llvm.ptr to i64
%one = llvm.mlir.constant(1 : i64) : i64
%a_next = llvm.add %a_int, %one : i64
%b_next = llvm.add %b_int, %one : i64
%ap = llvm.inttoptr %a_next : i64 to !llvm.ptr
%bp = llvm.inttoptr %b_next : i64 to !llvm.ptr
%first = llvm.load %a {alignment = 1 : i64} : !llvm.ptr -> i8
llvm.store %first, %b {alignment = 1 : i64} : i8, !llvm.ptr
%second = llvm.load %ap {alignment = 1 : i64} : !llvm.ptr -> i8
llvm.store %second, %bp {alignment = 1 : i64} : i8, !llvm.ptr"""


@pytest.fixture
def selected(tmp_path, request):
    extent = getattr(request, "param", 2)
    names = ("MERLIN_TEST_CLANG", "MERLIN_TEST_MLIR_TRANSLATE", "MERLIN_TEST_RISCV_GCC")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("requires three explicitly selected stock compile tools")
    tools = tuple(Path(os.environ[name]).resolve(strict=True) for name in names)
    assert Path(toolchain.clang()).resolve() == tools[0] and toolchain.mlir_translate().resolve() == tools[1]
    root = tmp_path / "corpus"
    member_root = root / "capsules" / "copy"
    member_root.mkdir(parents=True)
    program = {
        "inputs": [{"name": "A", "role": "input", "shape": [1, extent], "dtype": "operand"}],
        "nodes": [{"name": "P", "op": "copy", "inputs": ["A"]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }
    typed, source = component_program.render(program, operand_dtype="i8", accumulator_dtype="i32")
    descriptor = {
        "interface_mlir": "source.mlir",
        "inputs": typed["inputs"],
        "component_program": typed,
        "operation": {"op": "component_program", "attributes": {"program": program}},
    }
    (member_root / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    (member_root / "source.mlir").write_text(source)
    tree = exact_tree_record(member_root)
    member = C.PerformanceCapsule(
        "copy", "copy", member_root, "copy", descriptor, tree["sha256"], tree["n_files"], tree["n_bytes"]
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "capsules": [
                    {
                        "family": "copy",
                        "capsule": "copy",
                        "snapshot_sha256": tree["sha256"],
                        "n_files": tree["n_files"],
                        "n_bytes": tree["n_bytes"],
                    }
                ]
            }
        )
    )
    corpus = C.FrozenPerformanceCorpus(
        root,
        root / "capsules",
        manifest,
        sha256_file(manifest),
        exact_tree_record(root / "capsules")["sha256"],
        (member,),
    )
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    script = tmp_path / "link.ld"
    script.write_text(
        "ENTRY(main)\nSECTIONS { . = 0x80000000; .text : { *(.text*) } .data : { *(.data*) } .bss : { *(.bss*) } }\n"
    )

    def pin(p):
        return str(p), sha256_file(p)

    owner = Path(__file__).resolve()
    recipe = HarnessBuildRecipe(
        tools[2],
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
        ldflags=("-Wl,--gc-sections",),
        kernel_stack_frame=KernelStackFramePolicy("feature_entry", 1024),
    )
    build = BuildOnlyService("feature_fixture", recipe, renderer, tuple(pin(p) for p in (owner, tools[2], script)))
    gate = LinkedElfAdmissionService("feature_fixture", artifact_gate, (pin(owner),))
    selection = F.CompileFeatureSelection(
        build,
        gate,
        pin(tools[1]),
        pin(tools[0]),
        pin(Path(shutil.which("readelf")).resolve()),
        (pin(Path(sys.executable).resolve()),),
        64,
        StaticFeatureLimits(8192, 1 << 20, 128),
        SourceDemandLimits(8, 8, 8, 64, 64, 256),
        FeedbackValueLimits(),
        8192,
        64 << 20,
    )
    return SimpleNamespace(
        root=tmp_path,
        member=member,
        corpus=corpus,
        contract=contract,
        selection=selection,
        scope=ComponentCostScope(*(document_sha256(x) for x in ("timer", "accuracy", "input"))),
        extent=extent,
    )


def compile_case(selected, name, operations):
    package = selected.root / (name + "-package")
    package.mkdir()
    fixture = Path(__file__).with_name("compiled_feature_control.py")
    shutil.copyfile(fixture, package / "driver.py")
    commands = ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "feature_fixture",
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
    slots = {
        name: {"shape": [1, selected.extent], "dtype": "i8", "role": role}
        for name, role in (("A", "input"), ("Y", "output"))
    }
    cb = {
        "abi_version": "0.1",
        "target": "feature_fixture",
        "commands": [],
        "tensors": slots,
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
    }
    (package / "buffer.json").write_text(json.dumps(cb))
    (package / "emitted.mlir").write_text(
        "module { llvm.func @feature_entry(%a: !llvm.ptr, %b: !llvm.ptr) {\n" + operations + "\nllvm.return } }\n"
    )
    output = selected.root / (name + "-compiled")
    with P.scoped_package_executor(OwnedCommands()):
        compile_source_only(
            package_dir=package,
            source=selected.member.source_dir / "source.mlir",
            original_abi=CompileOnlySourceAbi(
                (CompileOnlyTensor("A", (1, selected.extent), "i8"),),
                (CompileOnlyTensor("Y", (1, selected.extent), "i8"),),
            ),
            contract_root=selected.contract,
            target="feature_fixture",
            output_root=output,
            build_service=selected.selection.build_service,
            elf_admission=selected.selection.elf_admission,
            readelf=Path(selected.selection.readelf[0]),
            timeout_s=45,
        )
    return package, output


def observe(selected, case):
    package, output = case
    return F.observe_component_compiled_features(
        report_path=output / "compile_only_result.json",
        package_dir=package,
        contract_root=selected.contract,
        corpus=selected.corpus,
        member=selected.member,
        scope=selected.scope,
        selection=selected.selection,
        evidence_root=selected.root / (output.name + "-features"),
    )


def test_actual_compiler_variants_change_emitted_sites_without_executing_values(selected):
    wide, split = compile_case(selected, "wide", WIDE), compile_case(selected, "split", SPLIT)
    a, b = observe(selected, wide), observe(selected, split)
    assert a["source_membership"] == b["source_membership"]
    am, bm = a["features"]["emitted"]["memory"], b["features"]["emitted"]["memory"]
    assert (am["load"]["sites"], bm["load"]["sites"]) == (1, 2)
    assert am["load"]["declared_access_bits"] == bm["load"]["declared_access_bits"] == 16
    assert a["features"]["linked"]["retained_entry"]["size_bytes"] > 0
    assert a["execution"] == b["numeric"] == "not_attempted"
    assert set(a["static_obligations"].values()) == {"UNKNOWN"}
    assert a["authority"] == b["authority"] == "none"


def test_eliminated_ir_site_is_not_an_execution_or_traffic_count(selected):
    ordinary = compile_case(selected, "ordinary", WIDE)
    extra = compile_case(selected, "unused", "%unused = llvm.load %a : !llvm.ptr -> i8\n" + WIDE)
    a, b = observe(selected, ordinary), observe(selected, extra)
    assert a["features"]["emitted"]["memory"]["load"]["sites"] == 1
    assert b["features"]["emitted"]["memory"]["load"]["sites"] == 2
    assert (ordinary[1] / "build/kernel.o").read_bytes() == (extra[1] / "build/kernel.o").read_bytes()


@pytest.mark.parametrize("product", ["generated/lowered.llvm.mlir", "build/kernel.o", "build/package_kernel.elf"])
def test_changed_compilation_products_refuse(selected, product):
    case = compile_case(selected, "changed", WIDE)
    target = case[1] / product
    backup = selected.root / "original-product"
    backup.write_bytes(target.read_bytes())
    target.write_bytes(b"changed product")
    with pytest.raises((ValueError, StageGateError), match="changed|membership|product"):
        observe(selected, case)


def test_resigned_entry_extent_cannot_replace_actual_native_symbol_output(selected):
    case = compile_case(selected, "entry", WIDE)
    report = case[1] / "compile_only_result.json"
    stored = json.loads(report.read_bytes())
    stored["retained_entry"]["size_bytes"] += 1
    report.write_text(json.dumps(stored))
    with pytest.raises(StageGateError, match="native symbol"):
        observe(selected, case)


def test_scanned_alias_refuses_before_reading_excluded_contents(selected):
    case = compile_case(selected, "alias", WIDE)
    excluded = selected.root / "excluded-original"
    excluded.write_text("owned excluded original bytes")
    (case[1] / "linked-invocation.json").symlink_to(excluded)
    with pytest.raises(StageGateError, match="alias"):
        observe(selected, case)
    assert excluded.read_text() == "owned excluded original bytes"


def test_unsupported_actual_emitted_body_cannot_return_an_incomplete_feature_vector(selected):
    case = compile_case(selected, "float", "%v = llvm.load %a : !llvm.ptr -> f32\nllvm.store %v, %b : f32, !llvm.ptr")
    with pytest.raises(ValueError, match="unsupported"):
        observe(selected, case)


def test_unselected_actual_native_tool_refuses(selected):
    case = compile_case(selected, "tool", WIDE)
    changed = Path("/usr/bin/true").resolve()
    selected.selection = replace(selected.selection, translator=(str(changed), sha256_file(changed)))
    with pytest.raises(StageGateError, match="unselected|consumption"):
        observe(selected, case)


@pytest.mark.parametrize("selected", [1 << 38], indirect=True)
def test_huge_original_extent_needs_no_tensor_allocation_and_keeps_coverage_unknown(selected, monkeypatch):
    from merlin.targetgen import capsule_inputs

    monkeypatch.setattr(capsule_inputs, "materialize_capsule_leaves", lambda *_args: pytest.fail("allocated values"))
    result = observe(selected, compile_case(selected, "large", WIDE))
    assert result["static_obligations"]["complete_output_coverage"] == "UNKNOWN"
    assert result["features"]["emitted"]["memory"]["store"]["sites"] == 1


def test_resigned_record_missing_actual_consumed_input_refuses(selected):
    case = compile_case(selected, "missing-consumption", WIDE)
    record = next(p for p in case[1].rglob("invocation.json") if json.loads(p.read_bytes())["stage"] == "object")
    (selected.root / "original-object-record").write_bytes(record.read_bytes())
    changed = json.loads(record.read_bytes())
    changed["inputs"] = []
    record.write_text(json.dumps(changed))
    report = case[1] / "compile_only_result.json"
    stored = json.loads(report.read_bytes())
    digest = sha256_file(record)
    stored["products"][record.relative_to(case[1]).as_posix()]["sha256"] = digest
    for row in stored["invocations"]:
        if row["path"] == str(record):
            row["sha256"] = digest
    report.write_text(json.dumps(stored))
    with pytest.raises(StageGateError, match="consumption"):
        observe(selected, case)


def test_whole_product_budget_is_checked_before_record_parsing(selected, monkeypatch):
    case = compile_case(selected, "budget", WIDE)
    selected.selection = replace(selected.selection, max_tree_bytes=1)
    monkeypatch.setattr(F, "_json", lambda *_args: pytest.fail("parsed before whole budget admission"))
    with pytest.raises(StageGateError, match="byte budget"):
        observe(selected, case)


def test_declared_member_identity_must_match_the_original_complete_source_tree(selected):
    case = compile_case(selected, "member", WIDE)
    selected.member = replace(selected.member, source_sha256=document_sha256("foreign member"))
    selected.corpus = replace(selected.corpus, capsules=(selected.member,))
    with pytest.raises(StageGateError, match="actual source"):
        observe(selected, case)


def test_report_cannot_introduce_an_excluded_elf_before_normal_verification(selected, monkeypatch):
    case = compile_case(selected, "foreign", WIDE)
    excluded = selected.root / "excluded-elf"
    excluded.write_text("owned excluded original bytes")
    report = case[1] / "compile_only_result.json"
    stored = json.loads(report.read_bytes())
    stored["elf"]["path"] = str(excluded)
    report.write_text(json.dumps(stored))
    original_hash = F.sha256_file

    def guarded_hash(path):
        assert Path(path) != excluded, "read an unselected ELF before membership refusal"
        return original_hash(path)

    monkeypatch.setattr(F, "sha256_file", guarded_hash)
    with pytest.raises(StageGateError, match="foreign linked"):
        observe(selected, case)
    assert excluded.read_text() == "owned excluded original bytes"
