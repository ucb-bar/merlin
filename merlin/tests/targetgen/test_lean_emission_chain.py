"""Actual source/emission transport controls, with native boundaries stubbed.

The selected class checker deliberately returns unsupported. These tests make
no mathematical, numerical, linked-ISA or hardware qualification claim.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

from merlin.common import invocation_record
from merlin.common.paths import data_path
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P
from merlin.verify import lean_emission_chain as E

SOURCE = (
    "module { func.func @source(%a: tensor<65537xf32>) -> tensor<65537xf32> { func.return %a : tensor<65537xf32> } }\n"
)
FINAL = "module { llvm.func @fixture_entry(%a: !llvm.ptr, %b: !llvm.ptr) { llvm.return } }\n"


class _Executor:
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
        return invocation_record.run(
            [sys.executable, "-I", "-B", *argv[1:]],
            directory=invocation_directory,
            stage=name,
            inputs=(Path(source),),
            outputs=(() if output is None else (Path(output),)),
            dependencies=tuple(package.directory.rglob("*.py")),
            cwd=package.directory,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@pytest.fixture
def emission(tmp_path, monkeypatch):
    package_dir = tmp_path / "candidate"
    package_dir.mkdir()
    commands = (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "lower_target_to_llvm",
        "emit_transformation_trace",
    )
    (package_dir / "manifest.yaml").write_text(
        json.dumps(
            {
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
                            *(
                                ["{output_json}"]
                                if name in {"emit_command_buffer", "emit_transformation_trace"}
                                else []
                            ),
                        ]
                    }
                    for name in commands
                },
            }
        )
    )
    (package_dir / "buffer.json").write_text(
        json.dumps(
            {
                "abi_version": "0.1",
                "target": "transport_fixture",
                "commands": [],
                "tensors": {
                    name: {"shape": [65537], "dtype": "f32", "role": role}
                    for name, role in (("input", "input"), ("output", "output"))
                },
                "kernel_abi": {
                    "kind": "whole_program",
                    "args": [{"tensor": "input", "access": "read"}, {"tensor": "output", "access": "write"}],
                    "outputs": ["output"],
                },
            }
        )
    )
    (package_dir / "emitted.mlir").write_text(FINAL)
    changed = SOURCE.replace("%a", "%renamed")
    trace = {
        "schema": E.SCHEMA,
        "checkpoints": [
            {"ir": SOURCE, "entry": "source"},
            {"ir": changed, "entry": "source"},
            {"ir": FINAL, "entry": "fixture_entry"},
        ],
    }
    (package_dir / "trace.json").write_text(json.dumps(trace))
    (package_dir / "driver.py").write_text(
        "from pathlib import Path\nimport sys\ncommand, source, *rest = sys.argv[1:]\n"
        "if command == 'lower_interface_to_target': sys.stdout.write(Path(source).read_text())\n"
        "elif command == 'emit_command_buffer': Path(rest[0]).write_bytes(Path('buffer.json').read_bytes())\n"
        "elif command == 'lower_target_to_llvm': sys.stdout.write(Path('emitted.mlir').read_text())\n"
        "elif command == 'emit_transformation_trace': Path(rest[0]).write_bytes(Path('trace.json').read_bytes())\n"
        "elif command != 'parse': raise ValueError(command)\n"
    )
    original = tmp_path / "original" / "source.mlir"
    original.parent.mkdir()
    original.write_text(SOURCE)
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    authority = tmp_path / "authority" / "bundle.json"
    authority.parent.mkdir()
    authority.write_text('{"immutable":true}\n')
    service = object.__new__(CommonTransformService)
    service.bundle_manifest, service.bundle_bytes = authority, authority.read_bytes()
    service.snapshot_digest = "original-snapshot"
    service.profile = {"sha256": "original-profile", "contents": ["original-host-policy"]}
    calls = []

    def verify(self):
        assert self.bundle_manifest.read_bytes() == self.bundle_bytes

    def check(self, request, *, work_root, timeout_s):
        calls.append(request)
        return {"status": "unsupported", "scope": "transport control only"}

    monkeypatch.setattr(CommonTransformService, "_verify", verify)
    monkeypatch.setattr(CommonTransformService, "check", check, raising=False)
    selection = E.EmissionChainSelection(service)
    package = P.load_package(package_dir, contract=contract)
    return dict(
        package=package,
        original_source=original,
        contract=contract,
        selection=selection,
        trace=trace,
        calls=calls,
        tmp=tmp_path,
    )


def _emit(fixture):
    generated = fixture["tmp"] / "generated"
    with P.scoped_package_executor(_Executor()):
        with invocation_record.observe_call(
            fixture["tmp"] / "observations",
            stage="actual_lowering",
            function=CC.lower_interface,
            arguments={},
            inputs=(fixture["original_source"],),
            outputs=(generated / "input.interface.mlir", generated / "lowered.llvm.mlir"),
            dependencies=tuple(fixture["package"].directory.rglob("*.*")),
        ) as observed:
            CC.lower_interface(
                fixture["package"], fixture["original_source"], generated, contract=fixture["contract"], timeout=30
            )
            observed.returned()
    return observed, generated


def _check(fixture):
    observed, generated = _emit(fixture)
    with P.scoped_package_executor(_Executor()):
        return E.check_emission_chain(
            selection=fixture["selection"],
            package=fixture["package"],
            original_source=fixture["original_source"],
            compiler_input=generated / "input.interface.mlir",
            emitted_artifact=generated / "lowered.llvm.mlir",
            emission_invocation=observed,
            target_entry="fixture_entry",
            evidence_root=fixture["tmp"] / "proof",
            remaining=lambda: 30,
        )


def _consume(fixture, result, monkeypatch, *, text=None):
    from merlin.llvmlower import codegen, pipeline
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

    # Real ordinary LLVM producer, only its native translator/compiler edges stubbed.
    monkeypatch.setattr(
        pipeline, "lower_to_llvm_ir", lambda *_args, **_kwargs: "define void @fixture_entry() { ret void }\n"
    )

    def compile_ll(source, output, *_args, **_kwargs):
        output.write_bytes(b"diagnostic object; not executable or qualified")
        return output

    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    artifact = E.reread_emitted_product(result, selection=fixture["selection"])
    build = fixture["tmp"] / "build"
    with invocation_record.observe_call(
        fixture["tmp"] / "consumer",
        stage="actual_object",
        function=llvm_mlir_to_object,
        arguments={},
        inputs=(Path(result.record()["emitted_artifact"]["path"]),),
        outputs=(build / "kernel.llvm.mlir", build / "kernel.o"),
    ) as observed:
        obj = llvm_mlir_to_object(artifact if text is None else text, build)
        observed.returned()
    return E.consume_emission_chain(
        result,
        selection=fixture["selection"],
        object_invocation=observed,
        object_path=obj,
        consumer_source=build / "kernel.llvm.mlir",
    )


def test_actual_lowering_fresh_check_and_actual_object_custody(emission, monkeypatch):
    result = _check(emission)
    assert result.record()["status"] == "partial"
    assert len(emission["calls"]) == 1
    assert emission["calls"][0]["before_ir"] == SOURCE
    report = _consume(emission, result, monkeypatch)
    assert report["emitted_artifact"]["sha256"] == report["actual_consumer_source"]["sha256"]
    assert id(result) not in E._LIVE
    assert (
        E.reopen_emission_custody(
            report,
            selection=emission["selection"],
            package_root=emission["package"].directory,
            evidence_root=emission["tmp"] / "proof",
        )["live_formal_acceptance"]
        == "not_issued_by_saved_transport_report"
    )


def test_saved_completion_cannot_recreate_live_authority(emission):
    result = _check(emission)
    try:
        with pytest.raises(ValueError, match="current original live"):
            E.reread_emitted_product(E.EmissionChainResult(result.report_json), selection=emission["selection"])
    finally:
        E.discard_emission_chain(result)


def test_actual_consumer_different_bytes_refuses(emission, monkeypatch):
    result = _check(emission)
    try:
        with pytest.raises(ValueError, match="different IR bytes"):
            _consume(emission, result, monkeypatch, text=FINAL + "\n")
    finally:
        E.discard_emission_chain(result)


@pytest.mark.parametrize(
    "member",
    ["source", "compiler_input", "emitted_artifact", "trace", "compiler_source_invocation", "trace_invocation"],
)
def test_changed_actual_chain_member_refuses(emission, member):
    result = _check(emission)
    try:
        path = Path(result.record()[member]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="changed"):
            E.reread_emitted_product(result, selection=emission["selection"])
    finally:
        E.discard_emission_chain(result)


@pytest.mark.parametrize("mutation", ["first", "last", "entry", "status", "duplicates"])
def test_false_or_assertion_carrying_trace_refuses(emission, mutation):
    trace = emission["trace"]
    if mutation in {"first", "last"}:
        trace["checkpoints"][0 if mutation == "first" else -1]["ir"] += "\n"
    elif mutation == "entry":
        trace["checkpoints"][1]["entry"] = "unrelated"
    elif mutation == "status":
        trace["status"] = "verified"
    raw = json.dumps(trace)
    if mutation == "duplicates":
        raw = raw.replace('"schema":', '"schema": "wrong", "schema":', 1)
    (emission["package"].directory / "trace.json").write_text(raw)
    with pytest.raises(ValueError):
        _check(emission)


@pytest.mark.parametrize("status", ["refuted", "error", "accepted", "unknown"])
def test_covered_live_checker_failure_hard_refuses(emission, monkeypatch, status):
    monkeypatch.setattr(CommonTransformService, "check", lambda *_args, **_kwargs: {"status": status})
    with pytest.raises(ValueError, match="fresh original live proof"):
        _check(emission)


def test_original_selection_mutation_and_instance_callbacks_refuse(emission):
    service = emission["selection"].service
    service.profile["contents"].append("different-policy-same-declared-hash")
    with pytest.raises(ValueError, match="authority changed"):
        emission["selection"].verify(
            package_root=emission["package"].directory, evidence_root=emission["tmp"] / "proof"
        )
    service.check = lambda *_args: {"status": "verified"}
    with pytest.raises(ValueError, match="instance callbacks"):
        E.EmissionChainSelection(service)


def test_missing_trace_is_explicit_uncovered_gap(emission):
    del emission["package"].manifest["commands"]["emit_transformation_trace"]
    result = _check(emission)
    assert result.record()["status"] == "not_declared"
    assert result.record()["outside_formal_coverage"] == "independent_checks_required"
    assert id(result) not in E._LIVE and not emission["calls"]


def test_original_bounded_trace_budget_refuses(emission):
    emission["selection"] = E.EmissionChainSelection(emission["selection"].service, maximum_trace_bytes=64)
    with pytest.raises(ValueError, match="byte budget"):
        _check(emission)


def _unused_renderer(*_args, **_kwargs):
    raise AssertionError("compile-only must not render numerical data")


def _diagnostic_gate(*, elf, evidence_root):
    from merlin.targetgen.contract.build_service import file_digest

    evidence_root.mkdir()
    path = evidence_root / "diagnostic.json"
    report = {"status": "accepted", "elf_sha256": file_digest(elf), "scope": "native-boundary stub; no ISA claim"}
    path.write_text(json.dumps(report))
    return {**report, "report_path": str(path), "report_sha256": file_digest(path)}


def test_connected_compile_source_only_and_reopen_same_host_selection(emission, monkeypatch):
    from merlin.llvmlower import codegen, toolchain
    from merlin.targetgen import compile_only_execution as C
    from merlin.targetgen.contract import compile as producer
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
    from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
    from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    owner = Path(__file__).resolve()
    compiler = Path(sys.executable).resolve()
    script = emission["tmp"] / "link.ld"
    script.write_text("diagnostic script; native boundary stub")
    recipe = HarnessBuildRecipe(
        compiler,
        (),
        (),
        script,
        0x80000000,
        ("-march=rv64gc", "-mabi=lp64d"),
        kernel_stack_frame=KernelStackFramePolicy("fixture_entry", 1024),
    )
    pins = tuple((str(path), file_digest(path)) for path in (owner, compiler, script))
    build = BuildOnlyService("transport_fixture", recipe, _unused_renderer, pins)
    gate = LinkedElfAdmissionService("transport_fixture", _diagnostic_gate, ((str(owner), file_digest(owner)),))

    def native_boundary(argv, **kwargs):
        for path in kwargs.get("outputs", ()):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".ll":
                path.write_text("define void @fixture_entry(ptr %a, ptr %b) { ret void }\n")
            else:
                path.write_bytes(b"diagnostic native boundary product; no executable claim")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def compile_ll(source, output, *_args, **_kwargs):
        output.write_bytes(b"diagnostic object")
        output.with_suffix(".su").write_text("kernel.ll:fixture_entry\t0\tstatic\n")
        return output

    monkeypatch.setattr(producer, "_observed_run", native_boundary)
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: compiler)
    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    monkeypatch.setattr(
        C, "_retained_entry", lambda **_kwargs: {"symbol": "fixture_entry", "scope": "native boundary stub"}
    )
    tensor = CompileOnlyTensor("input", (65537,), "f32")
    arguments = dict(
        package_dir=emission["package"].directory,
        source=emission["original_source"],
        original_abi=CompileOnlySourceAbi((tensor,), (CompileOnlyTensor("output", (65537,), "f32"),)),
        contract_root=emission["contract"],
        target="transport_fixture",
        output_root=emission["tmp"] / "complete",
        build_service=build,
        elf_admission=gate,
        readelf=compiler,
        timeout_s=60,
        emission_verification=emission["selection"],
    )
    with P.scoped_package_executor(_Executor()):
        report = C.compile_source_only(**arguments)
    assert report["compilation_status"] == "linked"
    assert report["emission_proof_chain"]["status"] == "partial"
    assert set(report["static_obligations"].values()) == {"UNKNOWN"}
    assert len(emission["calls"]) == 1
    path = arguments["output_root"] / "compile_only_result.json"
    assert (
        C.verify_compile_only_report(
            path, build_service=build, elf_admission=gate, emission_verification=emission["selection"]
        )["numeric"]
        == "not_attempted"
    )
    assert len(emission["calls"]) == 1  # saved transport does not issue new proof authority
    with pytest.raises(ValueError, match="same original host binding"):
        C.verify_compile_only_report(path, build_service=build, elf_admission=gate)
