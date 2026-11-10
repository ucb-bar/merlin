"""Actual source/emission transport controls, with native boundaries stubbed.

The selected class checker deliberately returns unsupported. These tests make
no mathematical, numerical, linked-ISA or hardware qualification claim.
"""

from __future__ import annotations

import json
import shutil
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


def _unused_renderer(*_args, **_kwargs):
    raise AssertionError("compile-only must not render numerical data")


def _diagnostic_gate(*, elf, evidence_root):
    from merlin.targetgen.contract.build_service import file_digest

    evidence_root.mkdir()
    path = evidence_root / "diagnostic.json"
    report = {"status": "accepted", "elf_sha256": file_digest(elf), "scope": "native-boundary stub; no ISA claim"}
    path.write_text(json.dumps(report))
    return {**report, "report_path": str(path), "report_sha256": file_digest(path)}
