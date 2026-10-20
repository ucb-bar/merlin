"""The GSIM certificate producer binds fresh same-ELF captures, never legacy inference."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import gsim_certificate as PRODUCER
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _capsule(tmp_path: Path, *, k: int = 16) -> Path:
    path = tmp_path / "capsule.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": f"matmul_k{k}",
                "inputs": [
                    {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
                    {"name": "X", "role": "input", "shape": [16, k], "dtype": "i8"},
                ],
                "operation": {
                    "op": "matmul",
                    "attributes": {
                        "lhs": "X",
                        "weight": "W",
                        "out": "Y0",
                        "epilogue": [],
                        "output_dtype": "i32",
                        "semantic": "non-functional-source-label",
                    },
                },
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _artifacts(tmp_path: Path) -> tuple[PRODUCER.ArtifactPaths, Path]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "ChipTop0.cpp").write_text("generated implementation\n", encoding="utf-8")
    (model_root / "ChipTop.h").write_text("generated interface\n", encoding="utf-8")
    model_manifest = PRODUCER.write_model_manifest(
        model_root, ["ChipTop0.cpp", "ChipTop.h"], tmp_path / "gsim_model_manifest.json"
    )
    paths = {}
    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_binary", "verilator_binary"):
        path = tmp_path / name
        path.write_text(f"exact {name}\n", encoding="utf-8")
        paths[name] = path
    artifacts = PRODUCER.ArtifactPaths(
        paths["gsim_firrtl"], paths["verilator_firrtl"], model_manifest, paths["gsim_binary"], paths["verilator_binary"]
    )
    tools = {}
    for name in ("emitter", "wrapper", "compiler", "harness", "library"):
        tool = tmp_path / name
        tool.write_text(f"exact {name}\n", encoding="utf-8")
        tools[name] = tool
    receipt = tmp_path / "gsim_build_receipt.json"
    PRODUCER.write_build_receipt(
        output=receipt,
        firrtl=paths["gsim_firrtl"],
        model_manifest=model_manifest,
        binary=paths["gsim_binary"],
        emitter=tools["emitter"],
        cxx_wrapper=tools["wrapper"],
        cxx_compiler=tools["compiler"],
        inputs=[("harness", tools["harness"]), ("static_library", tools["library"])],
        commands=[
            {"stage": "elaborate", "cwd": str(tmp_path), "argv": ["java", "Generator"]},
            {"stage": "emit", "cwd": str(tmp_path), "argv": [str(tools["emitter"].resolve()), "input.fir"]},
            {"stage": "compile", "cwd": str(tmp_path), "argv": [str(tools["wrapper"].resolve()), "ChipTop0.cpp"]},
            {
                "stage": "link",
                "cwd": str(tmp_path),
                "argv": [str(tools["wrapper"].resolve()), "ChipTop0.o", "harness.o", "-o", "gsim_binary"],
            },
        ],
    )
    return artifacts, receipt


class _Backend:
    def available(self, engine: str) -> bool:
        return engine in ("gsim", "verilator")

    def run_elf(self, elf: Path, *, simulator: str, timeout: int) -> str:
        assert elf.read_bytes() == b"one exact elf"
        assert timeout > 0
        return f"{simulator}:same-output"

    def parse_output(self, console: str):
        return {"Y0": [[1, 2], [3, 4]]}, console


def test_workload_is_derived_from_functional_manifest_fields(tmp_path: Path) -> None:
    workload = WORKLOAD.derive_workload(_capsule(tmp_path, k=32))
    assert workload["operation"] == "matmul"
    assert workload["shape"] == {"m": 16, "n": 16, "k": 32}
    assert workload["semantics"]["operand_dtypes"] == {"lhs": "i8", "weight": "i8"}
    assert "semantic" not in workload["semantics"]["operation_attributes"]
    assert workload["semantics"]["numeric_policy"]["compare"] == "exact_int"


def test_frozen_generated_corpus_is_verified_before_workloads_are_derived(tmp_path: Path) -> None:
    from merlin_experiments.phase2 import authoring as stage

    root = tmp_path / "frozen"
    capsule_dir = root / "capsules" / "performance" / "matmul_k16"
    capsule_dir.mkdir(parents=True)
    source = _capsule(tmp_path)
    (capsule_dir / "capsule.yaml").write_bytes(source.read_bytes())
    tree = P2_CONTRACTS.exact_tree_record(capsule_dir)
    aggregate = P2_CONTRACTS.exact_tree_record(root / "capsules")
    manifest = root / "performance_corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "test_target",
                "capsules_sha256": aggregate["sha256"],
                "capsules": [
                    {
                        "family": "prediction",
                        "capsule": "matmul_k16",
                        "source_relative_path": "performance/matmul_k16",
                        "snapshot_relative_path": "capsules/performance/matmul_k16",
                        "snapshot_sha256": tree["sha256"],
                        "n_files": tree["n_files"],
                        "n_bytes": tree["n_bytes"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    workloads = WORKLOAD.derive_frozen_corpus_workloads(
        root, manifest_sha256=_sha(manifest), capsules_sha256=aggregate["sha256"], expected_target="test_target"
    )
    assert workloads["matmul_k16"]["shape"] == {"m": 16, "n": 16, "k": 16}
    (capsule_dir / "capsule.yaml").write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(stage.StageGateError, match="bytes changed|member changed"):
        WORKLOAD.derive_frozen_corpus_workloads(
            root, manifest_sha256=_sha(manifest), capsules_sha256=aggregate["sha256"], expected_target="test_target"
        )


def test_generated_model_manifest_is_deterministic_and_detects_changed_sources(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "b.cpp").write_text("b", encoding="utf-8")
    (root / "a.h").write_text("a", encoding="utf-8")
    first = PRODUCER.build_model_manifest(root, ["b.cpp", "a.h"])
    second = PRODUCER.build_model_manifest(root, ["a.h", "b.cpp"])
    assert first == second
    (root / "b.cpp").write_text("changed", encoding="utf-8")
    assert PRODUCER.build_model_manifest(root, ["a.h", "b.cpp"])["files_sha256"] != first["files_sha256"]


def test_output_digest_is_declared_little_endian_tensor_bytes_not_json(tmp_path: Path) -> None:
    cb = {"tensors": {"Y0": {"role": "output", "shape": [2], "dtype": "i32"}}}
    digest, rows = WORKLOAD.encode_declared_outputs({"Y0": [1, -2]}, cb)
    raw = b"\x01\x00\x00\x00\xfe\xff\xff\xff"
    assert rows == [{"name": "Y0", "shape": [2], "dtype": "i32", "n_bytes": 8, "sha256": sha256(raw).hexdigest()}]
    assert digest != sha256(json.dumps({"Y0": [1, -2]}, sort_keys=True).encode()).hexdigest()
    with pytest.raises(WORKLOAD.ProducerError, match="requires 2"):
        WORKLOAD.encode_declared_outputs({"Y0": [1]}, cb)


def test_smoke_refuses_firrtl_and_generated_model_with_different_tops(tmp_path: Path) -> None:
    artifacts, _ = _artifacts(tmp_path)
    artifacts.gsim_firrtl.write_text("FIRRTL version 3.3.0\ncircuit NotChipTop :\n", encoding="utf-8")
    report = PRODUCER.smoke_legacy_evidence(
        target="test_target",
        legacy_root=tmp_path / "absent-legacy",
        v1_capture_root=None,
        artifacts=artifacts,
        build_receipt=None,
    )
    assert report["status"] == "refused"
    assert any("NotChipTop" in issue and "ChipTop" in issue for issue in report["issues"])


def test_offline_capture_and_producer_make_a_gate_qualifying_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _capsule(tmp_path)
    artifact_dir = tmp_path / "lowered"
    artifact_dir.mkdir()
    command_buffer = {"tensors": {"Y0": {"role": "output", "shape": [2, 2], "dtype": "i32"}}}
    (artifact_dir / "command_buffer.json").write_text(json.dumps(command_buffer), encoding="utf-8")
    (artifact_dir / "lowered.llvm.mlir").write_text("module {}", encoding="utf-8")
    artifacts, receipt = _artifacts(tmp_path)

    # Keep the test offline while exercising the same-ELF and reference-output logic.
    import merlin.runtime.reference as reference

    monkeypatch.setattr(reference, "reference_outputs", lambda cb: {"Y0": [[1, 2], [3, 4]]})
    monkeypatch.setattr(reference, "outputs_match", lambda got, expected: got == expected)

    def build_elf(cb, llvm, destination):
        assert cb == command_buffer and llvm == "module {}"
        elf = destination / "case.elf"
        elf.write_bytes(b"one exact elf")
        return elf

    capture = PRODUCER.capture_case(
        target="test_target",
        capsule_manifest=capsule,
        artifact_dir=artifact_dir,
        workdir=tmp_path / "work",
        artifacts=artifacts,
        backend=_Backend(),
        build_elf=build_elf,
    )
    assert capture["reference"]["elf_sha256"] == capture["candidate"]["elf_sha256"]
    capture_path = tmp_path / "case.json"
    capture_path.write_text(json.dumps(capture, sort_keys=True), encoding="utf-8")

    certificate = PRODUCER.produce_certificate(
        target="test_target", captures=[capture_path], artifacts=artifacts, build_receipt=receipt
    )
    certificate_path = tmp_path / "gsim_equivalence_certificate.json"
    certificate_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")
    record = GATE.load_certificate(certificate_path)
    assert record.target == "test_target" and len(record.members) == 1


def test_independent_float_host_lane_uses_canonical_inputs_and_capsule_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_dir = tmp_path / "float_host_lane"
    capsule_dir.mkdir()
    capsule = capsule_dir / "capsule.yaml"
    capsule.write_text(
        yaml.safe_dump(
            {
                "name": "float_host_lane",
                "inputs": [{"name": "X", "role": "input", "shape": [1, 1, 2, 2], "dtype": "bf16"}],
                "operation": {"op": "movement", "attributes": {"src": "X", "out": "Y0", "output_dtype": "bf16"}},
                "numeric_policy": {"compare": "tolerance_float", "dtype": "bf16", "atol": 0.01, "rtol": 0.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    values = [0.5, -1.0, 2.0, 3.0]
    (capsule_dir / "golden.yaml").write_text(
        yaml.safe_dump(
            {
                "golden_source": "independent_test_oracle",
                "oracle_provenance": {"inputs": {"X": {"shape": [1, 1, 2, 2], "decoded": values}}},
                "outputs": {"Y0": [[[[0.5, -1.0], [2.0, 3.0]]]]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "lowered-float"
    artifact_dir.mkdir()
    command_buffer = {
        "tensors": {
            "arg0": {"role": "input", "shape": [1, 1, 2, 2], "dtype": "bf16"},
            "Y0": {"role": "output", "shape": [1, 1, 2, 2], "dtype": "bf16"},
        },
        "commands": [],
    }
    (artifact_dir / "command_buffer.json").write_text(json.dumps(command_buffer), encoding="utf-8")
    (artifact_dir / "lowered.llvm.mlir").write_text("module {}", encoding="utf-8")
    artifacts, _receipt = _artifacts(tmp_path)

    import merlin.runtime.reference as reference

    monkeypatch.setattr(
        reference,
        "reference_outputs",
        lambda _cb: pytest.fail("an empty host-lane command stream is not the float program's semantic oracle"),
    )

    def build_elf(cb, llvm, destination):
        assert llvm == "module {}"
        assert cb["canonical_inputs"] == {"arg0": {"shape": [1, 1, 2, 2], "values": values}}
        elf = destination / "case.elf"
        elf.write_bytes(b"one exact elf")
        return elf

    from merlin.runtime import fp8_formats

    codes = [int(value) for value in fp8_formats.float_to_codes(values, "bf16")]

    class FloatBackend(_Backend):
        def parse_output(self, console: str):
            return {"Y0": [codes[:2], codes[2:]]}, console

    capture = PRODUCER.capture_case(
        target="test_target",
        capsule_manifest=capsule,
        artifact_dir=artifact_dir,
        workdir=tmp_path / "work-float",
        artifacts=artifacts,
        backend=FloatBackend(),
        build_elf=build_elf,
    )

    assert capture["semantic_reference"] == {
        "kind": "independent_capsule_golden",
        "golden_source": "independent_test_oracle",
        "numeric_policy": {"compare": "tolerance_float", "dtype": "bf16", "atol": 0.01, "rtol": 0.0},
        "operand_binding": "linalg_positional_declaration_order",
        "operand_source": "recorded_capsule_golden",
        "canonical_inputs_sha256": PRODUCER._document_sha({"arg0": {"shape": [1, 1, 2, 2], "values": values}}),
    }
    assert capture["reference"]["output_sha256"] == capture["candidate"]["output_sha256"]


def test_integer_whole_program_uses_materialized_inputs_and_complete_capsule_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole-program command list is an accelerator projection, not its semantic oracle."""
    capsule = _capsule(tmp_path)
    command_buffer = {
        "tensors": {
            "W": {"role": "weight", "shape": [16, 16], "dtype": "i8"},
            "X": {"role": "input", "shape": [16, 16], "dtype": "i8"},
            "Y0": {"role": "output", "shape": [16, 16], "dtype": "i32"},
        },
        # Deliberately empty: a projection cannot stand in for the submitted complete kernel.
        "commands": [],
        "kernel_abi": {
            "kind": "whole_program",
            "symbol": "test_kernel",
            "args": [
                {"tensor": "W", "access": "read"},
                {"tensor": "X", "access": "read"},
                {"tensor": "Y0", "access": "write"},
            ],
            "outputs": ["Y0"],
        },
    }
    import merlin.runtime.reference as reference

    monkeypatch.setattr(
        reference,
        "reference_outputs",
        lambda _cb: pytest.fail("a whole-program accelerator projection is not the complete kernel's semantic oracle"),
    )

    normalized, expected, matches, semantic = PRODUCER._semantic_oracle(capsule, command_buffer)

    from merlin.targetgen import capsule_golden as golden

    manifest = yaml.safe_load(capsule.read_text(encoding="utf-8"))
    canonical = golden.materialized_input_values(manifest)
    assert normalized["canonical_inputs"] == canonical
    assert expected == golden.golden(manifest, capsule.parent)
    assert matches(expected)
    assert semantic == {
        "kind": "whole_program_capsule_golden",
        "golden_source": "merlin_tensor_int",
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "operand_binding": "by_name",
        "operand_source": "recomputed_golden_materialization",
        "canonical_inputs_sha256": PRODUCER._document_sha(canonical),
    }


def test_model_whole_program_binds_returned_input_by_global_plan_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = tmp_path / "capsule.yaml"
    capsule.write_text(
        yaml.safe_dump(
            {
                "name": "tiny_model",
                "kind": "model",
                "inputs": [{"name": "I0", "role": "input", "shape": [2], "dtype": "f32"}],
                "operation": {"op": "model", "attributes": {"out": "Y0"}},
                "numeric_policy": {"compare": "tolerance_float", "dtype": "f32", "atol": 0.01, "rtol": 0.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    command_buffer = {
        "tensors": {
            "arg0": {"role": "weight", "shape": [2], "dtype": "i8"},
            "Y0": {"role": "output", "shape": [2], "dtype": "f32"},
            "tmp": {"role": "intermediate", "shape": [2], "dtype": "f32"},
        },
        "commands": [],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "arg0", "access": "read"},
                {"tensor": "Y0", "access": "readwrite"},
                {"tensor": "tmp", "access": "write"},
            ],
            "outputs": ["Y0"],
        },
        "params": {
            "global_program_plan": {
                "entry_bindings": ["arg0", "Y0"],
                "output_bindings": ["Y0"],
            }
        },
    }

    import contextlib

    import numpy as np

    from merlin.runtime import dispatch_runtime
    from merlin.targetgen import capsule_golden as golden
    from merlin.targetgen import capsule_runner

    provenance = {
        "source": {"content_sha256": "1" * 64},
        "bundle": {"content_sha256": "2" * 64},
        "construction": "frozen_capsule_assets_v1",
        "validation": {"weights_validated_exact": True, "golden_validated": True},
    }

    @contextlib.contextmanager
    def bundle(_capsule, *, timeout):
        assert timeout == 300
        yield tmp_path, provenance, lambda: None

    monkeypatch.setattr(capsule_runner, "_model_runtime_bundle", bundle)
    monkeypatch.setattr(
        dispatch_runtime,
        "resolve_forward_args",
        lambda _bundle: [
            np.asarray([-1, 1], dtype=np.int8),
            np.asarray([0.5, -0.25], dtype=np.float32),
        ],
    )
    monkeypatch.setattr(golden, "golden", lambda *_args: {"Y0": [1.0, 2.0]})
    monkeypatch.setattr(golden, "golden_source", lambda *_args: "pytorch_frozen_model")
    monkeypatch.setattr(
        golden,
        "compare",
        lambda expected, observed, *_args, **_kwargs: {"status": "pass" if expected == observed else "fail"},
    )

    normalized, expected, matches, semantic = PRODUCER._semantic_oracle(capsule, command_buffer)

    bound = {
        "arg0": {"shape": [2], "values": [-1, 1]},
        "Y0": {"shape": [2], "values": [0.5, -0.25]},
    }
    assert normalized["canonical_inputs"] == bound
    assert expected == {"Y0": [1.0, 2.0]}
    assert matches(expected)
    assert semantic["operand_binding"] == "by_name"
    assert semantic["operand_source"] == "validated_frozen_model_bundle"
    assert semantic["canonical_inputs_sha256"] == PRODUCER._document_sha(bound)
    assert semantic["model_source_sha256"] == "1" * 64
    assert semantic["model_bundle_sha256"] == "2" * 64
    assert semantic["model_bundle_validation"]["weights_validated_exact"] is True


def test_legacy_xval_cannot_be_promoted_or_fill_a_v1_capture(tmp_path: Path) -> None:
    legacy = tmp_path / "xval_bytes.json"
    legacy.write_text(
        json.dumps(
            {
                "target": "test_target",
                "reference_engine": "verilator",
                "candidate_engine": "gsim",
                "capsules": [
                    {
                        "capsule": "c",
                        "agreement": "AGREE",
                        "evidence": "output_bytes",
                        "bytes_match": True,
                        "reference": {"engine": "verilator", "ran": True, "verdict": "pass"},
                        "candidate": {"engine": "gsim", "ran": True, "verdict": "pass"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts, receipt = _artifacts(tmp_path)
    with pytest.raises(WORKLOAD.ProducerError, match="not a v1 capture"):
        PRODUCER.produce_certificate(
            target="test_target", captures=[legacy], artifacts=artifacts, build_receipt=receipt
        )


def test_smoke_report_fails_closed_on_missing_capture_and_build_receipt(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "xval_bytes.json").write_text(
        json.dumps(
            {
                "target": "test_target",
                "reference_engine": "verilator",
                "candidate_engine": "gsim",
                "capsules": [
                    {
                        "capsule": "c",
                        "agreement": "AGREE",
                        "evidence": "output_bytes",
                        "bytes_match": True,
                        "reference": {"engine": "verilator", "ran": True, "verdict": "pass"},
                        "candidate": {"engine": "gsim", "ran": True, "verdict": "pass"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts, _ = _artifacts(tmp_path)
    report = PRODUCER.smoke_legacy_evidence(
        target="test_target", legacy_root=legacy_root, v1_capture_root=None, artifacts=artifacts, build_receipt=None
    )
    assert report["status"] == "refused"
    assert report["v1_capture_count"] == 0
    assert any("legacy xval" in issue for issue in report["issues"])
    assert any("build receipt" in issue for issue in report["issues"])
    assert report["rule"].startswith("GSIM remains primary final timing")


def test_build_receipt_must_bind_exact_model_and_binary(tmp_path: Path) -> None:
    artifacts, receipt = _artifacts(tmp_path)
    doc = json.loads(receipt.read_text())
    doc["binary_sha256"] = "0" * 64
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WORKLOAD.ProducerError, match="binary_sha256"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())


def test_build_receipt_rehashes_tools_and_ordered_transcript(tmp_path: Path) -> None:
    artifacts, receipt = _artifacts(tmp_path)
    doc = json.loads(receipt.read_text())
    compiler = Path(doc["tools"]["cxx_compiler"]["path"])
    compiler.write_text("changed compiler\n", encoding="utf-8")
    with pytest.raises(WORKLOAD.ProducerError, match="tool pin"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())
    compiler.write_text("exact compiler\n", encoding="utf-8")
    doc["commands"][-1]["stage"] = "compile"
    doc["commands_sha256"] = PRODUCER._document_sha(doc["commands"])
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WORKLOAD.ProducerError, match="incomplete or unordered"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())


def test_build_receipt_can_honestly_adopt_preexisting_firrtl(tmp_path: Path) -> None:
    artifacts, original = _artifacts(tmp_path)
    doc = json.loads(original.read_text())
    tools = doc["tools"]
    inputs = [(row["role"], row["path"]) for row in doc["inputs"]]
    receipt = tmp_path / "adopted_build_receipt.json"
    commands = [row for row in doc["commands"] if row["stage"] != "elaborate"]
    PRODUCER.write_build_receipt(
        output=receipt,
        firrtl=artifacts.gsim_firrtl,
        model_manifest=artifacts.gsim_model,
        binary=artifacts.gsim_binary,
        emitter=tools["gsim_emitter"]["path"],
        cxx_wrapper=tools["cxx_wrapper"]["path"],
        cxx_compiler=tools["cxx_compiler"]["path"],
        inputs=inputs,
        commands=commands,
        firrtl_boundary=PRODUCER.FIRRTL_BOUNDARY_ADOPTED,
    )

    adopted = json.loads(receipt.read_text())
    assert adopted["schema_version"] == "merlin.gsim-model-build.v3"
    assert adopted["provenance"] == {
        "firrtl_boundary": "adopted_preexisting",
        "elaboration_performed": False,
        "warning": PRODUCER.ADOPTED_FIRRTL_WARNING,
    }
    assert all(row["stage"] != "elaborate" for row in adopted["commands"])
    PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())


def test_adopted_firrtl_receipt_rejects_claimed_elaboration(tmp_path: Path) -> None:
    artifacts, original = _artifacts(tmp_path)
    doc = json.loads(original.read_text())
    tools = doc["tools"]
    inputs = [(row["role"], row["path"]) for row in doc["inputs"]]
    with pytest.raises(WORKLOAD.ProducerError, match="must not claim an elaborate command"):
        PRODUCER.build_receipt_document(
            firrtl=artifacts.gsim_firrtl,
            model_manifest=artifacts.gsim_model,
            binary=artifacts.gsim_binary,
            emitter=tools["gsim_emitter"]["path"],
            cxx_wrapper=tools["cxx_wrapper"]["path"],
            cxx_compiler=tools["cxx_compiler"]["path"],
            inputs=inputs,
            commands=doc["commands"],
            firrtl_boundary=PRODUCER.FIRRTL_BOUNDARY_ADOPTED,
        )


def test_adopted_firrtl_receipt_warning_is_load_bearing(tmp_path: Path) -> None:
    artifacts, receipt = _artifacts(tmp_path)
    doc = json.loads(receipt.read_text())
    doc["commands"] = [row for row in doc["commands"] if row["stage"] != "elaborate"]
    doc["commands_sha256"] = PRODUCER._document_sha(doc["commands"])
    doc["provenance"] = {
        "firrtl_boundary": PRODUCER.FIRRTL_BOUNDARY_ADOPTED,
        "elaboration_performed": False,
        "warning": "the important caveat was dropped",
    }
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WORKLOAD.ProducerError, match="contradicts its provenance boundary"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())


def test_a_digest_capture_reads_values_once_on_spike_and_binds_both_engines_to_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A performance-scale output cannot be printed on the reference engine (about 210 cycles/s):
    the shared ELF prints one digest per output, the values come once from Spike's full-value build,
    and both engines' digests must equal the digest of those values. Cycles are recorded per engine."""
    from merlin.perf import capture_store
    from merlin.runtime.out_digest import container_bytes, xxh64

    (tmp_path / "captures" / "test_target").mkdir(parents=True)
    monkeypatch.setattr(capture_store, "store_root", lambda target: tmp_path / "captures" / target)
    capsule = _capsule(tmp_path)
    artifact_dir = tmp_path / "lowered"
    artifact_dir.mkdir()
    command_buffer = {"tensors": {"Y0": {"role": "output", "shape": [2, 2], "dtype": "i32"}}}
    from merlin.targetgen.contract import readback_policy as RB

    # This fixture buffer has no commands to derive a logical interface from; its declared output is the roster.
    monkeypatch.setattr(RB, "_console_output_roster", lambda cb: (["Y0"], {"Y0": {"shape": [2, 2], "dtype": "i32"}}))
    (artifact_dir / "command_buffer.json").write_text(json.dumps(command_buffer), encoding="utf-8")
    (artifact_dir / "lowered.llvm.mlir").write_text("module {}", encoding="utf-8")
    artifacts, receipt = _artifacts(tmp_path)
    import merlin.runtime.reference as reference

    values = [[1, 2], [3, 4]]
    monkeypatch.setattr(reference, "reference_outputs", lambda cb: {"Y0": values})
    monkeypatch.setattr(reference, "outputs_match", lambda got, expected: got == expected)
    digest = f"{xxh64(container_bytes([1, 2, 3, 4], 4)):016x}"
    built = []

    def build_elf(cb, llvm, destination, readback_policy=None):
        destination.mkdir(parents=True, exist_ok=True)
        elf = destination / ("digest.elf" if readback_policy else "values.elf")
        elf.write_bytes(b"digest elf" if readback_policy else b"values elf")
        built.append(readback_policy.transport if readback_policy else "full")
        return elf

    class DigestBackend:
        gsim_holds = [1, 2, 3, 4]

        def available(self, engine):
            return engine in ("spike", "gsim", "verilator")

        def run_elf(self, elf, *, simulator, timeout):
            if simulator == "spike":
                assert elf.read_bytes().startswith(b"values elf")
                return "values"
            assert elf.read_bytes().startswith(b"digest elf"), "both engines run the one digest ELF"
            held = self.gsim_holds if simulator == "gsim" else [1, 2, 3, 4]
            line = f"OUT_DIGEST Y0 16 {xxh64(container_bytes(held, 4)):016x}"
            return f"METRIC cycles {900 if simulator == 'gsim' else 900}\n{line}\nDONE\n"

        def parse_output(self, console):
            if console == "values":
                return {"Y0": values}, {}
            return {}, {"cycles": 900}

    capture = PRODUCER.capture_case(
        target="test_target",
        capsule_manifest=capsule,
        artifact_dir=artifact_dir,
        workdir=tmp_path / "work",
        artifacts=artifacts,
        backend=DigestBackend(),
        build_elf=build_elf,
    )
    assert built == ["out_digest_v1", "full"]
    for side in ("reference", "candidate"):
        assert capture[side]["readback"]["mode"] == "digest"
        assert capture[side]["readback"]["output_digests"] == {"Y0": digest}
        assert capture[side]["cycles"] == 900
    assert capture["reference"]["output_sha256"] == capture["candidate"]["output_sha256"]
    capture_path = tmp_path / "case.json"
    capture_path.write_text(json.dumps(capture, sort_keys=True), encoding="utf-8")
    PRODUCER.produce_certificate(
        target="test_target", captures=[capture_path], artifacts=artifacts, build_receipt=receipt
    )

    wrong = DigestBackend()
    wrong.gsim_holds = [1, 2, 3, 5]

    def build_other_elf(cb, llvm, destination, readback_policy=None):  # new bytes: no stored capture answers
        elf = build_elf(cb, llvm, destination, readback_policy)
        elf.write_bytes(elf.read_bytes() + b" v2")
        return elf

    with pytest.raises(WORKLOAD.ProducerError, match="digests differ"):
        PRODUCER.capture_case(
            target="test_target",
            capsule_manifest=capsule,
            artifact_dir=artifact_dir,
            workdir=tmp_path / "w2",
            artifacts=artifacts,
            backend=wrong,
            build_elf=build_other_elf,
            readback="digest",
        )


# ---------------------------------------------------------------------------------------------
# engine-level qualification (``--certification engine_qualified``)
# ---------------------------------------------------------------------------------------------
def _qualification_captures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cycles: dict[int, tuple[int, int]]):
    """One same-ELF capture per K, with (Verilator, gSIM) kernel cycles as given."""
    from merlin.perf import capture_store

    (tmp_path / "captures" / "test_target").mkdir(parents=True)
    monkeypatch.setattr(capture_store, "store_root", lambda target: tmp_path / "captures" / target)
    import merlin.runtime.reference as reference

    monkeypatch.setattr(reference, "reference_outputs", lambda cb: {"Y0": [[1, 2], [3, 4]]})
    monkeypatch.setattr(reference, "outputs_match", lambda got, expected: got == expected)
    artifacts, receipt = _artifacts(tmp_path)
    paths = []
    for k, (verilator_cycles, gsim_cycles) in cycles.items():
        case = tmp_path / f"case{k}"
        case.mkdir()
        capsule = _capsule(case, k=k)
        lowered = case / "lowered"
        lowered.mkdir()
        (lowered / "command_buffer.json").write_text(
            json.dumps({"tensors": {"Y0": {"role": "output", "shape": [2, 2], "dtype": "i32"}}}), encoding="utf-8"
        )
        (lowered / "lowered.llvm.mlir").write_text("module {}", encoding="utf-8")

        class CycleBackend(_Backend):
            def run_elf(self, elf, *, simulator, timeout):
                return simulator

            def parse_output(self, console):
                n = gsim_cycles if console == "gsim" else verilator_cycles
                return {"Y0": [[1, 2], [3, 4]]}, {"cycles": n}

        def build_elf(cb, llvm, destination, _k=k):
            elf = destination / "case.elf"
            elf.write_bytes(f"elf {_k}".encode())
            return elf

        capture = PRODUCER.capture_case(
            target="test_target",
            capsule_manifest=capsule,
            artifact_dir=lowered,
            workdir=case / "work",
            artifacts=artifacts,
            backend=CycleBackend(),
            build_elf=build_elf,
        )
        path = case / "capture.json"
        path.write_text(json.dumps(capture, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths, artifacts, receipt


def test_an_engine_qualification_admits_covered_strata_and_refuses_the_rest(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import engine_qualification as EQ

    paths, artifacts, receipt = _qualification_captures(tmp_path, monkeypatch, {32: (4100, 4100)})
    document = EQ.produce_qualification(
        target="test_target", captures=paths, artifacts=artifacts, build_receipt=receipt, cycle_budget=10_000
    )
    path = tmp_path / "engine_qualification.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    record = GATE.load_certificate(path)
    assert record.certification == "engine_qualified" and len(record.coverage) == 1

    qualified = WORKLOAD.derive_workload(_capsule(tmp_path / "case32", k=32))
    bigger = json.loads(json.dumps(qualified))
    bigger["shape"]["k"] = 8192  # same stratum and form, performance scale
    assert EQ.coverage_key(bigger) == EQ.coverage_key(qualified)
    other_form = json.loads(json.dumps(qualified))
    other_form["semantics"]["operation_attributes"]["epilogue"] = ["relu"]
    assert record.admits(bigger) and not record.admits(other_form)
    assert GATE.plan_evaluation(record, bigger, phase="final_performance", gsim_available=True).eligible
    refused = GATE.plan_evaluation(record, other_form, phase="final_performance", gsim_available=True)
    assert not refused.eligible and not refused.final_cycle_authority

    tampered = json.loads(path.read_text())
    tampered["coverage"][0]["key"]["form"]["epilogue"] = ["relu"]
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(GATE.GsimGateError, match="coverage"):
        GATE.load_certificate(path)


@pytest.mark.parametrize(
    "cycles, budget, why", [((4100, 4116), 10_000, "Verilator 4100"), ((4100, 4100), 4000, "over the 4000 budget")]
)
def test_engine_qualification_needs_identical_cycles_within_its_budget(tmp_path, monkeypatch, cycles, budget, why):
    from merlin_experiments.phase2 import engine_qualification as EQ

    paths, artifacts, receipt = _qualification_captures(tmp_path, monkeypatch, {32: cycles})
    with pytest.raises(EQ.EngineQualificationError, match=why):
        EQ.produce_qualification(
            target="test_target", captures=paths, artifacts=artifacts, build_receipt=receipt, cycle_budget=budget
        )


def test_the_host_suite_takes_the_largest_affordable_member_per_key_and_names_gaps(tmp_path):
    from merlin_experiments.phase2 import engine_qualification as EQ

    w = {}
    for k in (32, 64, 128):
        (tmp_path / f"k{k}").mkdir()
        w[k] = WORKLOAD.derive_workload(_capsule(tmp_path / f"k{k}", k=k))
    relu = json.loads(json.dumps(w[32]))
    relu["semantics"]["operation_attributes"]["epilogue"] = ["relu"]
    plan = EQ.plan_suite(
        {"big": w[128], "relu_member": relu},
        {"small": (w[32], 900), "medium": (w[64], 1800), "too_big": (w[128], 9000)},
        cycle_budget=2000,
    )
    assert plan["selected"] == ["medium"]
    assert plan["uncovered"] == [EQ.coverage_key(relu)]
