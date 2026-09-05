"""The GSIM certificate producer binds fresh same-ELF captures, never legacy inference."""
from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir


_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load("perf_gsim_gate")
PRODUCER = _load("produce_gsim_certificate")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _capsule(tmp_path: Path, *, k: int = 16) -> Path:
    path = tmp_path / "capsule.yaml"
    path.write_text(yaml.safe_dump({
        "name": f"matmul_k{k}",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
            {"name": "X", "role": "input", "shape": [16, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {
            "lhs": "X", "weight": "W", "out": "Y0", "epilogue": [],
            "output_dtype": "i32", "semantic": "non-functional-source-label"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }, sort_keys=False), encoding="utf-8")
    return path


def _artifacts(tmp_path: Path) -> tuple[PRODUCER.ArtifactPaths, Path]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "ChipTop0.cpp").write_text("generated implementation\n", encoding="utf-8")
    (model_root / "ChipTop.h").write_text("generated interface\n", encoding="utf-8")
    model_manifest = PRODUCER.write_model_manifest(
        model_root, ["ChipTop0.cpp", "ChipTop.h"], tmp_path / "gsim_model_manifest.json")
    paths = {}
    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_binary", "verilator_binary"):
        path = tmp_path / name
        path.write_text(f"exact {name}\n", encoding="utf-8")
        paths[name] = path
    artifacts = PRODUCER.ArtifactPaths(
        paths["gsim_firrtl"], paths["verilator_firrtl"], model_manifest,
        paths["gsim_binary"], paths["verilator_binary"])
    tools = {}
    for name in ("emitter", "wrapper", "compiler", "harness", "library"):
        tool = tmp_path / name
        tool.write_text(f"exact {name}\n", encoding="utf-8")
        tools[name] = tool
    receipt = tmp_path / "gsim_build_receipt.json"
    PRODUCER.write_build_receipt(
        output=receipt, firrtl=paths["gsim_firrtl"], model_manifest=model_manifest,
        binary=paths["gsim_binary"], emitter=tools["emitter"],
        cxx_wrapper=tools["wrapper"], cxx_compiler=tools["compiler"],
        inputs=[("harness", tools["harness"]), ("static_library", tools["library"])],
        commands=[
            {"stage": "elaborate", "cwd": str(tmp_path), "argv": ["java", "Generator"]},
            {"stage": "emit", "cwd": str(tmp_path), "argv": ["gsim", "input.fir"]},
            {"stage": "compile", "cwd": str(tmp_path), "argv": ["clang++", "ChipTop0.cpp"]},
            {"stage": "link", "cwd": str(tmp_path),
             "argv": ["clang++", "ChipTop0.o", "harness.o", "-o", "gsim_binary"]},
        ])
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
    workload = PRODUCER.derive_workload(_capsule(tmp_path, k=32))
    assert workload["operation"] == "matmul"
    assert workload["shape"] == {"m": 16, "n": 16, "k": 32}
    assert workload["semantics"]["operand_dtypes"] == {"lhs": "i8", "weight": "i8"}
    assert "semantic" not in workload["semantics"]["operation_attributes"]
    assert workload["semantics"]["numeric_policy"]["compare"] == "exact_int"


def test_frozen_generated_corpus_is_verified_before_workloads_are_derived(tmp_path: Path) -> None:
    import perf_agent_stage as stage

    root = tmp_path / "frozen"
    capsule_dir = root / "capsules" / "performance" / "matmul_k16"
    capsule_dir.mkdir(parents=True)
    source = _capsule(tmp_path)
    (capsule_dir / "capsule.yaml").write_bytes(source.read_bytes())
    tree = stage._exact_tree_record(capsule_dir)
    aggregate = stage._exact_tree_record(root / "capsules")
    manifest = root / "performance_corpus_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "target": "test_target", "capsules_sha256": aggregate["sha256"],
        "capsules": [{
            "family": "prediction", "capsule": "matmul_k16",
            "source_relative_path": "performance/matmul_k16",
            "snapshot_relative_path": "capsules/performance/matmul_k16",
            "snapshot_sha256": tree["sha256"], "n_files": tree["n_files"],
            "n_bytes": tree["n_bytes"],
        }],
    }, sort_keys=True), encoding="utf-8")
    workloads = PRODUCER.derive_frozen_corpus_workloads(
        root, manifest_sha256=_sha(manifest), capsules_sha256=aggregate["sha256"],
        expected_target="test_target")
    assert workloads["matmul_k16"]["shape"] == {"m": 16, "n": 16, "k": 16}
    (capsule_dir / "capsule.yaml").write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(stage.StageGateError, match="bytes changed|member changed"):
        PRODUCER.derive_frozen_corpus_workloads(
            root, manifest_sha256=_sha(manifest), capsules_sha256=aggregate["sha256"],
            expected_target="test_target")


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
    digest, rows = PRODUCER.encode_declared_outputs({"Y0": [1, -2]}, cb)
    raw = b"\x01\x00\x00\x00\xfe\xff\xff\xff"
    assert rows == [{"name": "Y0", "shape": [2], "dtype": "i32", "n_bytes": 8,
                     "sha256": sha256(raw).hexdigest()}]
    assert digest != sha256(json.dumps({"Y0": [1, -2]}, sort_keys=True).encode()).hexdigest()
    with pytest.raises(PRODUCER.ProducerError, match="requires 2"):
        PRODUCER.encode_declared_outputs({"Y0": [1]}, cb)


def test_smoke_refuses_firrtl_and_generated_model_with_different_tops(tmp_path: Path) -> None:
    artifacts, _ = _artifacts(tmp_path)
    artifacts.gsim_firrtl.write_text("FIRRTL version 3.3.0\ncircuit NotChipTop :\n", encoding="utf-8")
    report = PRODUCER.smoke_legacy_evidence(
        target="test_target", legacy_root=tmp_path / "absent-legacy",
        v1_capture_root=None, artifacts=artifacts, build_receipt=None)
    assert report["status"] == "refused"
    assert any("NotChipTop" in issue and "ChipTop" in issue for issue in report["issues"])


def test_offline_capture_and_producer_make_a_gate_qualifying_certificate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capsule = _capsule(tmp_path)
    artifact_dir = tmp_path / "lowered"
    artifact_dir.mkdir()
    command_buffer = {
        "tensors": {"Y0": {"role": "output", "shape": [2, 2], "dtype": "i32"}}}
    (artifact_dir / "command_buffer.json").write_text(
        json.dumps(command_buffer), encoding="utf-8")
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
        target="test_target", capsule_manifest=capsule, artifact_dir=artifact_dir,
        workdir=tmp_path / "work", artifacts=artifacts, backend=_Backend(), build_elf=build_elf)
    assert capture["reference"]["elf_sha256"] == capture["candidate"]["elf_sha256"]
    capture_path = tmp_path / "case.json"
    capture_path.write_text(json.dumps(capture, sort_keys=True), encoding="utf-8")

    certificate = PRODUCER.produce_certificate(
        target="test_target", captures=[capture_path], artifacts=artifacts, build_receipt=receipt)
    certificate_path = tmp_path / "gsim_equivalence_certificate.json"
    certificate_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")
    record = GATE.load_certificate(certificate_path)
    assert record.target == "test_target" and len(record.members) == 1


def test_legacy_xval_cannot_be_promoted_or_fill_a_v1_capture(tmp_path: Path) -> None:
    legacy = tmp_path / "xval_bytes.json"
    legacy.write_text(json.dumps({
        "target": "test_target", "reference_engine": "verilator", "candidate_engine": "gsim",
        "capsules": [{"capsule": "c", "agreement": "AGREE", "evidence": "output_bytes",
                      "bytes_match": True,
                      "reference": {"engine": "verilator", "ran": True, "verdict": "pass"},
                      "candidate": {"engine": "gsim", "ran": True, "verdict": "pass"}}],
    }), encoding="utf-8")
    artifacts, receipt = _artifacts(tmp_path)
    with pytest.raises(PRODUCER.ProducerError, match="not a v1 capture"):
        PRODUCER.produce_certificate(
            target="test_target", captures=[legacy], artifacts=artifacts, build_receipt=receipt)


def test_smoke_report_fails_closed_on_missing_capture_and_build_receipt(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "xval_bytes.json").write_text(json.dumps({
        "target": "test_target", "reference_engine": "verilator", "candidate_engine": "gsim",
        "capsules": [{"capsule": "c", "agreement": "AGREE", "evidence": "output_bytes",
                      "bytes_match": True,
                      "reference": {"engine": "verilator", "ran": True, "verdict": "pass"},
                      "candidate": {"engine": "gsim", "ran": True, "verdict": "pass"}}],
    }), encoding="utf-8")
    artifacts, _ = _artifacts(tmp_path)
    report = PRODUCER.smoke_legacy_evidence(
        target="test_target", legacy_root=legacy_root, v1_capture_root=None,
        artifacts=artifacts, build_receipt=None)
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
    with pytest.raises(PRODUCER.ProducerError, match="binary_sha256"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())


def test_build_receipt_rehashes_tools_and_ordered_transcript(tmp_path: Path) -> None:
    artifacts, receipt = _artifacts(tmp_path)
    doc = json.loads(receipt.read_text())
    compiler = Path(doc["tools"]["cxx_compiler"]["path"])
    compiler.write_text("changed compiler\n", encoding="utf-8")
    with pytest.raises(PRODUCER.ProducerError, match="tool pin"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())
    compiler.write_text("exact compiler\n", encoding="utf-8")
    doc["commands"][-1]["stage"] = "compile"
    doc["commands_sha256"] = PRODUCER._document_sha(doc["commands"])
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PRODUCER.ProducerError, match="incomplete or unordered"):
        PRODUCER.validate_build_receipt(receipt, pins=artifacts.pinned())
