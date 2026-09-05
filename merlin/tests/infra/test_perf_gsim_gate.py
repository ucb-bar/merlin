"""GSIM is fast only inside a pinned, byte-equivalent elaborated-RTL envelope."""
from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


_SOURCE = merlin_dir() / "experiments/gemmini_perf_bench/scripts/perf_gsim_gate.py"
_SPEC = importlib.util.spec_from_file_location("perf_gsim_gate_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
GATE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GATE
_SPEC.loader.exec_module(GATE)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _workload(k: int = 16) -> dict:
    return {
        "operation": "matmul",
        "shape": {"m": 16, "n": 16, "k": k},
        "semantics": {"input_dtype": "i8", "accumulator_dtype": "i32", "transpose": [False, False]},
    }


def _make_certificate(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    artifacts: dict[str, Path] = {}
    for name in sorted(GATE.REQUIRED_PINS):
        path = tmp_path / f"{name}.artifact"
        path.write_bytes(f"exact-{name}-bytes".encode())
        artifacts[name] = path
    workload = _workload()
    elf = "e" * 64
    output_tensors = [{"name": "Y0", "shape": [1], "dtype": "i32",
                       "n_bytes": 4, "sha256": "8" * 64}]
    member = {
        "workload": workload,
        "workload_sha256": GATE.workload_sha256(workload),
        "elf_sha256": elf,
        "agreement": "AGREE",
        "evidence": "output_bytes",
        "bytes_match": True,
        "reference": {
            "engine": "verilator", "ran": True, "verdict": "pass", "elf_sha256": elf,
            "binary_sha256": _sha(artifacts["verilator_binary"]),
            "firrtl_sha256": _sha(artifacts["verilator_firrtl"]),
            "output_sha256": "9" * 64,
            "output_encoding": GATE.OUTPUT_ENCODING, "output_tensors": output_tensors,
            "derived_from_rtl": True, "cycle_accurate": True,
        },
        "candidate": {
            "engine": "gsim", "ran": True, "verdict": "pass", "elf_sha256": elf,
            "binary_sha256": _sha(artifacts["gsim_binary"]),
            "firrtl_sha256": _sha(artifacts["gsim_firrtl"]),
            "model_sha256": _sha(artifacts["gsim_model"]),
            "output_sha256": "9" * 64,
            "output_encoding": GATE.OUTPUT_ENCODING, "output_tensors": output_tensors,
            "derived_from_rtl": True, "cycle_accurate": True,
        },
    }
    doc = {
        "schema_version": GATE.SCHEMA_VERSION,
        "status": "certified",
        "target": "test_target",
        "fidelity": GATE.FIDELITY,
        "primary_engine": "gsim",
        "reference_engine": "verilator",
        "pins": {name: {"path": path.name, "sha256": _sha(path)}
                 for name, path in artifacts.items()},
        "members": [member],
        "unresolved": [{"workload": _workload(128),
                        "reason": "deep-K Verilator corroboration has not completed"}],
    }
    extra: dict[str, Path] = {}
    for name in ("gsim_emitter", "cxx_wrapper", "cxx_compiler", "harness", "libfesvr"):
        extra[name] = tmp_path / name
        extra[name].write_text(f"exact-{name}", encoding="utf-8")
    commands = [
        {"stage": "elaborate", "cwd": str(tmp_path), "argv": ["java", "Generator"]},
        {"stage": "emit", "cwd": str(tmp_path), "argv": ["gsim", "input.fir"]},
        {"stage": "compile", "cwd": str(tmp_path), "argv": ["clang++", "model.cpp"]},
        {"stage": "link", "cwd": str(tmp_path), "argv": ["clang++", "model.o", "-o", "gsim"]},
    ]
    inputs = [
        {"role": "harness", "path": str(extra["harness"]), "sha256": _sha(extra["harness"]),
         "n_bytes": extra["harness"].stat().st_size},
        {"role": "static_library", "path": str(extra["libfesvr"]),
         "sha256": _sha(extra["libfesvr"]), "n_bytes": extra["libfesvr"].stat().st_size},
    ]
    build_receipt = tmp_path / "gsim_build_receipt.json"
    build_receipt.write_text(json.dumps({
        "schema_version": GATE.BUILD_RECEIPT_SCHEMA, "status": "complete",
        "firrtl_sha256": _sha(artifacts["gsim_firrtl"]),
        "model_manifest_sha256": _sha(artifacts["gsim_model"]),
        "binary_sha256": _sha(artifacts["gsim_binary"]),
        "artifacts": {
            "firrtl": {"path": str(artifacts["gsim_firrtl"]),
                       "sha256": _sha(artifacts["gsim_firrtl"])},
            "model_manifest": {"path": str(artifacts["gsim_model"]),
                               "sha256": _sha(artifacts["gsim_model"])},
            "binary": {"path": str(artifacts["gsim_binary"]),
                       "sha256": _sha(artifacts["gsim_binary"])},
        },
        "tools": {name: {"path": str(extra[name]), "sha256": _sha(extra[name])}
                  for name in ("gsim_emitter", "cxx_wrapper", "cxx_compiler")},
        "inputs": inputs,
        "inputs_sha256": sha256(GATE.canonical_json(inputs).encode()).hexdigest(),
        "commands": commands,
        "commands_sha256": sha256(GATE.canonical_json(commands).encode()).hexdigest(),
    }, sort_keys=True), encoding="utf-8")
    doc["build_binding"] = {
        "path": build_receipt.name, "sha256": _sha(build_receipt),
        "commands_sha256": sha256(GATE.canonical_json(commands).encode()).hexdigest(),
    }
    certificate = tmp_path / "gsim_equivalence_certificate.json"
    certificate.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    return certificate, artifacts


def _execution(record, *, engine: str = "gsim", cycles: int = 123) -> dict:
    pin = "gsim_binary" if engine == "gsim" else "verilator_binary"
    return {
        "engine": engine,
        "status": "pass",
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "cycles": cycles,
        "elf_sha256": "a" * 64,
        "binary_sha256": record.pins[pin]["sha256"],
        "firrtl_sha256": record.pins[
            "gsim_firrtl" if engine == "gsim" else "verilator_firrtl"]["sha256"],
        **({"model_sha256": record.pins["gsim_model"]["sha256"]} if engine == "gsim" else {}),
    }


def test_certificate_is_content_addressed_and_all_artifact_pins_are_verified(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    assert record.sha256 == _sha(path)
    assert set(record.pins) == GATE.REQUIRED_PINS
    assert set(record.members) == {GATE.workload_sha256(_workload())}
    assert record.to_dict()["certified_workloads"] == 1


def test_tampered_firrtl_or_model_binary_refuses_the_certificate(tmp_path: Path) -> None:
    path, artifacts = _make_certificate(tmp_path)
    artifacts["gsim_firrtl"].write_bytes(b"changed after certification")
    with pytest.raises(GATE.GsimGateError, match="gsim_firrtl digest mismatch"):
        GATE.load_certificate(path)


def test_cross_validation_must_run_one_exact_elf_on_both_engines(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    doc = json.loads(path.read_text())
    doc["members"][0]["candidate"]["elf_sha256"] = "f" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(GATE.GsimGateError, match="same ELF"):
        GATE.load_certificate(path)


@pytest.mark.parametrize("field,value", [
    ("agreement", "UNDETERMINABLE"),
    ("evidence", "verdict_only"),
    ("bytes_match", False),
])
def test_only_byte_identical_agreement_enters_the_envelope(
        tmp_path: Path, field: str, value: object) -> None:
    path, _ = _make_certificate(tmp_path)
    doc = json.loads(path.read_text())
    doc["members"][0][field] = value
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(GATE.GsimGateError):
        GATE.load_certificate(path)


def test_eligible_development_work_must_use_gsim_without_silent_fallback(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    decision = GATE.plan_evaluation(
        record, _workload(), phase="development_correctness", gsim_available=True)
    assert decision.eligible and decision.use_gsim and decision.selected_engine == "gsim"
    with pytest.raises(GATE.GsimGateError, match="must use GSIM"):
        GATE.validate_execution(record, decision, _execution(record, engine="verilator"))

    unavailable = GATE.plan_evaluation(
        record, _workload(), phase="development_correctness", gsim_available=False)
    assert not unavailable.admitted and unavailable.selected_engine is None
    assert "fallback is forbidden" in unavailable.refusal_reason


def test_outside_envelope_uses_recorded_verilator_fallback_and_never_gsim(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    decision = GATE.plan_evaluation(
        record, _workload(32), phase="development_correctness", gsim_available=True)
    assert not decision.eligible and not decision.use_gsim
    assert decision.selected_engine == "verilator"
    assert "outside" in decision.fallback_reason
    with pytest.raises(GATE.GsimGateError, match="recorded Verilator fallback"):
        GATE.validate_execution(record, decision, _execution(record, engine="gsim"))


def test_unfinished_deep_k_cross_validation_is_explicit_not_generalized(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    decision = GATE.plan_evaluation(
        record, _workload(128), phase="final_performance", gsim_available=True)
    assert not decision.eligible and decision.selected_engine is None
    assert decision.admitted is False and decision.final_cycle_authority is False
    assert "explicitly leaves" in decision.refusal_reason
    assert "deep-K" in decision.refusal_reason
    assert "final timing requires" in decision.refusal_reason


def test_certified_gsim_is_final_cycle_authority_with_exact_binary_pin(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    decision = GATE.plan_evaluation(
        record, _workload(), phase="final_performance", gsim_available=True)
    audit = GATE.validate_execution(record, decision, _execution(record, cycles=987))
    assert audit["admitted"]
    assert audit["cycle_claim_authority"] == GATE.FIDELITY
    assert audit["decision"]["final_cycle_authority"] is True

    bad = _execution(record)
    bad["binary_sha256"] = "0" * 64
    with pytest.raises(GATE.GsimGateError, match="pinned gsim_binary"):
        GATE.validate_execution(record, decision, bad)
    with pytest.raises(GATE.GsimGateError, match="positive integer cycle"):
        GATE.validate_execution(record, decision, _execution(record, cycles=0))


def test_discovery_refuses_post_hoc_choice_between_distinct_certificates(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path1, artifacts1 = _make_certificate(first)
    path2, artifacts2 = _make_certificate(second)
    doc = json.loads(path2.read_text())
    doc["unresolved"][0]["reason"] += " (second certificate)"
    path2.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    artifact_paths = {name: path for name, path in artifacts1.items()}
    # The two fixtures use byte-identical pin payloads, so either path set validates both documents.
    assert {name: _sha(path) for name, path in artifacts1.items()} == {
        name: _sha(path) for name, path in artifacts2.items()}
    with pytest.raises(GATE.GsimGateError, match="multiple distinct"):
        GATE.discover_certificate([tmp_path], target="test_target", artifact_paths=artifact_paths)
    selected = GATE.discover_certificate(
        [tmp_path], target="test_target", artifact_paths=artifact_paths,
        expected_sha256=_sha(path1))
    assert selected.sha256 == _sha(path1)


def test_campaign_predeclares_gsim_and_deterministic_verilator_subset(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    first = GATE.predeclare_campaign(
        record, [_workload(), _workload(128)], gsim_available=True, corroboration_count=1)
    second = GATE.predeclare_campaign(
        record, [_workload(128), _workload()], gsim_available=True, corroboration_count=1)
    assert first == second
    assert first["development"][0]["selected_engine"] in {"gsim", "verilator"}
    assert {row["selected_engine"] for row in first["development"]} == {"gsim", "verilator"}
    assert first["verilator_corroboration"] == [{
        "workload_sha256": GATE.workload_sha256(_workload()),
        "primary_engine": "gsim",
        "corroborating_engine": "verilator",
        "require_same_elf": True,
        "require_output_bytes_match": True,
    }]
    assert len(first["predeclaration_sha256"]) == 64


def test_legacy_cross_validation_is_discovered_but_never_implicitly_qualified(tmp_path: Path) -> None:
    report = {
        "target": "test_target", "reference_engine": "verilator", "candidate_engine": "gsim",
        "capsules": [
            {"capsule": "small", "agreement": "AGREE", "evidence": "output_bytes",
             "bytes_match": True,
             "reference": {"engine": "verilator", "ran": True, "verdict": "pass"},
             "candidate": {"engine": "gsim", "ran": True, "verdict": "pass"}},
            {"capsule": "deep_k", "agreement": "UNDETERMINABLE", "evidence": "none",
             "bytes_match": None,
             "reference": {"engine": "verilator", "ran": False, "verdict": "did_not_run"},
             "candidate": {"engine": "gsim", "ran": False, "verdict": "did_not_run"}},
        ],
    }
    path = tmp_path / "xval_gm_bytes.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    found = GATE.discover_cross_validation_reports([tmp_path], target="test_target")
    assert len(found) == 1
    assert found[0].agreeing_capsules == ("small",)
    assert found[0].unresolved_capsules == ("deep_k",)
    assert not found[0].qualifying
    assert "shared ELF SHA-256 per agreeing capsule" in found[0].missing_for_qualification


def test_predeclared_corroboration_requires_same_elf_and_output_bytes(tmp_path: Path) -> None:
    path, _ = _make_certificate(tmp_path)
    record = GATE.load_certificate(path)
    declaration = GATE.predeclare_campaign(
        record, [_workload()], gsim_available=True, corroboration_count=1)
    item = declaration["verilator_corroboration"][0]
    primary = {**_execution(record), "output_sha256": "b" * 64}
    reference = {**_execution(record, engine="verilator"), "output_sha256": "b" * 64}
    evidence = GATE.validate_corroboration(record, item, primary, reference)
    assert evidence["agreement"] == "AGREE"
    assert evidence["performance_selection"].startswith("none")

    wrong_elf = {**reference, "elf_sha256": "c" * 64}
    with pytest.raises(GATE.GsimGateError, match="same exact ELF"):
        GATE.validate_corroboration(record, item, primary, wrong_elf)
    wrong_output = {**reference, "output_sha256": "d" * 64}
    with pytest.raises(GATE.GsimGateError, match="output bytes"):
        GATE.validate_corroboration(record, item, primary, wrong_output)
