from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.compare import paper_report as report
from merlin.compare import paper_ablation_generator
from merlin.compare.paper_measurement_controller import produce_receipt


def _backend(name: str, kind: str, precisions=("fp32",)):
    return SimpleNamespace(name=name, kind=kind, precisions=precisions)


def _spec(*backends):
    model = SimpleNamespace(name="holdout", precisions=("fp32",))
    spec = SimpleNamespace(
        status="frozen", label="frozen", primary_precision="w8a8", control_precision="fp32",
        core_counts=(1,), models=(model,), backends=backends,
        reporting={"performance_claims": {
            "parity_median_ratio_band": [0.95, 1.05],
            "win_median_ratio_max": 0.95,
            "win_requires_nonoverlapping_observed_ranges": True,
            "win_requires_causal_attribution": True,
            "language": "descriptive_ratio_not_statistical_significance",
        }},
        sha256=lambda: "a" * 64,
    )
    spec.matrix = lambda: tuple(
        SimpleNamespace(model=model, backend=backend, precision="fp32", core_count=1)
        for backend in backends)
    return spec


def _result(backend: str, samples, *, status="pass", attribution=None,
            scope="end_to_end", stages=("encode", "predict"),
            timed_stages=None, stage_samples=None):
    timed_stages = tuple(timed_stages if timed_stages is not None else stages)
    result = {
        "model": "holdout", "backend": backend, "precision": "fp32", "core_count": 1,
        "target": "k1", "artifact_sha256": "b" * 64,
        "session": {"stages": list(stages)},
        "lifecycle": {"status": status, "reason": None},
        "correctness": {"gate_ok": status == "pass"},
        "quality": {"gate_ok": status == "pass", "metric": "test", "value": 1.0},
        "timing": {"scope": scope, "sample_unit": "complete_session",
                   "timed_stages": list(timed_stages), "samples": list(samples),
                   "stage_samples": stage_samples or {},
                   "median": int(sorted(samples)[len(samples) // 2]) if samples else None},
        "execution": {"mode": "test", "requested_mode": "test", "fallback_used": False},
        "measurement_receipt": {"path": "/retained/test-receipt.yaml",
                                "sha256": "f" * 64, "aet_run_id": "test-run",
                                "command_sha256": "e" * 64},
    }
    if attribution is not None:
        result["causal_attribution"] = attribution
    return result


@pytest.fixture(autouse=True)
def _isolate_result_contract(monkeypatch):
    # The v2 cross-field validator has its own extensive tests.  These fixtures intentionally contain
    # only the fields read by the analysis layer, so its claim semantics can be tested in isolation.
    monkeypatch.setattr(report, "validate_paper_result", lambda result: None)
    def fake_roots(_spec, results):
        for result in results:
            if not isinstance(result.get("measurement_receipt"), dict):
                raise ValueError("measurement receipt is required")
        return [{"index": index, "cell": result["backend"],
                 "receipt_sha256": result["measurement_receipt"]["sha256"]}
                for index, result in enumerate(results)]
    monkeypatch.setattr(report, "_measurement_roots", fake_roots)


def _claim_ready_record(comparator: str, *, why="whole-model visibility removes a materialization boundary",
                        how="the frozen fusion pass keeps producer and consumer in one compiled region"):
    return {"comparator": comparator, "status": "available", "why": why, "how": how,
            "evidence": {"binding_sha256": "a" * 64, "ablation_sha256": "b" * 64,
                         "structural_sha256": "c" * 64}}


def _document(spec, *results):
    return report.seal_results_document(spec, list(results))


def _controller_result(tmp_path: Path, spec) -> tuple[Path, dict]:
    root = tmp_path / "controller-input"
    root.mkdir()
    source = root / "cell.c"
    source.write_text(r'''
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *artifact = fopen(argv[1], "rb"), *input = fopen(argv[2], "rb");
  FILE *output = fopen(argv[3], "wb");
  if (!artifact || !input || !output) return 3;
  volatile unsigned long value = (unsigned long)fgetc(artifact);
  int byte; while ((byte = fgetc(input)) != EOF) fputc(byte, output);
  fclose(artifact); fclose(input); fclose(output);
  for (unsigned long index = 1; index < 50000000; ++index) value = value * 33u + index;
  return (int)(value & 0u);
}
''', encoding="utf-8")
    executable = root / "cell"
    subprocess.run(["/usr/bin/clang", "-O2", "-std=c11", str(source), "-o", str(executable)],
                   check=True)
    tool = root / "cc"
    shutil.copy2(Path("/usr/bin/clang").resolve(), tool)
    tool.chmod(0o755)
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority
    authority = write_toolchain_authority(
        root / "toolchain-authority.json", authority_id="report-controller-test",
        target="unit-test", build_tool=tool)
    authority_document = json.loads(authority.read_text(encoding="utf-8"))
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"frozen model artifact\n")
    session_input = root / "session-input.bin"
    import struct
    session_input.write_bytes(
        b"MRLNFRM1" + struct.pack("<Q", 8) + b"frame-0!"
        + struct.pack("<Q", 8) + b"frame-1!")
    reference = root / "reference.bin"
    reference.write_bytes(session_input.read_bytes())
    probe = root / "paper_k1_board_probe.c"
    probe.write_bytes(Path(paper_ablation_generator.__file__).with_name(
        "paper_k1_board_probe.c").read_bytes())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = root / "session-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "kind": "paper_session_inputs_v1",
        "session_kind": "continuous", "observations": 2,
        "inputs": {"session_input": digest(session_input)},
        "records": [
            {"index": 0, "payload_sha256": hashlib.sha256(b"frame-0!").hexdigest()},
            {"index": 1, "payload_sha256": hashlib.sha256(b"frame-1!").hexdigest()},
        ],
    }), encoding="utf-8")
    io_receipt = root / "measurement-io-receipt.json"
    io_receipt.write_text("{}\n", encoding="utf-8")
    cell = {"model": "holdout", "backend": "merlin_frozen", "precision": "fp32",
            "core_count": 1}
    study_root = root / "frozen-study.yaml"
    study_root.write_text(yaml.safe_dump({
        "schema_version": 1, "kind": "paper_unit_test_study_root_v1", "status": "frozen",
        "study_sha256": spec.sha256(), "cell": cell}), encoding="utf-8")
    backend_template = root / "backend-template.yaml"
    backend_template.write_text("kind: unit_test_backend_template\n", encoding="utf-8")
    contract = {
        "schema_version": 2, "kind": "paper_measurement_contract_v2", "status": "ready",
        "registry_id": "unit_test_v1", "backend_adapter": "unit_test",
        "study_spec": {"path": study_root.name, "sha256": digest(study_root)},
        "backend_template": {"path": backend_template.name,
                             "sha256": digest(backend_template)},
        "study_sha256": spec.sha256(), "run_id": "test-run", "target": "unit-test",
        "result_identity": {"timestamp": "20260831T000000Z", "git_sha": "deadbee",
                            "study_label": "study", "target": "unit-test",
                            "model": "holdout", "checkpoint": "checkpoint", "fidelity": "full",
                            "backend": "merlin_frozen", "runtime": "merlin",
                            "precision": "fp32", "quantization": "none", "core_count": 1},
        "session": {"kind": "continuous", "warmups": 1, "observations": 2,
                    "stages": ["step"], "carried_state": [],
                    "parameters": {"paper_primary_scope": "end_to_end"},
                    "measurement_repeats": 5},
        "frozen_provenance": {"study_sha256": spec.sha256()},
        "cell": cell, "artifact_sha256": digest(artifact),
        "artifact": {"path": artifact.name, "sha256": digest(artifact)},
        "inputs": {"session_input": {"path": session_input.name,
                                        "sha256": digest(session_input)}},
        "session_manifest": {"path": manifest.name, "sha256": digest(manifest)},
        "measurement_io_receipt": {"path": io_receipt.name, "sha256": digest(io_receipt)},
        "reference_output": {"path": reference.name, "sha256": digest(reference)},
        "oracle": {"kind": "bytes_exact", "metric": "unit", "threshold": 1.0,
                   "scope": "trajectory", "steps": 2},
        "build": {
            "study_sha256": spec.sha256(), "cell": cell,
            "frozen_provenance_sha256": hashlib.sha256(json.dumps(
                {"study_sha256": spec.sha256()}, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "model_artifact_sha256": digest(artifact), "source_identity_sha256": "b" * 64,
            "package_identity_sha256": "c" * 64,
            "expected_executable_sha256": digest(executable),
            "tool": {"path": tool.name, "sha256": digest(tool)},
            "toolchain_authority": {"path": authority.name, "sha256": digest(authority)},
            "build_tool_identity_sha256": authority_document["tool"]["identity_sha256"],
            "sources": {"cell": {"path": source.name, "sha256": digest(source)}},
            "inputs": {},
            "argv": ["{tool}", "-O2", "-std=c11", "{source:cell}", "-o", "{output}"],
            "cwd": ".", "environment": {"PATH": "/usr/bin:/bin"}, "timeout_seconds": 10},
        "argv": ["{executable}", "{artifact}", "{input:session_input}", "{observation}"],
        "cwd": ".", "environment": {},
        "execution": {"mode": "unit", "core_ids": [min(os.sched_getaffinity(0))],
                      "require_worker_threads": False},
        "memory_policy": "resident", "timeout_seconds": 10,
        "warmup_iterations": 1, "measured_iterations": 5,
        "timing": {"unit": "ns", "sample_unit": "complete_session", "scope": "end_to_end",
                   "timed_stages": ["step"], "excluded_stages": [], "stage_samples": {}},
        "board_probe_source": {"path": probe.name, "sha256": digest(probe)},
    }
    contract_path = root / "measurement.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    receipt_path = produce_receipt(contract_path, tmp_path / "controller-bundle")
    raw = json.loads((receipt_path.parent / "raw_measurement.json").read_text(encoding="utf-8"))
    result = {"run_id": "test-run", **contract["result_identity"],
              "artifact_sha256": digest(artifact),
              **{field: raw[field] for field in (
                  "session", "lifecycle", "correctness", "quality", "memory", "timing",
                  "execution", "provenance")}}
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    result["measurement_receipt"] = {
        "path": str(receipt_path), "sha256": digest(receipt_path),
        "aet_run_id": result["run_id"], "command_sha256": receipt["command_sha256"],
    }
    return receipt_path, result


def test_win_requires_pass_nonoverlap_and_frozen_comparator_specific_why_how(monkeypatch):
    ours = _backend("merlin_frozen", "compiler")
    xnn = _backend("executorch_xnnpack", "external_runtime")
    spec = _spec(ours, xnn)
    monkeypatch.setattr(report, "_evidence_causal_record",
                        lambda *_args: _claim_ready_record("executorch_xnnpack"))
    ours_result = _result("merlin_frozen", [70, 75, 80, 85, 90], attribution=[{
        "comparator": "executorch_xnnpack",
        "why": "whole-model visibility removes a materialization boundary",
        "how": "the frozen fusion pass keeps producer and consumer in one compiled region",
    }])
    other = _result("executorch_xnnpack", [100, 105, 110, 115, 120])

    generated = report.build_paper_report(
        spec, _document(spec, ours_result, other))
    comparison = generated["primary_end_to_end"]["rows"][0]["comparisons"][0]

    assert comparison["label"] == "win"
    assert comparison["e2e_win_claim"] is True
    assert comparison["ratio_ours_over_comparator"] == pytest.approx(80 / 110)
    assert comparison["causal_attribution"]["why"]
    assert comparison["causal_attribution"]["how"]


@pytest.mark.parametrize("ours_result, expected_note", [
    (_result("merlin_frozen", [70, 75, 80, 85, 90]), "causal attribution"),
    (_result("merlin_frozen", [50, 50, 50, 50, 110], attribution=[{
        "comparator": "executorch_xnnpack", "why": "fusion", "how": "one pass"}]),
     "observed ranges overlap"),
])
def test_a_low_median_without_all_win_gates_is_not_called_a_win(ours_result, expected_note):
    spec = _spec(_backend("merlin_frozen", "compiler"),
                 _backend("executorch_xnnpack", "external_runtime"))
    other = _result("executorch_xnnpack", [100, 105, 110, 115, 120])

    comparison = report.build_paper_report(
        spec, _document(spec, ours_result, other)
    )["primary_end_to_end"]["rows"][0]["comparisons"][0]

    assert comparison["label"] == "advantage_not_claimable"
    assert comparison["e2e_win_claim"] is False
    assert any(expected_note in note for note in comparison["claim_notes"])


def test_labels_parity_and_loss_from_frozen_median_ratio_band():
    ours_backend = _backend("merlin_frozen", "compiler")
    xnn = _backend("merlin_xnnpack", "kernel_swap")
    et = _backend("executorch_xnnpack", "external_runtime")
    spec = _spec(ours_backend, xnn, et)
    ours = _result("merlin_frozen", [98, 99, 100, 101, 102])
    parity = _result("merlin_xnnpack", [100, 101, 102, 103, 104])
    faster_comparator = _result("executorch_xnnpack", [75, 78, 80, 82, 85])

    comparisons = report.build_paper_report(
        spec, _document(spec, ours, parity, faster_comparator)
    )["primary_end_to_end"]["rows"][0]["comparisons"]

    assert {row["comparator"]: row["label"] for row in comparisons} == {
        "merlin_xnnpack": "parity", "executorch_xnnpack": "loss"}


def test_shared_threshold_boundary_can_be_a_win_only_when_extra_gates_pass(monkeypatch):
    spec = _spec(_backend("merlin_frozen", "compiler"),
                 _backend("executorch_xnnpack", "external_runtime"))
    monkeypatch.setattr(report, "_evidence_causal_record",
                        lambda *_args: _claim_ready_record("executorch_xnnpack",
                                                           why="less traffic", how="fusion"))
    ours = _result("merlin_frozen", [95] * 5, attribution=[{
        "comparator": "executorch_xnnpack", "why": "less traffic", "how": "fusion"}])
    other = _result("executorch_xnnpack", [100] * 5)
    comparison = report.build_paper_report(
        spec, _document(spec, ours, other)
    )["primary_end_to_end"]["rows"][0]["comparisons"][0]

    assert comparison["ratio_ours_over_comparator"] == 0.95
    assert comparison["label"] == "win"


def test_free_form_result_why_how_cannot_bypass_frozen_evidence_gate():
    spec = _spec(_backend("merlin_frozen", "compiler"),
                 _backend("executorch_xnnpack", "external_runtime"))
    ours = _result("merlin_frozen", [70, 75, 80, 85, 90], attribution=[{
        "comparator": "executorch_xnnpack", "status": "available",
        "why": "fabricated from the timing number", "how": "also fabricated",
        "evidence": {"binding_sha256": "a" * 64, "ablation_sha256": "b" * 64,
                     "structural_sha256": "c" * 64},
    }])
    other = _result("executorch_xnnpack", [100, 105, 110, 115, 120])

    comparison = report.build_paper_report(
        spec, _document(spec, ours, other)
    )["primary_end_to_end"]["rows"][0]["comparisons"][0]

    assert comparison["label"] == "advantage_not_claimable"
    assert "causal attribution" in comparison["claim_notes"][0]


def test_all_applicable_baselines_and_missing_not_run_cells_are_visible():
    backends = (
        _backend("merlin_frozen", "compiler"),
        _backend("hand_v0", "frozen_baseline"),
        _backend("merlin_openblas", "kernel_swap"),
        _backend("merlin_xnnpack", "kernel_swap"),
        _backend("executorch_xnnpack", "external_runtime"),
    )
    spec = _spec(*backends)
    ours = _result("merlin_frozen", [], status="not_run")
    generated = report.build_paper_report(spec, _document(spec, ours))
    row = generated["primary_end_to_end"]["rows"][0]

    assert row["ours"]["status"] == "not_run"
    assert {comparison["comparator"] for comparison in row["comparisons"]} == {
        "hand_v0", "merlin_openblas", "merlin_xnnpack", "executorch_xnnpack"}
    assert all(comparison["label"] == "not_comparable" for comparison in row["comparisons"])
    assert generated["coverage"]["not_run_cells"] == ["holdout/merlin_frozen/fp32/1c"]
    assert len(generated["coverage"]["missing_cells"]) == 4


def test_kernel_swap_coverage_is_explicit_in_report_and_markdown():
    spec = _spec(
        _backend("merlin_frozen", "compiler"),
        _backend("merlin_xnnpack", "kernel_swap"),
    )
    ours = _result("merlin_frozen", [100, 101, 102])
    xnn = _result("merlin_xnnpack", [90, 91, 92])
    xnn["execution"].update(n_candidates=7, n_eligible=5, n_routed=5)

    generated = report.build_paper_report(spec, _document(spec, ours, xnn))

    assert generated["kernel_swap_coverage"]["rows"] == [{
        "model": "holdout", "backend": "merlin_xnnpack", "precision": "fp32",
        "core_count": 1, "candidates": 7, "eligible": 5, "routed": 5,
        "eligible_coverage": 1.0, "complete_eligible_coverage": True,
        "status": "pass",
    }]
    markdown = report.render_markdown(generated)
    assert "Kernel-swap routing coverage" in markdown
    assert "| holdout | merlin_xnnpack | fp32 | 1 | 7 | 5 | 5 | 100.0% | pass |" in markdown


def test_stage_samples_and_subset_results_are_diagnostic_only():
    spec = _spec(_backend("merlin_frozen", "compiler"))
    e2e = _result("merlin_frozen", [100, 110, 120],
                  stage_samples={"encode": [30, 35, 40], "predict": [50, 55, 60]})
    subset = _result("merlin_frozen", [45, 50, 55], scope="stage_subset",
                     timed_stages=("predict",))

    generated = report.build_paper_report(
        spec, _document(spec, e2e, subset))

    assert generated["primary_end_to_end"]["rows"][0]["ours"]["median_ns"] == 110
    diagnostics = generated["stage_diagnostics"]
    assert "never used" in diagnostics["scope"]
    assert {row["stage"] for row in diagnostics["rows"]} == {"encode", "predict"}
    assert all(row["claim_eligible"] is False for row in diagnostics["rows"])
    markdown = report.render_markdown(generated)
    assert "Primary: end-to-end continuous sessions" in markdown
    assert "Stage diagnostics (not end-to-end claims)" in markdown


def test_thresholds_are_required_and_explicit():
    with pytest.raises(ValueError, match="performance_claims"):
        report.PerformanceClaims.parse({})
    with pytest.raises(ValueError, match="non-overlapping"):
        report.PerformanceClaims.parse({"performance_claims": {
            "parity_median_ratio_band": [0.95, 1.05], "win_median_ratio_max": 0.95,
            "win_requires_nonoverlapping_observed_ranges": False,
            "win_requires_causal_attribution": True,
            "language": "descriptive_ratio_not_statistical_significance",
        }})


def test_unsupported_comparisons_remain_explicit_in_report_and_markdown():
    spec = _spec(_backend("merlin_frozen", "compiler"))
    spec.reporting["unsupported_comparisons"] = [{
        "backend": "executorch_xnnpack", "precision": "w8a8",
        "status": "not_implemented", "reason": "quantized runtime path is unavailable",
    }]
    generated = report.build_paper_report(
        spec, _document(spec, _result("merlin_frozen", [100, 110, 120])))

    assert generated["unsupported_comparisons"] == spec.reporting["unsupported_comparisons"]
    markdown = report.render_markdown(generated)
    assert "Unsupported comparisons" in markdown
    assert "executorch_xnnpack/w8a8" in markdown
    assert "quantized runtime path is unavailable" in markdown


def test_report_rejects_timing_mutation_after_results_are_content_sealed():
    spec = _spec(_backend("merlin_frozen", "compiler"),
                 _backend("executorch_xnnpack", "external_runtime"))
    ours = _result("merlin_frozen", [98, 99, 100, 101, 102])
    comparator = _result("executorch_xnnpack", [100, 101, 102, 103, 104])
    document = report.seal_results_document(spec, [ours, comparator])
    changed = copy.deepcopy(document)
    timing = changed["results"][0]["timing"]
    timing.update(samples=[1, 1, 1, 1, 1], median=1)

    with pytest.raises(ValueError, match="content seal"):
        report.build_paper_report(spec, changed)


def test_report_retains_the_verified_measurement_results_seal():
    spec = _spec(_backend("merlin_frozen", "compiler"))
    document = report.seal_results_document(
        spec, [_result("merlin_frozen", [98, 99, 100, 101, 102])])

    generated = report.build_paper_report(spec, document)

    assert generated["results_content_seal"] == document["content_seal"]


def test_results_cannot_be_sealed_without_retained_run_measurement_receipt():
    spec = _spec(_backend("merlin_frozen", "compiler"))
    result = _result("merlin_frozen", [98, 99, 100, 101, 102])
    del result["measurement_receipt"]

    with pytest.raises(ValueError, match="measurement receipt|AET|raw measurement"):
        report.seal_results_document(spec, [result])


def test_result_contract_rejects_unknown_nested_agentic_fields(monkeypatch):
    monkeypatch.undo()
    from merlin.compare.paper import validate_paper_result

    # The full validator is exercised elsewhere; this attack demonstrates that provenance must
    # not be an open bag in which an agent can insert an authority claim.
    result = {
        "schema_version": 2, "run_id": "r", "timestamp": "t", "git_sha": "deadbee",
        "study_label": "s", "target": "k1", "model": "m", "checkpoint": "c",
        "artifact_sha256": "a" * 64, "fidelity": "full", "backend": "b",
        "runtime": "r", "precision": "fp32", "quantization": "none", "core_count": 1,
        "session": {"kind": "image_stream", "warmups": 0, "observations": 1,
                    "measurement_repeats": 1, "stages": ["step"], "carried_state": [],
                    "parameters": {}},
        "lifecycle": {"built": False, "ran": False, "status": "not_run", "reason": "test"},
        "correctness": {"gate_ok": False},
        "quality": {"gate_ok": False, "metric": "x", "value": None},
        "timing": {"unit": "ns", "sample_unit": "complete_session", "scope": "end_to_end",
                   "timed_stages": ["step"], "excluded_stages": [], "samples": [],
                   "stage_samples": {}, "median": None, "p95": None},
        "memory": {"policy": None, "peak_rss_bytes": None},
        "execution": {"mode": None, "requested_mode": None, "fallback_used": False},
        "provenance": {"study_sha256": "b" * 64, "compiler_policy_sha256": "c" * 64,
                       "compiler_source_sha256": "d" * 64, "runtime_sha256": "e" * 64,
                       "capture_session_identity_sha256": "f" * 64,
                       "vlen_bits": None, "vlen_source": None, "agentic": True},
    }

    with pytest.raises(ValueError, match="provenance.*unrecognized|closed"):
        validate_paper_result(result)
    del result["provenance"]["agentic"]
    result["session"]["parameters"]["agentic"] = True
    with pytest.raises(ValueError, match="session.parameters.*unrecognized|closed"):
        validate_paper_result(result)


def test_results_seal_is_rooted_in_retained_raw_and_aet_receipts(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(report, "validate_paper_result", lambda result: None)
    spec = _spec(_backend("merlin_frozen", "compiler"))
    receipt, result = _controller_result(tmp_path, spec)
    raw = receipt.parent / "raw_measurement.json"
    document = report.seal_results_document(spec, [result])

    assert document["content_seal"]["schema_version"] == 3
    assert document["content_seal"]["measurement_roots"][0][
        "raw_measurement_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()

    raw.write_text('{"samples_ns":[1]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="producer capability|raw measurement"):
        report._verify_results_seal(spec, document)


def test_results_seal_rejects_fabricated_raw_samples_after_all_hashes_are_refreshed(
        tmp_path, monkeypatch):
    """Raw/result reconciliation, not digest self-consistency, is the seal authority."""
    monkeypatch.undo()
    spec = _spec(_backend("merlin_frozen", "compiler"))
    result = _result("merlin_frozen", [98, 99, 100, 101, 102])
    result["run_id"] = "forged-run"
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({
        "schema_version": 3, "kind": "paper_cell_raw_measurement_v3", "status": "complete",
        "run_id": "forged-run", "study_sha256": spec.sha256(),
        "cell": {"model": "holdout", "backend": "merlin_frozen", "precision": "fp32",
                 "core_count": 1},
        "driver": {"id": "merlin.paper.cell_driver_v1", "argv": ["/bin/false"],
                   "command_sha256": report._canonical_sha256(["/bin/false"]), "exit_code": 0,
                   "started_unix_ns": 1, "ended_unix_ns": 2},
        "artifact_sha256": result["artifact_sha256"], "lifecycle": result["lifecycle"],
        "correctness": result["correctness"], "quality": result["quality"],
        "timing": {**result["timing"], "samples": [1, 1, 1], "median": 1},
        "execution": result["execution"],
    }, sort_keys=True), encoding="utf-8")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    # Even a fully refreshed outer receipt cannot make divergent raw samples authoritative.
    receipt = tmp_path / "receipt.yaml"
    receipt.write_text(yaml.safe_dump({
        "schema_version": 3, "kind": "paper_cell_measurement_receipt_v3",
        "status": "finalized", "aet_run_id": "forged-run", "study_sha256": spec.sha256(),
        "cell": {"model": "holdout", "backend": "merlin_frozen", "precision": "fp32",
                 "core_count": 1},
        "command_sha256": report._canonical_sha256(["/bin/false"]),
        "raw_measurement": {"path": raw.name, "sha256": raw_sha},
        "aet_lifecycle": {"run_record": {"path": "false", "sha256": "0" * 64},
                          "events": {"path": "false.events", "sha256": "0" * 64}},
    }), encoding="utf-8")
    result["measurement_receipt"] = {
        "path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "aet_run_id": "forged-run",
        "command_sha256": report._canonical_sha256(["/bin/false"]),
    }

    with pytest.raises(ValueError, match="producer capability|controller measurement receipt"):
        report.seal_results_document(spec, [result])


def test_results_document_and_seal_are_closed(monkeypatch):
    spec = _spec(_backend("merlin_frozen", "compiler"))
    document = _document(spec, _result("merlin_frozen", [98, 99, 100, 101, 102]))
    document["agentic"] = True
    with pytest.raises(ValueError, match="results document.*closed"):
        report.build_paper_report(spec, document)
    del document["agentic"]
    document["content_seal"]["narrative"] = "trust me"
    with pytest.raises(ValueError, match="results content seal.*closed"):
        report.build_paper_report(spec, document)
