"""The performance phase starts from one explicit, frozen Arm-4 compiler or not at all.

These tests pin the boundary that keeps a performance run attributable: no "latest" lookup, no live
submission directory, no vacuous 0/0 completion, and no untrusted entrypoint outside the derived bwrap
policy.  The expensive simulator is not launched here; the mount table and completion arithmetic are
pure and therefore exercise the refusal paths in CI.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root


def _load_campaign():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_perf_campaign_gate", scripts / "perf_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PC = _load_campaign()


def _load_runner():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_run_perf_bench_gate", scripts / "run_perf_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PR = _load_runner()


def _functional_run(tmp_path: Path, run_id: str = "arm4_explicit") -> tuple[Path, str]:
    run = tmp_path / "merlin_assisted" / run_id
    sub = run / "submission"
    sub.mkdir(parents=True)
    (sub / "manifest.yaml").write_text(yaml.safe_dump({
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "entrypoints": {"tool": "tool.py"},
        "commands": {},
    }))
    (sub / "tool.py").write_text("print('fixture')\n")
    digest = hash_tree(sub)["sha256"]

    (run / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "bundle_id": "merlin_assisted_rtlchecks_hwbringup_v0",
        "sandbox": "bwrap",
        "bundle_input_snapshot": {
            "version": 2, "content_sha256": "a" * 64, "n_files": 7, "n_bytes": 41,
        },
        "isolation_violations": [],
        "golden_mask_selftest": {"n_answer_files_masked": 3, "leaked_answer_files": []},
    }))
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump({
        "converged": True,
        "rounds": [{"answer_access_clean": True, "audit_hits": []}],
        "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True},
    }))
    (run / "freeze.json").write_text(json.dumps({
        "submission_sha256": digest, "submission_sha256_recheck": digest,
        "workspace_mutable_after_freeze": False, "frozen_at": "2026-08-31T00:00:00Z",
    }))
    (run / "run_manifest.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "submission_sha256": digest,
        "integrity_status": "clean",
        "integrity_exempt": False,
        "gradeable": True,
        "public_dev": {"functional_pass": 1, "passed": "2/2", "highest_tier": "L3"},
        "hidden": {"functional_pass": 1, "passed": "1/1"},
    }))
    for phase, names in (("public", ("p0", "p1")), ("hidden", ("h0",))):
        d = run / f"grading_{phase}"
        d.mkdir()
        rows = [{"capsule": n, "status": "pass", "tiers": {"L2": "pass", "L3": "pass"}}
                for n in names]
        (d / "score_capsule.json").write_text(json.dumps({
            "n_capsules": len(rows), "n_passed": len(rows), "functional_pass": 1,
            "gradeable": True, "integrity_status": "clean", "integrity_exempt": False,
            "per_capsule": rows,
        }))
    return run, digest


def test_one_exact_arm4_run_and_digest_are_required(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert rec.run_dir == run.resolve()
    assert rec.digest == digest
    assert rec.public_capsules == 2 and rec.hidden_capsules == 1

    with pytest.raises(PC.CampaignGateError, match="explicit functional run id"):
        PC.inspect_functional_run(tmp_path, "", digest)
    with pytest.raises(PC.CampaignGateError, match="simple directory name"):
        PC.inspect_functional_run(tmp_path, "../arm4_explicit", digest)
    with pytest.raises(PC.CampaignGateError, match="does not match"):
        PC.inspect_functional_run(tmp_path, run.name, "0" * 64)


def test_a_non_arm4_or_vacuous_functional_run_is_refused(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    env = yaml.safe_load((run / "environment.yaml").read_text())
    env["bundle_id"] = "merlin_assisted_hwbringup_v0"
    (run / "environment.yaml").write_text(yaml.safe_dump(env))
    with pytest.raises(PC.CampaignGateError, match="Arm-4 RTL-checks bundle"):
        PC.inspect_functional_run(tmp_path, run.name, digest)

    run, digest = _functional_run(tmp_path, "arm4_zero")
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    manifest["hidden"]["passed"] = "0/0"
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(PC.CampaignGateError, match="non-vacuous"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_functional_run_must_have_frozen_bundle_inputs_v2(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    environment = yaml.safe_load((run / "environment.yaml").read_text())
    environment.pop("bundle_input_snapshot")
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(PC.CampaignGateError, match="immutable bundle-input snapshot v2"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_the_perf_workspace_is_a_copy_and_is_digest_checked(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    snapshot = PC.materialize_perf_workspace(rec, tmp_path / "perf")
    assert snapshot != rec.submission_dir
    assert hash_tree(snapshot)["sha256"] == digest

    (rec.submission_dir / "tool.py").write_text("print('functional tree moved later')\n")
    assert hash_tree(snapshot)["sha256"] == digest
    assert "fixture" in (snapshot / "tool.py").read_text()


def test_digest_excluded_submission_state_is_refused(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    cache = run / "submission" / "__pycache__"
    cache.mkdir()
    (cache / "tool.pyc").write_bytes(b"unhashed executable bytes")
    with pytest.raises(PC.CampaignGateError, match="digest-excluded path"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_perf_fork_detects_snapshot_drift(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    snapshot = PC.materialize_perf_workspace(rec, tmp_path / "perf")
    fork = PC.functional_fork(rec)
    held = PC.check_fork(fork, snapshot)
    assert held.ok is True and held.state == "held"

    # The host can still change bytes; the bwrap mount is the runtime boundary and this digest check is
    # the before/after backstop. Restore one write bit to model a bug in that boundary.
    tool = snapshot / "tool.py"
    tool.chmod(0o644)
    tool.write_text("print('mutated')\n")
    assert PC.check_fork(fork, snapshot).ok is None


def test_completion_is_non_vacuous_and_every_expected_cell_must_finish() -> None:
    with pytest.raises(PC.CampaignGateError, match="zero expected"):
        PC.completion_counts([], ())

    expected = (
        PC.PerfCell("PC", "pc_k4", "spike", "r000"),
        PC.PerfCell("PC", "pc_k4", "verilator", "r000"),
    )
    complete = [
        {"family": "PC", "capsule": "pc_k4", "simulator": "spike", "replicate": "r000",
         "correct": True, "cycles": None},
        {"family": "PC", "capsule": "pc_k4", "simulator": "verilator", "replicate": "r000",
         "correct": True, "cycles": 9,
         "provenance": {"tier": "L3", "simulator": "verilator",
                        "derived_from_rtl": True, "cycle_accurate": True}},
    ]
    counts = PC.completion_counts(complete, expected)
    assert counts == {"expected": 2, "reported": 2, "correct": 2, "cycles_measured": 1,
                      "failed": 0, "missing": 0, "complete": True}


def test_completion_refuses_a_missing_exact_identity() -> None:
    expected = (
        PC.PerfCell("PC", "pc_k4", "spike", "r000"),
        PC.PerfCell("PC", "pc_k4", "verilator", "r000"),
    )
    spike_only = [
        {"family": "PC", "capsule": "pc_k4", "simulator": "spike", "replicate": "r000",
         "correct": True, "cycles": None},
    ]
    assert PC.completion_report(spike_only, expected) == {
        "expected": 2, "reported": 1, "correct": 1, "cycles_measured": 0,
        "failed": 0, "missing": 1, "complete": False,
    }
    with pytest.raises(PC.CampaignGateError, match="1 of 2 expected"):
        PC.completion_counts(spike_only, expected)


def test_completion_refuses_a_duplicate_reported_identity() -> None:
    identity = PC.PerfCell("PC", "pc_k4", "spike", "r000")
    row = {"family": "PC", "capsule": "pc_k4", "simulator": "spike", "replicate": "r000",
           "correct": True, "cycles": None}
    with pytest.raises(PC.CampaignGateError, match="repeat cell identity"):
        PC.completion_report([row, dict(row)], (identity,))


def test_completion_refuses_an_unexpected_reported_identity() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "spike", "r000"),)
    unexpected = {
        "family": "PC", "capsule": "pc_k8", "simulator": "spike", "replicate": "r000",
        "correct": True, "cycles": None,
    }
    with pytest.raises(PC.CampaignGateError, match="unexpected cell identities"):
        PC.completion_report([unexpected], expected)


def test_completion_refuses_a_duplicate_expected_identity() -> None:
    identity = PC.PerfCell("PC", "pc_k4", "spike", "r000")
    with pytest.raises(PC.CampaignGateError, match="expected cell identities contain duplicates"):
        PC.completion_report([], (identity, identity))


def test_completion_refuses_the_old_kernel_simulator_identity_shape() -> None:
    with pytest.raises(PC.CampaignGateError, match="expected identity must be a PerfCell"):
        PC.completion_report([], {"k0": ("spike", "verilator")})


def test_completion_refuses_a_report_without_the_full_exact_identity() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "spike", "r000"),)
    incomplete = {"family": "PC", "capsule": "pc_k4", "simulator": "spike",
                  "correct": True, "cycles": None}
    with pytest.raises(PC.CampaignGateError, match="invalid reported performance identity"):
        PC.completion_report([incomplete], expected)


def test_completion_does_not_coerce_identity_fields() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "spike", "1"),)
    coerced = {"family": "PC", "capsule": "pc_k4", "simulator": "spike",
               "replicate": 1, "correct": True, "cycles": None}
    with pytest.raises(PC.CampaignGateError, match="invalid reported performance identity"):
        PC.completion_report([coerced], expected)


def test_spike_correctness_screen_accepts_absent_cycles() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "spike", "r000"),)
    spike = {"family": "PC", "capsule": "pc_k4", "simulator": "spike",
             "replicate": "r000", "correct": True}
    assert PC.completion_counts([spike], expected)["complete"] is True


def test_spike_cycles_are_refused_as_performance_evidence() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "spike", "r000"),)
    timed_spike = {"family": "PC", "capsule": "pc_k4", "simulator": "spike",
                   "replicate": "r000", "correct": True, "cycles": 7}
    report = PC.completion_report([timed_spike], expected)
    assert report["cycles_measured"] == 0
    assert report["failed"] == 1 and report["complete"] is False


@pytest.mark.parametrize("simulator", ["spike", "verilator"])
def test_every_simulator_cell_requires_correctness(simulator: str) -> None:
    expected = (PC.PerfCell("PC", "pc_k4", simulator, "r000"),)
    row = {"family": "PC", "capsule": "pc_k4", "simulator": simulator,
           "replicate": "r000", "correct": False,
           "cycles": None if simulator == "spike" else 9,
           "provenance": {"tier": "L3", "simulator": "verilator",
                          "derived_from_rtl": True, "cycle_accurate": True}}
    assert PC.completion_report([row], expected)["complete"] is False


@pytest.mark.parametrize("cycles", [None, 0, -1, True, 1.5])
def test_verilator_cycles_must_be_a_positive_integer(cycles: object) -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "verilator", "r000"),)
    row = {"family": "PC", "capsule": "pc_k4", "simulator": "verilator",
           "replicate": "r000", "correct": True, "cycles": cycles,
           "provenance": {"tier": "L3", "simulator": "verilator",
                          "derived_from_rtl": True, "cycle_accurate": True}}
    assert PC.completion_report([row], expected)["complete"] is False


@pytest.mark.parametrize(("field", "value"), [
    ("tier", "L2"),
    ("simulator", "spike"),
    ("derived_from_rtl", False),
    ("cycle_accurate", False),
])
def test_verilator_timing_requires_cycle_accurate_rtl_provenance(field: str, value: object) -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "verilator", "r000"),)
    provenance = {"tier": "L3", "simulator": "verilator",
                  "derived_from_rtl": True, "cycle_accurate": True}
    provenance[field] = value
    row = {"family": "PC", "capsule": "pc_k4", "simulator": "verilator",
           "replicate": "r000", "correct": True, "cycles": 9,
           "provenance": provenance}
    report = PC.completion_report([row], expected)
    assert report["cycles_measured"] == 1
    assert report["failed"] == 1 and report["complete"] is False


def test_verilator_timing_without_provenance_is_not_complete() -> None:
    expected = (PC.PerfCell("PC", "pc_k4", "verilator", "r000"),)
    row = {"family": "PC", "capsule": "pc_k4", "simulator": "verilator",
           "replicate": "r000", "correct": True, "cycles": 9}
    assert PC.completion_report([row], expected)["complete"] is False


def test_fixed_profiler_builds_the_exact_27_spike_and_9_verilator_cells() -> None:
    expected = PR._expected_cells(PR._selected_corpus("all"), "auto")
    assert len(expected) == 36
    assert sum(cell.simulator == "spike" for cell in expected) == 27
    assert sum(cell.simulator == "verilator" for cell in expected) == 9
    assert {cell.family for cell in expected} == {"fixed_profile"}
    assert {cell.replicate for cell in expected} == {"r000"}
    assert len({cell.capsule for cell in expected}) == 27
    assert PR.PC.completion_report([], expected) == {
        "expected": 36, "reported": 0, "correct": 0, "cycles_measured": 0,
        "failed": 0, "missing": 36, "complete": False,
    }


def test_fixed_profiler_projects_tier_results_to_exact_completion_rows() -> None:
    arm = {"per_sim": {
        "spike": {"correct": True, "cycles": 71,
                  "provenance": {"tier": "L2", "simulator": "spike",
                                 "derived_from_rtl": False, "cycle_accurate": False}},
        "verilator": {"correct": True, "cycles": 109,
                      "provenance": {"tier": "L3", "simulator": "verilator",
                                     "derived_from_rtl": True, "cycle_accurate": True}},
    }}
    rows = PR._completion_rows("G00", arm, ("spike", "verilator"))
    assert rows == [
        {"family": "fixed_profile", "capsule": "G00", "simulator": "spike",
         "replicate": "r000", "correct": True, "cycles": None,
         "provenance": arm["per_sim"]["spike"]["provenance"]},
        {"family": "fixed_profile", "capsule": "G00", "simulator": "verilator",
         "replicate": "r000", "correct": True, "cycles": 109,
         "provenance": arm["per_sim"]["verilator"]["provenance"]},
    ]
    expected = PR._expected_cells([{"id": "G00", "sim_hint": "L2+L3"}], "auto")
    assert PR.PC.completion_counts(rows, expected)["complete"] is True


def test_fixed_profiler_copies_tier_provenance_into_simulator_results(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command_buffer = {"tensors": {}, "commands": []}
    monkeypatch.setattr(PR.CR, "load_capsule", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(PR.CR, "default_adapters", lambda: {"L2": object(), "L3": object()})
    monkeypatch.setattr(PR.CR, "run_capsule", lambda *_args, **_kwargs: {
        "status": "pass", "numeric": {"status": "pass"},
        "work_volume": {"exact_macs": 4096, "artifact_sha256": "a" * 64,
                        "basis": "compiler_command_buffer"},
        "command_buffer_artifact": {
            "command_buffer": command_buffer, "artifact_sha256": PR._canonical_sha256(command_buffer),
            "compiler_provenance": "fixture compiler"},
        "tiers": {
            "L2": {"status": "pass", "cycles": 71, "derived_from_rtl": False,
                   "cycle_accurate": False, "evidence": "spike.log"},
            "L3": {"status": "pass", "cycles": 109, "derived_from_rtl": True,
                   "cycle_accurate": True, "evidence": "verilator.log"},
        },
    })
    result = PR.run_arm4(
        tmp_path, {"id": "G00", "macs": 4096}, tmp_path,
        ("spike", "verilator"), tmp_path, 30, "gemmini",
        rtl_identity={"rtl_facts": {"sha256": "e" * 64},
                      "circt_core_hw": {"path": "/fixture/core.mlir", "sha256": "f" * 64}})
    assert result["per_sim"]["spike"]["cycles"] is None
    assert result["per_sim"]["spike"]["correctness_cycles"] == 71
    assert result["per_sim"]["verilator"]["provenance"] == {
        "tier": "L3", "simulator": "verilator", "derived_from_rtl": True,
        "cycle_accurate": True, "evidence": "verilator.log",
    }
    assert result["per_sim"]["verilator"]["achieved_macs_per_cycle"] == 4096 / 109
    assert result["command_buffer_artifact"]["command_buffer"] == command_buffer
    assert result["rtl_facts_sha256"] == "e" * 64
    assert result["per_sim"]["verilator"]["rtl_facts_sha256"] == "e" * 64


def _counter_pass_fixture(kind: str) -> dict:
    digest = "a" * 64
    identity = {
        "program": {"kind": "compiler_command_buffer", "sha256": "b" * 64},
        "inputs": {"kind": "frozen_capsule_tree", "sha256": "c" * 64},
        "toolchain": {"target": "gemmini", "frozen_submission_sha256": "d" * 64,
                      "recorded_revisions": {"merlin": "revision-1"}},
    }
    report = {
        "discovery": {"status": "derived", "header_sha256": digest},
        "capacity": {"status": "derived", "slots": 8,
                     "provenance": {"source": "rtl.hw.mlir", "sha256": "f" * 64}},
        "measured_header_sha256": digest,
    }
    if kind == "occupancy":
        report.update({
            "selection": {"kind": "joint_occupancy", "unit": None},
            "occupancy": {"by_combination": {"load": "TARGET_LOAD_CYCLES"}},
            "readings": {"TARGET_LOAD_CYCLES": 17},
        })
    else:
        report.update({
            "selection": {"kind": "unit", "unit": "BYTES"},
            "selected_counters": {"TARGET_DMA_BYTES": 41},
            "readings": {"TARGET_DMA_BYTES": 256},
        })
    return {
        "approach": "arm4",
        "measurement_identity": identity,
        "measurement_identity_refusals": [],
        "per_sim": {"verilator": {
            "correct": True, "cycles": 109,
            "measurement_conditions": {"cycle_window": "accelerator_region"},
            "counters": report,
        }},
    }


def test_counter_passes_link_only_as_raw_named_readings() -> None:
    linked = PR._link_counter_passes(
        _counter_pass_fixture("occupancy"), _counter_pass_fixture("physical_bytes"),
        physical_unit="BYTES")
    assert linked["status"] == "linked" and linked["refusals"] == []
    physical = linked["physical_byte_counters"]
    assert physical == {
        "unit_family": "BYTES",
        "semantic_resolution": "raw_named_readings_only",
        "selected_counters": {"TARGET_DMA_BYTES": 41},
        "readings": {"TARGET_DMA_BYTES": 256},
        "binding_status": "no counter-byte binding probe was supplied",
    }
    assert "total" not in physical and "read_bytes" not in physical and "write_bytes" not in physical


def test_counter_pass_linkage_refuses_identity_drift_and_missing_readings() -> None:
    occupancy = _counter_pass_fixture("occupancy")
    physical = _counter_pass_fixture("physical_bytes")
    physical["measurement_identity"]["program"]["sha256"] = "e" * 64
    physical["per_sim"]["verilator"]["counters"]["readings"] = {}
    linked = PR._link_counter_passes(occupancy, physical, physical_unit="BYTES")
    assert linked["status"] == "refused"
    assert linked["measurement_identity"] is None
    assert any("identities differ" in reason for reason in linked["refusals"])
    assert any("no raw named counter readings" in reason for reason in linked["refusals"])


def test_counter_pass_cycle_perturbation_is_recorded_not_used_as_a_linkage_failure() -> None:
    occupancy = _counter_pass_fixture("occupancy")
    physical = _counter_pass_fixture("physical_bytes")
    physical["per_sim"]["verilator"]["cycles"] = 106

    linked = PR._link_counter_passes(occupancy, physical, physical_unit="BYTES")

    assert linked["status"] == "linked" and linked["refusals"] == []
    assert linked["cycle_windows"] == {
        "occupancy": 109, "physical_bytes": 106, "instrumentation_delta": -3}


def test_counter_link_attaches_only_nonempty_exhaustive_exact_rtl_byte_facts() -> None:
    occupancy = _counter_pass_fixture("occupancy")
    physical = _counter_pass_fixture("physical_bytes")
    physical_report = physical["per_sim"]["verilator"]["counters"]
    physical_report["selected_counters"]["TARGET_DMA_WRITE_BYTES"] = 42
    physical_report["readings"]["TARGET_DMA_WRITE_BYTES"] = 0
    rtl_sha256 = "9" * 64
    facts = [
        {"fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha256,
         "counter_field": "TARGET_DMA_BYTES", "direction": "read", "unit_bytes": 1,
         "derived_from_rtl": True, "provenance": "fixture RTL proof"},
        {"fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha256,
         "counter_field": "TARGET_DMA_WRITE_BYTES", "direction": "write", "unit_bytes": 1,
         "derived_from_rtl": True, "provenance": "fixture RTL proof"},
    ]
    linked = PR._link_counter_passes(
        occupancy, physical, physical_unit="BYTES", rtl_facts_sha256=rtl_sha256,
        counter_binding={"status": "exact", "rtl_facts_sha256": rtl_sha256,
                         "counter_facts": facts})
    assert linked["physical_byte_counters"]["counter_facts"] == facts
    assert linked["physical_byte_counters"]["semantic_resolution"] == "rtl_bound_physical_bytes"

    unknown = PR._link_counter_passes(
        occupancy, physical, physical_unit="BYTES", rtl_facts_sha256=rtl_sha256,
        counter_binding={"status": "unknown", "counter_facts": [],
                         "why": "direction is not proved"})
    assert "counter_facts" not in unknown["physical_byte_counters"]
    assert unknown["physical_byte_counters"]["semantic_resolution"] == "raw_named_readings_only"


def test_resource_bindings_are_derived_only_from_exact_raw_artifacts() -> None:
    command = {"tensors": {"a": {"shape": [1, 1]}, "b": {"shape": [1, 1]}},
               "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}]}
    measurement = {
        "work_volume": {"exact_macs": 1, "artifact_sha256": PR._canonical_sha256(command),
                        "basis": "compiler_command_buffer", "unit": "macs"},
        "command_buffer_artifact": {
            "command_buffer": command, "artifact_sha256": PR._canonical_sha256(command),
            "compiler_provenance": "fixture compiler"},
        "linked_counter_evidence": {"physical_byte_counters": {
            "semantic_resolution": "raw_named_readings_only"}},
    }
    bindings = PR._resource_bindings(measurement)
    assert set(bindings) == {"compute"}
    assert bindings["compute"]["derived_from_tool"] is True
    measurement["work_volume"]["artifact_sha256"] = "0" * 64
    assert PR._resource_bindings(measurement) == {}


def test_auxiliary_roofline_requirements_fail_closed_without_empty_compiler_path() -> None:
    measurement = _counter_pass_fixture("occupancy")
    measurement["per_sim"]["verilator"]["provenance"] = {
        "derived_from_rtl": True, "cycle_accurate": True}
    measurement["per_sim"]["verilator"]["measurement_conditions"] = {
        "cache_protocol": "fixture-protocol"}
    measurement["linked_counter_evidence"] = {"status": "linked"}
    evidence = PR._roofline_auxiliary_requirements(
        [{"kernel": "G00", "approaches": {"only": measurement}}],
        {"rtl_facts": {"sha256": "a" * 64},
         "circt_core_hw": {"sha256": "b" * 64, "path": "/fixture/core.mlir"}})
    assert evidence["status"] == "NO_GO"
    assert evidence["empty_run_requirements"] == [{
        "measurement_protocol": "fixture-protocol", "required_replicates": 4,
        "status": "UNKNOWN", "receipts": [],
        "why": ("the performance corpus has no structurally-empty workload emitted by the frozen "
                "compiler; running a hand-authored empty kernel would not measure the same compiler path"),
    }]
    assert evidence["composition_probe"]["status"] == "UNKNOWN"
    assert evidence["partial_evidence_is_admissible"] is False


def test_auxiliary_composition_probe_retains_exact_circt_bound_occupancy() -> None:
    measurement = _counter_pass_fixture("occupancy")
    rtl = measurement["per_sim"]["verilator"]
    rtl["provenance"] = {"derived_from_rtl": True, "cycle_accurate": True}
    rtl["measurement_conditions"] = {"cache_protocol": "fixture-protocol"}
    report = rtl["counters"]
    report["discovery"]["event_codes"] = {"TARGET_LOAD_CYCLES": 7}
    report["overlap"] = {"partition_proof": {
        "status": "proved", "sha256": "b" * 64, "method": "fixture-proof"}}
    measurement["linked_counter_evidence"] = {
        "status": "linked", "rtl_facts_sha256": "a" * 64, "occupancy": report}

    evidence = PR._roofline_auxiliary_requirements(
        [{"kernel": "G00", "approaches": {"only": measurement}}],
        {"rtl_facts": {"sha256": "a" * 64},
         "circt_core_hw": {"sha256": "b" * 64, "path": "/fixture/core.mlir"}})

    raw = evidence["composition_probe"]["raw_probe"]
    assert raw["rtl_facts_sha256"] == "a" * 64
    assert raw["circt_core_hw"]["sha256"] == "b" * 64
    assert raw["cycles"] == 109
    assert raw["readings"] == {"TARGET_LOAD_CYCLES": 17}


def test_counter_collection_runs_both_scoped_passes_and_restores_environment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "original")
    monkeypatch.setenv("MERLIN_HW_COUNTER_UNIT", "ORIGINAL_UNIT")
    observed: list[tuple[str, str | None, str | None]] = []

    def run_one(pass_name: str) -> dict:
        observed.append((pass_name, os.environ.get("MERLIN_HW_COUNTERS"),
                         os.environ.get("MERLIN_HW_COUNTER_UNIT")))
        return _counter_pass_fixture(pass_name)

    result = PR._collect_linked_counter_passes(run_one, physical_unit="BYTES")
    assert observed == [
        ("occupancy", "1", None),
        ("physical_bytes", "1", "BYTES"),
    ]
    assert result["linked_counter_evidence"]["status"] == "linked"
    assert set(result["counter_passes"]) == {"occupancy", "physical_bytes"}
    assert os.environ["MERLIN_HW_COUNTERS"] == "original"
    assert os.environ["MERLIN_HW_COUNTER_UNIT"] == "ORIGINAL_UNIT"


def test_counter_cli_does_not_label_an_arbitrary_unit_as_physical_bytes() -> None:
    with pytest.raises(PR.PC.CampaignGateError, match="requires the BYTES unit"):
        PR.main([
            "--functional-run-id", "not-opened",
            "--functional-submission-sha256", "a" * 64,
            "--counter-unit", "EVENTS",
        ])


def test_rtl_identity_binds_full_facts_bytes_to_active_circt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    circt = tmp_path / "core.mlir"
    circt.write_text("hw.module @Fixture() {}\n", encoding="utf-8")
    circt_sha256 = hashlib.sha256(circt.read_bytes()).hexdigest()
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"core_hw_sha256": circt_sha256}}),
                     encoding="utf-8")
    monkeypatch.setattr(PR.mlc_bridge, "core_hw_mlir", lambda _target: circt)

    identity = PR._load_rtl_identity(facts, "fixture")

    assert identity["circt_core_hw"]["sha256"] == circt_sha256
    assert identity["rtl_facts"]["sha256"] == hashlib.sha256(facts.read_bytes()).hexdigest()
    facts.write_text(json.dumps({"inputs": {"core_hw_sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(PR.PC.CampaignGateError, match="does not match"):
        PR._load_rtl_identity(facts, "fixture")


def test_package_sandbox_is_answer_closed_networkless_and_submission_read_only(tmp_path: Path) -> None:
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    ws = tmp_path / "workspace"
    pkg = ws / "submission"
    pkg.mkdir(parents=True)
    policy = PC.package_sandbox_policy(te, ws, pkg)
    assert policy.coverage_gap == ()
    assert "--unshare-net" in policy.argv
    assert "--clearenv" in policy.argv
    assert str(Path.home() / ".claude") not in policy.argv
    pairs = list(zip(policy.argv, policy.argv[1:]))
    assert ("--ro-bind", str(pkg)) in pairs
    assert policy.required_tools, "tool enforcement must not be an empty loop"


def test_actual_entrypoint_and_every_tool_probe_use_the_bwrap_policy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from merlin.targetgen import oot_runner
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    ws = tmp_path / "workspace"
    pkg_dir = tmp_path / "frozen" / "submission"
    pkg_dir.mkdir(parents=True)
    tool = pkg_dir / "tool.py"
    tool.write_text("print('boxed')\n")
    package = oot_runner.Package(pkg_dir, {
        "language": "python", "build": None,
        "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}},
    }, tool)
    inp = ws / "generated" / "input.interface.mlir"
    inp.parent.mkdir(parents=True)
    inp.write_text("module {}\n")
    policy = PC.package_sandbox_policy(te, ws, pkg_dir)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(PC.subprocess, "run", fake_run)
    probe_rows = PC.run_tool_probes(policy)
    assert len(probe_rows) == len(policy.required_tools)
    with PC.boxed_entrypoints(policy):
        result = oot_runner.run_entrypoint(package, "parse", inp)
        untrusted_build = oot_runner.Package(pkg_dir, {
            **package.manifest, "build": {"command": ["touch", "escaped-host-build"]},
        }, tool)
        with pytest.raises(PC.CampaignGateError, match="no host build"):
            oot_runner.build_package(untrusted_build)
    assert result.returncode == 0
    assert len(calls) == len(policy.required_tools) + 1
    entry_call = calls[-1]
    assert entry_call[0] == "bwrap"
    assert "--unshare-net" in entry_call and "--clearenv" in entry_call
    assert "perf-package" in entry_call and str(tool) in entry_call


def test_perf_runner_has_no_latest_or_mtime_submission_selection() -> None:
    source = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/run_perf_bench.py").read_text()
    assert "_latest_submission" not in source
    assert "st_mtime" not in source
    assert "--functional-run-id" in source
    assert "--functional-submission-sha256" in source
