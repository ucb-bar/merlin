"""Synthetic contract tests for the generic RTL-to-roofline orchestration edge."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root


_SCRIPT = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/build_rtl_roofline.py"

_CIRCT_HW_TEXT = "\n".join([
    "hw.module @Device() {",
    "  %not_a = comb.xor bin %busy_a, %true : i1",
    "  %not_b = comb.xor bin %busy_b, %true : i1",
    "  %only_a = comb.and bin %busy_a, %not_b : i1",
    "  %only_b = comb.and bin %not_a, %busy_b : i1",
    "  %both = comb.and bin %busy_a, %busy_b : i1",
    "  %unused = hw.instance \"meter\" @Meter("
    "io_event_io_event_signal_1: %only_a: i1, "
    "io_event_io_event_signal_2: %only_b: i1, "
    "io_event_io_event_signal_3: %both: i1) -> (x: i1)",
    "}",
])
_CIRCT_HW_SHA256 = hashlib.sha256(_CIRCT_HW_TEXT.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _receipt(artifact: dict[str, Any]) -> dict[str, Any]:
    return {"artifact": artifact, "artifact_sha256": _digest(artifact)}


def _write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _rtl() -> dict[str, Any]:
    return {
        "generator": {"tool": "synthetic structural extractor", "revision": "fixture-r1"},
        "inputs": {"core_hw_sha256": _CIRCT_HW_SHA256},
        "facts": {
            "source": "synthetic elaboration receipt",
            "memories": [{"bytes": 64, "depth": 8, "provenance": "memory declaration"}],
            "arrays": [{
                "rows": 3, "cols": 5, "primary": True, "provenance": "array declaration",
            }],
        },
    }


def _capabilities() -> dict[str, Any]:
    return {
        "dma": {
            "directions": {
                "value": ["read", "write", "copy"],
                "path": "driver.capabilities.directions",
                "provenance": "synthetic driver probe",
                "derived_from_tool": True,
            },
            "measurement_protocols": {
                "value": ["fresh-process", "predecessor-run"],
                "path": "driver.capabilities.measurement_protocols",
                "provenance": "synthetic runner protocol probe",
                "derived_from_tool": True,
            },
            **{
                direction: {
                    "sizes_bytes": {
                        "value": [8, 16, 32, 64],
                        "path": f"driver.capabilities.{direction}_sizes",
                        "provenance": f"synthetic {direction} descriptor-emission probe",
                        "derived_from_tool": True,
                    },
                }
                for direction in ("read", "write", "copy")
            },
        },
        "compute": {
            "workload_emitter": {
                "value": True,
                "path": "driver.capabilities.compute_workload_emitter",
                "provenance": "synthetic workload-emitter probe",
                "derived_from_tool": True,
            },
            "tile_multiples": {
                "value": [1, 2, 4, 8],
                "path": "driver.capabilities.compute_tile_multiples",
                "provenance": "synthetic compiler-emission probe",
                "derived_from_tool": True,
            },
        },
    }


def _command_buffer(macs: int) -> dict[str, Any]:
    return {
        "tensors": {"lhs": {"shape": [macs, 1]}, "rhs": {"shape": [1, 1]}},
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "lhs", "rhs": "rhs"}}],
    }


def _cycle_receipt(cycles: int, context: dict[str, Any]) -> dict[str, Any]:
    return _receipt({
        "cycles": cycles, "provenance": "synthetic cycle counter log", "context": context})


def _work_receipt(resource: str, macs: int, context: dict[str, Any]) -> dict[str, Any]:
    return _receipt({
        "resource": resource,
        "compiler_provenance": "synthetic compiler command-buffer artifact",
        "command_buffer": _command_buffer(macs),
        "context": context,
    })


def _empty_run_receipt(protocol: str, rtl_sha256: str, replicate: int) -> dict[str, Any]:
    context = {"kind": "empty_run", "measurement_protocol": protocol,
               "rtl_facts_sha256": rtl_sha256}
    cycle = _receipt({
        "cycles": 1, "provenance": "raw empty-process cycle counter log",
        "run_id": f"{protocol}-empty-{replicate}", "context": context,
    })
    command = _receipt({
        "compiler_provenance": "compiler-emitted empty command buffer",
        "command_buffer": {"tensors": {}, "commands": []}, "context": context,
    })
    return {"measurement_protocol": protocol, "cycle_receipt": cycle,
            "command_buffer_receipt": command}


def _traffic_receipt(resource: str, byte_count: int, rtl_sha256: str,
                     context: dict[str, Any]) -> dict[str, Any]:
    return _receipt({
        "resource": resource,
        "rtl_facts_sha256": rtl_sha256,
        "readings": {"read_count": byte_count, "write_count": 0},
        "counter_facts": [
            {
                "fact_kind": "counter_byte_binding",
                "artifact_sha256": rtl_sha256,
                "counter_field": "read_count",
                "direction": "read",
                "unit_bytes": 1,
                "provenance": "RTL counter binding for reads",
                "derived_from_rtl": True,
            },
            {
                "fact_kind": "counter_byte_binding",
                "artifact_sha256": rtl_sha256,
                "counter_field": "write_count",
                "direction": "write",
                "unit_bytes": 1,
                "provenance": "RTL counter binding for writes",
                "derived_from_rtl": True,
            },
        ],
        "context": context,
    })


def _calibration(rtl_sha256: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for direction in ("read", "write", "copy"):
        for protocol in ("fresh-process", "predecessor-run"):
            sweep = f"dma.{direction}.{protocol}"
            resource = f"resource-{sweep}"
            for size in (8, 16, 32, 64):
                coordinates = {"transfer_bytes": size, "measurement_protocol": protocol}
                context = {
                    "sweep_id": sweep, "coordinates": coordinates,
                    "measurement_protocol": protocol,
                    "rtl_facts_sha256": rtl_sha256,
                }
                samples.append({
                    "sweep_id": sweep,
                    "coordinates": coordinates,
                    "measurement_protocol": protocol,
                    "kind": "movement",
                    "cycle_receipt": _cycle_receipt(size + 5, context),
                    "traffic_receipt": _traffic_receipt(resource, size, rtl_sha256, context),
                })
    resource = "resource-compute.saturation"
    protocol = "fresh-process"
    for multiple in (1, 2, 4, 8):
        coordinates = {"tile_multiple": multiple, "tile_shape": [3, 5]}
        context = {
            "sweep_id": "compute.saturation", "coordinates": coordinates,
            "measurement_protocol": protocol,
            "rtl_facts_sha256": rtl_sha256,
        }
        samples.append({
            "sweep_id": "compute.saturation",
            "coordinates": coordinates,
            "measurement_protocol": protocol,
            "kind": "compute",
            "cycle_receipt": _cycle_receipt(2 * multiple + 2, context),
            "command_buffer_receipt": _work_receipt(resource, multiple, context),
        })
    composition_artifact = {
        "rtl_facts_sha256": rtl_sha256,
        "source": "synthetic elaborated counters.mlir",
        "circt_hw_sha256": _CIRCT_HW_SHA256,
        "counter_layout": {
            "prefix": "probe",
            "engines": ["engine-a", "engine-b"],
            "kinds": {"engine-a": "compute", "engine-b": "movement"},
            "by_combination": {
                "engine-a": "only_a", "engine-b": "only_b",
                "engine-a+engine-b": "both",
            },
        },
        "readings": {"only_a": 20, "only_b": 10, "both": 0},
        "cycles": 40,
        "codes": {"only_a": 1, "only_b": 2, "both": 3},
        "module": "Device",
        "counter_module": "Meter",
    }
    return {
        "samples": samples,
        "empty_run_receipts": [
            _empty_run_receipt(protocol, rtl_sha256, replicate)
            for protocol in ("fresh-process", "predecessor-run")
            for replicate in range(4)
        ],
        "composition_receipt": _receipt(composition_artifact),
    }


def _observations(rtl_sha256: str) -> dict[str, Any]:
    protocol = "fresh-process"
    context = {"workload": "case-0", "measurement_protocol": protocol,
               "rtl_facts_sha256": rtl_sha256}
    return {
        "expected_workloads": ["case-0"],
        "observations": [{
            "workload": "case-0",
            "measurement_protocol": protocol,
            "cycle_receipt": _cycle_receipt(2600, context),
            "command_buffer_receipt": _work_receipt(
                "resource-compute.saturation", 1000, context),
            "traffic_receipt": _traffic_receipt(
                "resource-dma.read.fresh-process", 400, rtl_sha256, context),
        }],
    }


def _run(tmp_path: Path, *, rtl: Any = None, capabilities: Any = None,
         calibration_factory: Any = _calibration, observations_factory: Any = _observations,
         markdown: bool = False, circt_text: str = _CIRCT_HW_TEXT) \
        -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    rtl_path = _write(tmp_path / "rtl.json", _rtl() if rtl is None else rtl)
    circt_path = tmp_path / "device.hw.mlir"
    circt_path.write_text(circt_text, encoding="utf-8")
    rtl_sha256 = hashlib.sha256(rtl_path.read_bytes()).hexdigest()
    inputs = {
        "rtl": rtl_path,
        "capabilities": _write(
            tmp_path / "capabilities.json",
            _capabilities() if capabilities is None else capabilities,
        ),
        "calibration": _write(tmp_path / "calibration.json", calibration_factory(rtl_sha256)),
        "observations": _write(tmp_path / "observations.json", observations_factory(rtl_sha256)),
    }
    output = tmp_path / "explicit-output.json"
    argv = [
        sys.executable, str(_SCRIPT),
        "--rtl-facts", str(inputs["rtl"]),
        "--harness-capabilities", str(inputs["capabilities"]),
        "--calibration", str(inputs["calibration"]),
        "--observations", str(inputs["observations"]),
        "--circt-hw", str(circt_path),
        "--output-json", str(output),
    ]
    if markdown:
        argv.extend(("--output-markdown", str(tmp_path / "explicit-output.md")))
    env = dict(os.environ)
    python_path = str(repo_root() / "merlin/python")
    env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"]
                                       if env.get("PYTHONPATH") else "")
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
    assert output.is_file(), result.stderr
    return result, json.loads(output.read_text(encoding="utf-8")), rtl_path


def test_cli_derives_resolved_artifact_from_plan_matched_raw_receipts(tmp_path: Path) -> None:
    result, artifact, rtl_path = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert artifact["status"] == "resolved"
    assert artifact["refusals"] == []
    assert artifact["calibration_plan"]["ready_sweeps"] == 7
    calibration = artifact["calibration_measurements"]
    assert calibration["sample_count"] == 28
    compute_peak = calibration["derived_peaks"]["resource-compute.saturation"]
    assert compute_peak["value"] == 8 / 17
    assert compute_peak["is_observed_ceiling"] is True
    assert calibration["unattached_fit_intercepts"]["resource-dma.read.fresh-process"]["cycles"] == 4
    assert artifact["composition_evidence"]["state"] == "measured"
    assert artifact["roofline"]["composition"] == "sum"
    point = artifact["roofline"]["points"]["case-0"]
    assert point["work"] == 1000.0 and point["work_unit"] == "macs"
    assert point["moved_bytes"] == {"resource-dma.read.fresh-process": 400.0}
    assert point["bound_cycles"] == 2551.0
    assert artifact["inputs"]["rtl_facts"]["sha256"] == hashlib.sha256(
        rtl_path.read_bytes()).hexdigest()


def test_cli_emits_markdown_only_at_explicit_optional_path(tmp_path: Path) -> None:
    result, artifact, _ = _run(tmp_path, markdown=True)

    assert result.returncode == 0 and artifact["status"] == "resolved"
    markdown = (tmp_path / "explicit-output.md").read_text(encoding="utf-8")
    assert "Status: **resolved**" in markdown
    assert "| case-0 | 2551.0 | resource-compute.saturation | True |" in markdown


def test_shared_empty_baseline_is_applied_once_not_once_per_resource(tmp_path: Path) -> None:
    result, artifact, _ = _run(tmp_path)

    assert result.returncode == 0
    point = artifact["roofline"]["points"]["case-0"]
    assert point["envelope"]["terms"] == {
        "resource-compute.saturation": 2125.0,
        "resource-dma.read.fresh-process": 425.0,
    }
    # The individual affine residuals (1 and 4 cycles) remain diagnostics.  Only the single
    # independently measured empty-run cost belongs to the whole workload.
    assert point["envelope"]["workload_fixed_cycles"] == 1
    assert set(artifact["calibration_measurements"]["unattached_fit_intercepts"]) >= {
        "resource-compute.saturation", "resource-dma.read.fresh-process",
    }


def test_absent_empty_run_baseline_refuses_ambiguous_intercepts(tmp_path: Path) -> None:
    def calibration(rtl_sha256: str) -> dict[str, Any]:
        document = _calibration(rtl_sha256)
        document.pop("empty_run_receipts")
        return document

    result, artifact, _ = _run(tmp_path, calibration_factory=calibration)

    assert result.returncode == 1 and artifact["status"] == "refused"
    assert any(item["code"] == "UNRESOLVED_FIXED_SCOPE" for item in artifact["refusals"])
    assert artifact["roofline"]["points"]["case-0"]["bound_cycles"] == "UNKNOWN"


def test_caller_authored_aggregates_are_refused_instead_of_trusted(tmp_path: Path) -> None:
    def calibration(rtl_sha256: str) -> dict[str, Any]:
        document = _calibration(rtl_sha256)
        document["peaks"] = [{"resource": "invented", "value": 999999}]
        document["fixed_terms"] = {"case-0": []}
        document["composition"] = {"operator": "max", "eta": 1.0}
        return document

    def observations(rtl_sha256: str) -> dict[str, Any]:
        document = _observations(rtl_sha256)
        document["observations"][0].update({
            "cycles": 1,
            "work": {"amount": 999999},
            "moved_bytes": [{"amount": 0}],
        })
        return document

    result, artifact, _ = _run(
        tmp_path, calibration_factory=calibration, observations_factory=observations)

    assert result.returncode == 1 and artifact["status"] == "refused"
    untrusted = [item for item in artifact["refusals"]
                 if item["code"] == "UNTRUSTED_AGGREGATE"]
    assert {item["source"] for item in untrusted} == {
        "calibration.peaks", "calibration.fixed_terms", "calibration.composition",
        "observations.observations[0].cycles", "observations.observations[0].work",
        "observations.observations[0].moved_bytes",
    }
    assert artifact["roofline"]["points"]["case-0"]["bound_cycles"] == 2551.0


def test_receipt_or_plan_coordinate_mismatch_fails_closed(tmp_path: Path) -> None:
    def calibration(rtl_sha256: str) -> dict[str, Any]:
        document = _calibration(rtl_sha256)
        document["samples"][0]["coordinates"]["transfer_bytes"] = 7
        document["samples"][1]["cycle_receipt"]["artifact_sha256"] = "0" * 64
        return document

    result, artifact, _ = _run(tmp_path, calibration_factory=calibration)

    assert result.returncode == 1
    codes = {item["code"] for item in artifact["refusals"]}
    assert "RECEIPT_MISMATCH" in codes
    assert "PLAN_COVERAGE_MISMATCH" in codes
    assert artifact["roofline"]["points"]["case-0"]["bound_cycles"] == "UNKNOWN"


def test_unverified_or_incomplete_overlap_counters_never_default_composition(tmp_path: Path) -> None:
    def calibration(rtl_sha256: str) -> dict[str, Any]:
        document = _calibration(rtl_sha256)
        artifact = document["composition_receipt"]["artifact"]
        artifact["circt_hw_sha256"] = "0" * 64
        document["composition_receipt"] = _receipt(artifact)
        return document

    result, artifact, _ = _run(tmp_path, calibration_factory=calibration)

    assert result.returncode == 1
    assert artifact["roofline"]["composition"] is None
    assert artifact["roofline"]["composition_eta"] == "UNKNOWN"
    assert artifact["roofline"]["points"]["case-0"]["bound_cycles"] == "UNKNOWN"
    assert any(item["code"] == "UNVERIFIED_COUNTER_PARTITION"
               for item in artifact["refusals"])


def test_explicit_circt_must_match_the_full_extractor_digest(tmp_path: Path) -> None:
    """A receipt cannot replace the elaborated RTL while retaining the facts-file digest."""
    substituted = _CIRCT_HW_TEXT.replace("%both =", "%different_both =")
    result, artifact, _ = _run(tmp_path, circt_text=substituted)

    assert result.returncode == 1 and artifact["status"] == "refused"
    assert any(item["code"] == "UNVERIFIED_CIRCT_BINDING"
               for item in artifact["refusals"])


def test_truncated_or_legacy_circt_digest_cannot_bind_a_roofline(tmp_path: Path) -> None:
    rtl = _rtl()
    rtl["inputs"]["core_hw_sha256"] = _CIRCT_HW_SHA256[:16]

    result, artifact, _ = _run(tmp_path, rtl=rtl)

    assert result.returncode == 1 and artifact["status"] == "refused"
    assert artifact["composition_evidence"]["state"] == "unknown"
    assert any(item["code"] == "UNVERIFIED_CIRCT_BINDING"
               for item in artifact["refusals"])


def test_embedded_circt_text_is_refused_even_when_it_is_hash_consistent(tmp_path: Path) -> None:
    def calibration(rtl_sha256: str) -> dict[str, Any]:
        document = _calibration(rtl_sha256)
        receipt = document["composition_receipt"]["artifact"]
        receipt["hw_text"] = _CIRCT_HW_TEXT
        document["composition_receipt"] = _receipt(receipt)
        return document

    result, artifact, _ = _run(tmp_path, calibration_factory=calibration)

    assert result.returncode == 1 and artifact["status"] == "refused"
    assert any(item["code"] == "UNVERIFIED_COUNTER_PARTITION"
               for item in artifact["refusals"])


def test_work_and_traffic_are_unknown_when_raw_derivations_are_incomplete(tmp_path: Path) -> None:
    def observations(rtl_sha256: str) -> dict[str, Any]:
        document = _observations(rtl_sha256)
        command = document["observations"][0]["command_buffer_receipt"]["artifact"]
        command["command_buffer"]["commands"][0]["opcode"] = "UNBOUND_OPERATION"
        document["observations"][0]["command_buffer_receipt"] = _receipt(command)
        traffic = document["observations"][0]["traffic_receipt"]["artifact"]
        traffic["counter_facts"][0]["derived_from_rtl"] = False
        document["observations"][0]["traffic_receipt"] = _receipt(traffic)
        return document

    result, artifact, _ = _run(tmp_path, observations_factory=observations)

    assert result.returncode == 1
    point = artifact["roofline"]["points"]["case-0"]
    assert point["work"] == "UNKNOWN"
    assert point["moved_bytes"] == {}
    assert point["bound_cycles"] == "UNKNOWN"
    codes = {item["code"] for item in artifact["refusals"]}
    assert {"UNKNOWN_WORK", "UNPROVEN_COUNTER_BINDING"} <= codes
