"""Target-owned calibration capability probes for Gemmini.

The generic calibration planner deliberately cannot infer an executable DMA ABI or a compiler's
accepted shape range from storage geometry.  This module is the narrow Gemmini boundary that asks the
real native emitter instead.  It records raw command-buffer and emitted-artifact digests for every
accepted probe, so the result can be passed directly to the generic planner without hand-entered
transfer sizes or conventional tile dimensions.

This is an *emission/compile* capability probe, not a performance measurement.  It never reports
cycles, cache residency, DMA bandwidth, or traffic.  In particular Gemmini's current fresh-process /
one-predecessor protocol does not observe cache state, so cache conditions are explicitly UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf.calibration_plan import MIN_POINTS_PER_PARAMETER
from merlin.targetgen.rtl.facts import ensure_facts


SCHEMA = "gemmini_calibration_capability_probe_v1"
_TARGET = "gemmini"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"RTL facts artifact {path} is not a JSON object")
    return value, {"path": str(path), "sha256": _sha256_bytes(raw)}


def _mesh_shape(rtl_facts: Mapping[str, Any]) -> tuple[int, int] | None:
    """Select the one declared primary/only array; ambiguity remains UNKNOWN."""
    body = rtl_facts.get("facts")
    arrays = body.get("arrays") if isinstance(body, Mapping) else None
    if not isinstance(arrays, list) or not arrays or not all(isinstance(item, Mapping) for item in arrays):
        return None
    selected = [item for item in arrays if item.get("primary") is True]
    if not selected and len(arrays) == 1:
        selected = [arrays[0]]
    if len(selected) != 1:
        return None
    rows, cols = selected[0].get("rows"), selected[0].get("cols")
    if (isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            or isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0):
        return None
    return rows, cols


def _compute_probe_buffer(rows: int, cols: int, multiple: int) -> dict[str, Any]:
    """One target-native resident-matmul accepted only if the actual emitter accepts it."""
    height = rows * multiple
    return {
        "tensors": {
            "probe_weight": {"shape": [cols, cols], "dtype": "i8", "role": "weight"},
            "probe_input": {"shape": [height, cols], "dtype": "i8", "role": "input"},
            "probe_output": {"shape": [height, cols], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "probe_weight", "dst": "probe_resident"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {
                "lhs": "probe_input", "rhs": "probe_resident", "dst": "probe_acc"}},
            {"opcode": "COMMIT", "operands": {"src": "probe_acc", "dst": "probe_output"},
             "attributes": {"output_dtype": "i32", "epilogue": []}},
            {"opcode": "EVICT", "operands": {"handle": "probe_resident"}},
        ],
    }


def _core_circt_receipt() -> dict[str, str]:
    """Record, but do not invent, the elaborated CIRCT artifact behind this target's emitter."""
    from merlin.targetgen.rtl import mlc_bridge

    path = mlc_bridge.core_hw_mlir(_TARGET)
    if path is None or not Path(path).is_file():
        return {"path": "UNKNOWN", "sha256": "UNKNOWN"}
    resolved = Path(path)
    return {"path": str(resolved), "sha256": _sha256_bytes(resolved.read_bytes())}


def _facts_bind_exact_core(rtl_facts: Mapping[str, Any], circt_receipt: Mapping[str, str]) -> bool:
    """Accept extracted facts only when they name the exact CIRCT bytes read by this probe.

    A digest of the facts JSON proves only which *claims* were consumed.  Without this second
    equality check, an old facts bundle could supply a stale mesh while the capability receipt named
    a newer core artifact.  Short legacy digests are deliberately insufficient for this boundary.
    """
    inputs = rtl_facts.get("inputs")
    recorded = inputs.get("core_hw_sha256") if isinstance(inputs, Mapping) else None
    actual = circt_receipt.get("sha256")
    return (isinstance(recorded, str) and len(recorded) == 64
            and isinstance(actual, str) and len(actual) == 64
            and recorded == actual)


def _emit_or_compile(command_buffer: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Run the actual target emitter, optionally its lowering/compiler, preserving failure text."""
    from . import gemmini_codegen_mlir as codegen

    record: dict[str, Any] = {
        "command_buffer": command_buffer,
        "command_buffer_sha256": _canonical_sha256(command_buffer),
        "stage": stage,
    }
    try:
        emitted, arguments = codegen.emit_kernel_mlir(command_buffer)
        record.update({"status": "accepted", "emitted_mlir_sha256": _sha256_bytes(
            emitted.encode("utf-8")), "argument_order": arguments})
        if stage == "compile":
            with tempfile.TemporaryDirectory(prefix="merlin_gemmini_calibration_probe_") as directory:
                obj = Path(codegen.build_object(command_buffer, directory))
                record["object_sha256"] = _sha256_bytes(obj.read_bytes())
    except Exception as exc:  # an unavailable compiler is a probe outcome, not an invented capability
        record.update({"status": "unknown", "why": f"{type(exc).__name__}: {exc}"})
    return record


def _measurement_protocols() -> tuple[list[dict[str, Any]], str | None]:
    """Ask the real harness fragment generator which protocols it can emit.

    This deliberately proves only the *protocol shape* (fresh process versus an unmeasured
    predecessor).  Its own return record must still say cache state is unobserved; otherwise a
    request mechanism would be mislabeled as a cache-residency measurement.
    """
    from . import gemmini_codegen_mlir as codegen

    previous_counters = os.environ.get("MERLIN_HW_COUNTERS")
    previous_condition = os.environ.get("MERLIN_CACHE_STATE")
    rows: list[dict[str, Any]] = []
    try:
        os.environ["MERLIN_HW_COUNTERS"] = "0"
        for condition in ("cold", "warm"):
            os.environ["MERLIN_CACHE_STATE"] = condition
            fragment = codegen._measurement_c_fragments("")
            protocol = fragment.get("cache_protocol")
            observed = fragment.get("cache_state_observed")
            state = fragment.get("cache_state")
            if not isinstance(protocol, str) or not protocol:
                return rows, "harness fragment did not report a measurement protocol"
            if observed is not False or state != "unknown":
                return rows, "harness fragment claimed cache state observability unexpectedly"
            rows.append({"requested_cache_condition": condition,
                         "measurement_protocol": protocol,
                         "cache_state": state,
                         "cache_state_observed": observed})
    except Exception as exc:  # unavailable generator -> unknown capability, never a fallback protocol
        return rows, f"{type(exc).__name__}: {exc}"
    finally:
        if previous_counters is None:
            os.environ.pop("MERLIN_HW_COUNTERS", None)
        else:
            os.environ["MERLIN_HW_COUNTERS"] = previous_counters
        if previous_condition is None:
            os.environ.pop("MERLIN_CACHE_STATE", None)
        else:
            os.environ["MERLIN_CACHE_STATE"] = previous_condition
    return rows, None


def _dma_probe_requests(rtl_facts: Mapping[str, Any], requested_sizes: tuple[int, ...], *,
                        stage: str) -> tuple[dict[str, Any], tuple[int, ...], bool, str]:
    """Probe every direction/size through the independent target-header DMA emitter.

    When no coordinates were supplied, the target boundary derives exactly the number required by the
    fit from the generated header's per-command payload capability.  Every resulting coordinate is
    still passed through the real emitter/compiler; the header number alone never establishes it.
    """
    from . import gemmini_dma_calibration as dma

    try:
        sizes = requested_sizes or dma.derived_transfer_ladder(
            rtl_facts, points=2 * MIN_POINTS_PER_PARAMETER)
    except Exception as exc:
        why = f"could not derive DMA probe coordinates: {type(exc).__name__}: {exc}"
        rows = {direction: {"status": "unknown", "why": why,
                            "requested_payload_sizes_bytes": list(requested_sizes),
                            "physical_traffic_bytes": {"status": "unmeasured", "value": None}}
                for direction in dma.DIRECTIONS}
        return rows, requested_sizes, False, why

    probes: dict[str, Any] = {}
    all_accepted = True
    failures: list[str] = []
    for direction in dma.DIRECTIONS:
        receipts = [dma.probe_dma_capability(direction, size, rtl_facts, stage=stage)
                    for size in sizes]
        rejected = [row for row in receipts if row.get("status") != "accepted"]
        if rejected:
            all_accepted = False
            failures.extend(f"{direction}/{row.get('requested_payload_bytes')}: "
                            f"{row.get('why', 'unknown probe failure')}" for row in rejected)
        probes[direction] = {
            "status": "accepted" if not rejected else "unknown",
            "requested_payload_sizes_bytes": list(sizes),
            "physical_traffic_bytes": {
                "status": "unmeasured", "value": None,
                "why": "payload extent is not physical traffic; measure the RTL byte counters",
            },
            "probe_receipts": receipts,
            **({} if not rejected else {"why": "; ".join(
                str(row.get("why", "unknown probe failure")) for row in rejected)}),
        }
    return probes, sizes, all_accepted, "; ".join(failures)


def probe_calibration_capabilities(*, stage: str = "emission",
                                   facts_path: str | Path | None = None,
                                   dma_transfer_sizes: tuple[int, ...] = ()) -> dict[str, Any]:
    """Return a content-addressed, planner-consumable capability artifact.

    ``stage='emission'`` proves native command-buffer emission and has no external compiler dependency.
    ``stage='compile'`` additionally lowers/compiles every probe.  Neither mode promotes a failed or
    unavailable probe into a negative hardware fact: only a complete successful set establishes the
    compute capability; all other mechanisms stay UNKNOWN.
    """
    if stage not in {"emission", "compile"}:
        raise ValueError("stage must be 'emission' or 'compile'")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
           for size in dma_transfer_sizes):
        raise ValueError("DMA transfer-size probes must be positive integer byte counts")
    if len(set(dma_transfer_sizes)) != len(dma_transfer_sizes):
        raise ValueError("DMA transfer-size probes must be unique")
    facts_file = ensure_facts(_TARGET, explicit=facts_path)
    rtl_facts, facts_receipt = _read_json(facts_file)
    circt_receipt = _core_circt_receipt()
    exact_core_binding = _facts_bind_exact_core(rtl_facts, circt_receipt)
    shape = _mesh_shape(rtl_facts) if exact_core_binding else None
    required_points = 2 * MIN_POINTS_PER_PARAMETER  # rate and intercept, both independently identified
    probe_rows: list[dict[str, Any]] = []
    if not exact_core_binding:
        compute_why = ("RTL facts do not carry a full core_hw_sha256 matching the exact CIRCT "
                       "artifact read by the capability probe")
    elif shape is None:
        compute_why = "RTL facts do not identify one positive primary/only compute array"
    else:
        rows, cols = shape
        probe_rows = [_emit_or_compile(_compute_probe_buffer(rows, cols, multiple), stage=stage)
                      | {"tile_shape": [rows, cols], "tile_multiple": multiple}
                      for multiple in range(1, required_points + 1)]
        rejected = [row for row in probe_rows if row["status"] != "accepted"]
        compute_why = ("; ".join(str(row.get("why", "unknown emitter failure")) for row in rejected)
                       if rejected else "")

    accepted = bool(shape) and len(probe_rows) == required_points and not compute_why
    protocol_rows, protocol_why = _measurement_protocols()
    protocols = sorted({row["measurement_protocol"] for row in protocol_rows})
    if exact_core_binding:
        dma_requests, admitted_dma_sizes, dma_accepted, dma_why = _dma_probe_requests(
            rtl_facts, dma_transfer_sizes, stage=stage)
    else:
        admitted_dma_sizes, dma_accepted = dma_transfer_sizes, False
        dma_why = ("RTL facts do not carry a full core_hw_sha256 matching the exact CIRCT artifact; "
                   "DMA geometry was not sent to the emitter")
        dma_requests = {direction: {
            "status": "unknown", "why": dma_why,
            "requested_payload_sizes_bytes": list(dma_transfer_sizes),
            "physical_traffic_bytes": {"status": "unmeasured", "value": None},
        } for direction in ("read", "write", "copy")}
    evidence = {
        "tool": "gemmini native command-buffer emitter",
        "stage": stage,
        "rtl_facts_sha256": facts_receipt["sha256"],
        "circt_core_hw_sha256": circt_receipt["sha256"],
        "probe_receipts_sha256": _canonical_sha256(probe_rows),
    }
    compute: dict[str, Any]
    if accepted:
        compute = {
            "workload_emitter": {"value": True, "derived_from_tool": True,
                           "source": "target-owned native emission/compile probes", "evidence": evidence},
            "tile_multiples": {"value": [row["tile_multiple"] for row in probe_rows],
                               "derived_from_tool": True,
                               "source": "target-owned native emission/compile probes", "evidence": evidence},
        }
    else:
        compute = {"status": "unknown", "why": compute_why or "compute probes were not complete"}

    dma_evidence = {
        "tool": "target-header expansion plus production MLIR-to-LLVM compiler",
        "stage": stage,
        "rtl_facts_sha256": facts_receipt["sha256"],
        "circt_core_hw_sha256": circt_receipt["sha256"],
        "direction_probe_receipts_sha256": _canonical_sha256(dma_requests),
        "coordinate_semantics": "requested payload bytes; physical traffic remains unmeasured",
    }
    dma_capability: dict[str, Any] = {
        "status": ("accepted" if dma_accepted and protocol_why is None else "unknown"),
        "direction_probes": dma_requests,
        "measurement_protocols": ({
            "value": protocols, "derived_from_tool": True,
            "source": "target-owned harness-fragment probe", "evidence": {
                "protocol_receipts_sha256": _canonical_sha256(protocol_rows),
                "rtl_facts_sha256": facts_receipt["sha256"],
            },
        } if protocol_why is None else {"status": "unknown", "why": protocol_why}),
    }
    if dma_accepted:
        dma_capability["directions"] = {
            "value": ["read", "write", "copy"], "derived_from_tool": True,
            "source": "direction-pure target-header expansion probes", "evidence": dma_evidence,
        }
        for direction in ("read", "write", "copy"):
            dma_capability[direction] = {"sizes_bytes": {
                "value": list(admitted_dma_sizes), "derived_from_tool": True,
                "source": "target-owned emission/compile probes of requested payload extents",
                "evidence": dict(dma_evidence, direction=direction),
            }}
    else:
        dma_capability["why"] = dma_why or "one or more direction-pure DMA probes were unavailable"

    # Auxiliary capabilities are proved at this same target boundary.  The empty path is sent through
    # the real native compiler.  Joint occupancy additionally requires the complete header/CIRCT
    # partition plus all direction-pure DMA and native-compute emission probes; semantic engine roles
    # are intentionally deferred to differential RTL measurements by the execution runner.
    from . import gemmini_roofline_auxiliary as auxiliary

    empty_probe = auxiliary.probe_empty_workload(
        rtl_facts, rtl_facts_sha256=facts_receipt["sha256"], stage=stage)
    occupancy_probe = auxiliary.probe_joint_occupancy(
        rtl_facts, rtl_facts_sha256=facts_receipt["sha256"],
        measurement_protocol=protocols[0] if protocols else "")
    empty_ready = empty_probe.get("status") == "accepted"
    occupancy_ready = (occupancy_probe.get("status") == "accepted"
                       and accepted and dma_accepted and protocol_why is None)

    complete = (accepted and dma_accepted and protocol_why is None
                and empty_ready and occupancy_ready)
    artifact = {
        "schema": SCHEMA,
        "status": "complete" if complete else "partial" if accepted or dma_accepted else "unknown",
        "target": _TARGET,
        "inputs": {"rtl_facts": facts_receipt, "circt_core_hw": circt_receipt},
        "compute": compute,
        "dma": dma_capability,
        "measurement_auxiliary": {
            "empty_workload_emitter": {
                "value": empty_ready, "derived_from_tool": True,
                "source": "target-owned native compiler empty-workload probe",
                **({} if empty_ready else {"why": empty_probe.get("why", "probe unavailable")}),
                "evidence": {"rtl_facts_sha256": facts_receipt["sha256"],
                             "circt_core_hw_sha256": circt_receipt["sha256"],
                             "probe_receipt": empty_probe},
            },
            "joint_occupancy_probe": {
                "value": occupancy_ready, "derived_from_tool": True,
                "source": "target-owned differential RTL occupancy runner probe",
                **({} if occupancy_ready else {"why": (
                    occupancy_probe.get("why") or compute_why or dma_why or protocol_why
                    or "supporting probes are incomplete")}),
                "evidence": {"rtl_facts_sha256": facts_receipt["sha256"],
                             "circt_core_hw_sha256": circt_receipt["sha256"],
                             "partition_probe_receipt": occupancy_probe},
            },
        },
        # The two rows demonstrate requests, not cache residency.  Keep the distinction visible.
        "cache_states": {"status": "unknown", "why": "cache state is not observed by this harness"},
        "measurement_protocol_receipts": protocol_rows,
        "probe_receipts": probe_rows,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact
