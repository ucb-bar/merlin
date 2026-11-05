"""Compact operator diagnostics from pinned observations; never optimization proof.

The caller supplies existing receipt projections, not raw IR/logs. Matching pins
bind those projections to a compiler/capsule/scope; this adapter does not infer
causal savings, independently validate schedules, or grant model qualification.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

FIELDS = {
    "packing_reuse": ("pack_calls", "packed_bytes", "reused_bytes", "layout_conversions"),
    "crosslayer_lifetimes": ("allocations", "peak_live_bytes", "reused_buffers", "spilled_bytes",
                             "arena_bytes", "bound_heap_allocations"),
    "cpu_sync": ("host_instructions", "submission_cycles", "completion_wait_cycles", "fences"),
    "schedule_queues": ("commands_total", "preload_commands", "load_commands", "compute_commands", "store_commands", "config_commands",
                        "dependency_wait_cycles", "ready_stall_cycles", "queue_highwater",
                        "device_read_bytes", "device_write_bytes", "useful_macs", "padded_macs"),
    "connected_fusion": ("source_regions", "emitted_regions", "host_device_transfers", "intermediate_bytes"),
    "wholechain": ("input_conversion_cycles", "model_cycles", "output_conversion_cycles", "checked_outputs"),
}


def _metrics(value, keys):
    return {key: number for key in keys if type(number := value.get(key)) in (int, float)
            and 0 <= number <= 2**63 - 1 and math.isfinite(number)}


def _artifact(value):
    path = Path(value["path"])
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError("receipt must be a direct regular artifact")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != value.get("sha256"):
        raise ValueError("receipt bytes differ from supplied pin")
    return {"path": str(path), "sha256": value["sha256"]}


def project(performance, *, compiler_sha256, capsule_sha256, correctness, observations):
    """Consume the existing joint_performance_report.project result unchanged.

    observations[section] has pins, artifact, planned and emitted mappings.
    Pins always include compiler/capsule/scope; timing sections also bind the
    selected model. Scalar values are caller-observed receipt projections, not
    decoded instruction semantics or a trusted performance qualification.
    """
    expected = {"compiler_sha256": compiler_sha256, "capsule_sha256": capsule_sha256,
                "scope_sha256": performance.get("scope_sha256")}
    report = {"schema": "merlin.joint_compiler_diagnostics.v1", "promotion_authority": False,
              "optimization_savings_estimated": False, "resource_cycles_additive": False,
              "correctness": correctness if correctness in {"pass", "fail", "unknown"} else "unknown",
              "pins": {**expected, "model_sha256": performance.get("model_sha256"),
                       "selection_sha256": performance.get("selection_sha256")},
              "performance_status": performance.get("status", "UNKNOWN"),
              "timing_qualified": False, "sections": {},
              "evidence_authority": "caller-observed pinned receipt projections; not independent optimization proof"}
    for name, keys in FIELDS.items():
        section = {"status": "UNKNOWN", "reason": "no matching emitted receipt", "planned": {}, "emitted": {},
                   "missing_metrics": list(keys), "artifact": None,
                   "origin": "unavailable", "maturity": "unknown", "limits": []}
        record = observations.get(name)
        if isinstance(record, dict):
            try:
                pins = record.get("pins", {})
                if any(not value or pins.get(key) != value for key, value in expected.items()):
                    raise ValueError("compiler, capsule or timing scope is absent/mismatched")
                if name in {"cpu_sync", "schedule_queues", "wholechain"} and (
                    not performance.get("model_sha256") or pins.get("model_sha256") != performance["model_sha256"]
                ):
                    raise ValueError("selected timing model is absent/mismatched")
                if name == "wholechain" and performance.get("scope_kind") != "whole_model":
                    raise ValueError("kernel-call or whole-trace evidence cannot establish whole-model scope")
                section["artifact"] = _artifact(record["artifact"])
                section["origin"] = record.get("origin", "caller_receipt")
                section["maturity"] = str(record.get("maturity", "unspecified_receipt"))[:160]
                limits = record.get("limits", [])
                section["limits"] = [str(v)[:240] for v in limits[:8]] if isinstance(limits, list) else []
                section["planned"] = _metrics(record.get("planned", {}), keys)
                section["emitted"] = _metrics(record.get("emitted", {}), keys)
                section["missing_metrics"] = [key for key in keys if key not in section["emitted"]]
                if section["emitted"]:
                    section.update(status="OBSERVED_PARTIAL" if section["missing_metrics"] else "OBSERVED",
                                   reason="matching pinned receipt; metrics remain observations, not causal savings")
                elif section["planned"]:
                    section.update(status="PLANNED_ONLY", reason="source analysis only; no emitted or runtime observation")
            except (OSError, ValueError, TypeError, KeyError) as error:
                section["reason"] = str(error)[:240]
        report["sections"][name] = section
    if (performance.get("modeled_resource_activity") or {}).get("scope") == "whole_trace":
        report["resource_scope_note"] = "Whole-trace resource busy cycles overlap and cannot decompose the selected call."
    return report
