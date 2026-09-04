#!/usr/bin/env python3
"""Materialize the exact calibration requests admitted by RTL/tool evidence.

This is deliberately a *planning edge*, not a microbenchmark generator.  The generic planner can
prove which coordinates an available compiler/harness may execute, but it cannot invent a target's
DMA ABI, cache flush protocol, or command-buffer lowering.  Writing those details here would turn a
portable calibration plan into a target-specific collection of guessed constants.

The emitted manifest is therefore the hand-off between discovery and execution: a target-owned
runner must execute every request exactly once per declared measurement protocol and return raw,
content-addressed receipts to ``build_rtl_roofline.py``.  An incomplete plan produces a manifest for
diagnosis but is explicitly non-dispatchable; consumers must not cherry-pick its ready subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir
from merlin.perf.calibration_plan import CalibrationPlan, build_calibration_plan_from_rtl


_SCHEMA = "rtl_calibration_campaign_v1"
_AUXILIARY_CAPABILITIES = (
    "empty_workload_emitter",
    "joint_occupancy_probe",
)


def _json(path: Path, label: str) -> tuple[Mapping[str, Any] | None, dict[str, str], str | None]:
    """Read one exact object input while retaining the bytes identity in every outcome."""
    receipt = {"path": str(path), "sha256": "UNKNOWN"}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, receipt, f"{label}: cannot read explicit path {path}: {exc}"
    receipt["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, receipt, f"{label}: input is not UTF-8 JSON: {exc}"
    if not isinstance(value, Mapping):
        return None, receipt, f"{label}: input must be a JSON object"
    return value, receipt, None


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _request(plan: CalibrationPlan, rtl_sha256: str, capability_sha256: str) -> list[dict[str, Any]]:
    """Create stable identities from an already-admitted plan; never synthesize a coordinate."""
    requests: list[dict[str, Any]] = []
    protocols = sorted({
        str(point.to_dict()["measurement_protocol"])
        for sweep in plan.sweeps for point in sweep.points
        if point.to_dict().get("measurement_protocol")
    })
    for sweep in plan.sweeps:
        if not sweep.ready:
            continue
        facts = [fact.to_dict() for fact in sweep.facts]
        for ordinal, point in enumerate(sweep.points):
            coordinates = point.to_dict()
            identity = {
                "rtl_facts_sha256": rtl_sha256,
                "harness_capabilities_sha256": capability_sha256,
                "sweep_id": sweep.sweep_id,
                "ordinal": ordinal,
                "coordinates": coordinates,
            }
            # Compute coordinates do not carry a protocol dimension in the calibration plan.  Choose
            # one deterministically from the tool-derived protocols and make it part of the exact
            # request identity; this is scheduling metadata, not a cache-state claim.
            if sweep.mechanism == "compute" and protocols:
                identity["measurement_protocol"] = protocols[0]
            requests.append({
                "request_sha256": _canonical_digest(identity),
                "identity": identity,
                "mechanism": sweep.mechanism,
                "condition": sweep.condition,
                "objective": sweep.objective,
                "fit": sweep.fit.to_dict(),
                "facts": facts,
                **({"measurement_protocol": identity["measurement_protocol"]}
                   if "measurement_protocol" in identity else {}),
                "required_raw_receipts": (
                    ["rtl_cycle_measurement", "compiler_command_buffer"]
                    if sweep.mechanism == "compute"
                    else ["rtl_cycle_measurement", "physical_counter"]
                ),
            })
    return requests


def _tool_capability(capabilities: Mapping[str, Any], name: str) -> tuple[bool, str]:
    auxiliary = capabilities.get("measurement_auxiliary")
    fact = auxiliary.get(name) if isinstance(auxiliary, Mapping) else None
    provenance = (fact.get("provenance", fact.get("source"))
                  if isinstance(fact, Mapping) else None)
    ready = (isinstance(fact, Mapping) and fact.get("value") is True
             and fact.get("derived_from_tool") is True and bool(provenance))
    return ready, (str(provenance) if ready else
                   f"measurement_auxiliary.{name} is not positively established by a tool probe")


def _auxiliary_requests(plan: CalibrationPlan, capabilities: Mapping[str, Any],
                        rtl_sha256: str, capability_sha256: str) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Schedule fixed-cost and overlap evidence; never assume the harness can produce either."""
    protocols = sorted({
        str(point.to_dict()["measurement_protocol"])
        for sweep in plan.sweeps for point in sweep.points
        if point.to_dict().get("measurement_protocol")
    })
    refusals: list[dict[str, Any]] = []
    standing: dict[str, bool] = {}
    for capability in _AUXILIARY_CAPABILITIES:
        standing[capability], why = _tool_capability(capabilities, capability)
        if not standing[capability]:
            refusals.append({
                "sweep_id": f"measurement_auxiliary.{capability}",
                "disposition": "UNKNOWN",
                "issues": [{"code": "UNKNOWN_CAPABILITY", "reason": why,
                            "fact_paths": [f"measurement_auxiliary.{capability}"]}],
            })
    if not protocols:
        refusals.append({
            "sweep_id": "measurement_auxiliary.empty_workload_emitter",
            "disposition": "UNKNOWN",
            "issues": [{"code": "UNKNOWN_PROTOCOL",
                        "reason": "no tool-derived measurement protocol is present in the plan",
                        "fact_paths": ["dma.measurement_protocols"]}],
        })
    requests: list[dict[str, Any]] = []
    if standing.get("empty_workload_emitter") and protocols:
        for protocol in protocols:
            for replicate in range(4):
                identity = {
                    "rtl_facts_sha256": rtl_sha256,
                    "harness_capabilities_sha256": capability_sha256,
                    "kind": "empty_run", "measurement_protocol": protocol,
                    "replicate": replicate,
                }
                requests.append({
                    "request_sha256": _canonical_digest(identity), "identity": identity,
                    "required_raw_receipts": ["rtl_cycle_measurement", "compiler_command_buffer"],
                })
    if standing.get("joint_occupancy_probe") and protocols:
        identity = {
            "rtl_facts_sha256": rtl_sha256,
            "harness_capabilities_sha256": capability_sha256,
            "kind": "composition_probe",
            "measurement_protocol": protocols[0],
        }
        requests.append({
            "request_sha256": _canonical_digest(identity), "identity": identity,
            "required_raw_receipts": ["rtl_cycle_measurement", "joint_occupancy_partition"],
        })
    return requests, refusals


def build(rtl_path: Path, capabilities_path: Path) -> tuple[dict[str, Any], int]:
    rtl, rtl_input, rtl_error = _json(rtl_path, "rtl_facts")
    capabilities, capabilities_input, capabilities_error = _json(
        capabilities_path, "harness_capabilities")
    errors = [error for error in (rtl_error, capabilities_error) if error]
    if rtl is None or capabilities is None:
        return {
            "schema": _SCHEMA,
            "status": "refused",
            "dispatchable": False,
            "inputs": {"rtl_facts": rtl_input, "harness_capabilities": capabilities_input},
            "refusals": errors,
            "calibration_plan": None,
            "measurement_requests": [],
            "auxiliary_measurement_requests": [],
        }, 1

    plan = build_calibration_plan_from_rtl(rtl, capabilities)
    requests = _request(plan, rtl_input["sha256"], capabilities_input["sha256"])
    auxiliary_requests, auxiliary_refusals = _auxiliary_requests(
        plan, capabilities, rtl_input["sha256"], capabilities_input["sha256"])
    refusals = [
        {"sweep_id": sweep.sweep_id, "disposition": sweep.disposition.value,
         "issues": [issue.to_dict() for issue in sweep.issues]}
        for sweep in plan.sweeps if not sweep.ready
    ] + auxiliary_refusals
    dispatchable = plan.ready and not auxiliary_refusals
    return {
        "schema": _SCHEMA,
        "status": "ready" if dispatchable else "refused",
        "dispatchable": dispatchable,
        "inputs": {"rtl_facts": rtl_input, "harness_capabilities": capabilities_input},
        "refusals": refusals,
        "calibration_plan": plan.to_dict(),
        "measurement_requests": requests,
        "auxiliary_measurement_requests": auxiliary_requests,
        "execution_contract": {
            "rule": ("execute every primary and auxiliary request with its exact identity and return "
                     "only raw receipts"),
            "receipt_builder": "build_rtl_roofline.py",
            "partial_execution_is_admissible": False,
        },
    }, 0 if dispatchable else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit the exact runner-ready calibration requests derived from RTL/tool evidence.")
    parser.add_argument("--rtl-facts", required=True, help="exact CIRCT-extracted RTL facts JSON")
    parser.add_argument("--harness-capabilities", required=True,
                        help="exact compiler/harness probe capabilities JSON")
    parser.add_argument("--output-json", required=True, help="exact manifest output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {Path(args.rtl_facts).resolve(), Path(args.harness_capabilities).resolve()}
    output = Path(args.output_json).resolve()
    if output in inputs:
        _parser().error("output path must be distinct from both explicit input paths")
    try:
        output.relative_to(out_dir().resolve())
    except ValueError:
        _parser().error(
            f"generated campaign must be below the configured output root {out_dir().resolve()}")
    artifact, status = build(Path(args.rtl_facts), Path(args.harness_capabilities))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
