"""Pure descriptive joins for ordinary public compiler exploration.

The original caller owns source, executable, numerical and measurement admission.
This projection issues no authority and never turns counts or predictions into
measurements. Structural lower bounds remain partial complete-call evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

from merlin.perf import capsule_roofline as R
from merlin.perf.candidate_source import public_projection

SCHEMA = "merlin.phase2.ordinary_capsule_feedback.v1"
_MISSING = {
    "complete_call_host": "accelerator-feeding host work and outside-kernel call costs are unpriced",
    "packing_layout": "packing, layout and transfer materialization have no complete-call floor",
    "ports": "physical interface crossings, port sharing and transfer granularity are unqualified",
    "overlap": "request order supplies no independently established resource/completion timeline",
    "placement": "source work and declarations alone do not qualify accelerator execution",
}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _count(value):
    return value if type(value) is int and 0 <= value < 2**64 else None


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _dynamic(report, compiled):
    from merlin_timing.observations import REPORT_V2, feature_complete

    unknown = {"status": "UNKNOWN", "feature_counts": None, "timing_cycles": None}
    if not report:
        return {**unknown, "reason": "complete original dynamic count report unavailable"}
    if compiled.get("status") != "observed" or report.get("elf_sha256") != compiled.get("elf_sha256"):
        return {**unknown, "reason": "dynamic counts do not join the original observed executable"}
    counts = report.get("feature_counts")
    try:
        valid = (
            report.get("schema") == REPORT_V2
            and feature_complete(report)
            and isinstance(counts, dict)
            and len(counts) <= 1024
            and all(type(key) is str and len(key) <= 256 and _count(value) is not None for key, value in counts.items())
        )
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        return {**unknown, "reason": "dynamic count coverage or vocabulary is incomplete or malformed"}
    return {
        "status": "observed",
        "feature_counts": deepcopy(counts),
        "scope": deepcopy(report.get("feature_observation")),
        "scope_sha256": report.get("scope_sha256"),
        "device_sha256": report.get("device_sha256"),
        "model_sha256": report.get("model_sha256"),
        "native_timing_unresolved": deepcopy(report.get("unresolved")),
        "timing_cycles": None,
        "physical_traffic_bytes": None,
        "interpretation": "executed descriptive counts and requested bytes; not physical traffic or utilization",
    }


def _screen(prediction):
    if not prediction:
        return {"status": "UNKNOWN", "estimated_cycles": None, "reason": "screen unavailable"}
    if not _positive(prediction.get("estimated_cycles")):
        return {
            "status": "UNKNOWN",
            "estimated_cycles": None,
            "reason": prediction.get("reason") or "finite positive screen estimate unavailable",
        }
    result = {
        key: deepcopy(prediction.get(key))
        for key in ("status", "estimated_cycles", "interval", "diagnostic_interval", "reason", "screen_terms")
    }
    result["interpretation"] = "correlated empirical screening estimates; not physical prices or measured cycles"
    result["attainment"] = None
    return result


def _measured(measurement, compiled, machine):
    """Require an explicit same-ELF/source-boundary/domain observation, not a prediction.

    Matching fields do not issue measurement qualification. The original caller
    must reopen its independently admitted cycle receipt before passing it here.
    """
    value = _mapping(measurement)
    if (
        value.get("kind") != "measured_target_cycles"
        or value.get("measured_now") is not True
        or value.get("functional_status") != "PASS"
        or compiled.get("status") != "observed"
        or not compiled.get("elf_sha256")
        or value.get("elf_sha256") != compiled["elf_sha256"]
        or value.get("complete_source_scope") is not True
        or any(not machine.get(key) or value.get(key) != machine[key] for key in ("scope_sha256", "device_sha256"))
        or not _positive(value.get("cycles"))
    ):
        return None
    return value["cycles"]


def join_capsule_feedback(
    member, compiled, raw_report=None, prediction=None, *, target, facts=None, machine=None, measurement=None
):
    """Join caller-owned public data; launch/read no files and grant no qualification.

    ``compiled`` is the ordinary compiled observation. ``raw_report`` is its
    complete original count-v2 report, not a saved timing verdict. ``prediction``
    is one existing timing-screen comparison case. Optional ``machine`` is the
    selected capsule_roofline machine document; omission derives it from the
    supplied facts only, never an ambient target. For measured partial-floor
    diagnostics, its explicit scope/device identities must match ``measurement``.
    Source logical payload and requested DMA bytes never price movement here.
    """
    member, compiled = _mapping(member), _mapping(compiled)
    diagnostics = _mapping(compiled.get("candidate_diagnostics"))
    features = _mapping(_mapping(diagnostics.get("candidate_features")).get("features"))
    macs = _count(features.get("source_macs")) if compiled.get("status") == "observed" else None
    selected = (
        deepcopy(dict(machine)) if isinstance(machine, Mapping) else R.machine_bounds(target, facts=_mapping(facts))
    )
    # A supplied machine cannot replace an arithmetic rate derivable from facts.
    if facts is not None:
        peak, reason = R.compute_peak_from_facts(_mapping(facts))
        selected["peak_macs_per_cycle"] = peak
        if peak is None:
            selected.setdefault("unresolved", {})["compute"] = reason
    contractions = None if macs is None else [(1, macs, 1)] if macs else []
    floor = R.roofline(contractions, None, None, selected)
    floor["unresolved"].update(_MISSING)
    floor["domain"] = "conditional useful source MAC demand against the selected arithmetic-site upper bound"
    floor["convergence_eligible"] = False
    floor["authority"] = "partial lower-bound diagnostic only"
    measured = _measured(measurement, compiled, selected)
    bound = floor["roofline_cycles"]
    comparable = measured is not None and _positive(bound)
    refuted = comparable and measured < bound
    roofline = {
        "floor": floor,
        "measured_cycles": measured,
        "status": "floor_refuted" if refuted else "partial_measured" if comparable else "UNKNOWN",
        "attainment": bound / measured if comparable and not refuted else None,
        "gap_cycles": measured - bound if comparable else None,
        "convergence_eligible": False,
        "measurement_qualification": "caller-owned; not issued by this join",
    }
    return public_projection(
        {
            "schema": SCHEMA,
            "authority": "none",
            "promotion_authority": "NONE",
            "member": {
                key: deepcopy(member.get(key))
                for key in ("name", "kind", "label", "source_reference", "source_sha256", "numeric_policy")
            },
            "source_work": {
                "status": "observed" if macs is not None else "UNKNOWN",
                "macs": macs,
                "features": deepcopy(dict(features)),
                "operation": _mapping(member.get("operation")).get("op"),
                "inputs": [
                    {key: deepcopy(_mapping(row).get(key)) for key in ("name", "shape", "dtype")}
                    for row in member.get("inputs", [])
                ]
                if isinstance(member.get("inputs"), list)
                else None,
                "basis": "original compiled candidate_diagnostics.candidate_features.source_macs",
                "logical_payload": "source tensor/host payload is not physical transfer volume",
                "semantic_equivalence": "caller-owned complete numerical gate required",
            },
            "compiled": {
                "status": compiled.get("status", "UNKNOWN"),
                "elf_sha256": compiled.get("elf_sha256"),
                "llvm_mlir_sha256": compiled.get("llvm_mlir_sha256"),
                "static": deepcopy(compiled.get("report")),
                "placement": deepcopy(diagnostics.get("placement")),
                "representations": deepcopy(diagnostics.get("representations")),
                "dependency_stages": deepcopy(diagnostics.get("dependency_stages")),
                "interpretation": "emitted declarations and static sites do not prove dynamic effects",
            },
            "host_and_packing": {
                "host": deepcopy(diagnostics.get("host")),
                "cost_graph": deepcopy(diagnostics.get("cost_graph")),
                "complete_call_cycles": None,
                "physical_traffic_bytes": None,
                "status": "UNKNOWN",
                "missing": deepcopy(_MISSING),
            },
            "dynamic": _dynamic(_mapping(raw_report), compiled),
            "screen": _screen(_mapping(prediction)),
            "machine": selected,
            "roofline": roofline,
            "timing_cycles": measured,
            "convergence_eligible": False,
        }
    )
