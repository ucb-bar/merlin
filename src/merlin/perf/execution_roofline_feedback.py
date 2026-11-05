"""Pure same-window roofline/count arithmetic; no execution or timing authority."""

import math
from copy import deepcopy

SCHEMA = "merlin.perf.execution_roofline_feedback.v1"


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _count(value):
    return type(value) is int and 0 <= value < 2**64


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _join(counts, estimate, model, scope):
    """Check transport consistency, never reconstruct an owner from its projection."""
    if (
        type(counts) is not dict
        or counts.get("schema") != "merlin.public_native_observed_counts.v1"
        or counts.get("status") != "observed"
        or not _digest(model)
        or type(counts.get("provenance")) is not dict
    ):
        raise ValueError("require original observed-count projection and explicit selected model digest")
    provenance = counts["provenance"]
    if (
        type(scope) is not dict
        or scope != provenance.get("scope_binding")
        or scope.get("schema") != "merlin.spike_scope_binding.v1"
        or any(not _digest(scope.get(k)) for k in ("scope_sha256", "elf_sha256", "input_sha256", "evidence_sha256"))
        or any(not _count(scope.get(k)) for k in ("begin_pc", "end_pc", "occurrence"))
        or scope["begin_pc"] >= scope["end_pc"]
        or any(
            not _digest(provenance.get(k))
            for k in (
                "report_sha256",
                "receipt_sha256",
                "runtime_sha256",
                "numerical_receipt_sha256",
                "elf_sha256",
                "input_sha256",
                "capsule_sha256",
            )
        )
        or any(scope[k] != provenance[k] for k in ("elf_sha256", "input_sha256"))
    ):
        raise ValueError("source roofline and counts require the same explicit original counter-window binding")
    if (
        type(estimate) is not dict
        or estimate.get("schema") != "merlin.public_native_exploratory_estimate.v1"
        or estimate.get("qualified_for_promotion") is not False
        or estimate.get("model_sha256") != model
        or estimate.get("report_sha256") != provenance["report_sha256"]
        or estimate.get("scope_sha256") != scope["scope_sha256"]
        or estimate.get("status") not in {"estimated", "unavailable"}
    ):
        raise ValueError("exploratory model/report/scope do not match original observed counts")
    cycles = estimate.get("estimated_cycles")
    if (estimate["status"] == "estimated" and not _positive(cycles)) or (
        estimate["status"] == "unavailable" and cycles is not None
    ):
        raise ValueError("same-window estimate needs finite positive cycles or explicit unavailability")
    metrics = counts.get("metrics")
    if type(metrics) is not dict or not 1 <= len(metrics) <= 256 or "scope_instructions" not in metrics:
        raise ValueError("require a bounded original metric inventory including scoped instructions")
    for name, row in metrics.items():
        if (
            type(name) is not str
            or type(row) is not dict
            or set(row) != {"value", "unit"}
            or not _count(row["value"])
            or row["unit"]
            not in {
                "instructions",
                "events",
                "commands",
                "fences",
                "encoded requested bytes",
                "encoded local operand bytes",
            }
        ):
            raise ValueError("count metrics require explicit original units and bounded integer values")
    if metrics["scope_instructions"]["unit"] != "instructions":
        raise ValueError("scoped host retirement inventory must have instruction units")
    return provenance, metrics, cycles


def _work(partial):
    if (
        type(partial) is not dict
        or partial.get("schema") != "merlin.perf.partial_roofline.v1"
        or partial.get("whole_call_cycles") is not None
        or partial.get("complete_attainment") is not None
        or partial.get("convergence_eligible") is not False
    ):
        raise ValueError("require original partial roofline without whole-call or attainment claims")
    useful, peak, floor = (
        partial.get(k)
        for k in (
            "useful_macs",
            "structural_peak_macs_per_cycle",
            "structural_compute_floor_cycles",
        )
    )
    if (
        useful is not None
        and not _count(useful)
        or peak is not None
        and (not _count(peak) or peak == 0)
        or floor is not None
        and not _count(floor)
    ):
        raise ValueError("structural work, rate and floor must be bounded counts or UNKNOWN")
    expected = 0 if useful == 0 else (useful + peak - 1) // peak if useful is not None and peak else None
    if floor != expected:
        raise ValueError("partial compute floor differs from its original useful work and structural peak")
    padded = partial.get("padded_work")
    if (
        type(padded) is not dict
        or padded.get("compulsory_workload_floor") is not False
        or padded.get("macs") is not None
        and (not _count(padded["macs"]) or useful is not None and padded["macs"] < useful)
    ):
        raise ValueError("padded schedule work cannot replace useful compulsory work")
    return useful, floor


def _requests(partial, metrics, names):
    logical = partial.get("logical_request_ideal_service")
    if type(logical) is not dict or set(logical) != {"read", "write"}:
        raise ValueError("partial roofline requires both explicit logical request directions")
    if names is not None and (type(names) is not dict or set(names) != {"read", "write"}):
        raise ValueError("select both original request metric names explicitly; do not mix byte namespaces")
    supplied = partial.get("provided_client_request_ideal_service")
    if supplied is not None and names is None:
        raise ValueError("provided client requests need explicit matching original observed metric names")
    result = {}
    for direction in ("read", "write"):
        row = logical[direction]
        if type(row) is not dict:
            raise ValueError("logical request direction is missing")
        demand, rate = row.get("requested_bytes"), row.get("client_peak_bytes_per_cycle")
        if demand is not None and not _count(demand) or rate is not None and (not _count(rate) or rate == 0):
            raise ValueError("request bytes/rate must be bounded counts or UNKNOWN")
        observed = None
        if names is not None:
            if type(names[direction]) is not str:
                raise ValueError("observed request metric names must be explicit strings")
            metric = metrics.get(names[direction])
            if type(metric) is not dict or metric["unit"] != "encoded requested bytes":
                raise ValueError("selected observed request metric is missing or has different byte units")
            observed = metric["value"]
            if supplied is not None and supplied["service"][direction]["requested_bytes"] != observed:
                raise ValueError("partial provided requests differ from the original scoped count report")
        result[direction] = {
            "observed_metric": names[direction] if names is not None else None,
            "logical_bytes": demand,
            "observed_encoded_request_bytes": observed,
            "client_peak_bytes_per_cycle": rate,
            "logical_ideal_service_cycles": demand / rate if demand is not None and rate else None,
            "observed_ideal_service_cycles": observed / rate if observed is not None and rate else None,
            "request_to_logical_ratio": observed / demand if observed is not None and demand else None,
            "compulsory_floor": False,
            "physical_traffic": None,
        }
    return result


def _host_service(selection, instructions):
    if selection is None:
        return None
    if (
        type(selection) is not dict
        or set(selection) != {"widths", "rtl_proof", "counter_boundary_policy"}
        or type(selection["widths"]) is not dict
        or not selection["widths"]
        or set(selection["widths"]) - {"issue", "retire"}
        or type(selection["rtl_proof"]) is not dict
        or set(selection["rtl_proof"]) != {"source_sha256", "statement"}
        or not _digest(selection["rtl_proof"]["source_sha256"])
        or type(selection["rtl_proof"]["statement"]) is not str
        or not selection["rtl_proof"]["statement"]
        or type(selection["counter_boundary_policy"]) is not str
        or not selection["counter_boundary_policy"]
        or any(not _count(n) or n == 0 for n in selection["widths"].values())
    ):
        raise ValueError("host service requires explicit RTL widths, source proof and counter-boundary policy")
    return {
        **deepcopy(selection),
        "observed_scope_instructions": instructions,
        "conditional_service_cycles": {k: instructions / n for k, n in selection["widths"].items()},
        "universal_host_floor": False,
        "boundary_warning": "counter opening/closing start and retirement conventions may differ",
        "added_to_compute_or_DMA": False,
    }


def build_feedback(
    partial_roofline,
    observed_counts,
    exploratory_estimate,
    *,
    model_sha256,
    roofline_scope_binding,
    request_metrics=None,
    host_service=None,
    modeled_cycle_breakdown=None,
):
    """Join already verified projections without issuing proof from saved data.

    The caller must independently derive source work/geometry and bind it to the
    selected complete window. Equality of supplied hashes proves no semantics.
    Optional request names choose one count interface; byte namespaces are never
    guessed. Optional host widths need caller-owned RTL and counter-boundary
    proof. Their service ideals are never added to device terms or refutations.
    """
    provenance, metrics, cycles = _join(
        observed_counts,
        exploratory_estimate,
        model_sha256,
        roofline_scope_binding,
    )
    useful, floor = _work(partial_roofline)
    requests = _requests(partial_roofline, metrics, request_metrics)
    host = _host_service(host_service, metrics["scope_instructions"]["value"])
    warnings = []
    if modeled_cycle_breakdown is not None:
        if type(modeled_cycle_breakdown) is not dict or type(modeled_cycle_breakdown.get("scope")) is not str:
            raise ValueError("modeled breakdown must declare its own scope")
        if modeled_cycle_breakdown["scope"] == "whole_trace":
            warnings.append("whole_trace breakdown has a different scope; no kernel decomposition or percentages")
        else:
            warnings.append(
                "modeled breakdown is not additive without independent same-window dependency/overlap proof"
            )
    if exploratory_estimate.get("event_accounting_complete") is not True:
        warnings.append("fixed-model event accounting is incomplete; unresolved event prices remain UNKNOWN")
    refuted = cycles is not None and floor is not None and cycles < floor
    ratio = floor / cycles if floor is not None and cycles else None
    if ratio is not None and not math.isfinite(ratio):
        raise ValueError("compute floor/estimate ratio exceeds finite report bounds")
    return {
        "schema": SCHEMA,
        "status": "REFUTED" if refuted else "EXPLORATORY" if cycles is not None else "UNKNOWN",
        "refutation": "estimate is below the structural useful-compute floor" if refuted else None,
        "provenance": {**deepcopy(provenance), "model_sha256": model_sha256},
        "estimated_total_cycles_same_window": cycles,
        "useful_macs": useful,
        "padded_work": deepcopy(partial_roofline["padded_work"]),
        "missing_roofline_terms": deepcopy(partial_roofline.get("missing_terms", [])),
        "estimate_caveats": {
            k: deepcopy(exploratory_estimate.get(k))
            for k in (
                "event_accounting_complete",
                "unresolved_events",
                "domain_deviations",
                "uncertainty_interval_cycles",
            )
        },
        "structural_compute_floor_cycles": floor,
        "partial_compute_floor_to_estimate_ratio": ratio,
        "ratio_is_complete_attainment": False,
        "conditional_requested_service": requests,
        "conditional_host_service": host,
        "observed_metrics": deepcopy(metrics),
        "command_class_counts": {k: deepcopy(v) for k, v in metrics.items() if k.startswith("target.class.")},
        "host_instruction_counts": {
            k: deepcopy(v) for k, v in metrics.items() if k == "scope_instructions" or k.startswith("cpu.")
        },
        "modeled_breakdown_used_for_kernel_decomposition": False,
        "warnings": warnings,
        "unknown": {
            "host_exact_cost": None,
            "cache_or_DRAM_traffic": None,
            "physical_ports": None,
            "overlap": None,
            "complete_attainment": None,
            "measured_cycles": None,
        },
        "composition": "no CPU + DMA + compute sum; no dependency/overlap proof supplied",
        "caller_obligations": (
            "original live owners must verify products, model, source work, geometry and complete window"
        ),
        "promotion_authority": "NONE",
        "incumbent_eligible": False,
        "convergence_eligible": False,
        "authority": "none; pure input consistency and conditional arithmetic only",
    }
