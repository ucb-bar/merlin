"""Keep necessary service bounds separate from independently feasible schedules.

Work divided by a structural rate ceiling bounds optimal execution from below.
A demonstrated independent schedule can bound it from above in its exact cell.
A slow microprobe is not a universal throughput ceiling. This projection never
fits prices, extrapolates probe geometries, or issues qualification/promotion.
"""

from __future__ import annotations

import math
from copy import deepcopy

SCHEMA = "merlin.perf.attainable_envelope.v1"


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cycles(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def project(
    service_floor,
    *,
    workload_sha256,
    scope_sha256,
    observed_cycles,
    observation_kind,
    independent_reference=None,
    compute_floor_cycles=None,
):
    """Project caller-checked same-cell inputs, with UNKNOWNs retained explicitly.

    ``service_floor`` is the existing ``compose_service_floor`` result, using
    compulsory workload demands. A reference must cover the *same complete
    workload and scope*, not a different microprobe normalized by useful work.
    Its finite upper cost is a feasible-schedule witness for its stated domain;
    it is not a necessary cost or a price for another geometry. Callers own all
    source, measurement, independence and validity-domain checks.

    Even a complete pure arithmetic projection confers no timing authority.
    Estimated and measured observations retain distinct labels. Below-bound
    observations keep their raw values but supply no attainment fraction.
    """
    if (
        type(service_floor) is not dict
        or service_floor.get("schema") != "merlin.perf.resource_service_floor.v1"
        or service_floor.get("scope") != "workload"
        or service_floor.get("convergence_eligible") is not False
        or type(service_floor.get("unresolved")) is not dict
        or not _sha(workload_sha256)
        or not _sha(scope_sha256)
        or observation_kind not in {"estimated", "measured"}
        or observed_cycles is not None
        and not _cycles(observed_cycles)
    ):
        raise ValueError("explicit compulsory service floor, exact workload/scope and cycle observation required")
    lower = service_floor.get("floor_cycles")
    if lower is not None and not _cycles(lower):
        raise ValueError("theoretical service floor must be positive or UNKNOWN")
    if compute_floor_cycles is not None and not _cycles(compute_floor_cycles):
        raise ValueError("compute-only floor must be positive or UNKNOWN")
    if lower is not None and compute_floor_cycles is not None and compute_floor_cycles > lower:
        raise ValueError("complete service floor cannot be below its compute-only term")
    reference = deepcopy(independent_reference)
    upper = None
    if reference is not None:
        if (
            type(reference) is not dict
            or set(reference)
            != {
                "workload_sha256",
                "scope_sha256",
                "complete_invocation",
                "independent",
                "cycles_lower",
                "cycles_upper",
                "domain",
                "evidence_sha256",
            }
            or reference["workload_sha256"] != workload_sha256
            or reference["scope_sha256"] != scope_sha256
            or reference["complete_invocation"] is not True
            or reference["independent"] is not True
            or type(reference["domain"]) is not str
            or not reference["domain"]
            or not _sha(reference["evidence_sha256"])
            or not _cycles(reference["cycles_lower"])
            or not _cycles(reference["cycles_upper"])
            or reference["cycles_lower"] > reference["cycles_upper"]
        ):
            raise ValueError(
                "independent feasible schedule must cover the same complete workload/scope and explicit domain"
            )
        upper = reference["cycles_upper"]
        if lower is not None and upper < lower:
            raise ValueError("feasible schedule refutes the theoretical service floor")
    inconsistent = observed_cycles is not None and lower is not None and observed_cycles < lower
    partial = bool(service_floor["unresolved"])
    ratio = lower / observed_cycles if lower is not None and observed_cycles is not None and not inconsistent else None
    interval = [ratio, upper / observed_cycles] if ratio is not None and upper is not None else None
    return dict(
        schema=SCHEMA,
        workload_sha256=workload_sha256,
        scope_sha256=scope_sha256,
        observation_kind=observation_kind,
        observed_cycles=observed_cycles,
        theoretical_floor_cycles=lower,
        theoretical_floor_ratio=ratio,
        independent_feasible_cycles_upper=upper,
        conditional_optimal_cycles_interval=[lower, upper],
        conditional_attainment_interval=interval,
        # A structural bound and independent witness do not qualify the entire
        # machine/domain or establish a physically attainable universal rate.
        complete_attainable_roofline_cycles=None,
        complete_attainment=None,
        compute_peak_utilization_fraction=(
            compute_floor_cycles / observed_cycles
            if compute_floor_cycles is not None and observed_cycles is not None and not inconsistent
            else None
        ),
        comparison_consistent=not inconsistent,
        reason=(("estimate" if observation_kind == "estimated" else "measurement") + "_below_structural_bound")
        if inconsistent
        else "incomplete_service_inventory"
        if partial
        else "theoretical_floor_unavailable"
        if lower is None
        else None,
        unresolved=deepcopy(service_floor["unresolved"]),
        independent_reference=reference,
        convergence_eligible=False,
        qualification="none; conditional arithmetic over caller-bound evidence",
        interpretation=(
            "necessary lower bound and independently feasible upper witness; "
            "never a fitted throughput ceiling"
        ),
    )
