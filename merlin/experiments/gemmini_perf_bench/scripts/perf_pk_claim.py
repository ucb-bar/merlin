"""Pure decision procedure for the Arm4 PK affine-reduction claim.

The expensive runner owns files, sandboxes, and sealing.  This module owns only
the mathematical decision over already-frozen capsule descriptors and already-
authored result rows.  It deliberately performs no discovery and no I/O: a
caller must pass the exact descriptor and result objects whose digests it has
verified.  Invalid or incomplete evidence is ``REFUSED``; complete evidence
that misses a predeclared bound is ``REFUTED``; no missing fact is inferred.
"""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any


_SIMPLE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_FAMILY = "PK"
_CLAIM = "PREDICTS"
_SCHEMA_VERSION = 1

# This mirrors the reviewed declaration in contract/capsules/profiles/_perf.yaml.
# Requiring exact equality makes a profile edit a deliberate analyzer-version
# edit too; a relaxed threshold cannot silently become eligible for promotion.
_EXPECTED_ACCEPTANCE: dict[str, Any] = {
    "schema_version": 1,
    "analyzer": "perf_pk_claim.analyze_pk_claim/v1",
    "fit": {
        "form": "affine",
        "method": "ordinary_least_squares_all_L3_replicates",
        "independent_variable": "K",
        "dependent_metric": "verilator_L3_cycles",
    },
    "cohort": {
        "operation": "matmul",
        "fixed_shape_axes": ["M", "N"],
        "fixed_fields": ["operation", "operand_dtype", "accum_dtype", "epilogue"],
        "exact_points": 4,
        "K_multipliers_of_tile": [1, 2, 4, 8],
    },
    "replicates": {
        "exact_count": 3,
        "identities": ["r000", "r001", "r002"],
    },
    "evidence": {
        "correctness_simulator": "spike",
        "correctness_tier": "L2",
        "timing_simulator": "verilator",
        "timing_tier": "L3",
        "spike_cycles_citable": False,
    },
    "thresholds": {
        "slope_cycles_per_K_element_min_exclusive": 0.0,
        "r_squared_min_inclusive": 0.995,
        "residual_bound": {
            "predicate": "abs_residual_le_max_of_floor_and_fraction_of_observed",
            "absolute_floor_cycles": 8,
            "observed_cycle_fraction": 0.03,
        },
    },
}


class _Refusal(ValueError):
    pass


def supported_acceptance() -> dict[str, Any]:
    """Return a copy of the only claim contract this analyzer can decide."""
    return copy.deepcopy(_EXPECTED_ACCEPTANCE)


def _exact_declaration_equal(observed: object, expected: object) -> bool:
    """Compare declaration data without Python's ``False == 0`` coercion."""
    if type(observed) is not type(expected):  # noqa: E721 - type identity is the security property
        return False
    if isinstance(expected, dict):
        return (observed.keys() == expected.keys()
                and all(_exact_declaration_equal(observed[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(observed) == len(expected)
                and all(_exact_declaration_equal(left, right)
                        for left, right in zip(observed, expected, strict=True)))
    return observed == expected


def _refused(reason: str, *, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": "REFUSED",
        "method": copy.deepcopy(_EXPECTED_ACCEPTANCE["fit"]),
        "declaration": (copy.deepcopy(preflight.get("declaration"))
                        if preflight is not None and preflight.get("declaration") is not None
                        else None),
        "cohort": (copy.deepcopy(preflight.get("cohort"))
                   if preflight is not None and preflight.get("cohort") is not None else None),
        "evidence": None,
        "fit": None,
        "checks": None,
        "refusal_reasons": [reason],
        "refutation_reasons": [],
    }


def _mapping(value: object, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{owner} must be a mapping")
    return value


def _sequence(value: object, owner: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _Refusal(f"{owner} must be a sequence")
    return value


def _positive_extent(value: object, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _Refusal(f"{owner} must be a positive integer")
    return value


def _descriptor_point(raw: object) -> dict[str, Any]:
    descriptor = _mapping(raw, "PK descriptor")
    name = descriptor.get("name")
    if not isinstance(name, str) or not _SIMPLE_NAME.fullmatch(name):
        raise _Refusal("every PK descriptor must have a simple non-empty name")
    if descriptor.get("kind") != "model_slice" or descriptor.get("label") != "dev":
        raise _Refusal(f"PK descriptor {name!r} is not the generated dev model_slice")

    performance = _mapping(descriptor.get("performance"), f"PK descriptor {name} performance")
    required_perf = {
        "level": "L1_tile",
        "family": _FAMILY,
        "lever": "reduction_depth",
        "claim": _CLAIM,
    }
    if any(performance.get(key) != value for key, value in required_perf.items()):
        raise _Refusal(f"PK descriptor {name!r} changes its family, level, lever, or claim")
    if not _exact_declaration_equal(performance.get("acceptance"), _EXPECTED_ACCEPTANCE):
        raise _Refusal(f"PK descriptor {name!r} has a missing, malformed, or unsupported acceptance contract")
    comparand = _mapping(performance.get("comparand"), f"PK descriptor {name} comparand")
    if (comparand.get("kind") != "fitted_prediction"
            or comparand.get("against") != "measured_cycles_same_member"):
        raise _Refusal(f"PK descriptor {name!r} changes the fitted-prediction comparand")
    falsifier = _mapping(performance.get("falsifier"), f"PK descriptor {name} falsifier")
    if (falsifier.get("observation") != "residual_cycles_by_K"
            or falsifier.get("negative_control") != "fixed_M_and_N_across_all_K_points"):
        raise _Refusal(f"PK descriptor {name!r} changes its residual falsifier or negative control")

    operation = _mapping(descriptor.get("operation"), f"PK descriptor {name} operation")
    attributes = _mapping(operation.get("attributes"), f"PK descriptor {name} attributes")
    if operation.get("op") != "matmul" or attributes.get("epilogue") != []:
        raise _Refusal(f"PK descriptor {name!r} must be an unfused matmul")
    lhs_name, weight_name = attributes.get("lhs"), attributes.get("weight")
    if not isinstance(lhs_name, str) or not isinstance(weight_name, str) or lhs_name == weight_name:
        raise _Refusal(f"PK descriptor {name!r} has no unambiguous lhs/weight binding")

    inputs = _sequence(descriptor.get("inputs"), f"PK descriptor {name} inputs")
    tensors: dict[str, Mapping[str, Any]] = {}
    for index, raw_tensor in enumerate(inputs):
        tensor = _mapping(raw_tensor, f"PK descriptor {name} input {index}")
        tensor_name = tensor.get("name")
        if not isinstance(tensor_name, str) or not tensor_name or tensor_name in tensors:
            raise _Refusal(f"PK descriptor {name!r} has missing or duplicate input names")
        tensors[tensor_name] = tensor
    if lhs_name not in tensors or weight_name not in tensors:
        raise _Refusal(f"PK descriptor {name!r} lhs/weight binding does not name its inputs")
    lhs_shape = _sequence(tensors[lhs_name].get("shape"), f"PK descriptor {name} lhs shape")
    weight_shape = _sequence(tensors[weight_name].get("shape"), f"PK descriptor {name} weight shape")
    if len(lhs_shape) != 2 or len(weight_shape) != 2:
        raise _Refusal(f"PK descriptor {name!r} operands must be rank-two")
    m, lhs_k = (_positive_extent(lhs_shape[0], f"{name}.M"),
                _positive_extent(lhs_shape[1], f"{name}.lhs_K"))
    weight_k, n = (_positive_extent(weight_shape[0], f"{name}.weight_K"),
                   _positive_extent(weight_shape[1], f"{name}.N"))
    if lhs_k != weight_k:
        raise _Refusal(f"PK descriptor {name!r} has inconsistent contracted K dimensions")

    lhs_dtype = tensors[lhs_name].get("dtype")
    weight_dtype = tensors[weight_name].get("dtype")
    if not isinstance(lhs_dtype, str) or not lhs_dtype or lhs_dtype != weight_dtype:
        raise _Refusal(f"PK descriptor {name!r} operands must share one declared dtype")
    accum_dtype = attributes.get("output_dtype")
    numeric = _mapping(descriptor.get("numeric_policy"), f"PK descriptor {name} numeric policy")
    if (not isinstance(accum_dtype, str) or not accum_dtype
            or numeric.get("dtype") != accum_dtype or numeric.get("compare") != "exact_int"):
        raise _Refusal(f"PK descriptor {name!r} has no exact common accumulator/output dtype")
    tiers = _sequence(descriptor.get("required_oracle_tiers"), f"PK descriptor {name} tiers")
    if not {"L2", "L3"}.issubset(set(tiers)):
        raise _Refusal(f"PK descriptor {name!r} does not require both L2 and L3")
    return {
        "capsule": name,
        "M": m,
        "N": n,
        "K": lhs_k,
        "operation": "matmul",
        "operand_dtype": lhs_dtype,
        "accum_dtype": accum_dtype,
        "epilogue": [],
    }


def _validate_descriptors(descriptors: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_descriptors = _sequence(descriptors, "PK descriptors")
    if len(raw_descriptors) != 4:
        raise _Refusal("PK requires exactly four frozen capsule descriptors")
    points = [_descriptor_point(raw) for raw in raw_descriptors]
    if len({point["capsule"] for point in points}) != 4:
        raise _Refusal("PK frozen descriptors repeat a capsule name")

    fixed_fields = ("M", "N", "operation", "operand_dtype", "accum_dtype", "epilogue")
    for field in fixed_fields:
        values = {repr(point[field]) for point in points}
        if len(values) != 1:
            raise _Refusal(f"PK cohort control changed fixed field {field!r}")
    tile = points[0]["M"]
    if points[0]["N"] != tile:
        raise _Refusal("PK cohort requires fixed single-tile M and N")
    by_k = {point["K"]: point for point in points}
    expected_k = [tile * factor for factor in (1, 2, 4, 8)]
    if sorted(by_k) != expected_k:
        raise _Refusal(
            f"PK cohort must contain exactly K={expected_k}, observed {sorted(by_k)}")
    ordered = [by_k[k] for k in expected_k]
    for index, point in enumerate(ordered):
        expected_name = f"PK{index:02d}_k{point['K']}"
        if point["capsule"] != expected_name:
            raise _Refusal(
                f"PK K={point['K']} member is {point['capsule']!r}, expected {expected_name!r}")

    cohort = {
        "negative_control": "fixed_M_and_N_across_all_K_points",
        "operation": points[0]["operation"],
        "M": tile,
        "N": tile,
        "operand_dtype": points[0]["operand_dtype"],
        "accum_dtype": points[0]["accum_dtype"],
        "epilogue": [],
        "capsules": [point["capsule"] for point in ordered],
        "K_values": expected_k,
        "K_multipliers_of_tile": [1, 2, 4, 8],
        "replicates": ["r000", "r001", "r002"],
    }
    return ordered, cohort


def preflight_pk_claim(descriptors: object) -> dict[str, Any]:
    """Validate frozen declarations before a claim-bearing run is admitted."""
    try:
        points, cohort = _validate_descriptors(descriptors)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": _SCHEMA_VERSION,
            "family": _FAMILY,
            "claim": _CLAIM,
            "status": "REFUSED",
            "declaration": None,
            "cohort": None,
            "expected_identities": [],
            "refusal_reasons": [str(exc)],
        }
    expected = []
    for point in points:
        for replicate in _EXPECTED_ACCEPTANCE["replicates"]["identities"]:
            for simulator, tier in (("spike", "L2"), ("verilator", "L3")):
                expected.append({
                    "family": _FAMILY,
                    "capsule": point["capsule"],
                    "simulator": simulator,
                    "replicate": replicate,
                    "tier": tier,
                })
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": "READY",
        "declaration": supported_acceptance(),
        "cohort": cohort,
        "expected_identities": expected,
        "refusal_reasons": [],
    }


def _validate_results(results: object, points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _sequence(results, "PK result rows")
    expected_capsules = {str(point["capsule"]) for point in points}
    pk_rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"result row {index}")
        identity = _mapping(row.get("identity"), f"result row {index} identity")
        if identity.get("family") == _FAMILY:
            pk_rows.append(row)
    expected_count = len(points) * 3 * 2
    if len(pk_rows) != expected_count:
        raise _Refusal(
            f"PK requires exactly {expected_count} result rows, observed {len(pk_rows)}")

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in pk_rows:
        identity = _mapping(row.get("identity"), "PK result identity")
        capsule = identity.get("capsule")
        simulator = identity.get("simulator")
        replicate = identity.get("replicate")
        if (capsule not in expected_capsules or simulator not in ("spike", "verilator")
                or replicate not in ("r000", "r001", "r002")):
            raise _Refusal(f"PK result has an undeclared identity: {dict(identity)!r}")
        key = (str(capsule), str(simulator), str(replicate))
        if key in indexed:
            raise _Refusal(f"PK result repeats identity {key!r}")
        indexed[key] = row

    observations: list[dict[str, Any]] = []
    for point in points:
        capsule, k = str(point["capsule"]), int(point["K"])
        for replicate in ("r000", "r001", "r002"):
            for simulator, tier, purpose, citable in (
                    ("spike", "L2", "correctness_screen", False),
                    ("verilator", "L3", "performance_certification", True)):
                key = (capsule, simulator, replicate)
                row = indexed.get(key)
                if row is None:
                    raise _Refusal(f"PK result is missing identity {key!r}")
                identity = _mapping(row.get("identity"), f"PK result {key!r} identity")
                if not {"family", "capsule", "simulator", "replicate"}.issubset(identity):
                    raise _Refusal(f"PK result {key!r} has an incomplete identity")
                if (row.get("approach") != "arm4" or row.get("tier") != tier
                        or row.get("purpose") != purpose or row.get("citable") is not citable):
                    raise _Refusal(f"PK result {key!r} changes its Arm4 L2/L3 evidence semantics")
                if (row.get("correct") is not True or row.get("tier_status") != "pass"
                        or row.get("grade_status") != "pass"
                        or row.get("numeric_status") != "pass"
                        or row.get("error") is not None or row.get("failure") is not None):
                    raise _Refusal(f"PK result {key!r} is not a correct successful measurement")
                cycles = row.get("cycles")
                if simulator == "spike":
                    if cycles is not None:
                        raise _Refusal(f"PK Spike L2 result {key!r} exposes citable cycles")
                else:
                    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
                        raise _Refusal(f"PK Verilator L3 result {key!r} lacks positive integer cycles")
                    observations.append({
                        "capsule": capsule,
                        "replicate": replicate,
                        "K": k,
                        "cycles": cycles,
                    })
    return observations


def _fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _fit(observations: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    xs = [int(row["K"]) for row in observations]
    ys = [int(row["cycles"]) for row in observations]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denominator = n * sxx - sx * sx
    if denominator == 0:
        raise _Refusal("PK K values are degenerate; an affine rate is not determined")
    slope = Fraction(n * sxy - sx * sy, denominator)
    intercept = Fraction(sy, n) - slope * Fraction(sx, n)
    residuals = [Fraction(y) - (slope * x + intercept)
                 for x, y in zip(xs, ys, strict=True)]
    mean_y = Fraction(sy, n)
    ss_res = sum((residual * residual for residual in residuals), Fraction())
    ss_total = sum(((Fraction(y) - mean_y) ** 2 for y in ys), Fraction())
    r_squared = None if ss_total == 0 else Fraction(1) - ss_res / ss_total
    floor = Fraction(8)
    fraction = Fraction(3, 100)
    allowed = [max(floor, fraction * y) for y in ys]
    within = [abs(residual) <= bound for residual, bound in zip(residuals, allowed, strict=True)]
    details = []
    for row, residual, bound, accepted in zip(
            observations, residuals, allowed, within, strict=True):
        details.append({
            "capsule": row["capsule"],
            "replicate": row["replicate"],
            "K": row["K"],
            "observed_cycles": row["cycles"],
            "predicted_cycles": float(Fraction(int(row["cycles"])) - residual),
            "residual_cycles": float(residual),
            "absolute_residual_cycles": float(abs(residual)),
            "allowed_absolute_residual_cycles": float(bound),
            "within_predeclared_bound": accepted,
        })
    rmse = math.sqrt(float(ss_res / n))
    max_abs = max(abs(value) for value in residuals)
    max_relative = max(abs(value) / y for value, y in zip(residuals, ys, strict=True))
    by_k = []
    for k in sorted(set(xs)):
        values = [y for x, y in zip(xs, ys, strict=True) if x == k]
        by_k.append({
            "K": k,
            "replicate_cycles": values,
            "mean_cycles": float(Fraction(sum(values), len(values))),
            "min_cycles": min(values),
            "max_cycles": max(values),
        })
    checks = {
        "positive_rate": slope > 0,
        "r_squared_at_least_0_995": r_squared is not None and r_squared >= Fraction(995, 1000),
        "every_residual_within_max_of_8_cycles_and_3_percent": all(within),
    }
    fit = {
        "form": "affine",
        "method": "ordinary_least_squares_all_L3_replicates",
        "equation": "verilator_L3_cycles = rate_cycles_per_K_element * K + intercept_cycles",
        "n_observations": n,
        "n_distinct_K": len(set(xs)),
        "rate_cycles_per_K_element": float(slope),
        "intercept_cycles": float(intercept),
        "exact_rate": _fraction_record(slope),
        "exact_intercept": _fraction_record(intercept),
        "r_squared": None if r_squared is None else float(r_squared),
        "rmse_cycles": rmse,
        "max_absolute_residual_cycles": float(max_abs),
        "max_relative_residual_fraction": float(max_relative),
        "sum_squared_residual_cycles": float(ss_res),
        "cycles_by_K": by_k,
        "residuals": details,
    }
    return fit, checks


def analyze_pk_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Decide PK from exact frozen descriptors and exact run-authored rows.

    Spike L2 rows are validated as correctness admissions but never enter the
    fit.  Every fitted observation is a correct, positive-cycle Verilator L3
    row with one of the exact predeclared identities.
    """
    preflight = preflight_pk_claim(descriptors)
    if preflight["status"] != "READY":
        return _refused(str(preflight["refusal_reasons"][0]), preflight=preflight)
    try:
        points, _cohort = _validate_descriptors(descriptors)
        observations = _validate_results(results, points)
        fit, checks = _fit(observations)
    except (KeyError, TypeError, ValueError) as exc:
        return _refused(str(exc), preflight=preflight)
    failed = [name for name, passed in checks.items() if not passed]
    status = "ESTABLISHED" if not failed else "REFUTED"
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": status,
        "method": copy.deepcopy(_EXPECTED_ACCEPTANCE["fit"]),
        "declaration": copy.deepcopy(preflight["declaration"]),
        "cohort": copy.deepcopy(preflight["cohort"]),
        "evidence": {
            "l2_correctness_rows_validated": 12,
            "l2_cycles_consumed": 0,
            "l3_positive_cycle_rows_consumed": len(observations),
            "timing_source": "verilator_L3_only",
            "arm": "arm4",
        },
        "fit": fit,
        "checks": checks,
        "refusal_reasons": [],
        "refutation_reasons": failed,
    }


# ---------------------------------------------------------------------------------------------------
# the refutation as a RESULT, not as a failure to fix
# ---------------------------------------------------------------------------------------------------
#
# A REFUTED decision names which checks missed their bounds and stops there. That reads as a tolerance
# problem, and a tolerance problem has an obvious repair -- move the tolerance -- which is exactly the
# repair this contract's own comment forbids. What distinguishes the two cases is the MECHANISM, and
# the mechanism is measurable on a target whose RTL counts the cycles each combination of its engines
# was busy.
#
# So this is deliberately a SEPARATE function and not a field of `analyze_pk_claim`. The decision is
# recomputed byte-for-byte by the reporting gate from the sealed rows; anything folded into it that the
# gate cannot reproduce from those rows alone would make every sealed campaign unverifiable. The
# mechanism is additive evidence beside the verdict, and it may never be the reason one moves.


def refutation_mechanism(decision: Mapping[str, Any], counter_values_by_capsule: Mapping[str, Any],
                         counters: Any) -> dict[str, Any]:
    """Why a REFUTED PK decision missed, from the same run's own joint-occupancy counters.

    ``counter_values_by_capsule`` maps each cohort capsule to the ``{counter name: value}`` readings a
    bracketed run of THAT capsule printed, and ``counters`` is the combination-counter set derived from
    the target's own shipped header (:func:`merlin.perf.hw_counters.counters_for_target`). Both arrive
    as parameters: this module decides, it does not discover.

    Two findings, and the first does not depend on the second:

    * ``affine_form_contradicted`` -- the marginal cost per unit of the fit axis falls strictly across
      every consecutive interval. An affine law asserts ONE constant marginal cost, so the cohort
      contradicts the model's form directly, with no threshold involved. This is the finding that
      settles what the repair is: a form the data contradicts is not repaired by a wider bound.
    * ``fill_transient`` -- realised overlap is still rising at the deepest point measured, so no point
      in the cohort priced a settled machine. That is the MECHANISM for the curvature, and it locates
      the repair in the ``cohort`` block (which depths were chosen), not in ``thresholds``.

    Refuses rather than guesses whenever a capsule's replicates disagree: collapsing them to a mean
    would invent a point the deterministic-simulator control was there to make unnecessary.
    """
    from merlin.perf import fill_transient as FT

    if not isinstance(decision, Mapping) or decision.get("family") != _FAMILY:
        raise _Refusal("the mechanism can only be read off a PK decision")
    fit = decision.get("fit")
    cohort = decision.get("cohort")
    if not isinstance(fit, Mapping) or not isinstance(cohort, Mapping):
        raise _Refusal("a REFUSED decision carries no fit or cohort, so it has no mechanism to read")

    capsule_of_k = dict(zip(cohort.get("K_values") or [], cohort.get("capsules") or [], strict=True))
    points, unread = [], []
    for entry in fit.get("cycles_by_K") or []:
        k = int(entry["K"])
        replicates = list(entry.get("replicate_cycles") or [])
        if not replicates or len(set(replicates)) != 1:
            raise _Refusal(
                f"PK K={k} replicates disagree ({sorted(set(replicates))}); the affine fit's own "
                "deterministic-simulator control did not hold, so this cohort has no single "
                "per-point cycle count to attribute a mechanism to")
        capsule = str(capsule_of_k[k])
        values = counter_values_by_capsule.get(capsule)
        if not values:
            unread.append(capsule)
            points.append(FT.Point(label=capsule, axis=k, cycles=int(replicates[0]),
                                   overlap_detail="no bracketed counter reading for this capsule"))
            continue
        points.append(FT.point_from_counter_values(capsule, k, int(replicates[0]), values, counters))

    verdict = FT.transient_verdict(points)
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "decision_status": decision.get("status"),
        "refutation_reasons": list(decision.get("refutation_reasons") or []),
        "independent_variable": _EXPECTED_ACCEPTANCE["fit"]["independent_variable"],
        "dependent_metric": _EXPECTED_ACCEPTANCE["fit"]["dependent_metric"],
        "capsules_without_counter_readings": unread,
        "mechanism": verdict,
        "repair_locus": (
            "cohort.K_multipliers_of_tile" if verdict["affine_form_contradicted"]
            and verdict["state"] == FT.IN_FILL_TRANSIENT else "undetermined"),
        "repair_note": (
            "every declared depth lies inside the machine's overlap fill transient, so the affine "
            "form was never testable on this cohort. Widening a threshold would accept a model the "
            "marginals contradict; the only honest repair is a NEW contract whose depths reach past "
            "saturation, and the acceptance block's own comment says that is a new experiment."
            if verdict["affine_form_contradicted"] and verdict["state"] == FT.IN_FILL_TRANSIENT
            else "the mechanism is not established from this evidence, so no repair is indicated"),
    }
