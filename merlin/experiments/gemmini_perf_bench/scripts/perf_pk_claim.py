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
_DEFAULT_TIMING_SIMULATOR = "gsim"
_TIMING_ENGINES = {
    "gsim": {"oracle_kind": "rtl_gsim", "dependent_metric": "gsim_L3_cycles"},
    "verilator": {
        "oracle_kind": "rtl_verilator", "dependent_metric": "verilator_L3_cycles"},
}

# This mirrors the reviewed declaration in contract/capsules/profiles/_perf.yaml.
# Requiring exact equality makes a profile edit a deliberate analyzer-version
# edit too; a relaxed threshold cannot silently become eligible for promotion.
_ACCEPTANCE_BASE: dict[str, Any] = {
    "schema_version": 2,
    "analyzer": "perf_pk_claim.analyze_pk_claim/v3",
    "fit": {
        "form": "affine",
        "method": "ordinary_least_squares_all_L3_replicates",
        "independent_variable": "K",
        "dependent_metric": None,
    },
    "cohort": {
        "operation": "matmul",
        "fixed_shape_axes": ["M", "N"],
        "fixed_fields": ["operation", "operand_dtype", "accum_dtype", "epilogue"],
        "exact_points": 4,
        "K_multipliers_of_tile": [1, 2, 4, 8],
    },
    "replicates": {
        # TWO, BECAUSE THE SECOND IS THE WITNESS AND THE THIRD MEASURES NOTHING. The timing engine
        # is deterministic -- verified over 392 repeated measurements of byte-identical programs with
        # zero disagreement -- so replicate three re-derives a number replicate two already agreed
        # on. Two is the floor and not one: one leaves the replicate dispersion UNDETERMINABLE, and
        # assuming it zero on a deterministic simulator is the assumption these contracts refuse.
        # At one replicate the paired families' band collapses to a fabricated 0.0 and their
        # negative control can never fire.
        #
        # This count is campaign-wide, not PK's alone: the stage reads it out of this declaration to
        # build the expected cells for every frozen member, so it sets the whole replicate schedule.
        "exact_count": 2,
        "identities": ["r000", "r001"],
    },
    "evidence": {
        "correctness_simulator": "spike",
        "correctness_tier": "L2",
        "timing_simulator": None,
        "timing_tier": "L3",
        "spike_cycles_citable": False,
        "timing_oracle_kind": None,
        "timing_fidelity": "elaborated_rtl",
        "required_identity_sha256_fields": [
            "simulator_binary_sha256",
            "elaborated_firrtl_sha256",
            "program_elf_sha256",
        ],
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


def supported_acceptance(
        timing_simulator: str = _DEFAULT_TIMING_SIMULATOR) -> dict[str, Any]:
    """Return a copy of the only claim contract this analyzer can decide."""
    if timing_simulator not in _TIMING_ENGINES:
        raise ValueError(
            f"timing simulator must be one of {sorted(_TIMING_ENGINES)}, got {timing_simulator!r}")
    acceptance = copy.deepcopy(_ACCEPTANCE_BASE)
    engine = _TIMING_ENGINES[timing_simulator]
    acceptance["fit"]["dependent_metric"] = engine["dependent_metric"]
    acceptance["evidence"]["timing_simulator"] = timing_simulator
    acceptance["evidence"]["timing_oracle_kind"] = engine["oracle_kind"]
    return acceptance


def _validated_acceptance(value: object) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise _Refusal("PK acceptance must be a mapping")
    evidence = value.get("evidence")
    simulator = evidence.get("timing_simulator") if isinstance(evidence, Mapping) else None
    if simulator not in _TIMING_ENGINES:
        raise _Refusal(
            f"PK acceptance timing simulator must be one of {sorted(_TIMING_ENGINES)}")
    expected = supported_acceptance(str(simulator))
    if not _exact_declaration_equal(value, expected):
        raise _Refusal(
            f"PK descriptor has a malformed or unsupported acceptance contract for {simulator}")
    return copy.deepcopy(expected), str(simulator)


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
    declaration = (copy.deepcopy(preflight.get("declaration"))
                   if preflight is not None and preflight.get("declaration") is not None
                   else None)
    method = declaration.get("fit") if isinstance(declaration, Mapping) else None
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": "REFUSED",
        "method": copy.deepcopy(method),
        "declaration": declaration,
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


def _sha256(value: object, owner: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)):
        raise _Refusal(f"{owner} must be an exact lowercase SHA-256")
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
    acceptance, timing_simulator = _validated_acceptance(performance.get("acceptance"))
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
        "acceptance": acceptance,
        "timing_simulator": timing_simulator,
    }


def _validate_descriptors(
        descriptors: object) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw_descriptors = _sequence(descriptors, "PK descriptors")
    if len(raw_descriptors) != 4:
        raise _Refusal("PK requires exactly four frozen capsule descriptors")
    points = [_descriptor_point(raw) for raw in raw_descriptors]
    if len({point["capsule"] for point in points}) != 4:
        raise _Refusal("PK frozen descriptors repeat a capsule name")

    fixed_fields = ("M", "N", "operation", "operand_dtype", "accum_dtype", "epilogue",
                    "acceptance", "timing_simulator")
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
        "replicates": ["r000", "r001"],
    }
    return ordered, cohort, copy.deepcopy(ordered[0]["acceptance"])


def preflight_pk_claim(descriptors: object) -> dict[str, Any]:
    """Validate frozen declarations before a claim-bearing run is admitted."""
    try:
        points, cohort, acceptance = _validate_descriptors(descriptors)
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
    timing_simulator = str(acceptance["evidence"]["timing_simulator"])
    expected = []
    for point in points:
        for replicate in acceptance["replicates"]["identities"]:
            for simulator, tier in (("spike", "L2"), (timing_simulator, "L3")):
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
        "declaration": acceptance,
        "cohort": cohort,
        "expected_identities": expected,
        "refusal_reasons": [],
    }


def _validate_results(
        results: object, points: Sequence[Mapping[str, Any]],
        acceptance: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _sequence(results, "PK result rows")
    evidence_contract = _mapping(acceptance.get("evidence"), "PK acceptance evidence")
    timing_simulator = str(evidence_contract.get("timing_simulator"))
    timing_oracle_kind = str(evidence_contract.get("timing_oracle_kind"))
    expected_capsules = {str(point["capsule"]) for point in points}
    pk_rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"result row {index}")
        identity = _mapping(row.get("identity"), f"result row {index} identity")
        if identity.get("family") == _FAMILY:
            pk_rows.append(row)
    expected_count = len(points) * len(_ACCEPTANCE_BASE["replicates"]["identities"]) * 2
    if len(pk_rows) != expected_count:
        raise _Refusal(
            f"PK requires exactly {expected_count} result rows, observed {len(pk_rows)}")

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in pk_rows:
        identity = _mapping(row.get("identity"), "PK result identity")
        capsule = identity.get("capsule")
        simulator = identity.get("simulator")
        replicate = identity.get("replicate")
        if (capsule not in expected_capsules or simulator not in ("spike", timing_simulator)
                or replicate not in tuple(_ACCEPTANCE_BASE["replicates"]["identities"])):
            raise _Refusal(f"PK result has an undeclared identity: {dict(identity)!r}")
        key = (str(capsule), str(simulator), str(replicate))
        if key in indexed:
            raise _Refusal(f"PK result repeats identity {key!r}")
        indexed[key] = row

    observations: list[dict[str, Any]] = []
    for point in points:
        capsule, k = str(point["capsule"]), int(point["K"])
        for replicate in tuple(_ACCEPTANCE_BASE["replicates"]["identities"]):
            for simulator, tier, purpose, citable in (
                    ("spike", "L2", "correctness_screen", False),
                    (timing_simulator, "L3", "performance_certification", True)):
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
                        raise _Refusal(
                            f"PK {timing_simulator} L3 result {key!r} lacks positive integer cycles")
                    oracle = _mapping(row.get("oracle"), f"PK result {key!r} oracle")
                    if (oracle.get("kind") != timing_oracle_kind
                            or oracle.get("derived_from_rtl") is not True
                            or oracle.get("fidelity") != "elaborated_rtl"):
                        raise _Refusal(
                            f"PK result {key!r} does not prove its selected elaborated-RTL oracle")
                    timing_identity = _mapping(
                        row.get("timing_identity"), f"PK result {key!r} timing identity")
                    expected_identity_fields = {
                        "simulator", "oracle_kind", "fidelity",
                        "simulator_binary_sha256", "elaborated_firrtl_sha256",
                        "program_elf_sha256",
                    }
                    if set(timing_identity) != expected_identity_fields:
                        raise _Refusal(
                            f"PK result {key!r} timing identity does not contain the exact fields")
                    if (timing_identity.get("simulator") != timing_simulator
                            or timing_identity.get("oracle_kind") != timing_oracle_kind
                            or timing_identity.get("fidelity") != "elaborated_rtl"):
                        raise _Refusal(
                            f"PK result {key!r} timing identity names a different engine or fidelity")
                    for field in evidence_contract["required_identity_sha256_fields"]:
                        _sha256(timing_identity.get(field), f"PK result {key!r} {field}")
                    observations.append({
                        "capsule": capsule,
                        "replicate": replicate,
                        "K": k,
                        "cycles": cycles,
                        "timing_identity": dict(timing_identity),
                    })
    if len({row["timing_identity"]["simulator_binary_sha256"] for row in observations}) != 1:
        raise _Refusal("PK timing rows do not share one exact simulator binary identity")
    if len({row["timing_identity"]["elaborated_firrtl_sha256"] for row in observations}) != 1:
        raise _Refusal("PK timing rows do not share one exact elaborated FIRRTL identity")
    return observations


def _fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _fit(
        observations: Sequence[Mapping[str, Any]],
        timing_simulator: str) -> tuple[dict[str, Any], dict[str, bool]]:
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
        "equation": (
            f"{timing_simulator}_L3_cycles = "
            "rate_cycles_per_K_element * K + intercept_cycles"),
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
    fit. Every fitted observation is a correct, positive-cycle row from the
    exact elaborated-RTL engine selected by the frozen acceptance contract.
    """
    preflight = preflight_pk_claim(descriptors)
    if preflight["status"] != "READY":
        return _refused(str(preflight["refusal_reasons"][0]), preflight=preflight)
    try:
        points, _cohort, acceptance = _validate_descriptors(descriptors)
        timing_simulator = str(acceptance["evidence"]["timing_simulator"])
        observations = _validate_results(results, points, acceptance)
        fit, checks = _fit(observations, timing_simulator)
    except (KeyError, TypeError, ValueError) as exc:
        return _refused(str(exc), preflight=preflight)
    failed = [name for name, passed in checks.items() if not passed]
    status = "ESTABLISHED" if not failed else "REFUTED"
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": status,
        "method": copy.deepcopy(acceptance["fit"]),
        "declaration": copy.deepcopy(preflight["declaration"]),
        "cohort": copy.deepcopy(preflight["cohort"]),
        "evidence": {
            # COUNTED, NOT ASSERTED. This was the literal 12 -- the count of one corpus at one
            # replicate schedule -- so the evidence block reported a number it had not counted and
            # would have kept reporting it whatever was validated. Every cohort point is admitted at
            # the correctness tier once per declared replicate, which is what this counts.
            "l2_correctness_rows_validated": (
                len(points) * len(acceptance["replicates"]["identities"])),
            "l2_cycles_consumed": 0,
            "l3_positive_cycle_rows_consumed": len(observations),
            "timing_source": f"{timing_simulator}_L3_only",
            "timing_simulator": timing_simulator,
            "timing_oracle_kind": acceptance["evidence"]["timing_oracle_kind"],
            "timing_fidelity": "elaborated_rtl",
            "simulator_binary_sha256": observations[0]["timing_identity"][
                "simulator_binary_sha256"],
            "elaborated_firrtl_sha256": observations[0]["timing_identity"][
                "elaborated_firrtl_sha256"],
            "program_elf_sha256_by_observation": [
                {"capsule": row["capsule"], "replicate": row["replicate"],
                 "sha256": row["timing_identity"]["program_elf_sha256"]}
                for row in observations
            ],
            "arm": "arm4",
        },
        "fit": fit,
        "checks": checks,
        "refusal_reasons": [],
        "refutation_reasons": failed,
    }
