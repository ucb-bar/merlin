"""Pure decision procedure for the PR operand-residency DIFFERENTIAL claim.

Sibling of :mod:`perf_pk_claim`, and deliberately the same shape: no discovery, no I/O, no file
ownership. A caller passes the exact frozen capsule descriptors and the exact run-authored result
rows whose digests it has already verified, plus the target's OWN derived overlap-counter set, and
gets back a verdict that the reporting gate can recompute byte-for-byte from those same inputs.

What differs from PK is the claim, and therefore the whole decision:

* PK claims ``PREDICTS`` -- a fitted prediction against the same member's measured cycles -- so its
  verdict is a goodness of fit against predeclared bounds.
* PR claims ``DIFFERENTIAL`` -- the same work at the same extents with only the operand-residency
  regime changed -- so its verdict is whether the fitted per-depth RATE differs across a residency
  boundary. Nothing is compared to an absolute prediction, and no bound on a residual exists to miss.

⚠️ PK was REFUTED because all four of its depths sat inside the machine's overlap fill transient, and
PR's cheapest band starts at the very same depth. So this analyzer runs
:func:`merlin.perf.fill_transient.transient_verdict` PER BAND and refuses to quote a rate for a band
whose points sit inside the transient. That refusal is a result, and it is reported per band.

⚠️ PR's declaration carries NO ``acceptance`` block today, so
:func:`merlin.perf.claim_reach.has_decision_procedure` reports it as undecidable and nothing computes
its verdict. :func:`supported_acceptance` returns the exact block that freezes THIS analyzer; a
descriptor that carries an acceptance block must match it exactly, and one that carries none is still
decided but is reported with ``contract_frozen: false`` so a reader cannot mistake a reproducible
decision for a contractually bound one.

The arithmetic lives in :mod:`merlin.perf.residency_claim`, which names no target, no store and no
band. This module owns only the declaration and evidence contract.
"""
from __future__ import annotations

import copy
import string
from collections.abc import Mapping, Sequence
from typing import Any

from merlin.perf import fill_transient as FT
from merlin.perf import residency_claim as RC

_FAMILY = "PR"
_CLAIM = "DIFFERENTIAL"
_LEVEL = "L2_intra_layer"
_LEVER = "operand_residency"
_SCHEMA_VERSION = 1
_DERIVATION = "memory_regime_reduction_depth"
_NAME_CHARS = frozenset(string.ascii_letters + string.digits + "._-")

#: The exact block that would freeze this analyzer into the profile. Proposed, NOT yet declared --
#: `merlin/contract/capsules/profiles/_perf.yaml` is owned elsewhere. Every value is a claim decision:
#: `minimum_replicate_count: 2` is what makes the noise band MEASURABLE rather than assumed, and
#: `points_per_regime: 3` is what makes the declared negative control (two disjoint depth ranges
#: inside one regime) checkable rather than exact by construction.
_PROPOSED_ACCEPTANCE: dict[str, Any] = {
    "schema_version": 1,
    "analyzer": "perf_pr_claim.analyze_pr_claim/v1",
    "fit": {
        "form": "affine_per_residency_band",
        "method": "ordinary_least_squares_over_the_replicate_agreed_cycles_of_one_band",
        "independent_variable": "K",
        "dependent_metric": "verilator_L3_cycles",
        "comparands": "the_fitted_rate_of_one_band_against_the_fitted_rate_of_another",
    },
    "cohort": {
        "operation": "matmul",
        "fixed_shape_axes": ["M", "N"],
        "fixed_fields": ["operation", "operand_dtype", "accum_dtype", "epilogue"],
        "axis_derivation": _DERIVATION,
        "minimum_reachable_bands": 2,
        "points_per_regime": 3,
    },
    "replicates": {
        "minimum_count": 2,
        "identities_authored_by": "run",
        "why": ("two replicates are the smallest set that MEASURES the dispersion this family's "
                "noise band is read from; one replicate leaves it UNDETERMINABLE, and assuming it "
                "zero on a deterministic simulator is the assumption this contract refuses"),
    },
    "evidence": {
        "correctness_simulator": "spike",
        "correctness_tier": "L2",
        "timing_simulator": "verilator",
        "timing_tier": "L3",
        "spike_cycles_citable": False,
        "overlap_instrument": "target_derived_combination_counters",
        "overlap_required_per_member": True,
    },
    "noise_band": {
        "kind": "measured_replicate_dispersion",
        "declared_constant": None,
        "predicate": "fitted_rates_agree_iff_exactly_equal_when_the_measured_dispersion_is_zero",
        "why": ("a constant written here would be a knob somebody can turn until the answer "
                "changes. The band is read off the evidence instead: replicates that are observed "
                "IDENTICAL make the dispersion zero by measurement, so agreement means exact "
                "rational equality. A member with one replicate has an UNDETERMINABLE dispersion "
                "and its band is refused rather than assumed noiseless"),
    },
    "transient_guard": {
        "module": "merlin.perf.fill_transient.transient_verdict",
        "applied": "per_residency_band",
        "on_in_fill_transient": "refuse_the_band_and_quote_no_rate",
        "why": ("the sibling reduction-depth family was refuted because every one of its depths lay "
                "inside the machine's overlap fill transient. This family's cheapest band starts at "
                "the same depth, so a band is put through the guard before any rate is quoted"),
    },
}


class _Refusal(ValueError):
    pass


def supported_acceptance() -> dict[str, Any]:
    """A copy of the only claim contract this analyzer can decide (proposed, not yet frozen)."""
    return copy.deepcopy(_PROPOSED_ACCEPTANCE)


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


def _simple_name(value: object) -> bool:
    """A capsule name is a path component, checked by character membership, not by pattern."""
    return isinstance(value, str) and bool(value) and set(value) <= _NAME_CHARS


def _descriptor_point(raw: object) -> dict[str, Any]:
    descriptor = _mapping(raw, "PR descriptor")
    name = descriptor.get("name")
    if not _simple_name(name):
        raise _Refusal("every PR descriptor must have a simple non-empty name")
    if descriptor.get("kind") != "model_slice" or descriptor.get("label") != "dev":
        raise _Refusal(f"PR descriptor {name!r} is not the generated dev model_slice")

    performance = _mapping(descriptor.get("performance"), f"PR descriptor {name} performance")
    required_perf = {"level": _LEVEL, "family": _FAMILY, "lever": _LEVER, "claim": _CLAIM}
    if any(performance.get(key) != value for key, value in required_perf.items()):
        raise _Refusal(f"PR descriptor {name!r} changes its family, level, lever, or claim")

    acceptance = performance.get("acceptance")
    if acceptance is not None and not _exact_declaration_equal(acceptance, _PROPOSED_ACCEPTANCE):
        raise _Refusal(
            f"PR descriptor {name!r} declares an acceptance contract this analyzer does not "
            "implement; a profile edit is an analyzer-version edit")

    comparand = _mapping(performance.get("comparand"), f"PR descriptor {name} comparand")
    if (comparand.get("kind") != "group_arithmetic"
            or comparand.get("against") != "the_same_affine_fit_in_a_different_residency_regime"):
        raise _Refusal(f"PR descriptor {name!r} changes the group-arithmetic comparand")
    falsifier = _mapping(performance.get("falsifier"), f"PR descriptor {name} falsifier")
    if (falsifier.get("observation") != "per_regime_fitted_rate_and_intercept"
            or falsifier.get("fires_when")
            != "the_rates_fitted_in_different_residency_regimes_agree_within_the_noise_band"
            or falsifier.get("negative_control") != "two_disjoint_depth_ranges_inside_one_regime"):
        raise _Refusal(f"PR descriptor {name!r} changes its falsifier or its negative control")
    gate = _mapping(performance.get("gate"), f"PR descriptor {name} gate")
    if gate.get("instrument") != "cycle_count":
        raise _Refusal(f"PR descriptor {name!r} changes its declared instrument")

    operation = _mapping(descriptor.get("operation"), f"PR descriptor {name} operation")
    attributes = _mapping(operation.get("attributes"), f"PR descriptor {name} attributes")
    if operation.get("op") != "matmul" or attributes.get("epilogue") != []:
        raise _Refusal(f"PR descriptor {name!r} must be an unfused matmul")
    lhs_name, weight_name = attributes.get("lhs"), attributes.get("weight")
    if not isinstance(lhs_name, str) or not isinstance(weight_name, str) or lhs_name == weight_name:
        raise _Refusal(f"PR descriptor {name!r} has no unambiguous lhs/weight binding")

    inputs = _sequence(descriptor.get("inputs"), f"PR descriptor {name} inputs")
    tensors: dict[str, Mapping[str, Any]] = {}
    for index, raw_tensor in enumerate(inputs):
        tensor = _mapping(raw_tensor, f"PR descriptor {name} input {index}")
        tensor_name = tensor.get("name")
        if not isinstance(tensor_name, str) or not tensor_name or tensor_name in tensors:
            raise _Refusal(f"PR descriptor {name!r} has missing or duplicate input names")
        tensors[tensor_name] = tensor
    if lhs_name not in tensors or weight_name not in tensors:
        raise _Refusal(f"PR descriptor {name!r} lhs/weight binding does not name its inputs")
    lhs_shape = _sequence(tensors[lhs_name].get("shape"), f"PR descriptor {name} lhs shape")
    weight_shape = _sequence(tensors[weight_name].get("shape"), f"PR descriptor {name} weight shape")
    if len(lhs_shape) != 2 or len(weight_shape) != 2:
        raise _Refusal(f"PR descriptor {name!r} operands must be rank-two")
    m = _positive_extent(lhs_shape[0], f"{name}.M")
    n = _positive_extent(weight_shape[1], f"{name}.N")

    lhs_dtype = tensors[lhs_name].get("dtype")
    if not isinstance(lhs_dtype, str) or not lhs_dtype or lhs_dtype != tensors[weight_name].get("dtype"):
        raise _Refusal(f"PR descriptor {name!r} operands must share one declared dtype")
    accum_dtype = attributes.get("output_dtype")
    numeric = _mapping(descriptor.get("numeric_policy"), f"PR descriptor {name} numeric policy")
    if (not isinstance(accum_dtype, str) or not accum_dtype
            or numeric.get("dtype") != accum_dtype or numeric.get("compare") != "exact_int"):
        raise _Refusal(f"PR descriptor {name!r} has no exact common accumulator/output dtype")
    tiers = _sequence(descriptor.get("required_oracle_tiers"), f"PR descriptor {name} tiers")
    if not {"L2", "L3"}.issubset(set(tiers)):
        raise _Refusal(f"PR descriptor {name!r} does not require both L2 and L3")

    # The band is DERIVED and carried by the capsule's own emitter record. It is never read out of the
    # capsule NAME: a name is a convenience, and parsing one would make a rename change a verdict.
    emitter = _mapping(performance.get("emitter"), f"PR descriptor {name} emitter")
    derived = _mapping(emitter.get("derived_axes"), f"PR descriptor {name} derived_axes")
    axis = _mapping(derived.get("K"), f"PR descriptor {name} derived K axis")
    if axis.get("derive") != _DERIVATION:
        raise _Refusal(
            f"PR descriptor {name!r} does not derive its depth from {_DERIVATION!r}; the residency "
            "band would then be an assertion rather than a derivation")
    k = _positive_extent(axis.get("value"), f"{name}.K")
    band = axis.get("label")
    if not isinstance(band, str) or not band.strip():
        raise _Refusal(f"PR descriptor {name!r} carries no derived residency band label")
    derivation = _mapping(axis.get("derivation"), f"PR descriptor {name} band derivation")
    by_regime = _mapping(derivation.get("by_regime"), f"PR descriptor {name} by_regime")
    own = _mapping(by_regime.get(band), f"PR descriptor {name} band {band}")
    depths = [int(entry.get("K")) for entry in _sequence(own.get("points"), f"{name} band points")
              if isinstance(entry, Mapping) and isinstance(entry.get("K"), int)]
    if k not in depths:
        raise _Refusal(
            f"PR descriptor {name!r} declares band {band!r} but its own derivation does not place "
            f"K={k} in that band; the label and the derivation disagree")
    capacity = derivation.get("capacity_rows")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise _Refusal(f"PR descriptor {name!r} band derivation carries no positive capacity")

    return {
        "capsule": name,
        "M": m,
        "N": n,
        "K": k,
        "band": band,
        "capacity_rows": capacity,
        "operation": "matmul",
        "operand_dtype": lhs_dtype,
        "accum_dtype": accum_dtype,
        "epilogue": [],
        "acceptance_declared": acceptance is not None,
    }


def _validate_descriptors(descriptors: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _sequence(descriptors, "PR descriptors")
    if not raw:
        raise _Refusal("PR requires at least one frozen capsule descriptor")
    points = [_descriptor_point(entry) for entry in raw]
    if len({point["capsule"] for point in points}) != len(points):
        raise _Refusal("PR frozen descriptors repeat a capsule name")

    for field in ("M", "N", "operation", "operand_dtype", "accum_dtype", "epilogue",
                  "capacity_rows"):
        values = {repr(point[field]) for point in points}
        if len(values) != 1:
            raise _Refusal(f"PR cohort control changed fixed field {field!r}")
    frozen = {point["acceptance_declared"] for point in points}
    if len(frozen) != 1:
        raise _Refusal("PR cohort mixes descriptors that do and do not declare an acceptance block")

    bands: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        bands.setdefault(str(point["band"]), []).append(point)
    for band, members in bands.items():
        depths = sorted(int(member["K"]) for member in members)
        if len(set(depths)) != len(depths):
            raise _Refusal(f"PR band {band!r} repeats a reduction depth")
        # gate.capacity asks for two depths per band; the declared negative control asks for two
        # DISJOINT depth ranges inside one band, which needs three. The stricter of the two is the
        # one a verdict actually depends on, so it is the one enforced here.
        if len(depths) < 3:
            raise _Refusal(
                f"PR band {band!r} carries {len(depths)} depth(s); the declared negative control "
                "needs two disjoint depth ranges inside one regime, so a band needs three")
    if len(bands) < 2:
        raise _Refusal(
            f"PR reaches {len(bands)} residency band(s); a residency differential needs two, one on "
            "each side of a boundary")

    ordered = sorted(points, key=lambda point: (int(point["K"]), point["capsule"]))
    cohort = {
        "negative_control": "two_disjoint_depth_ranges_inside_one_regime",
        "operation": points[0]["operation"],
        "M": points[0]["M"],
        "N": points[0]["N"],
        "operand_dtype": points[0]["operand_dtype"],
        "accum_dtype": points[0]["accum_dtype"],
        "epilogue": [],
        "axis_derivation": _DERIVATION,
        "capacity_rows": points[0]["capacity_rows"],
        "capsules": [point["capsule"] for point in ordered],
        "K_values": [int(point["K"]) for point in ordered],
        "bands": {band: {"capsules": [member["capsule"] for member in
                                      sorted(members, key=lambda row: int(row["K"]))],
                         "K_values": sorted(int(member["K"]) for member in members)}
                  for band, members in sorted(
                      bands.items(), key=lambda item: min(int(row["K"]) for row in item[1]))},
    }
    return ordered, cohort


def _refused(reason: str, *, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": RC.REFUSED,
        "method": copy.deepcopy(_PROPOSED_ACCEPTANCE["fit"]),
        "contract_frozen": (bool(preflight.get("contract_frozen"))
                            if preflight is not None else False),
        "declaration": (copy.deepcopy(preflight.get("declaration"))
                        if preflight is not None else None),
        "cohort": (copy.deepcopy(preflight.get("cohort")) if preflight is not None else None),
        "evidence": None,
        "verdict": None,
        "refusal_reasons": [reason],
        "refutation_reasons": [],
    }


def preflight_pr_claim(descriptors: object, *, replicates: Sequence[str]) -> dict[str, Any]:
    """Validate frozen declarations and the exact replica schedule before a run is admitted.

    ``replicates`` is REQUIRED and has no default: PR declares no replicate count, and inventing one
    here would make the run's own schedule an analyzer assumption. Two is the minimum because that is
    what makes the noise band a measurement rather than a hope.
    """
    try:
        identities = tuple(replicates)
        if any(not _simple_name(entry) for entry in identities):
            raise _Refusal("every replicate identity must be a simple non-empty name")
        if len(set(identities)) != len(identities):
            raise _Refusal("the replicate schedule repeats an identity")
        if len(identities) < 2:
            raise _Refusal(
                f"PR was offered {len(identities)} replicate(s); this family's noise band is the "
                "MEASURED replicate dispersion, and one replicate leaves it UNDETERMINABLE rather "
                "than zero")
        points, cohort = _validate_descriptors(descriptors)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": _SCHEMA_VERSION,
            "family": _FAMILY,
            "claim": _CLAIM,
            "status": "REFUSED",
            "contract_frozen": False,
            "declaration": None,
            "cohort": None,
            "replicates": [],
            "expected_identities": [],
            "refusal_reasons": [str(exc)],
        }
    expected = [{"family": _FAMILY, "capsule": point["capsule"], "simulator": simulator,
                 "replicate": replicate, "tier": tier}
                for point in points
                for replicate in identities
                for simulator, tier in (("spike", "L2"), ("verilator", "L3"))]
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": "READY",
        "contract_frozen": bool(points[0]["acceptance_declared"]),
        "declaration": supported_acceptance(),
        "cohort": cohort,
        "replicates": list(identities),
        "expected_identities": expected,
        "refusal_reasons": [],
    }


def _validate_results(results: object, points: Sequence[Mapping[str, Any]],
                      replicates: Sequence[str]) -> dict[str, dict[str, Any]]:
    rows = _sequence(results, "PR result rows")
    expected_capsules = {str(point["capsule"]) for point in points}
    family_rows = [row for row in rows
                   if isinstance(row, Mapping)
                   and isinstance(row.get("identity"), Mapping)
                   and row["identity"].get("family") == _FAMILY]
    expected_count = len(points) * len(replicates) * 2
    if len(family_rows) != expected_count:
        raise _Refusal(
            f"PR requires exactly {expected_count} result rows, observed {len(family_rows)}")

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    approaches: set[Any] = set()
    for row in family_rows:
        identity = _mapping(row.get("identity"), "PR result identity")
        key = (identity.get("capsule"), identity.get("simulator"), identity.get("replicate"))
        if (key[0] not in expected_capsules or key[1] not in ("spike", "verilator")
                or key[2] not in set(replicates)):
            raise _Refusal(f"PR result has an undeclared identity: {dict(identity)!r}")
        if key in indexed:
            raise _Refusal(f"PR result repeats identity {key!r}")
        indexed[key] = row
        approaches.add(row.get("approach"))
    if len(approaches) != 1 or not isinstance(next(iter(approaches)), str) or not next(iter(approaches)):
        raise _Refusal(
            f"PR rows name {len(approaches)} approach(es) {sorted(map(repr, approaches))}; the "
            "cohort control requires one named approach across every member")

    observed: dict[str, dict[str, Any]] = {}
    for point in points:
        capsule = str(point["capsule"])
        cycles_by_replicate: list[int] = []
        counter_readings: list[Any] = []
        for replicate in replicates:
            for simulator, tier, purpose, citable in (
                    ("spike", "L2", "correctness_screen", False),
                    ("verilator", "L3", "performance_certification", True)):
                key = (capsule, simulator, replicate)
                row = indexed.get(key)
                if row is None:
                    raise _Refusal(f"PR result is missing identity {key!r}")
                if (row.get("tier") != tier or row.get("purpose") != purpose
                        or row.get("citable") is not citable):
                    raise _Refusal(f"PR result {key!r} changes its L2/L3 evidence semantics")
                if (row.get("correct") is not True or row.get("tier_status") != "pass"
                        or row.get("grade_status") != "pass"
                        or row.get("numeric_status") != "pass"
                        or row.get("error") is not None or row.get("failure") is not None):
                    raise _Refusal(f"PR result {key!r} is not a correct successful measurement")
                cycles = row.get("cycles")
                if simulator == "spike":
                    if cycles is not None:
                        raise _Refusal(f"PR Spike L2 result {key!r} exposes citable cycles")
                    continue
                if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
                    raise _Refusal(f"PR Verilator L3 result {key!r} lacks positive integer cycles")
                cycles_by_replicate.append(cycles)
                counter_readings.append(row.get("counter_values"))
        observed[capsule] = {
            "band": str(point["band"]),
            "K": int(point["K"]),
            "replicate_cycles": tuple(cycles_by_replicate),
            "counter_readings": counter_readings,
        }
    return observed


def _counter_set(counters: Any) -> Any:
    """Accept either a live ``OccupancyCounters`` or the mapping its ``to_dict`` produces.

    :func:`merlin.perf.hw_counters.counters_for_target` hands its caller the DICT form, and a sealed
    campaign manifest carries that same dict. Rebuilding it here -- structurally, by splitting each
    combination key on the separator ``to_dict`` joined it with -- means a caller cannot accidentally
    pass a mapping that silently resolves no counters and reads back as a machine with no overlap.
    """
    from merlin.perf.hw_counters import OccupancyCounters

    if isinstance(counters, OccupancyCounters):
        return counters
    if not isinstance(counters, Mapping):
        raise _Refusal("the derived counter set must be an OccupancyCounters or its to_dict mapping")
    engines = tuple(_sequence(counters.get("engines"), "counter engines"))
    combinations = _mapping(counters.get("by_combination"), "counter by_combination")
    if len(engines) < 2 or not combinations:
        raise _Refusal(
            "the supplied counter set resolves fewer than two engines or no combinations; one engine "
            "cannot overlap with itself, and an empty set would report every member as unmeasured")
    by_combination = {frozenset(str(key).split("+")): str(name)
                      for key, name in combinations.items()}
    return OccupancyCounters(prefix=str(counters.get("prefix") or ""), engines=engines,
                             by_combination=by_combination)


def _member(capsule: str, record: Mapping[str, Any], counters: Any, partition: Any) -> RC.Member:
    """One :class:`~merlin.perf.residency_claim.Member`, with its overlap reading or its reason.

    ``partition`` is the target's CIRCT counter-partition evidence (:data:`FT.PARTITION_FIELDS`).
    This module is a pure decision procedure over frozen descriptors and sealed result rows, so it
    holds no RTL artifact of its own and cannot derive that evidence -- the caller reads it off the
    target boundary and hands it in. Absent or malformed evidence produces a member with NO overlap
    reading and the reason attached, which its band reports as ``BAND_OVERLAP_UNDETERMINABLE``. It is
    never substituted for, and it never reaches the report as a bare exception string.
    """
    from merlin.perf.hw_counters import eta_from_counters

    try:
        proof = FT.partition_kwargs(partition)
    except FT.PartitionEvidenceError as exc:
        return RC.Member(
            label=capsule, band=str(record["band"]), axis=int(record["K"]),
            replicate_cycles=tuple(record["replicate_cycles"]),
            overlap_detail=str(exc))

    readings = list(record["counter_readings"])
    absent = [index for index, values in enumerate(readings) if not isinstance(values, Mapping)]
    if absent:
        return RC.Member(
            label=capsule, band=str(record["band"]), axis=int(record["K"]),
            replicate_cycles=tuple(record["replicate_cycles"]),
            overlap_detail=(f"{len(absent)} of {len(readings)} timing replicate(s) carry no "
                            "combination-counter reading; an unread counter is UNKNOWN, never zero"))
    normalised = [{str(name): int(value) for name, value in sorted(dict(values).items())}
                  for values in readings]
    if any(entry != normalised[0] for entry in normalised[1:]):
        return RC.Member(
            label=capsule, band=str(record["band"]), axis=int(record["K"]),
            replicate_cycles=tuple(record["replicate_cycles"]),
            overlap_detail=("the replicates disagree about their own combination counters, so this "
                            "member has no single overlap reading to attribute"))
    # THE COUNTER WINDOW, and why it is the SMALLEST replicate rather than an average or the first.
    # The readings above are already known identical across replicates; the cycle counts need not be,
    # and a member whose replicates disagree is refused by its band rather than collapsed here. The
    # exclusive partition has to fit inside EVERY window it was read in, so the binding one is the
    # shortest. Taking a longer window would let a partition that overflowed one replicate's run pass
    # the delegate's corruption check.
    window = min(int(value) for value in record["replicate_cycles"])
    reading = eta_from_counters(dict(normalised[0]), counters, measurement_cycles=window, **proof)
    if reading.get("state") != "measured":
        return RC.Member(
            label=capsule, band=str(record["band"]), axis=int(record["K"]),
            replicate_cycles=tuple(record["replicate_cycles"]),
            overlap_detail=str(reading.get("why") or "the counter reading is not measured"))
    return RC.Member(
        label=capsule, band=str(record["band"]), axis=int(record["K"]),
        replicate_cycles=tuple(record["replicate_cycles"]),
        realised_overlap=int(reading["realised_cycles"]),
        available_overlap=int(reading["available_cycles"]),
        overlap_detail=str(reading.get("note") or ""))


def analyze_pr_claim(descriptors: object, results: object, *,
                     replicates: Sequence[str] | None = None,
                     counters: Any = None, partition: Any = None) -> dict[str, Any]:
    """Decide PR from exact frozen descriptors, exact run-authored rows, and derived counters.

    ``counters`` is the combination-counter set derived from the target's OWN shipped header
    (:func:`merlin.perf.hw_counters.counters_for_target`). It is REQUIRED: without it the per-band
    fill-transient guard cannot run, and a rate quoted past a guard that did not run is exactly the
    result that refuted the sibling family. Absent counters is a refusal, never a skipped check.

    ``partition`` is that target's CIRCT counter-partition evidence -- the elaborated HW text, the
    header's event codes, and the two module identities -- which
    :func:`merlin.perf.hw_counters.eta_from_counters` requires before it will call an overlap reading
    measured. A counter block's NAMES only suggest that it partitions busy time; the proof is in the
    RTL, and without it every member would report UNKNOWN overlap and every band would be refused for
    a reason that pointed at the instrument rather than at the missing artifact. So it is refused
    up-front, by name, and the reason says exactly which input the run did not supply.

    ⚠️ **The three of them default to ``None`` so that a caller which cannot supply them gets a
    VERDICT rather than a TypeError.** :mod:`perf_claim_dispatch` routes every family as
    ``analyzer(descriptors, results)``, which is a shape this family does not fit -- it needs the
    run's replicate schedule and the target's counter evidence, and neither is recoverable from the
    descriptors. Nothing is invented to fill the gap: each absence is refused by the name of the input
    that is missing, so a run whose PR family went undecided says WHICH artifact it lacked. That is
    also why the defaults are not an excuse to omit them; they are the report of an omission.
    """
    if replicates is None:
        return _refused(
            "the run's replicate schedule was not supplied; PR declares no replicate count of its "
            "own and one cannot be invented here, because this family's noise band IS the measured "
            "dispersion across the identities the run actually executed")
    preflight = preflight_pr_claim(descriptors, replicates=replicates)
    if preflight["status"] != "READY":
        return _refused(str(preflight["refusal_reasons"][0]), preflight=preflight)
    if counters is None:
        return _refused(
            "the target's derived combination-counter set was not supplied, so the per-band "
            "fill-transient guard cannot run; a rate quoted without that guard repeats the "
            "refutation this family was built to avoid", preflight=preflight)
    try:
        FT.partition_kwargs(partition)
    except FT.PartitionEvidenceError as exc:
        return _refused(str(exc), preflight=preflight)
    try:
        counters = _counter_set(counters)
        points, _cohort = _validate_descriptors(descriptors)
        observed = _validate_results(results, points, list(preflight["replicates"]))
        members = [_member(str(point["capsule"]), observed[str(point["capsule"])], counters,
                           partition)
                   for point in points]
        verdict = RC.residency_verdict(members)
    # NARROW ON PURPOSE. These two are the analyzer's own refusal vocabulary: `_Refusal` is every
    # contract/evidence rejection raised above, and `ResidencyEvidenceError` is the arithmetic
    # module's. A `KeyError`/`TypeError`/`ValueError` from anywhere else is a PROGRAMMING error, and
    # catching those here is how a call made with the wrong arity was reported for a whole family as
    # a considered REFUSED whose reason was the text of a TypeError. It must crash instead.
    except (_Refusal, RC.ResidencyEvidenceError) as exc:
        return _refused(str(exc), preflight=preflight)

    status = verdict["status"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "family": _FAMILY,
        "claim": _CLAIM,
        "status": status,
        "method": copy.deepcopy(_PROPOSED_ACCEPTANCE["fit"]),
        "contract_frozen": bool(preflight["contract_frozen"]),
        "declaration": copy.deepcopy(preflight["declaration"]),
        "cohort": copy.deepcopy(preflight["cohort"]),
        "evidence": {
            "l2_correctness_rows_validated": len(points) * len(preflight["replicates"]),
            "l2_cycles_consumed": 0,
            "l3_positive_cycle_rows_consumed": len(points) * len(preflight["replicates"]),
            "timing_source": "verilator_L3_only",
            "replicates": list(preflight["replicates"]),
            "overlap_engines": list(getattr(counters, "engines", ()) or ()),
        },
        "verdict": verdict,
        "refusal_reasons": [verdict["reason"]] if status == RC.REFUSED else [],
        "refutation_reasons": (
            [f"{row['lower_band']}|{row['upper_band']}" for row in verdict["boundaries"]
             if row["falsifier_fired"]] if status == RC.REFUTED else []),
    }


def promotion_status(decision: Mapping[str, Any]) -> str:
    """The campaign's promotion vocabulary for a PR decision -- the same mapping PK's gate uses."""
    if not isinstance(decision, Mapping) or decision.get("family") != _FAMILY:
        raise _Refusal("promotion status can only be read off a PR decision")
    return "PROMOTED" if decision.get("status") == RC.ESTABLISHED else "BLOCKED"


def decision_boundary(decision: Mapping[str, Any] | None) -> dict[str, Any]:
    """The campaign-manifest boundary record, shaped exactly as PK's so one gate reads both."""
    if decision is None:
        return {
            "module": "perf_pr_claim",
            "identity_bridge": "not_invoked",
            "promotion_integration": "blocked",
            "promotion_status": "BLOCKED",
            "reason": "the PR analyzer has not consumed the sealed result rows",
        }
    status = str(decision.get("status"))
    return {
        "module": "perf_pr_claim",
        "identity_bridge":
            "analyze_pr_claim(frozen_descriptors,sealed_result_rows,derived_counters,"
            "circt_counter_partition)",
        "promotion_integration": "integrated",
        "promotion_status": promotion_status(decision),
        "reason": str((decision.get("verdict") or {}).get("reason")
                      or (decision.get("refusal_reasons") or [status])[0]),
    }


def falsifier_evidence(decision: Mapping[str, Any], *, member_of: Any = None) -> list[Any]:
    """This family's negative-control evidence as :class:`merlin.perf.campaign.FalsifierEvidence`.

    One record per band that reached its own negative control, so the campaign's own promotion gate
    sees the same three-state answer this analyzer computed: ``True`` the control fired (the two
    disjoint depth ranges inside one regime agreed), ``False`` it did not, and ``None`` never -- a
    band whose control could not be evaluated produces no record at all rather than an undetermined
    one, because the reason it could not is already a named band refusal.

    ``member_of`` maps a band name to the :class:`~merlin.perf.campaign.ReplicaIdentity` the run used
    for that band's control member. It is required: this module does not invent identities.
    """
    from merlin.perf.campaign import FalsifierEvidence

    if not isinstance(decision, Mapping) or decision.get("family") != _FAMILY:
        raise _Refusal("falsifier evidence can only be read off a PR decision")
    if member_of is None:
        raise _Refusal("falsifier evidence needs the run's own replica identity per band")
    verdict = decision.get("verdict") or {}
    out = []
    for band in verdict.get("bands") or []:
        control = band.get("negative_control")
        if not isinstance(control, Mapping):
            continue
        identity = member_of(str(band["band"])) if callable(member_of) else member_of[str(band["band"])]
        out.append(FalsifierEvidence(identity=identity, negative_control=True,
                                     fired=bool(control["fired"]), reason=str(control["reason"])))
    return out
