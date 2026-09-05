"""Family-agnostic decision procedure for an affine PREDICTS claim.

``perf_pk_claim`` proves one family's affine law, but it is welded to that family: the independent
variable is the literal ``"K"``, and the cohort check reconstructs member names as ``PK{i:02d}_k{K}``.
A second predictive family therefore had no procedure at all -- which is how ``PM`` and ``PV`` reached
the corpus declaring ``PREDICTS`` with nothing able to evaluate them.  A claim no procedure can refute
is not a claim, so this module supplies the general form: the family, the independent variable, and
every threshold come from the frozen acceptance contract, and nothing about a particular target,
family, or axis is written here.

Same verdict discipline as the PK analyzer, deliberately:
  * ``REFUSED``  -- the evidence is incomplete or malformed; no fit is attempted and nothing is
                    inferred.  Absence of proof is never reported as proof.
  * ``REFUTED``  -- complete evidence that misses a predeclared bound.
  * ``ACCEPTED`` -- complete evidence inside every predeclared bound.

The contract is immutable launch data.  Thresholds are read, never defaulted: a contract that omits
one is REFUSED rather than silently scored against a bound this module chose.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = 1
ANALYZER = "perf_affine_claim.analyze_affine_claim/v1"

ACCEPTED, REFUTED, REFUSED = "ACCEPTED", "REFUTED", "REFUSED"

#: How the independent variable is read off a capsule descriptor.  Each kind is a STRUCTURAL read of
#: the declared operand shapes -- never a parse of the capsule's name, which is a label and not data.
_VARIABLE_KINDS = ("output_elements", "input_elements", "input_dim")


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def _is_number(value: Any) -> bool:
    """A contract number, excluding bool -- `True` is not a threshold of 1.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _inputs_by_name(descriptor: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = descriptor.get("inputs")
    if not isinstance(rows, Sequence):
        return {}
    return {str(r.get("name")): r for r in rows if isinstance(r, Mapping)}


def independent_value(descriptor: Mapping[str, Any], source: Mapping[str, Any]) -> int | None:
    """Read the declared independent variable from one descriptor, or None if underivable."""
    kind = source.get("kind")
    named = _inputs_by_name(descriptor)
    if kind == "output_elements":
        # For a contraction the output extent is the LHS rows by the weight columns.  Both are read
        # from the declared operand shapes, so this holds for any mesh size and any tiling.
        lhs, weight = source.get("lhs"), source.get("weight")
        a, w = named.get(str(lhs)), named.get(str(weight))
        if not a or not w:
            return None
        ash, wsh = a.get("shape"), w.get("shape")
        if not (isinstance(ash, Sequence) and isinstance(wsh, Sequence) and ash and wsh):
            return None
        if len(ash) < 1 or len(wsh) < 2:
            return None
        return int(ash[0]) * int(wsh[1])
    if kind == "input_elements":
        row = named.get(str(source.get("input")))
        shape = row.get("shape") if row else None
        if not isinstance(shape, Sequence) or not shape:
            return None
        total = 1
        for extent in shape:
            total *= int(extent)
        return total
    if kind == "input_dim":
        row = named.get(str(source.get("input")))
        shape = row.get("shape") if row else None
        axis = source.get("axis")
        if not isinstance(shape, Sequence) or not isinstance(axis, int):
            return None
        if not (0 <= axis < len(shape)):
            return None
        return int(shape[axis])
    return None


def _ols(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Ordinary least squares over (x, y); returns (intercept, slope, r_squared)."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = n * sxx - sx * sx
    if denominator == 0:                     # every point shares one x: a slope is not identifiable
        return None
    slope = (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r_squared = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return intercept, slope, r_squared


def analyze_affine_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Decide one family's affine claim over frozen descriptors and authored result rows."""
    if not isinstance(descriptors, Sequence) or not descriptors:
        return _fail("no capsule descriptors were supplied")
    if not isinstance(results, Sequence) or not results:
        return _fail("no result rows were supplied")

    contracts, families = [], set()
    for d in descriptors:
        if not isinstance(d, Mapping):
            return _fail("a descriptor is not a mapping")
        perf = d.get("performance")
        if not isinstance(perf, Mapping):
            return _fail("a descriptor carries no performance block")
        if perf.get("claim") != "PREDICTS":
            return _fail("this procedure decides PREDICTS claims only",
                         observed_claim=perf.get("claim"))
        families.add(str(perf.get("family")))
        contracts.append(perf.get("acceptance"))
    if len(families) != 1:
        return _fail("descriptors span more than one family", families=sorted(families))
    if any(not isinstance(c, Mapping) for c in contracts):
        return _fail("a member carries no frozen acceptance contract")
    first = contracts[0]
    if any(c != first for c in contracts):
        return _fail("members disagree about the frozen acceptance contract")

    contract: Mapping[str, Any] = first
    if contract.get("analyzer") != ANALYZER:
        return _fail("the contract names a different analyzer",
                     declared=contract.get("analyzer"), this=ANALYZER)
    fit = contract.get("fit")
    thresholds = contract.get("thresholds")
    cohort = contract.get("cohort")
    if not all(isinstance(x, Mapping) for x in (fit, cohort, thresholds)):
        return _fail("the contract is missing its fit, cohort, or thresholds block")
    source = fit.get("variable_source")
    if not isinstance(source, Mapping) or source.get("kind") not in _VARIABLE_KINDS:
        return _fail("the contract does not declare how to read its independent variable")
    metric = fit.get("dependent_metric")
    if not isinstance(metric, str) or not metric:
        return _fail("the contract does not name its dependent metric")

    exact_points = cohort.get("exact_points")
    if isinstance(exact_points, bool) or not isinstance(exact_points, int) or len(descriptors) != exact_points:
        return _fail("the cohort is not the exact predeclared size",
                     expected=exact_points, observed=len(descriptors))

    xs: dict[str, int] = {}
    for d in descriptors:
        name = str(d.get("name") or "")
        value = independent_value(d, source)
        if value is None or value <= 0:
            return _fail("the independent variable is not derivable from a member's declared shapes",
                         capsule=name)
        xs[name] = value

    observations: list[tuple[float, float]] = []
    seen: set[tuple[str, str]] = set()
    for row in results:
        if not isinstance(row, Mapping):
            return _fail("a result row is not a mapping")
        capsule = str(row.get("capsule") or "")
        if capsule not in xs:
            return _fail("a result row names a capsule outside the cohort", capsule=capsule)
        replicate = str(row.get("replicate") or "")
        if (capsule, replicate) in seen:
            return _fail("a (capsule, replicate) pair is reported twice",
                         capsule=capsule, replicate=replicate)
        seen.add((capsule, replicate))
        cycles = row.get(metric)
        if not isinstance(cycles, (int, float)) or isinstance(cycles, bool) or cycles <= 0:
            return _fail("a result row carries no positive value for the dependent metric",
                         capsule=capsule, metric=metric)
        observations.append((float(xs[capsule]), float(cycles)))

    # COMPLETENESS IS PER MEMBER, NOT A ROW TOTAL. Counting rows lets evidence that covers only
    # part of the cohort reach the fit: 12 rows drawn from 2 of 4 members, carrying invented
    # replicate ids, matches the total and is then fitted as a TWO-POINT line -- which is
    # unrefutable by construction, and is precisely this module's stated failure mode inverted
    # (absence of proof reported as proof). The contract already freezes the replicate identities,
    # so the exact (capsule, replicate) set is derivable and is what must be matched.
    replicates = contract.get("replicates")
    if not isinstance(replicates, Mapping):
        return _fail("the contract does not declare its replicate cohort")
    identities = replicates.get("identities")
    if (not isinstance(identities, Sequence) or isinstance(identities, str) or not identities
            or any(not isinstance(i, str) or not i for i in identities)):
        return _fail("the contract does not freeze its replicate identities")
    count = replicates.get("exact_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(identities):
        return _fail("the contract's replicate count does not match its frozen identities",
                     exact_count=count, identities=len(identities))
    expected_pairs = {(name, str(identity)) for name in xs for identity in identities}
    if seen != expected_pairs:
        absent = sorted(expected_pairs - seen)
        extra = sorted(seen - expected_pairs)
        return _fail("the evidence is not the exact predeclared (capsule, replicate) cohort",
                     missing=absent[:8], unexpected=extra[:8],
                     n_missing=len(absent), n_unexpected=len(extra))

    fitted = _ols(observations)
    if fitted is None:
        return _fail("the cohort does not vary its independent variable, so no slope is identifiable")
    intercept, slope, r_squared = fitted

    slope_min = thresholds.get("slope_min_exclusive")
    r2_min = thresholds.get("r_squared_min_inclusive")
    bound = thresholds.get("residual_bound")
    if not _is_number(slope_min) or not _is_number(r2_min):
        return _fail("the contract omits a slope or r-squared threshold")
    if not isinstance(bound, Mapping):
        return _fail("the contract omits its residual bound")
    floor = bound.get("absolute_floor_cycles")
    fraction = bound.get("observed_cycle_fraction")
    if not _is_number(floor) or not _is_number(fraction):
        return _fail("the residual bound is not fully specified")

    breaches = []
    for x, y in observations:
        residual = abs(y - (intercept + slope * x))
        allowed = max(float(floor), float(fraction) * y)
        if residual > allowed:
            breaches.append({"x": x, "observed_cycles": y, "residual": residual,
                             "allowed": allowed})

    measured = {"intercept_cycles": intercept, "slope_cycles_per_unit": slope,
                "r_squared": r_squared, "n_observations": len(observations),
                "independent_variable": fit.get("independent_variable"),
                "distinct_x": sorted({x for x, _ in observations})}
    reasons = []
    if not slope > float(slope_min):
        reasons.append(f"slope {slope:.6g} is not above the predeclared floor {slope_min}")
    if r_squared < float(r2_min):
        reasons.append(f"r_squared {r_squared:.6g} is below the predeclared minimum {r2_min}")
    if breaches:
        reasons.append(f"{len(breaches)} observation(s) exceed the predeclared residual bound")
    if reasons:
        return {"verdict": REFUTED, "reasons": reasons, "measured": measured,
                "breaches": breaches[:8], "family": sorted(families)[0]}
    return {"verdict": ACCEPTED, "measured": measured, "family": sorted(families)[0]}


# ---------------------------------------------------------------------------------------------
# PREFLIGHT: is this cohort admissible BEFORE a single cycle is spent measuring it?
#
# The decision procedure above refuses bad evidence.  That is too late to be the only guard: a
# campaign that discovers at report time that its contract omits a threshold, or that its cohort
# holds the independent variable constant, has already paid for every L3 cell.  The preflight asks
# the same questions of the DECLARATION alone, so an inadmissible family is refused at launch.
#
# Everything below is read from the frozen declaration.  Nothing about a family, an axis, an
# operation or a target is written here: the independent variable comes from ``fit.variable_source``,
# the cohort size from ``cohort.exact_points``, the schedule from ``replicates``, and the bounds from
# ``thresholds`` -- and a declaration that omits any of them is REFUSED rather than completed with a
# value this module chose.
# ---------------------------------------------------------------------------------------------


class _Refusal(ValueError):
    """One named reason a declaration cannot be admitted."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise _Refusal(f"{label} must be a sequence")
    return value


def _family_of(descriptors: object) -> str | None:
    """The one family a cohort names, or None when it names zero or several."""
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str):
        return None
    names = set()
    for descriptor in descriptors:
        performance = descriptor.get("performance") if isinstance(descriptor, Mapping) else None
        if isinstance(performance, Mapping):
            names.add(str(performance.get("family")))
    return names.pop() if len(names) == 1 else None


def _free_axes(source: Mapping[str, Any]) -> dict[str, object]:
    """Which declared operand axes the independent variable is allowed to move.

    This is the cohort control, DERIVED: a member differs from its siblings only in the quantity the
    fit varies, so every operand axis the variable source does not name must be identical across the
    cohort.  Each family's own falsifier names the same control in prose -- one holds K fixed while
    the output extent moves, another holds the window and image fixed while the channel depth moves
    -- and reading it off ``variable_source`` gets both without either being written here.
    """
    kind = source.get("kind")
    if kind == "output_elements":
        # The output extent is lhs rows by weight columns, so exactly those two axes may move.
        return {str(source.get("lhs")): {0}, str(source.get("weight")): {1}}
    if kind == "input_elements":
        return {str(source.get("input")): "all"}
    if kind == "input_dim":
        return {str(source.get("input")): {int(source.get("axis"))}}
    raise _Refusal("the contract does not declare how to read its independent variable")


def _shape_of(descriptor: Mapping[str, Any], name: str) -> tuple[int, ...]:
    row = _inputs_by_name(descriptor).get(name)
    shape = row.get("shape") if row else None
    if not isinstance(shape, Sequence) or isinstance(shape, str) or not shape:
        raise _Refusal(f"member {descriptor.get('name')!r} declares no shape for operand {name!r}")
    extents = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise _Refusal(
                f"member {descriptor.get('name')!r} operand {name!r} has a non-positive extent")
        extents.append(int(extent))
    return tuple(extents)


#: The cohort-control fields a descriptor states about itself, each read STRUCTURALLY from the block
#: that declares it.  A contract naming a field absent from this table is REFUSED rather than having
#: that part of its control silently skipped -- an unread control is not a control.
def _fixed_field(descriptor: Mapping[str, Any], field: str) -> Any:
    operation = _mapping(descriptor.get("operation"), f"member {descriptor.get('name')!r} operation")
    attributes = _mapping(operation.get("attributes"),
                          f"member {descriptor.get('name')!r} operation attributes")
    if field == "operation":
        return operation.get("op")
    if field == "operand_dtype":
        rows = _sequence(descriptor.get("inputs"), f"member {descriptor.get('name')!r} inputs")
        return tuple(sorted({str(row.get("dtype")) for row in rows if isinstance(row, Mapping)}))
    if field == "accum_dtype":
        return attributes.get("output_dtype")
    if field == "epilogue":
        return list(attributes.get("epilogue") or [])
    raise _Refusal(
        f"the cohort control names a fixed field {field!r} this procedure cannot read from a "
        "descriptor; it would be declared and never checked")


def _validated_declaration(descriptors: object) -> tuple[list[Mapping[str, Any]],
                                                         dict[str, Any], dict[str, Any]]:
    """Admit the cohort, or raise the first reason it is inadmissible."""
    rows = _sequence(descriptors, "affine descriptors")
    if not rows:
        raise _Refusal("no capsule descriptors were supplied")
    members: list[Mapping[str, Any]] = []
    contracts: list[Any] = []
    families: set[str] = set()
    names: list[str] = []
    for index, raw in enumerate(rows):
        descriptor = _mapping(raw, f"descriptor {index}")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name:
            raise _Refusal(f"descriptor {index} has no capsule name")
        performance = _mapping(descriptor.get("performance"), f"descriptor {name!r} performance")
        if performance.get("claim") != "PREDICTS":
            raise _Refusal(
                f"descriptor {name!r} declares {performance.get('claim')!r}; this procedure "
                "decides PREDICTS claims only")
        families.add(str(performance.get("family")))
        contracts.append(performance.get("acceptance"))
        names.append(name)
        members.append(descriptor)
    if len(set(names)) != len(names):
        raise _Refusal("the cohort repeats a capsule name")
    if len(families) != 1:
        raise _Refusal(f"descriptors span {len(families)} families {sorted(families)}")
    contract = _mapping(contracts[0], "the frozen acceptance contract")
    if any(entry != contract for entry in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        raise _Refusal(
            f"the contract names analyzer {contract.get('analyzer')!r}, not {ANALYZER!r}")

    fit = _mapping(contract.get("fit"), "the contract's fit block")
    if fit.get("form") != "affine":
        raise _Refusal(
            f"the contract fits a {fit.get('form')!r} form; this procedure fits an affine law and "
            "will not score a declaration against a law it did not state")
    source = _mapping(fit.get("variable_source"), "the contract's fit.variable_source")
    if source.get("kind") not in _VARIABLE_KINDS:
        raise _Refusal("the contract does not declare how to read its independent variable")
    metric = fit.get("dependent_metric")
    if not isinstance(metric, str) or not metric:
        raise _Refusal("the contract does not name its dependent metric")

    cohort = _mapping(contract.get("cohort"), "the contract's cohort block")
    exact_points = cohort.get("exact_points")
    if isinstance(exact_points, bool) or not isinstance(exact_points, int) or exact_points < 2:
        raise _Refusal("the contract does not declare a cohort of at least two points")
    if len(members) != exact_points:
        raise _Refusal(
            f"the cohort is {len(members)} member(s) against a predeclared {exact_points}")

    thresholds = _mapping(contract.get("thresholds"), "the contract's thresholds block")
    if not _is_number(thresholds.get("slope_min_exclusive")) or not _is_number(
            thresholds.get("r_squared_min_inclusive")):
        raise _Refusal("the contract omits a slope or r-squared threshold; a bound this module "
                       "chose itself would be its opinion wearing the contract's authority")
    bound = _mapping(thresholds.get("residual_bound"), "the contract's residual bound")
    if not _is_number(bound.get("absolute_floor_cycles")) or not _is_number(
            bound.get("observed_cycle_fraction")):
        raise _Refusal("the residual bound is not fully specified")

    from merlin.perf import claim_reach
    try:
        schedule = claim_reach.replicate_contract(members[0]["performance"])
    except ValueError as exc:
        raise _Refusal(f"the replicate contract is malformed: {exc}") from exc
    if schedule is None:
        raise _Refusal("the contract declares no replicate schedule")
    if schedule.exact_count is None or not schedule.identities:
        raise _Refusal(
            "an affine law is fitted over ALL of its replicates, so the cohort it is fitted over "
            "must be frozen; this contract declares only a floor")
    if schedule.minimum_count < 2:
        raise _Refusal(
            f"the contract schedules {schedule.minimum_count} replicate(s); one leaves the "
            "replicate dispersion UNDETERMINABLE rather than zero")

    evidence = _mapping(contract.get("evidence"), "the contract's evidence block")
    lanes = []
    for simulator_key, tier_key in (("correctness_simulator", "correctness_tier"),
                                    ("timing_simulator", "timing_tier")):
        simulator, tier = evidence.get(simulator_key), evidence.get(tier_key)
        if not isinstance(simulator, str) or not simulator or not isinstance(tier, str) or not tier:
            raise _Refusal(f"the contract's evidence omits its {simulator_key}/{tier_key}")
        lanes.append((simulator, tier))

    # --- the cohort control: only the declared independent variable may move ---
    free = _free_axes(source)
    xs: dict[str, int] = {}
    for descriptor in members:
        name = str(descriptor.get("name"))
        value = independent_value(descriptor, source)
        if value is None or value <= 0:
            raise _Refusal(
                f"the independent variable is not derivable from member {name!r}'s declared shapes")
        xs[name] = int(value)
    if len(set(xs.values())) < 2:
        raise _Refusal(
            "every member shares one value of the independent variable, so no slope is "
            "identifiable and the law could not be refuted by any measurement")

    for field in _sequence(cohort.get("fixed_fields") or (), "the cohort's fixed_fields"):
        observed = {repr(_fixed_field(descriptor, str(field))) for descriptor in members}
        if len(observed) != 1:
            raise _Refusal(f"the cohort control does not hold {str(field)!r} fixed")
    declared_operation = cohort.get("operation")
    if declared_operation is not None:
        for descriptor in members:
            if _fixed_field(descriptor, "operation") != declared_operation:
                raise _Refusal(
                    f"member {descriptor.get('name')!r} is not the declared "
                    f"{declared_operation!r} cohort operation")

    operands = {name for descriptor in members for name in _inputs_by_name(descriptor)}
    tracking: list[dict[str, Any]] = []
    for operand in sorted(operands):
        allowed = free.get(operand, set())
        shapes = {str(descriptor.get("name")): _shape_of(descriptor, operand)
                  for descriptor in members}
        ranks = {len(shape) for shape in shapes.values()}
        if len(ranks) != 1:
            raise _Refusal(f"operand {operand!r} changes rank across the cohort")
        if allowed == "all":
            continue
        for axis in range(ranks.pop()):
            if axis in allowed:
                continue
            extents = {name: shape[axis] for name, shape in shapes.items()}
            if len(set(extents.values())) == 1:
                continue
            # An axis may still move, but ONLY because the independent variable moved: an im2col
            # weight's contracted extent is the window times the channel depth, so it is a fixed
            # multiple of the depth the fit varies and holding it "fixed" would admit no cohort at
            # all. Anything else moving is a second variable, and a two-variable cohort cannot
            # attribute its slope to either.
            ratios = {Fraction(extents[name], xs[name]) for name in extents}
            if len(ratios) != 1:
                raise _Refusal(
                    f"the cohort control does not hold operand {operand!r} axis {axis} fixed, and "
                    "it does not move in proportion to the declared independent variable either; "
                    "more than one quantity varies across this cohort")
            tracking.append({"operand": operand, "axis": axis,
                             "ratio_to_independent_variable": str(ratios.pop())})

    falsifier = _mapping(members[0]["performance"].get("falsifier"),
                         "the family's falsifier")
    control = falsifier.get("negative_control")
    if not isinstance(control, str) or not control:
        raise _Refusal("the family declares no negative control")

    cohort_record = {
        "negative_control": control,
        "operation": declared_operation,
        "capsules": [str(descriptor.get("name")) for descriptor in members],
        "independent_variable": fit.get("independent_variable"),
        "independent_values": [xs[str(descriptor.get("name"))] for descriptor in members],
        "dependent_metric": metric,
        "fixed_fields": [str(field) for field in (cohort.get("fixed_fields") or ())],
        "axes_tracking_the_independent_variable": tracking,
        "replicates": list(schedule.identities),
        "replicate_source": schedule.source,
        "evidence_lanes": [{"simulator": simulator, "tier": tier} for simulator, tier in lanes],
    }
    return members, cohort_record, {"contract": contract, "lanes": lanes,
                                    "identities": schedule.identities}


def preflight_affine_claim(descriptors: object) -> dict[str, Any]:
    """Validate a frozen affine declaration before a claim-bearing run is admitted.

    Takes the descriptors alone: this family's contract freezes its own replicate identities, so a
    run has no schedule to offer and none is accepted.
    """
    family = _family_of(descriptors)
    try:
        members, cohort, resolved = _validated_declaration(descriptors)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "claim": "PREDICTS",
            "status": REFUSED,
            "declaration": None,
            "cohort": None,
            "replicates": [],
            "expected_identities": [],
            "refusal_reasons": [str(exc)],
        }
    expected = [{"family": family, "capsule": str(descriptor.get("name")),
                 "simulator": simulator, "replicate": replicate, "tier": tier}
                for descriptor in members
                for replicate in resolved["identities"]
                for simulator, tier in resolved["lanes"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "claim": "PREDICTS",
        "status": "READY",
        "declaration": copy.deepcopy(dict(resolved["contract"])),
        "cohort": cohort,
        "replicates": list(resolved["identities"]),
        "expected_identities": expected,
        "refusal_reasons": [],
    }
