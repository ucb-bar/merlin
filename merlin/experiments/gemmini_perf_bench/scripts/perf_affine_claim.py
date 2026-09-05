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

from collections.abc import Mapping, Sequence
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
