"""Conservative complete-plan comparison under an original measurement owner.

Units are preserved: host nanoseconds never become target cycles. The required
original verifier owns runtime/timer qualification, current input/state/cache
and emission joins. This leaf supplies conditional arithmetic and freshness,
not legality, placement permission, timing authority or an authoring grant.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

from merlin.common.digest import is_sha256
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

from .component_cost import ComponentCostScope, ComponentFeatureObservation, component_region_accounting
from .term import UNKNOWN, Bounds, PerformanceTerm, Provenance, Validity

_REGIMES = ("cold", "warm")
_COMPARISON_RESULT = object()


def _bounded_json(value, maximum):
    chunks, size = [], 0
    try:
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > maximum:
                raise ValueError("offload cost evidence exceeds its byte budget")
            chunks.append(chunk)
    except (TypeError, RecursionError) as error:
        raise ValueError("offload cost evidence requires bounded JSON data") from error
    return "".join(chunks)


def _term(term):
    if (
        type(term) is not PerformanceTerm
        or type(term.bounds) is not Bounds
        or type(term.provenance) is not Provenance
        or type(term.validity) is not Validity
        or type(term.unit) is not str
        or not term.unit.strip()
    ):
        raise TypeError("offload costs require original unit-bearing PerformanceTerm values")
    term.__post_init__()
    term.provenance.__post_init__()
    term.validity.__post_init__()
    term.bounds.__post_init__()
    for value in (term.value, term.bounds.lower, term.bounds.upper):
        if value is UNKNOWN:
            continue
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("offload cost values and known bounds must be finite and positive")
    return term.to_dict()


def _inputs(query):
    original, candidate, scope, stamp = (query[key] for key in ("original", "candidate", "scope", "stamp"))
    if (
        type(scope) is not ComponentCostScope
        or type(stamp) is not AnalysisStamp
        or type(stamp.context) is not PipelineContext
        or not is_sha256(stamp.source_sha256)
        or type(stamp.ir_level) is not str
        or not stamp.ir_level.strip()
    ):
        raise ValueError("offload costs require the original complete scope and current source/context stamp")
    scope.__post_init__()
    stamp.context.__post_init__()
    if query["regime"] not in _REGIMES or type(query["regime"]) is not str:
        raise ValueError("offload cost policy must explicitly select cold or warm")
    if not callable(query["verify_original"]):
        raise ValueError("offload costs require the original live measurement verifier")
    observations, costs, accounting, regions = {}, {}, {}, 0
    for arm, observation in (("original", original), ("candidate", candidate)):
        if type(observation) is not ComponentFeatureObservation:
            raise TypeError("offload costs require typed complete observations, not screening reports")
        observation.__post_init__()
        if observation.scope_sha256 != scope.sha256:
            raise ValueError("offload observation differs from the original timer/numerical/input scope")
        files = observation.artifact_files
        if (
            type(files) is not tuple
            or not files
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or not row[0]
                or not is_sha256(row[1])
                for row in files
            )
            or observation.executable_sha256 not in {digest for _, digest in files}
        ):
            raise ValueError("offload observations must identify their actual executable artifacts")
        observations[arm] = dict(vars(observation))
        observations[arm].update(artifact_files=files)
        accounting[arm] = {}
        for regime in _REGIMES:
            rows = getattr(observation, regime)
            if type(rows) is not tuple:
                raise TypeError("offload complete regions require an immutable roster")
            regions += len(rows)
            if regions > query["max_regions"]:
                raise ValueError("offload complete regions exceed their analysis budget")
            accounting[arm][regime] = component_region_accounting(rows, stages=scope.stages)
            observations[arm][regime] = [dict(vars(row)) for row in rows]
        terms = query[arm + "_costs"]
        if not isinstance(terms, Mapping) or set(terms) != set(_REGIMES):
            raise ValueError("offload costs require exactly the original cold and warm regimes")
        costs[arm] = {regime: _term(terms[regime]) for regime in _REGIMES}
    for key in ("member_sha256", "corpus_sha256", "target_sha256", "scope_sha256", "inputs_sha256"):
        if getattr(original, key) != getattr(candidate, key):
            raise ValueError("offload observations compare different original " + key)
    units = {terms[regime].unit for terms in (query["original_costs"], query["candidate_costs"]) for regime in _REGIMES}
    if len(units) != 1:
        raise ValueError("offload costs must use one exact unit; implicit conversions are unavailable")
    record = dict(
        observations=observations, costs=costs, scope=scope.to_dict(), stamp=asdict(stamp), regime=query["regime"]
    )
    return record, accounting, units.pop()


def _assess(query, expected=None):
    maximum = query["max_evidence_bytes"]
    record, accounting, unit = _inputs(query)
    snapshot = _bounded_json(record, maximum)
    if expected is not None and snapshot != expected:
        raise ValueError("offload source, complete costs or original context changed; replan")
    arguments = {
        key: value
        for key, value in query.items()
        if key not in ("max_regions", "max_evidence_bytes", "verify_original")
    }

    def verify():
        if query["verify_original"](**arguments) is not None:
            raise ValueError("original offload verifier must raise on failure and return None on success")
        current, _, _ = _inputs(query)
        if _bounded_json(current, maximum) != snapshot:
            raise ValueError("offload source, complete costs or original context changed during verification")

    verify()
    missing = [
        "current " + name + " context is unknown"
        for name, value in vars(query["stamp"].context).items()
        if value is None
    ]
    for arm in ("original", "candidate"):
        observation = query[arm]
        for gate in ("functional_status", "legality_status"):
            if getattr(observation, gate) != "PASS":
                missing.append(arm + " " + gate + " is " + getattr(observation, gate))
        for regime in _REGIMES:
            missing.extend(arm + " " + regime + ": " + reason for reason in accounting[arm][regime]["missing"])
    regime = query["regime"]
    terms = [query[arm + "_costs"][regime] for arm in ("original", "candidate")]
    for arm, term in zip(("original", "candidate"), terms, strict=True):
        if term.is_unknown or not term.bounds.known:
            missing.append(arm + " complete " + regime + " cost is UNKNOWN or lacks complete bounds")
        elif term.provenance.kind not in ("measured", "calibrated"):
            missing.append(arm + " complete cost lacks measured or calibrated provenance")
    status = (
        "REFUSAL"
        if any(
            getattr(query[arm], gate) in ("FAIL", "REFUSAL")
            for arm in ("original", "candidate")
            for gate in ("functional_status", "legality_status")
        )
        else "UNRESOLVED"
    )
    if not missing:
        before, after = terms
        # Same conservative dominance used by component screening, preserving
        # the original unit rather than relabeling a duration as CycleInterval.
        status = (
            "OPPORTUNITY"
            if after.bounds.upper < before.bounds.lower
            else ("REGRESSION" if after.bounds.lower > before.bounds.upper else "TIE_OR_OVERLAP")
        )
    verify()
    bindings = {}
    for arm in ("original", "candidate"):
        observation = query[arm]
        bindings[arm] = {
            key: value
            for key, value in vars(observation).items()
            if key.endswith("sha256") or key == "evidence_sha256s"
        }
    result = dict(
        schema="merlin.offload_profitability.v1",
        choose_candidate=status == "OPPORTUNITY",
        status=status,
        regime=regime,
        unit=unit,
        stamp=asdict(query["stamp"]),
        scope=query["scope"].to_dict(),
        costs=record["costs"],
        accounting=accounting,
        observations=bindings,
        input_identity=hashlib.sha256(snapshot.encode()).hexdigest(),
        missing=missing,
        scope_note="conditional complete-cost selection only; original verifier owns timing/evidence authority; "
        "legality, required placement and actual emission conformance remain independent",
    )
    return _bounded_json(result, maximum), snapshot


@dataclass(frozen=True)
class OffloadCostDecision:
    """Revalidated conditional data, never an admission or timing authority."""

    _record: str = field(repr=False)
    _check: Callable[[], str] = field(repr=False, compare=False)
    _construction: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self._construction is not _COMPARISON_RESULT or not callable(self._check):
            raise ValueError("offload decision data must come from the live complete-cost comparison")

    def revalidate(self):
        self.__post_init__()
        if self._check() != self._record:
            raise ValueError("offload cost decision changed; replan")

    def record(self):
        self.revalidate()
        return json.loads(self._record)

    @property
    def choose_candidate(self):
        return self.record()["choose_candidate"]

    @property
    def status(self):
        return self.record()["status"]

    @property
    def stamp(self):
        record = self.record()["stamp"]
        return AnalysisStamp(record["source_sha256"], PipelineContext(**record["context"]), record["ir_level"])


def compare_complete_offload_cost(
    *,
    original,
    candidate,
    original_costs,
    candidate_costs,
    scope,
    regime,
    stamp,
    verify_original,
    max_regions=4096,
    max_evidence_bytes=4 << 20,
):
    """Compare two original-owned complete observations with no unit conversion.

    ``verify_original`` receives the exact observations, term maps, complete
    scope, regime and current AnalysisStamp as keyword arguments before/after
    comparison and again on revalidation. It must reopen qualified actual
    source/runtime/cache/input/timer/emission evidence, raise on failure, and
    return None. A permissive validator or screening JSON is no qualification.
    The caller separately checks legal plans, required coverage and final output.
    """
    if (
        type(max_regions) is not int
        or not 0 < max_regions <= 1 << 20
        or type(max_evidence_bytes) is not int
        or not 0 < max_evidence_bytes <= 64 << 20
    ):
        raise ValueError("offload costs require explicit bounded analysis controls")
    query = dict(
        original=original,
        candidate=candidate,
        original_costs=original_costs,
        candidate_costs=candidate_costs,
        scope=scope,
        regime=regime,
        stamp=stamp,
        verify_original=verify_original,
        max_regions=max_regions,
        max_evidence_bytes=max_evidence_bytes,
    )
    record, snapshot = _assess(query)
    return OffloadCostDecision(record, lambda: _assess(query, snapshot)[0], _COMPARISON_RESULT)
