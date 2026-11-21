"""Frozen joint semantic cells for independent component cost applicability.

These objects are declarations, not hardware or timing capabilities. Exact joint
cells prevent an unmeasured combination of individually observed axis ranges
from acquiring authority. General repetition/range proofs are separate work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from merlin.analysis.optimization.evaluation.analytical_calibration import (
    _feature_cycles,
    _json_pointer,
    _parse_calibration,
)
from merlin.analysis.perf.bounds.decomposition import ResourceKind, compose_resource_intervals
from merlin.common.digest import canonical_sha256, is_sha256
from merlin.compiler.partition.global_plan import CycleInterval

# ---- component_applicability -------------------------------------------------------------------


def _atom(value):
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise ValueError("applicability requires structural semantic vocabulary")
    return value


def _pairs(values, *, positive):
    if type(values) is not tuple or not values:
        raise ValueError("applicability resource/unit coordinates are unavailable")
    names = []
    for row in values:
        if type(row) is not tuple or len(row) != 2 or type(row[1]) is not int or row[1] < int(positive):
            raise ValueError("applicability coordinates require exact nonnegative integers")
        names.append(_atom(row[0]))
    if names != sorted(set(names)):
        raise ValueError("applicability coordinate membership must be canonical and distinct")


@dataclass(frozen=True)
class ComponentApplicabilityCoordinates:
    hardware_sha256: str
    timer_sha256: str
    accuracy_sha256: str
    input_policy_sha256: str
    unit_meaning_sha256: str
    payload_bytes: tuple[tuple[str, int], ...]
    capacity_bytes: tuple[tuple[str, int], ...]
    live_bytes: tuple[tuple[str, int], ...]
    working_set_bytes: tuple[tuple[str, int], ...]
    tile_shape: tuple[int, ...]
    tile_counts: tuple[int, ...]
    tails: tuple[int, ...]
    streaming: bool
    dependency_depth: int
    reuse_state: str
    reuse_count: int
    chain_length: int

    def verify(self):
        if not all(
            is_sha256(value)
            for value in (
                self.hardware_sha256,
                self.timer_sha256,
                self.accuracy_sha256,
                self.input_policy_sha256,
                self.unit_meaning_sha256,
            )
        ):
            raise ValueError("applicability coordinates omit HW/numeric/input/timer/unit identity")
        _pairs(self.payload_bytes, positive=False)
        _pairs(self.capacity_bytes, positive=True)
        _pairs(self.live_bytes, positive=False)
        _pairs(self.working_set_bytes, positive=False)
        capacity, live, working = map(dict, (self.capacity_bytes, self.live_bytes, self.working_set_bytes))
        if set(capacity) != set(live) or set(capacity) != set(working):
            raise ValueError("applicability omitted a physical resource coordinate")
        if any(live[name] > capacity[name] or working[name] < live[name] for name in capacity):
            raise ValueError("applicability live bytes violate capacity or working-set bounds")
        if type(self.streaming) is not bool or (
            not self.streaming and any(working[name] > capacity[name] for name in capacity)
        ):
            raise ValueError("applicability omitted its streaming/capacity regime")
        vectors = (self.tile_shape, self.tile_counts, self.tails)
        if (
            any(type(vector) is not tuple or not vector for vector in vectors)
            or len({len(vector) for vector in vectors}) != 1
            or any(type(value) is not int or value <= 0 for vector in vectors[:2] for value in vector)
            or any(
                type(tail) is not int or not 0 <= tail < tile
                for tail, tile in zip(self.tails, self.tile_shape, strict=True)
            )
        ):
            raise ValueError("applicability tile/tail coordinates are incomplete")
        if (
            any(type(value) is not int or value < 0 for value in (self.dependency_depth, self.reuse_count))
            or type(self.chain_length) is not int
            or self.chain_length < 1
            or self.reuse_state not in ("fresh", "within_invocation", "cross_invocation")
            or (self.reuse_state == "fresh") != (self.reuse_count == 0)
        ):
            raise ValueError("applicability dependency/reuse/composition state is incomplete")
        return self.sha256

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ComponentApplicabilityStratum:
    id: str
    cells: tuple[ComponentApplicabilityCoordinates, ...]
    held_groups: tuple[str, ...]


@dataclass(frozen=True)
class ComponentApplicabilityDomain:
    strata: tuple[ComponentApplicabilityStratum, ...]
    calibration_groups: tuple[str, ...]

    def verify(self):
        if (
            type(self.strata) is not tuple
            or not self.strata
            or any(type(row) is not ComponentApplicabilityStratum for row in self.strata)
            or type(self.calibration_groups) is not tuple
            or not self.calibration_groups
        ):
            raise ValueError("applicability needs independently frozen strata and calibration groups")
        for group in self.calibration_groups:
            _atom(group)
        if len(set(self.calibration_groups)) != len(self.calibration_groups):
            raise ValueError("applicability calibration groups are duplicated")
        ids, cells, scopes = [], [], []
        for row in self.strata:
            ids.append(_atom(row.id))
            if (
                type(row.cells) is not tuple
                or not row.cells
                or any(type(cell) is not ComponentApplicabilityCoordinates for cell in row.cells)
                or type(row.held_groups) is not tuple
                or not row.held_groups
            ):
                raise ValueError("applicability required stratum omits joint cells or held transfer groups")
            for group in row.held_groups:
                _atom(group)
            if len(set(row.held_groups)) != len(row.held_groups) or set(row.held_groups) & set(self.calibration_groups):
                raise ValueError("applicability held transfer groups leak calibration membership")
            for cell in row.cells:
                cells.append(cell.verify())
                scopes.append(
                    (
                        cell.hardware_sha256,
                        cell.timer_sha256,
                        cell.accuracy_sha256,
                        cell.input_policy_sha256,
                        cell.unit_meaning_sha256,
                        cell.capacity_bytes,
                    )
                )
        if len(set(ids)) != len(ids) or len(set(cells)) != len(cells):
            raise ValueError("applicability joint cell/stratum membership is ambiguous")
        if len(set(scopes)) != 1:
            raise ValueError("applicability cells select different HW/numeric/input/timer/unit/resource scope")
        return self.sha256

    @property
    def sha256(self):
        return canonical_sha256(asdict(self))

    def lookup(self, coordinates):
        """Return an exact stratum or explicit UNKNOWN; never enlarge a cell."""
        self.verify()
        if type(coordinates) is not ComponentApplicabilityCoordinates:
            return {"status": "UNKNOWN", "reason": "independently observed applicability coordinates unavailable"}
        coordinates.verify()
        rows = [row for row in self.strata if any(cell == coordinates for cell in row.cells)]
        if not rows:
            return {"status": "UNKNOWN", "reason": "joint payload/capacity/tiling/tail/dependency/reuse outside domain"}
        return {"status": "IN_DOMAIN", "stratum": rows[0].id, "domain_sha256": self.sha256}


# ---- component_cost ----------------------------------------------------------------------------
# Complete generated-program cost scopes and target-provided feature evidence.
#
# Providers decode emitted instructions and physical resources. This owner only prices
# explicit feature counts with controlled calibration and composes declared regions.

COMPLETE_STAGES = (
    "preparation",
    "allocation",
    "packing",
    "proof",
    "device",
    "readout",
    "reconstruction",
    "replay",
    "dispatch",
    "publication",
    "cleanup",
)


@dataclass(frozen=True)
class ComponentCostScope:
    """An exact timer/accuracy/input regime, with all complete-cost obligations."""

    timer_sha256: str
    accuracy_sha256: str
    input_policy_sha256: str
    stages: tuple[str, ...] = COMPLETE_STAGES

    def __post_init__(self):
        if any(not is_sha256(value) for value in (self.timer_sha256, self.accuracy_sha256, self.input_policy_sha256)):
            raise ValueError("component cost scope requires exact timer, accuracy and input policy")
        if self.stages != COMPLETE_STAGES:
            raise ValueError("component cost scope must include every complete execution stage")

    def to_dict(self):
        return {
            "timer_sha256": self.timer_sha256,
            "accuracy_sha256": self.accuracy_sha256,
            "input_policy_sha256": self.input_policy_sha256,
            "stages": list(self.stages),
        }

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ComponentCostRegion:
    """Feature context for one exclusive or inclusive execution region.

    Inclusive regions replace their contained regions when composing totals. A zero
    region still needs explicit calibrated features whose observed counts are zero.
    """

    id: str
    stages: tuple[str, ...]
    feature_ids: tuple[str, ...]
    context: Mapping[str, Any]
    parent: str | None = None
    accounting: str = "exclusive"

    def __post_init__(self):
        if not self.id or not self.stages or not set(self.stages) <= set(COMPLETE_STAGES):
            raise ValueError("cost region must identify its complete-cost stages")
        if self.accounting not in ("exclusive", "inclusive"):
            raise ValueError("region accounting must be exclusive or inclusive")
        if len(set(self.feature_ids)) != len(self.feature_ids) or not self.feature_ids:
            raise ValueError("cost region requires distinct calibrated feature identifiers")
        if not isinstance(self.context, Mapping):
            raise ValueError("feature context must be a mapping")


@dataclass(frozen=True)
class ComponentFeatureObservation:
    """One exact member/compiler arm, carrying separate cold and warm regions.

    The source provider must verify actual build/run joins and returns complete
    dependency identities; no instruction count is itself a cycle observation.
    """

    compiler_sha256: str
    member_sha256: str
    corpus_sha256: str
    target_sha256: str
    scope_sha256: str
    domain_sha256: str
    executable_sha256: str
    dependencies_sha256: str
    inputs_sha256: str
    evidence_sha256s: tuple[str, ...]
    cold: tuple[ComponentCostRegion, ...]
    warm: tuple[ComponentCostRegion, ...]
    functional_status: str
    artifact_files: tuple[tuple[str, str], ...] = ()
    legality_status: str = "UNKNOWN"

    def __post_init__(self):
        digests = (
            self.compiler_sha256,
            self.member_sha256,
            self.corpus_sha256,
            self.target_sha256,
            self.scope_sha256,
            self.domain_sha256,
            self.executable_sha256,
            self.dependencies_sha256,
            self.inputs_sha256,
            *self.evidence_sha256s,
        )
        if not self.evidence_sha256s or any(not is_sha256(value) for value in digests):
            raise ValueError("component observations must bind every input and execution dependency")
        statuses = ("PASS", "FAIL", "UNKNOWN", "REFUSAL")
        if self.functional_status not in statuses or self.legality_status not in statuses:
            raise ValueError("component observation needs an explicit functional status")


def _price(region, calibration):
    features = {feature.id: feature for feature in calibration.features}
    unknown = set(region.feature_ids) - set(features)
    if unknown:
        return CycleInterval.unknown("unpriced features: " + ",".join(sorted(unknown)))
    resources = {}
    for ident in region.feature_ids:
        feature = features[ident]
        try:
            count = _json_pointer(region.context, feature.pointer)
            if isinstance(count, bool) or not isinstance(count, (float, int)) or not math.isfinite(count) or count < 0:
                raise ValueError("invalid count")
        except (KeyError, TypeError, ValueError):
            return CycleInterval.unknown(f"{region.id}: feature {ident} is UNKNOWN")
        interval = _feature_cycles(feature, float(count), calibration.movement)
        if interval is None:
            return CycleInterval.unknown(f"{region.id}: feature {ident} is outside calibrated domain")
        kind = (
            ResourceKind.COMPUTE
            if feature.kind == "compute"
            else ResourceKind.FIXED
            if feature.kind == "fixed"
            else ResourceKind.MOVEMENT
        )
        prior = resources.get(feature.resource)
        if prior is not None:
            if prior[0] != kind:
                raise ValueError("resource has incompatible calibration kinds")
            interval = CycleInterval(
                prior[1].lo + interval.lo,
                prior[1].hi + interval.hi,
                provenance=prior[1].provenance + interval.provenance,
            )
        resources[feature.resource] = (kind, interval)
    return compose_resource_intervals(
        resources,
        operator=calibration.composition,
        eta=calibration.composition_eta,
        provenance="calibration sha256:" + calibration.document_sha256,
    )


def component_region_accounting(regions, *, stages=COMPLETE_STAGES):
    """Check the existing complete-region ownership independently of cost units.

    This is accounting data, not evidence that a timer or region boundary has
    the declared meaning. Inclusive ancestors suppress their contained regions
    from the sum; repeated stage labels do not imply overlap or zero work.
    """
    if type(stages) is not tuple or stages != COMPLETE_STAGES:
        raise ValueError("component accounting must include every complete execution stage")
    if any(type(region) is not ComponentCostRegion for region in regions):
        raise TypeError("component accounting requires typed regions")
    by_id = {region.id: region for region in regions}
    if len(by_id) != len(regions):
        raise ValueError("cost region identities are duplicated")
    covered, included = set(), []
    for region in regions:
        region.__post_init__()
        ancestry, parent, contained = {region.id}, region.parent, False
        while parent is not None:
            if parent not in by_id or parent in ancestry:
                raise ValueError("cost region containment is missing or cyclic")
            ancestry.add(parent)
            owner = by_id[parent]
            if owner.accounting == "inclusive" and not set(region.stages) <= set(owner.stages):
                raise ValueError("inclusive parent does not cover contained complete-cost stages")
            contained |= owner.accounting == "inclusive"
            parent = owner.parent
        covered.update(region.stages)
        if not contained:
            included.append(region.id)
    return {
        "included": tuple(included),
        "covered": tuple(stage for stage in stages if stage in covered),
        "missing": tuple("missing complete-cost stage " + stage for stage in stages if stage not in covered),
    }


def complete_component_cost(
    observation,
    calibration_document,
    *,
    scope,
    qualified_domains,
    applicability_domain=None,
    applicability_coordinates=None,
):
    """Price both regimes, exposing contained regions without counting them twice."""
    if type(observation) is not ComponentFeatureObservation or type(scope) is not ComponentCostScope:
        raise TypeError("component cost requires typed observation and scope")
    calibration = _parse_calibration(calibration_document, canonical_sha256(calibration_document), "canonical_mapping")
    if observation.target_sha256 != calibration.target_sha256 or observation.scope_sha256 != scope.sha256:
        raise ValueError("component feature scope differs from calibration or timer selection")
    domain_missing = []
    if observation.domain_sha256 not in qualified_domains:
        domain_missing.append("feature domain has not passed held ranking and interval validation")
    if applicability_domain is not None:
        if type(applicability_domain) is not ComponentApplicabilityDomain:
            raise TypeError("component cost requires a typed semantic applicability declaration")
        lookup = applicability_domain.lookup(applicability_coordinates)
        if lookup["status"] != "IN_DOMAIN" or observation.domain_sha256 != applicability_domain.sha256:
            domain_missing.append(lookup.get("reason", "feature domain differs from semantic applicability"))
    if observation.legality_status != "PASS":
        domain_missing.append("complete member legality qualification is " + observation.legality_status)
    if observation.functional_status != "PASS":
        domain_missing.append("complete member functional qualification is " + observation.functional_status)
    reports, totals = {}, {}
    for regime in ("cold", "warm"):
        regions = getattr(observation, regime)
        accounting = component_region_accounting(regions, stages=scope.stages)
        prices = {region.id: _price(region, calibration) for region in regions}
        included = accounting["included"]
        missing = list(domain_missing)
        missing.extend(accounting["missing"])
        missing.extend(reason for ident in included for reason in prices[ident].missing)
        if missing:
            total = CycleInterval.unknown(*dict.fromkeys(missing))
        else:
            # Regions describe ordered complete-program work. Cross-region overlap needs
            # one inclusive calibrated parent; an overlap operator never defaults here.
            total = CycleInterval(
                sum(prices[i].lo for i in included),
                sum(prices[i].hi for i in included),
                provenance=(
                    "complete cost scope sha256:" + scope.sha256,
                    "calibration sha256:" + calibration.document_sha256,
                    *observation.evidence_sha256s,
                ),
            )
        totals[regime] = total
        reports[regime] = {
            "total": total.to_dict(),
            "regions": [
                {
                    "id": region.id,
                    "stages": list(region.stages),
                    "accounting": region.accounting,
                    "parent": region.parent,
                    "contained": region.id not in included,
                    "cycles": prices[region.id].to_dict(),
                }
                for region in regions
            ],
        }
    return totals, {
        "schema": "component_complete_cost_v1",
        "scope_sha256": scope.sha256,
        "domain_sha256": observation.domain_sha256,
        "regimes": reports,
        "promotion": "SCREENING_ONLY",
    }


def validate_complete_cost_report(report):
    """Check public arithmetic and complete stage accounting without private context."""
    if not isinstance(report, Mapping) or set(report) != {
        "schema",
        "scope_sha256",
        "domain_sha256",
        "regimes",
        "promotion",
    }:
        raise ValueError("complete component cost report schema is invalid")
    if (
        report["schema"] != "component_complete_cost_v1"
        or report["promotion"] != "SCREENING_ONLY"
        or not is_sha256(report["scope_sha256"])
        or not is_sha256(report["domain_sha256"])
        or set(report["regimes"]) != {"cold", "warm"}
    ):
        raise ValueError("complete component report identity/regime is invalid")

    def interval(raw):
        if set(raw) != {"lo", "hi", "resolved", "provenance", "missing"} or type(raw["resolved"]) is not bool:
            raise ValueError("complete component interval schema is invalid")
        value = CycleInterval(raw["lo"], raw["hi"], tuple(raw["provenance"]), tuple(raw["missing"]))
        if value.to_dict() != dict(raw) or (
            value.resolved
            and (not math.isfinite(value.lo) or not math.isfinite(value.hi) or not value.provenance or value.missing)
        ):
            raise ValueError("complete component interval evidence is invalid")
        return value

    for regime in report["regimes"].values():
        if not isinstance(regime, Mapping) or set(regime) != {"total", "regions"}:
            raise ValueError("complete component regime schema is invalid")
        rows = regime["regions"]
        if not isinstance(rows, list):
            raise ValueError("complete component regions must be a roster")
        by_id = {}
        for row in rows:
            if set(row) != {"id", "stages", "accounting", "parent", "contained", "cycles"} or row["id"] in by_id:
                raise ValueError("complete component cost region schema is invalid")
            if (
                not row["id"]
                or not row["stages"]
                or not set(row["stages"]) <= set(COMPLETE_STAGES)
                or row["accounting"] not in ("inclusive", "exclusive")
                or type(row["contained"]) is not bool
            ):
                raise ValueError("complete component cost region scope is invalid")
            by_id[row["id"]] = row
        included, covered = [], set()
        for row in rows:
            parent, ancestry, contained = row["parent"], {row["id"]}, False
            while parent is not None:
                if parent not in by_id or parent in ancestry:
                    raise ValueError("complete component region containment is invalid")
                ancestry.add(parent)
                owner = by_id[parent]
                if owner["accounting"] == "inclusive":
                    if not set(row["stages"]) <= set(owner["stages"]):
                        raise ValueError("complete parent omits contained stage scope")
                    contained = True
                parent = owner["parent"]
            if row["contained"] != contained:
                raise ValueError("complete component containment claim differs from ownership")
            covered.update(row["stages"])
            price = interval(row["cycles"])
            if not contained:
                included.append(price)
        total = interval(regime["total"])
        if total.resolved:
            if covered != set(COMPLETE_STAGES) or any(not value.resolved for value in included):
                raise ValueError("complete cost total omits unpriced work")
            if total.lo != sum(v.lo for v in included) or total.hi != sum(v.hi for v in included):
                raise ValueError("complete cost total double counts or omits a region")
    return report
