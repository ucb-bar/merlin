"""Complete generated-program cost scopes and target-provided feature evidence.

Providers decode emitted instructions and physical resources. This owner only prices
explicit feature counts with controlled calibration and composes declared regions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_sha256
from merlin.perf.analytical_resources import compose_resource_intervals
from merlin.perf.component_applicability import ComponentApplicabilityDomain
from merlin.perf.decompose import ResourceKind
from merlin.perf.phase2_analytical_provider import _feature_cycles, _json_pointer, _parse_calibration
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

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


def complete_component_cost(observation, calibration_document, *, scope, qualified_domains,
                            applicability_domain=None, applicability_coordinates=None):
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
        by_id = {region.id: region for region in regions}
        if len(by_id) != len(regions):
            raise ValueError("cost region identities are duplicated")
        covered, prices, included = set(), {}, []
        for region in regions:
            ancestry, parent = {region.id}, region.parent
            contained = False
            while parent is not None:
                if parent not in by_id or parent in ancestry:
                    raise ValueError("cost region containment is missing or cyclic")
                ancestry.add(parent)
                owner = by_id[parent]
                if owner.accounting == "inclusive" and not set(region.stages) <= set(owner.stages):
                    raise ValueError("inclusive parent does not cover contained complete-cost stages")
                contained |= owner.accounting == "inclusive"
                parent = owner.parent
            prices[region.id] = _price(region, calibration)
            covered.update(region.stages)
            if not contained:
                included.append(region.id)
        missing = list(domain_missing)
        missing.extend("missing complete-cost stage " + stage for stage in scope.stages if stage not in covered)
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
