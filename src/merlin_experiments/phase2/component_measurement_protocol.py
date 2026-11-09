"""Closed observations consumed by independent measurement qualification.

These data objects identify actual products; constructing them grants no runtime,
timer or performance authority. The fixed independently prepared context must
rederive their source/HW/timer and semantic correspondence from actual records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.perf.component_cost import ComponentCostScope, ComponentFeatureObservation
from merlin_experiments.phase1.component_witness import ComponentStageWitness

from . import corpus as C
from .component_applicability import ComponentApplicabilityObservation
from .component_runtime_qualification import _argument_binding
from .component_variants import ComponentVariantSnapshot
from .contracts import StageGateError, document_sha256

MEASUREMENT_ROLES = ("feature_provider", "cca_provider", "rtl_executor")
MEASUREMENT_MECHANISMS = (
    "source_to_executable", "hardware_timer_scope", "complete_stage_accounting",
    "cold_warm_regime", "feature_observation", "cca_observation", "applicability_scope",
)
MEASUREMENT_CASES = tuple(f"{mechanism}.{direction}" for mechanism in MEASUREMENT_MECHANISMS
                          for direction in ("positive", "negative"))


@dataclass(frozen=True)
class ComponentMeasurementControl:
    case_id: str
    arguments: dict[str, dict]
    evidence_root: Path
    required_invocation_stages: tuple[str, ...]


@dataclass(frozen=True)
class ComponentScreenSample:
    id: str
    member: C.PerformanceCapsule
    variant: ComponentVariantSnapshot
    group: str
    arguments: dict[str, dict]
    evidence_root: Path
    required_invocation_stages: tuple[str, ...]
    applicability_stratum: str | None = None


@dataclass(frozen=True)
class MatchedComponentMeasurement:
    features: ComponentFeatureObservation
    stage_witness: ComponentStageWitness
    member_binding: dict
    frontend: str
    candidate_membership_sha256: str
    required_effects: tuple[str, ...]
    hardware_intake_sha256: str
    scope: ComponentCostScope
    cold_cycles: int
    warm_cycles: int
    measurement_record: Path
    measurement_record_sha256: str
    applicability: ComponentApplicabilityObservation | None = None


def measurement_argument_binding(value):
    if type(value) is ComponentCostScope:
        return {"complete_cost_scope": value.to_dict()}
    if type(value) is C.FrozenPerformanceCorpus:
        C.verify_frozen_performance_corpus(value)
        return {"corpus_root": str(value.root), "manifest_sha256": value.manifest_sha256,
                "capsules_sha256": value.capsules_sha256}
    if type(value) is C.PerformanceCapsule:
        if C.CONTRACTS.exact_tree_record(value.source_dir)["sha256"] != value.source_sha256:
            raise StageGateError("component measurement argument source membership changed")
        return {"family": value.family, "capsule": value.capsule,
                "source_dir": str(value.source_dir), "source_sha256": value.source_sha256}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise StageGateError("component measurement arguments require closed string keys")
        return {key: measurement_argument_binding(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [measurement_argument_binding(item) for item in value]
    return _argument_binding(value)


def control_binding(control):
    return document_sha256({
        "case_id": control.case_id, "arguments": measurement_argument_binding(control.arguments),
        "evidence_root": str(control.evidence_root), "stages": control.required_invocation_stages,
    })


def sample_binding(sample):
    sample.variant.verify()
    return document_sha256({
        "id": sample.id, "member_sha256": sample.member.source_sha256,
        "variant_sha256": sample.variant.sha256, "group": sample.group,
        "applicability_stratum": sample.applicability_stratum,
        "arguments": measurement_argument_binding(sample.arguments),
        "evidence_root": str(sample.evidence_root), "stages": sample.required_invocation_stages,
    })


def measurement_binding(observation):
    if type(observation) is not MatchedComponentMeasurement:
        raise StageGateError("measurement binding requires a typed actual observation")
    return document_sha256(measurement_argument_binding(asdict(observation)))
