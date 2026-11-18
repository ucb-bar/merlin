"""Join semantic applicability to independent measured execution and held strata.

Arithmetic and these data objects cannot grant physical roles. The measurement
qualifier executes its fixed source/output/effect/timer controls and owns issuance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.analysis.optimization.evaluation.component_cost import (
    COMPLETE_STAGES,
    ComponentApplicabilityCoordinates,
    ComponentApplicabilityDomain,
    ComponentCostScope,
    ComponentFeatureObservation,
    complete_component_cost,
)
from merlin.analysis.optimization.evaluation.component_screen import (
    qualify_component_screen,
    validate_component_screen_report,
)
from merlin.common.process import invocation_records as invocation_record
from merlin.development.backend.optimization.components.execution import _member_binding
from merlin.development.backend.optimization.components.stage import ComponentVariantSnapshot
from merlin.development.backend.optimization.contracts import StageGateError, document_sha256, mapping_file, sha256_file
from merlin.development.backend.optimization.controls.runtime_qualification import (
    _argument_binding,
    _evidence_file,
    _invocations,
)
from merlin.development.backend.optimization.search import corpus as C
from merlin.development.backend.qualification.components.witnesses import (
    REQUIRED_EXECUTION_EFFECTS,
    ComponentStageWitness,
    verify_component_stage_witness,
)

# ---- component_applicability -------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentApplicabilityObservation:
    coordinates: ComponentApplicabilityCoordinates
    product: Path
    product_sha256: str
    invocations: tuple[Path, ...]


def verify_domain_scope(domain, *, runtime, scope):
    if type(domain) is not ComponentApplicabilityDomain:
        raise StageGateError("component measurement requires an independently frozen applicability domain")
    try:
        domain.verify()
    except ValueError as error:
        raise StageGateError("component applicability declaration is incomplete") from error
    runtime.qualification.context.verify_applicability_domain(domain)
    point = domain.strata[0].cells[0]
    if (point.hardware_sha256, point.timer_sha256, point.accuracy_sha256, point.input_policy_sha256) != (
        runtime.hardware_intake.sha256,
        scope.timer_sha256,
        scope.accuracy_sha256,
        scope.input_policy_sha256,
    ):
        raise StageGateError("component applicability selected different HW/numeric/input/timer scope")
    return domain.sha256


def applicability_observation(*, observation, features, domain, runtime, owner):
    """Reopen an actual derived product; source/HW semantics remain context-owned."""
    if type(observation) is not ComponentApplicabilityObservation:
        raise StageGateError("component features omit actual independently derived applicability coordinates")
    try:
        lookup = domain.lookup(observation.coordinates)
    except ValueError as error:
        raise StageGateError("component applicability coordinates violate resource/shape semantics") from error
    runtime.qualification.context.verify_component_applicability(observation, features)
    if features.domain_sha256 != domain.sha256:
        raise StageGateError("component feature domain differs from its complete semantic applicability")
    product = _evidence_file(observation.product, observation.product_sha256, owner)
    expected = {
        "schema": "merlin.component_applicability_observation.v1",
        "domain_sha256": domain.sha256,
        "coordinates": observation.coordinates.to_dict(),
        "compiler_sha256": features.compiler_sha256,
        "member_sha256": features.member_sha256,
        "executable_sha256": features.executable_sha256,
        "inputs_sha256": features.inputs_sha256,
        "dependencies_sha256": features.dependencies_sha256,
    }
    # Normal JSON encoding converts coordinate tuples to lists.
    if mapping_file(product) != json.loads(json.dumps(expected)):
        raise StageGateError("component applicability product differs from actual source/ELF/input/unit coordinates")
    if not isinstance(observation.invocations, tuple) or not observation.invocations:
        raise StageGateError("component applicability has no actual independent derivation invocations")
    pins, records = {product: observation.product_sha256}, []
    for value in observation.invocations:
        path = _evidence_file(value, sha256_file(value), owner)
        records.append(invocation_record.verify(path))
        pins[path] = sha256_file(path)
    producers = [
        record
        for record in records
        if any(pin["path"] == str(product) and pin["sha256"] == observation.product_sha256 for pin in record["outputs"])
    ]
    if not any(pin["sha256"] == features.executable_sha256 for record in producers for pin in record["inputs"]):
        raise StageGateError("component applicability lacks actual exact executed-ELF derivation")
    return lookup, pins


def qualify_required_applicability_strata(domain, rows, *, calibration_sha256):
    """Statistical arithmetic on matched full-output/timer observations, not issuance."""
    domain.verify()
    strata = {row.id: row for row in domain.strata}
    grouped = {(ident, regime): [] for ident in strata for regime in ("cold", "warm")}
    seen_cells = {key: set() for key in grouped}
    cell_groups = {key: set() for key in grouped}
    for ident, cell, regime, observation in rows:
        if (
            ident not in strata
            or regime not in ("cold", "warm")
            or cell not in {point.sha256 for point in strata[ident].cells}
            or observation.group not in strata[ident].held_groups
            or observation.group in domain.calibration_groups
            or observation.domain != domain.sha256
        ):
            raise StageGateError("applicability held observation leaks or selects another semantic transfer stratum")
        grouped[ident, regime].append(observation)
        seen_cells[ident, regime].add(cell)
        cell_groups[ident, regime].add((cell, observation.group))
    reports = {}
    from merlin.development.backend.optimization.controls.measurement_qualification import (
        _CalibratedPrediction,
    )

    for (ident, regime), observed in grouped.items():
        if {row.group for row in observed} != set(strata[ident].held_groups):
            raise StageGateError("applicability omitted a required held stratum/transfer group in " + regime)
        expected_cells = {point.sha256 for point in strata[ident].cells}
        if seen_cells[ident, regime] != expected_cells:
            raise StageGateError("applicability omitted a required joint cell in " + ident + "/" + regime)
        if cell_groups[ident, regime] != {
            (cell, group) for cell in expected_cells for group in strata[ident].held_groups
        }:
            raise StageGateError("applicability omitted held transfer coverage for a required joint cell")
        report = qualify_component_screen(
            observed, lambda _training: _CalibratedPrediction(), calibration_sha256=calibration_sha256
        )
        report.update(stratum=ident, regime=regime, applicability_cells=sorted(expected_cells))
        validate_component_screen_report(report)
        if not report["exposable"]:
            raise StageGateError("applicability stratum failed ranking/error/coverage: " + ident + "/" + regime)
        reports[ident, regime] = report
    return reports


# ---- component_measurement_protocol ------------------------------------------------------------
# Closed observations consumed by independent measurement qualification.
#
# These data objects identify actual products; constructing them grants no runtime,
# timer or performance authority. The fixed independently prepared context must
# rederive their source/HW/timer and semantic correspondence from actual records.

MEASUREMENT_ROLES = ("feature_provider", "cca_provider", "rtl_executor")

MEASUREMENT_MECHANISMS = (
    "source_to_executable",
    "hardware_timer_scope",
    "complete_stage_accounting",
    "cold_warm_regime",
    "feature_observation",
    "cca_observation",
    "applicability_scope",
)

MEASUREMENT_CASES = tuple(
    f"{mechanism}.{direction}" for mechanism in MEASUREMENT_MECHANISMS for direction in ("positive", "negative")
)


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
        return {
            "corpus_root": str(value.root),
            "manifest_sha256": value.manifest_sha256,
            "capsules_sha256": value.capsules_sha256,
        }
    if type(value) is C.PerformanceCapsule:
        if C.CONTRACTS.exact_tree_record(value.source_dir)["sha256"] != value.source_sha256:
            raise StageGateError("component measurement argument source membership changed")
        return {
            "family": value.family,
            "capsule": value.capsule,
            "source_dir": str(value.source_dir),
            "source_sha256": value.source_sha256,
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise StageGateError("component measurement arguments require closed string keys")
        return {key: measurement_argument_binding(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [measurement_argument_binding(item) for item in value]
    return _argument_binding(value)


def control_binding(control):
    return document_sha256(
        {
            "case_id": control.case_id,
            "arguments": measurement_argument_binding(control.arguments),
            "evidence_root": str(control.evidence_root),
            "stages": control.required_invocation_stages,
        }
    )


def sample_binding(sample):
    sample.variant.verify()
    return document_sha256(
        {
            "id": sample.id,
            "member_sha256": sample.member.source_sha256,
            "variant_sha256": sample.variant.sha256,
            "group": sample.group,
            "applicability_stratum": sample.applicability_stratum,
            "arguments": measurement_argument_binding(sample.arguments),
            "evidence_root": str(sample.evidence_root),
            "stages": sample.required_invocation_stages,
        }
    )


def measurement_binding(observation):
    if type(observation) is not MatchedComponentMeasurement:
        raise StageGateError("measurement binding requires a typed actual observation")
    return document_sha256(measurement_argument_binding(asdict(observation)))


# ---- component_measurement_evidence ------------------------------------------------------------
# Reopen actual source, executable and complete timer observations.


def actual_measurement_product(invocations, *, measurement, measurement_sha256, executable_sha256):
    producers = [
        invocation
        for invocation in invocations
        if any(pin["path"] == str(measurement) and pin["sha256"] == measurement_sha256 for pin in invocation["outputs"])
    ]
    if not producers:
        raise StageGateError("independent timer observation was not produced at an actual invoked boundary")
    if not any(
        pin["sha256"] == executable_sha256
        for invocation in invocations
        if invocation["kind"] == "subprocess"
        for pin in invocation["inputs"]
    ):
        raise StageGateError("independent measured executable was not consumed by an actual runtime process")
    if not any(pin["sha256"] == executable_sha256 for producer in producers for pin in producer["inputs"]):
        raise StageGateError("independent timer producer did not consume the exact executed ELF")


def matched_measurement(
    *, observation, owner, runtime, admission, scope, calibration, variants, applicability_domain, sample=None
):
    """Join observed timers to the same complete source/ELF/output execution.

    The independently prepared context must separately rederive the timer's
    physical semantics; these generic checks establish product correspondence.
    """
    if type(observation) is not MatchedComponentMeasurement:
        raise StageGateError("independent measurement requires a complete typed source/HW/timer observation")
    runtime.qualification.context.verify_measurement(observation)
    features = observation.features
    if type(features) is not ComponentFeatureObservation or not features.artifact_files:
        raise StageGateError("independent measurement omits actual emitted/executed feature artifacts")
    lookup, applicability_pins = applicability_observation(
        observation=observation.applicability,
        features=features,
        domain=applicability_domain,
        runtime=runtime,
        owner=owner,
    )
    if lookup["status"] != "IN_DOMAIN":
        raise StageGateError("independent measured execution is outside its frozen semantic applicability")
    if sample is not None and lookup["stratum"] != sample.applicability_stratum:
        raise StageGateError("held measurement selected another required semantic applicability stratum")
    members = [member for member in admission.corpus.capsules if member.source_sha256 == features.member_sha256]
    if len(members) != 1:
        raise StageGateError("independent measurement selected an unqualified source member")
    member = members[0]
    binding = _member_binding(admission, member)
    source_memberships = {variant.compiler_sha256: variant.membership_sha256 for variant in variants}
    from merlin.development.backend.optimization.contracts import exact_tree_record

    source_memberships[admission.baseline_sha256] = exact_tree_record(admission.baseline)["sha256"]
    if (
        observation.member_binding != binding
        or source_memberships.get(features.compiler_sha256) != observation.candidate_membership_sha256
        or observation.hardware_intake_sha256 != runtime.hardware_intake.sha256
        or observation.scope != scope
        or features.scope_sha256 != scope.sha256
        or features.target_sha256 != admission.target_sha256
        or features.corpus_sha256 != admission.corpus.capsules_sha256
        or features.functional_status != "PASS"
        or features.legality_status != "PASS"
        or not set(REQUIRED_EXECUTION_EFFECTS) <= set(observation.required_effects)
    ):
        raise StageGateError("independent measurement differs from fresh source/HW/complete scope authority")
    if sample is not None and (member != sample.member or features.compiler_sha256 != sample.variant.compiler_sha256):
        raise StageGateError("held measurement selected another source/compiler variant")
    evidence = dict(applicability_pins)
    for path, digest in features.artifact_files:
        evidence[_evidence_file(path, digest, owner)] = digest
    witness = verify_component_stage_witness(
        observation.stage_witness,
        member=binding,
        candidate_sha256=observation.candidate_membership_sha256,
        target_descriptor_sha256=admission.target_sha256,
        frontend=observation.frontend,
        capsule_root=member.source_dir,
        evidence_root=owner,
        required_effects=observation.required_effects,
    )
    elf = [row for row in observation.stage_witness.stages if row.stage == "elf"]
    if len(elf) != 1 or elf[0].sha256 != features.executable_sha256:
        raise StageGateError("independent measured executable differs from complete ordinary verification")
    for row in observation.stage_witness.stages:
        if row.stage != "source":
            evidence[_evidence_file(row.path, row.sha256, owner)] = row.sha256
    records = _invocations(_InvocationOwner(owner))
    invocations = [invocation_record.verify(path) for path, _digest in records]
    measurement = _evidence_file(observation.measurement_record, observation.measurement_record_sha256, owner)
    expected = {
        "schema": "merlin.independent_component_measurement.v1",
        "hardware_intake_sha256": runtime.hardware_intake.sha256,
        "target_sha256": admission.target_sha256,
        "scope_sha256": scope.sha256,
        "compiler_sha256": features.compiler_sha256,
        "member_sha256": member.source_sha256,
        "executable_sha256": features.executable_sha256,
        "inputs_sha256": features.inputs_sha256,
        "dependencies_sha256": features.dependencies_sha256,
        "cold_cycles": observation.cold_cycles,
        "warm_cycles": observation.warm_cycles,
    }
    if mapping_file(measurement) != expected:
        raise StageGateError("independent cycle record differs from the actually observed full timer scope")
    if any(type(cycles) is not int or cycles <= 0 for cycles in (observation.cold_cycles, observation.warm_cycles)):
        raise StageGateError("independent complete cold/warm cycle measurements are unavailable")
    actual_measurement_product(
        invocations,
        measurement=measurement,
        measurement_sha256=observation.measurement_record_sha256,
        executable_sha256=features.executable_sha256,
    )
    for regime in ("cold", "warm"):
        if {stage for region in getattr(features, regime) for stage in region.stages} != set(COMPLETE_STAGES):
            raise StageGateError("independent measurement omits a complete cold/warm execution stage")
    totals, _report = complete_component_cost(
        features,
        calibration,
        scope=scope,
        qualified_domains=(features.domain_sha256,),
        applicability_domain=applicability_domain,
        applicability_coordinates=observation.applicability.coordinates,
    )
    if any(not totals[regime].resolved for regime in ("cold", "warm")):
        raise StageGateError("independent measurement has unpriced complete cold/warm work")
    evidence[measurement] = observation.measurement_record_sha256
    evidence.update(records)
    for row in witness["stages"]:
        if row["stage"] != "source":
            evidence[Path(row["path"])] = row["sha256"]
    return totals, evidence


class _InvocationOwner:
    def __init__(self, evidence_root):
        self.evidence_root = evidence_root
        self.required_invocation_stages = ()


def calibration_evidence(*, runtime, adapter, samples):
    """Require independent physical calibration provenance, not pinned history."""
    from merlin.analysis.optimization.evaluation.calibration_bundle import prepare_phase2_calibration

    pins = dict(runtime.source_pins)
    if adapter not in pins or sha256_file(adapter) != pins[adapter]:
        raise StageGateError("measurement calibration is outside independently prepared membership")
    runtime.qualification.context.verify_measurement_calibration(adapter, samples)
    prepared = prepare_phase2_calibration(adapter)
    if prepared.get("status") != "ready" or not isinstance(prepared.get("calibration"), dict):
        raise StageGateError("independent controlled measurement calibration is unavailable")
    evidence = {adapter: pins[adapter]}
    for row in prepared.get("evidence_files", []):
        path, digest = Path(row["path"]), row["sha256"]
        if pins.get(path) != digest or path.resolve() != path or path.is_symlink() or sha256_file(path) != digest:
            raise StageGateError("measurement calibration evidence is outside independent source/provenance authority")
        evidence[path] = digest
    return prepared["calibration"], evidence
